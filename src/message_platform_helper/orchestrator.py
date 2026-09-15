"""Top-level multi-agent orchestrator."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .agents import AgentRegistry, build_default_agent_registry
from .config import ConfigCatalog, Settings, load_platform_config, load_settings
from .application.security import TenantContext, permission_policy_from_config
from .decision import DecisionEngine, build_default_decision_engine, to_reasoning_decision
from .decision import KnowledgePolicy, WorkflowRouter
from .infrastructure.observability import (
    RequestMetrics,
    TraceContext,
    elapsed_ms,
    get_observer,
    log_request_completed,
    update_observation,
)
from .llm import LLMClient, build_llm, describe_llm
from .memory import MemoryManager, build_memory_store, memory_from_dict
from .models import AgentResult, AssistantRequest, HelperResponse, JsonDict, KnowledgeChunk, deep_merge, to_jsonable
from .platform import PlatformGateway
from .rag.embedding import build_embedding_provider
from .rate_limit import CounterStore, build_counter_store
from .react import AgentContext, RuleBasedAgent
from .tools import ToolRegistry
from .workflow import WorkflowExecutor, WorkflowRegistry, build_default_workflow_registry, build_workflow_registry_from_config


@dataclass
class TaskSlice:
    agent: str
    workflow: str
    text: str
    payload: JsonDict
    need_retrieval: bool = False
    search_query: str = ""


@dataclass
class MessagePlatformHelper:
    settings: Settings
    llm: LLMClient
    memory_manager: MemoryManager
    knowledge_base: object
    platform: PlatformGateway
    counter_store: CounterStore
    config_catalog: ConfigCatalog | None = None
    rag_service: object | None = None
    agent_registry: AgentRegistry | None = None
    tool_registry: ToolRegistry | None = None
    workflow_registry: WorkflowRegistry | None = None
    workflow_executor: WorkflowExecutor | None = None
    decision_engine: DecisionEngine | None = None

    def __post_init__(self) -> None:
        if self.agent_registry is None:
            self.agent_registry = build_default_agent_registry(self.counter_store)
        if self.tool_registry is None:
            self.tool_registry = ToolRegistry()
        if self.config_catalog and self.config_catalog.policies and self.tool_registry is not None:
            self.tool_registry.permission_policy = permission_policy_from_config(self.config_catalog.policies)
        if self.rag_service is None:
            self.rag_service = _build_rag_service(self.knowledge_base)
        if self.workflow_registry is None:
            self.workflow_registry = (
                build_workflow_registry_from_config(self.config_catalog.workflows)
                if self.config_catalog and self.config_catalog.workflows
                else build_default_workflow_registry()
            )
        if self.workflow_executor is None:
            self.workflow_executor = WorkflowExecutor(self.agent_registry, self.workflow_registry)
        if self.decision_engine is None:
            knowledge_policy = _knowledge_policy_from_config(self.config_catalog)
            workflow_router = _workflow_router_from_config(self.config_catalog)
            self.decision_engine = build_default_decision_engine(
                self.llm,
                agent_registry=self.agent_registry,
                tool_registry=self.tool_registry,
                workflow_registry=self.workflow_registry,
                knowledge_policy=knowledge_policy,
                workflow_router=workflow_router,
                intent_prompt=_intent_prompt_from_config(self.config_catalog),
            )

    @classmethod
    def from_env(cls) -> "MessagePlatformHelper":
        settings = load_settings()
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        database_url = settings.rag_database_url or _rag_database_url()
        llm = build_llm(settings)
        memory_store = build_memory_store(database_url, settings.redis_url)
        from .config import load_rag_config

        rag_config = load_rag_config(settings.config_dir)
        embedding_provider = build_embedding_provider(rag_config)
        kb = _build_knowledge_base(database_url, embedding_provider)
        platform = PlatformGateway(
            platform_base_url=settings.platform_base_url,
            mock_database_url=database_url,
            headers=settings.platform_headers or {},
        )
        counter_store = build_counter_store(database_url, settings.redis_url)
        config_catalog = load_platform_config(settings.config_dir)
        rag_service = _build_rag_service(kb, rag_config)
        return cls(
            settings=settings,
            llm=llm,
            memory_manager=MemoryManager(memory_store, llm),
            knowledge_base=kb,
            platform=platform,
            counter_store=counter_store,
            config_catalog=config_catalog,
            rag_service=rag_service,
        )

    def handle(self, request: AssistantRequest, on_step=None) -> HelperResponse:
        trace = TraceContext.from_request(request)
        observer = get_observer()
        with observer.propagate(trace, request), observer.start_request(trace, request) as observation:
            try:
                response = self._handle(request, on_step=on_step, trace=trace)
            except Exception as exc:
                update_observation(
                    observation,
                    level="ERROR",
                    status_message=str(exc),
                    metadata={"error_type": type(exc).__name__},
                )
                observer.score_current_trace(
                    "task_success",
                    0.0,
                    data_type="NUMERIC",
                    comment="Request raised an exception before a HelperResponse was produced.",
                )
                observer.flush()
                raise
            update_observation(
                observation,
                output={
                    "ok": response.ok,
                    "trace_id": response.trace_id,
                    "request_id": response.request_id,
                    "results": [{"agent": result.agent, "ok": result.ok} for result in response.results],
                },
                metadata={
                    "intent": response.decision.intent,
                    "workflow": response.decision.workflow,
                    "selected_agents": response.decision.selected_agents,
                    "retrieved_chunks": len(response.retrieved),
                    "metrics": response.metrics,
                },
                level="DEFAULT" if response.ok else "ERROR",
            )
            observer.score_current_trace(
                "task_success",
                1.0 if response.ok else 0.0,
                data_type="NUMERIC",
                comment="1 when all selected agents completed without error-level issues.",
            )
            observer.score_current_trace("request_latency_ms", response.metrics.get("latency_ms", 0.0), data_type="NUMERIC")
            observer.flush()
            return response

    def _handle(self, request: AssistantRequest, on_step=None, trace: TraceContext | None = None) -> HelperResponse:
        trace = trace or TraceContext.from_request(request)
        tenant_context = TenantContext.from_request(request)
        metrics = RequestMetrics()
        storage_session_id = _tenant_scoped_session_id(request.tenant_id, request.session_id)
        memory = self.memory_manager.load(storage_session_id)
        memory.profile.user_id = request.user_id or memory.profile.user_id
        memory.profile.tenant_id = request.tenant_id or memory.profile.tenant_id
        memory.profile.locale = request.locale or memory.profile.locale
        memory = self.memory_manager.remember_request(memory, request.text, request.payload)

        if self.decision_engine is None:
            raise RuntimeError("Decision engine is not configured.")
        started = time.perf_counter()
        decision_result = self.decision_engine.decide(request, memory)
        metrics.model_time_ms += elapsed_ms(started)
        decision = to_reasoning_decision(decision_result)
        task_slices = _task_slices_for_request(request, decision)
        if task_slices:
            _apply_task_slices_to_decision(decision, task_slices)
        retrieved: List[KnowledgeChunk] = []
        if not task_slices and (decision.need_retrieval or "knowledge" in decision.selected_agents):
            if self.rag_service is None:
                raise RuntimeError("RAG service is not configured.")
            started = time.perf_counter()
            retrieved = self.rag_service.retrieve(decision.search_query or request.text, limit=5, tenant_context=tenant_context)
            metrics.rag_time_ms += elapsed_ms(started)

        context = AgentContext(
            request=request,
            memory=memory,
            retrieved=retrieved,
            llm=self.llm,
            platform=self.platform,
            tool_registry=self.tool_registry,
            rag_service=self.rag_service,
            tenant_context=tenant_context,
            on_step=on_step,
        )
        if self.workflow_executor is None:
            raise RuntimeError("Workflow executor is not configured.")
        if task_slices:
            results, retrieved = self._execute_task_slices(task_slices, context, tenant_context, metrics)
        else:
            started = time.perf_counter()
            results = self.workflow_executor.execute(decision.workflow, decision.selected_agents, context)
            metrics.tool_time_ms += elapsed_ms(started)
        if not results and decision.request_type == "chat":
            memory = self._hydrate_memory_from_runs(memory)
            started = time.perf_counter()
            answer = self.llm.answer(request.text, system_prompt=_chat_system_prompt(memory))
            metrics.model_time_ms += elapsed_ms(started)
            results = [
                AgentResult(
                    agent="chat",
                    ok=bool(answer.get("ok", True)),
                    output=answer,
                )
            ]

        issues = [issue for result in results for issue in result.issues]
        next_actions = [action for result in results for action in result.next_actions]
        ok = all(result.ok for result in results) and not any(issue.get("severity") == "error" for issue in issues)
        metrics.finish()
        response = HelperResponse(
            ok=ok,
            session_id=request.session_id,
            decision=decision,
            memory=_memory_for_display(memory, request.session_id),
            retrieved=retrieved,
            results=results,
            issues=issues,
            next_actions=next_actions,
            trace_id=trace.trace_id,
            request_id=trace.request_id,
            conversation_id=trace.conversation_id,
            metrics=metrics.as_dict(),
        )
        self.memory_manager.remember_response(
            memory,
            response_summary=self._response_summary(results),

            facts_patch=deep_merge(
                {"lastAgents": decision.selected_agents, "lastOk": ok},
                _operation_history_patch(request, decision, results),
            ),
        )
        self._save_run(request, response)
        log_request_completed(
            trace,
            metrics,
            {
                "ok": ok,
                "intent": decision.intent,
                "workflow": decision.workflow,
                "agents": decision.selected_agents,
                "retrieved_chunks": len(retrieved),
            },
        )
        return response

    def _execute_task_slices(
        self,
        task_slices: List[TaskSlice],
        base_context: AgentContext,
        tenant_context: TenantContext,
        metrics: RequestMetrics,
    ) -> tuple[List[AgentResult], List[KnowledgeChunk]]:
        from contextvars import copy_context
        from concurrent.futures import ThreadPoolExecutor

        retrieved: List[KnowledgeChunk] = []
        prepared: List[tuple[TaskSlice, AgentContext]] = []
        for task in task_slices:
            task_retrieved: List[KnowledgeChunk] = []
            if task.need_retrieval or task.agent == "knowledge":
                if self.rag_service is None:
                    raise RuntimeError("RAG service is not configured.")
                started = time.perf_counter()
                task_retrieved = self.rag_service.retrieve(task.search_query or task.text, limit=5, tenant_context=tenant_context)
                metrics.rag_time_ms += elapsed_ms(started)
                retrieved.extend(task_retrieved)
            task_request = AssistantRequest(
                text=task.text,
                session_id=base_context.request.session_id,
                user_id=base_context.request.user_id,
                tenant_id=base_context.request.tenant_id,
                locale=base_context.request.locale,
                payload=task.payload,
                dry_run=base_context.request.dry_run,
                request_id=base_context.request.request_id,
                trace_id=base_context.request.trace_id,
            )
            prepared.append(
                (
                    task,
                    AgentContext(
                        request=task_request,
                        memory=base_context.memory,
                        retrieved=task_retrieved,
                        llm=base_context.llm,
                        platform=base_context.platform,
                        tool_registry=base_context.tool_registry,
                        rag_service=base_context.rag_service,
                        tenant_context=base_context.tenant_context,
                        scratch={},
                        on_step=base_context.on_step,
                    ),
                )
            )

        # RAG retrieval above stays serial on the main thread: the postgres
        # session behind rag_service is not thread-safe. Agent execution is
        # parallel because each task has its own scratch and only touches
        # thread-safe boundaries (LLM, platform HTTP, pre-retrieved chunks).
        def run_task(task_context: AgentContext, task: TaskSlice) -> List[AgentResult]:
            started = time.perf_counter()
            try:
                executor = self.workflow_executor
                if executor is None:
                    raise RuntimeError("Workflow executor is not configured.")
                return executor.execute(task.workflow, [task.agent], task_context)
            finally:
                metrics.tool_time_ms += elapsed_ms(started)

        results: List[AgentResult] = []
        max_workers = min(len(prepared), 4)
        if max_workers <= 1 or len(prepared) == 1:
            for task, task_context in prepared:
                results.extend(run_task(task_context, task))
            return results, retrieved
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(copy_context().run, run_task, task_context, task)
                for task, task_context in prepared
            ]
            for future in futures:
                results.extend(future.result())
        return results, retrieved

    def load_memory_for_display(self, session_id: str, tenant_id: str = ""):
        storage_session_id = _tenant_scoped_session_id(tenant_id, session_id)
        memory = self.memory_manager.load(storage_session_id)
        list_runs = getattr(self.memory_manager.store, "list_runs", None)
        if not callable(list_runs):
            return _memory_for_display(memory, session_id)
        recent_messages = _messages_from_runs(list_runs(storage_session_id, limit=20))
        if recent_messages:
            memory.recent_messages = recent_messages[-self.memory_manager.policy.recent_message_limit :]
        return _memory_for_display(memory, session_id)

    def list_memory_sessions_for_display(self, tenant_id: str = "") -> List[JsonDict]:
        list_sessions = getattr(self.memory_manager.store, "list_sessions", None)
        if not callable(list_sessions):
            return []
        return [
            item
            for item in (_session_for_display(session, tenant_id) for session in list_sessions())
            if item
        ]

    def ingest_knowledge(self, title: str, content: str, source: str = "manual", tags: List[str] | None = None, replace: bool = True) -> JsonDict:
        before = self.knowledge_base.count()
        chunks = self.knowledge_base.ingest_text(title, content, source=source, tags=tags, replace=replace)
        after = self.knowledge_base.count()
        return {
            "ok": True,
            "database_url": self.knowledge_base.database_url,
            "before_chunks": before,
            "after_chunks": after,
            "delta_chunks": after - before,
            "chunks": [to_jsonable(chunk) for chunk in chunks],
            "ingestion": self.knowledge_base.last_ingestion.to_dict() if self.knowledge_base.last_ingestion else None,
            "stats": self.knowledge_base.stats(),
        }

    def ingest_knowledge_file(self, path: str, tags: List[str] | None = None, replace: bool = True, document_key: str = "") -> JsonDict:
        before = self.knowledge_base.count()
        chunks = self.knowledge_base.ingest_file(Path(path), tags=tags, replace=replace, document_key=document_key)
        after = self.knowledge_base.count()
        return {
            "ok": True,
            "database_url": self.knowledge_base.database_url,
            "before_chunks": before,
            "after_chunks": after,
            "delta_chunks": after - before,
            "chunks": [to_jsonable(chunk) for chunk in chunks],
            "ingestion": self.knowledge_base.last_ingestion.to_dict() if self.knowledge_base.last_ingestion else None,
            "stats": self.knowledge_base.stats(),
        }

    def search_knowledge(self, query: str, limit: int = 5, tags: List[str] | None = None) -> JsonDict:
        if self.rag_service is None:
            raise RuntimeError("RAG service is not configured.")
        chunks = self.rag_service.retrieve(query, limit=limit, tags=tags)
        return {
            "ok": True,
            "database_url": self.knowledge_base.database_url,
            "query": query,
            "chunks": [to_jsonable(chunk) for chunk in chunks],
            "total_chunks": self.knowledge_base.count(),
        }

    def knowledge_stats(self) -> JsonDict:
        return {"ok": True, **self.knowledge_base.stats()}

    def health(self) -> JsonDict:
        return {
            "ok": True,
            "service": "message-platform-helper",
            "llm": describe_llm(self.llm, self.settings),
            "observability": get_observer().status(),
            "dataDir": str(self.settings.data_dir),
            "configDir": str(self.settings.config_dir) if self.settings.config_dir else "",
            "database": {"type": "postgresql", "urlConfigured": bool(self.knowledge_base.database_url)},
        }

    def _agent_sequence(self, names: List[str]) -> List[RuleBasedAgent]:
        if self.agent_registry is None:
            return []
        return self.agent_registry.resolve(names)

    def _response_summary(self, results: List[AgentResult]) -> str:
        parts = [_result_text(result.agent, result.ok, result.output, result.issues) for result in results]
        return "\n\n".join(part for part in parts if part)

    def _save_run(self, request: AssistantRequest, response: HelperResponse) -> None:
        store = self.memory_manager.store
        if hasattr(store, "save_run"):
            store.save_run(_tenant_scoped_session_id(request.tenant_id, request.session_id), to_jsonable(request), to_jsonable(response))

    def _hydrate_memory_from_runs(self, memory) -> object:
        if memory.facts.get("operationHistory"):
            return memory
        list_runs = getattr(self.memory_manager.store, "list_runs", None)
        if not callable(list_runs):
            return memory
        history: List[JsonDict] = []
        for run in list_runs(memory.session_id, limit=20):
            response = run.get("response") or {}
            request = run.get("request") or {}
            results = [
                AgentResult(
                    agent=str(result.get("agent") or ""),
                    ok=bool(result.get("ok", True)),
                    output=dict(result.get("output") or {}),
                )
                for result in response.get("results") or []
            ]
            decision = response.get("decision") or {}
            history.extend(
                _operation_history_patch(
                    AssistantRequest(text=str(request.get("text") or ""), payload=dict(request.get("payload") or {})),
                    decision,
                    results,
                ).get("operationHistory") or []
            )
        if history:
            memory.facts = deep_merge(memory.facts, {"operationHistory": history[-20:]})
            return self.memory_manager.store.save(memory)
        return memory


class _EmptyKnowledgeBase:
    database_url = ""
    embedding_provider = None
    last_ingestion = None

    def count(self) -> int:
        return 0

    def stats(self) -> JsonDict:
        return {"database_url": "", "documents": 0, "chunks": 0}

    def ingest_text(self, title: str, content: str, source: str = "manual", tags: List[str] | None = None, replace: bool = True) -> list[KnowledgeChunk]:
        return []

    def ingest_file(self, path: Path, tags: List[str] | None = None, replace: bool = True) -> list[KnowledgeChunk]:
        return []


class _EmptyRagService:
    def retrieve(self, query: str, limit: int = 5, **kwargs) -> list[KnowledgeChunk]:
        return []


def _has_sqlalchemy() -> bool:
    try:
        import sqlalchemy  # noqa: F401
    except ImportError:
        return False
    return True


def _rag_database_url() -> str:
    if not _has_sqlalchemy():
        return ""
    from .rag.db import rag_database_url

    return rag_database_url()


def _build_knowledge_base(database_url: str, embedding_provider) -> object:
    if not database_url or not _has_sqlalchemy():
        return _EmptyKnowledgeBase()
    from .rag import KnowledgeBase, seed_default_knowledge

    kb = KnowledgeBase(database_url, embedding_provider=embedding_provider)
    try:
        seed_default_knowledge(kb)
    except RuntimeError:
        pass
    return kb


def _build_rag_service(knowledge_base: object, rag_config: dict | None = None) -> object:
    database_url = str(getattr(knowledge_base, "database_url", "") or "")
    embedding_provider = getattr(knowledge_base, "embedding_provider", None)
    if not database_url or not _has_sqlalchemy():
        return _EmptyRagService()
    from .rag import build_postgres_rag_service
    from .rag.db import build_session_factory

    session = build_session_factory(database_url)()
    return build_postgres_rag_service(session, embedding_provider, config=rag_config)


def _chat_system_prompt(memory) -> str:
    return (
        "Answer the user's chat message using the provided conversation memory. "
        "你是企业消息平台助手。请用用户当前语言直接回答。"
        "如果用户询问之前发生过什么，只能依据 memory.summary、memory.facts 和 memory.recent_messages。"
        "当相关记忆已提供时，不要声称没有记忆；当信息不足时说明缺口。"
        "不要编造未提供的历史、配置、邮箱、模板或执行结果。"
        f"memory={_memory_prompt_view(memory)}"
    )


def _memory_prompt_view(memory) -> JsonDict:
    payload = to_jsonable(memory)
    facts = dict(payload.get("facts") or {})
    if isinstance(facts.get("operationHistory"), list):
        facts["operationHistory"] = facts["operationHistory"][-20:]
    return {
        "session_id": payload.get("session_id") or payload.get("sessionId"),
        "summary": payload.get("summary") or "",
        "facts": facts,
        "recent_messages": list(payload.get("recent_messages") or payload.get("recentMessages") or [])[-12:],
    }


def _task_slices_for_request(request: AssistantRequest, decision) -> List[TaskSlice]:
    payload = dict(request.payload or {})
    raw_slices = (decision.metadata or {}).get("taskSlices") or (decision.metadata or {}).get("task_slices") or []
    if not isinstance(raw_slices, list) or len(raw_slices) <= 1:
        return []
    tasks: List[TaskSlice] = []
    for raw in raw_slices:
        if not isinstance(raw, dict):
            continue
        agent = _task_agent(raw)
        if agent not in {"knowledge", "template", "channel_config"}:
            continue
        text = str(raw.get("text") or request.text or "")
        workflow = str(raw.get("workflow") or _default_workflow_for_agent(agent))
        search_query = str(raw.get("searchQuery") or raw.get("search_query") or text)
        tasks.append(
            TaskSlice(
                agent=agent,
                workflow=workflow,
                text=text,
                payload=_payload_for_task(payload, raw, agent),
                need_retrieval=bool(raw.get("needRag") if "needRag" in raw else raw.get("need_rag", agent == "knowledge")),
                search_query=search_query,
            )
        )
    return tasks if len(tasks) > 1 else []


def _apply_task_slices_to_decision(decision, task_slices: List[TaskSlice]) -> None:
    selected_agents = [task.agent for task in task_slices]
    decision.selected_agents = selected_agents
    decision.need_retrieval = any(task.need_retrieval for task in task_slices)
    decision.workflow = "message_platform_workflow"
    decision.request_type = "action" if any(task.agent != "knowledge" for task in task_slices) else "query"
    decision.domain = "implementation"
    decision.operation = "execute"
    decision.intent = "workflow"
    decision.search_query = next((task.search_query for task in task_slices if task.search_query), decision.search_query)
    metadata = dict(decision.metadata or {})
    metadata["taskSlices"] = [
        {
            "agent": task.agent,
            "workflow": task.workflow,
            "text": task.text,
            "needRetrieval": task.need_retrieval,
            "searchQuery": task.search_query,
        }
        for task in task_slices
    ]
    metadata["multiAgentSplit"] = True
    decision.metadata = metadata


def _task_agent(raw: JsonDict) -> str:
    selected = raw.get("selectedAgents") or raw.get("selected_agents") or []
    if isinstance(selected, list) and selected:
        return str(selected[0])
    domain = str(raw.get("payloadDomain") or raw.get("payload_domain") or raw.get("domain") or "")
    if domain == "knowledge":
        return "knowledge"
    if domain == "template":
        return "template"
    if domain == "channel":
        return "channel_config"
    return ""


def _default_workflow_for_agent(agent: str) -> str:
    return {
        "knowledge": "knowledge_answer",
        "template": "template_workflow",
        "channel_config": "implementation_workflow",
    }.get(agent, "")


def _payload_for_task(payload: JsonDict, raw: JsonDict, agent: str) -> JsonDict:
    payload_domain = str(raw.get("payloadDomain") or raw.get("payload_domain") or "")
    if agent == "template":
        scoped = dict(payload.get("template") or {})
        for key in ("template", "templates", "scope", "targetLanguage", "sourceLanguage", "businessObject", "documentId", "groupCode"):
            if key in payload and key not in scoped:
                scoped[key] = payload[key]
        return {"template": scoped} if scoped or payload_domain == "template" else payload
    if agent == "channel_config":
        scoped = dict(payload.get("channel") or {})
        for key in ("email", "configName", "mailHost", "mailPort", "mailUsername", "username", "mailPwd", "password", "verifyUser", "receiver", "test"):
            if key in payload and key not in scoped:
                scoped[key] = payload[key]
        return {"channel": scoped} if scoped or payload_domain == "channel" else payload
    return payload


def _tenant_scoped_session_id(tenant_id: str, session_id: str) -> str:
    tenant = str(tenant_id or "").strip()
    session = str(session_id or "default").strip() or "default"
    return f"{tenant}::{session}" if tenant else session


def _unscoped_session_id(storage_session_id: str, tenant_id: str = "") -> str:
    prefix = f"{tenant_id}::" if tenant_id else ""
    if prefix and storage_session_id.startswith(prefix):
        return storage_session_id[len(prefix) :]
    return storage_session_id.split("::", 1)[1] if "::" in storage_session_id else storage_session_id


def _memory_for_display(memory, session_id: str):
    payload = to_jsonable(memory)
    payload["session_id"] = session_id
    payload["sessionId"] = session_id
    return memory_from_dict(payload)


def _session_for_display(session: JsonDict, tenant_id: str) -> JsonDict | None:
    storage_session_id = str(session.get("sessionId") or session.get("session_id") or "")
    if not storage_session_id:
        return None
    if tenant_id:
        prefix = f"{tenant_id}::"
        if not storage_session_id.startswith(prefix):
            return None
        display_session_id = storage_session_id[len(prefix) :]
    else:
        if "::" in storage_session_id:
            return None
        display_session_id = storage_session_id
    return {**session, "sessionId": display_session_id, "session_id": display_session_id}


def _messages_from_runs(runs: List[JsonDict]) -> List[JsonDict]:
    messages: List[JsonDict] = []
    for run in reversed(runs):
        request = run.get("request") or {}
        response = run.get("response") or {}
        created_at = run.get("created_at") or now_from_response(response)
        request_text = str(request.get("text") or "").strip()
        if request_text:
            messages.append({"role": "user", "text": request_text, "at": created_at})
        response_text = _response_text(response)
        if response_text:
            messages.append({"role": "assistant", "text": response_text, "at": created_at})
    return messages


def _response_text(response: JsonDict) -> str:
    results = response.get("results") or []
    issues = response.get("issues") or []
    parts = []
    for result in results:
        parts.append(
            _result_text(
                str(result.get("agent") or ""),
                bool(result.get("ok", True)),
                dict(result.get("output") or {}),
                list(result.get("issues") or []),
            )
        )
    parts.extend(_issue_text(issue) for issue in issues)
    return "\n\n".join(part for part in parts if part)


def _result_text(agent: str, ok: bool, output: JsonDict, issues: List[JsonDict] | None = None) -> str:
    for key in ("answer", "message", "text", "content"):
        value = output.get(key)
        if value:
            return str(value)
    summary = output.get("summary")
    details = _human_details(output)
    if summary and details:
        return f"{summary}\n{details}"
    if summary:
        return str(summary)
    issue_lines = [_issue_text(issue) for issue in issues or []]
    if issue_lines:
        return "\n".join(line for line in issue_lines if line)
    return f"{agent}: {'completed' if ok else 'failed'}" if agent else ""


def _human_details(output: JsonDict) -> str:
    lines = []
    for item in output.get("translatedTemplates") or []:
        if isinstance(item, dict):
            label = item.get("templateName") or item.get("templateCode") or item.get("templateId")
            if label:
                lines.append(f"- {label}")
    for key in ("emailConfig", "channelSaveResponse", "channelTestResponse", "citations"):
        value = output.get(key)
        if value:
            lines.append(f"{key}: {json.dumps(_compact_value(value), ensure_ascii=False, default=str)}")
    return "\n".join(lines)


def _issue_text(issue: JsonDict) -> str:
    message = str(issue.get("message") or "").strip()
    if not message:
        return ""
    severity = str(issue.get("severity") or "info")
    return f"{severity}: {message}"


def now_from_response(response: JsonDict) -> object:
    return (response.get("memory") or {}).get("updated_at") or (response.get("memory") or {}).get("updatedAt") or time.time()


def _operation_history_patch(request: AssistantRequest, decision, results: List[AgentResult]) -> JsonDict:
    entries: List[JsonDict] = []
    for result in results:
        if result.agent == "chat":
            continue
        output = dict(result.output or {})
        entries.append(
            {
                "agent": result.agent,
                "ok": result.ok,
                "workflow": _decision_value(decision, "workflow"),
                "intent": _decision_value(decision, "intent"),
                "requestText": request.text,
                "requestPayload": _compact_value(request.payload),
                "summary": output.get("summary") or "",
                "details": _compact_value(output),
            }
        )
    return {"operationHistory": entries[-20:]} if entries else {}


def _decision_value(decision, key: str) -> object:
    if isinstance(decision, dict):
        return decision.get(key) or decision.get(key.replace("_", ""))
    return getattr(decision, key, "")


def _compact_value(value: object, depth: int = 0) -> object:
    if depth >= 4:
        return _brief(value)
    if isinstance(value, dict):
        result: JsonDict = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text in {"platformPayload", "saveResponse", "templateListResponse", "executionTrace", "content", "raw", "templateContent"}:
                continue
            result[key_text] = _compact_value(item, depth + 1)
        return result
    if isinstance(value, list):
        return [_compact_value(item, depth + 1) for item in value[:10]]
    return _brief(value)


def _brief(value: object) -> object:
    if isinstance(value, str) and len(value) > 300:
        return value[:300]
    return value


def request_from_payload(payload: JsonDict) -> AssistantRequest:
    return AssistantRequest(
        text=str(payload.get("text") or ""),
        session_id=str(payload.get("sessionId") or payload.get("session_id") or "default"),
        user_id=str(payload.get("userId") or payload.get("user_id") or "anonymous"),
        tenant_id=str(payload.get("tenantId") or payload.get("tenant_id") or ""),
        locale=str(payload.get("locale") or "zh_CN"),
        payload=dict(payload.get("payload") or {}),
        dry_run=bool(payload.get("dryRun", payload.get("dry_run", True))),
        request_id=str(payload.get("requestId") or payload.get("request_id") or ""),
        trace_id=str(payload.get("traceId") or payload.get("trace_id") or ""),
    )


def _knowledge_policy_from_config(config: ConfigCatalog | None) -> KnowledgePolicy | None:
    rules = (config.policies.get("rag_intents") if config else None) or {}
    request_type_rules = (config.policies.get("rag_request_types") if config else None) or {}
    if rules or request_type_rules:
        return KnowledgePolicy(
            intent_rules={str(key): bool(value) for key, value in rules.items()},
            request_type_rules={str(key): bool(value) for key, value in request_type_rules.items()},
        )
    return None


def _workflow_router_from_config(config: ConfigCatalog | None) -> WorkflowRouter | None:
    routes = (config.policies.get("intent_workflows") if config else None) or {}
    return WorkflowRouter(intent_workflows={str(key): str(value) for key, value in routes.items()}) if routes else None


def _intent_prompt_from_config(config: ConfigCatalog | None) -> str:
    return str((config.prompts.get("intent_classifier") if config else "") or "")
