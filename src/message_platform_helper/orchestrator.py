"""Top-level multi-agent orchestrator."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .agents import AgentRegistry, build_default_agent_registry
from .config import ConfigCatalog, Settings, load_platform_config, load_settings
from .application.security import TenantContext, permission_policy_from_config
from .decision import DecisionEngine, build_default_decision_engine, to_reasoning_decision
from .decision import KnowledgePolicy, WorkflowRouter
from .infrastructure.observability import RequestMetrics, TraceContext, elapsed_ms, log_request_completed
from .llm import LLMClient, build_llm, describe_llm
from .memory import MemoryManager, build_memory_store
from .models import AgentResult, AssistantRequest, HelperResponse, JsonDict, KnowledgeChunk, to_jsonable
from .platform import PlatformGateway
from .rag import KnowledgeBase, RagService, build_postgres_rag_service, seed_default_knowledge
from .rag.db import build_session_factory, rag_database_url
from .rag.embedding import build_embedding_provider
from .rate_limit import CounterStore, build_counter_store
from .react import AgentContext, ReActAgent
from .tools import ToolRegistry
from .workflow import WorkflowExecutor, WorkflowRegistry, build_default_workflow_registry, build_workflow_registry_from_config


@dataclass
class MessagePlatformHelper:
    settings: Settings
    llm: LLMClient
    memory_manager: MemoryManager
    knowledge_base: KnowledgeBase
    platform: PlatformGateway
    counter_store: CounterStore
    config_catalog: ConfigCatalog | None = None
    rag_service: RagService | None = None
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
            session = build_session_factory(self.knowledge_base.database_url)()
            self.rag_service = build_postgres_rag_service(session, self.knowledge_base.embedding_provider)
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
            )

    @classmethod
    def from_env(cls) -> "MessagePlatformHelper":
        settings = load_settings()
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        database_url = settings.rag_database_url or rag_database_url()
        llm = build_llm(settings)
        memory_store = build_memory_store(database_url, settings.redis_url)
        embedding_provider = build_embedding_provider()
        kb = KnowledgeBase(database_url, embedding_provider=embedding_provider)
        seed_default_knowledge(kb)
        platform = PlatformGateway(
            platform_base_url=settings.platform_base_url,
            business_agent_url=settings.business_agent_url,
            mock_database_url=database_url,
            headers=settings.platform_headers or {},
        )
        counter_store = build_counter_store(database_url, settings.redis_url)
        config_catalog = load_platform_config(settings.config_dir)
        session = build_session_factory(database_url)()
        rag_service = build_postgres_rag_service(session, embedding_provider)
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

    def handle(self, request: AssistantRequest) -> HelperResponse:
        trace = TraceContext.from_request(request)
        tenant_context = TenantContext.from_request(request)
        metrics = RequestMetrics()
        memory = self.memory_manager.load(request.session_id)
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
        retrieved: List[KnowledgeChunk] = []
        if decision.need_retrieval or "knowledge" in decision.selected_agents:
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
        )
        if self.workflow_executor is None:
            raise RuntimeError("Workflow executor is not configured.")
        started = time.perf_counter()
        results = self.workflow_executor.execute(decision.workflow, decision.selected_agents, context)
        metrics.tool_time_ms += elapsed_ms(started)
        if not results and decision.request_type == "chat":
            started = time.perf_counter()
            answer = self.llm.answer(request.text)
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
            memory=memory,
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
            facts_patch={"lastAgents": decision.selected_agents, "lastOk": ok},
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

    def ingest_knowledge_file(self, path: str, tags: List[str] | None = None, replace: bool = True) -> JsonDict:
        before = self.knowledge_base.count()
        chunks = self.knowledge_base.ingest_file(Path(path), tags=tags, replace=replace)
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
            "dataDir": str(self.settings.data_dir),
            "configDir": str(self.settings.config_dir) if self.settings.config_dir else "",
            "database": {"type": "postgresql", "urlConfigured": bool(self.knowledge_base.database_url)},
        }

    def _agent_sequence(self, names: List[str]) -> List[ReActAgent]:
        if self.agent_registry is None:
            return []
        return self.agent_registry.resolve(names)

    def _response_summary(self, results: List[AgentResult]) -> str:
        parts = []
        for result in results:
            status = "ok" if result.ok else "failed"
            parts.append(f"{result.agent}:{status}")
        return ", ".join(parts)

    def _save_run(self, request: AssistantRequest, response: HelperResponse) -> None:
        store = self.memory_manager.store
        if hasattr(store, "save_run"):
            store.save_run(request.session_id, to_jsonable(request), to_jsonable(response))


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
