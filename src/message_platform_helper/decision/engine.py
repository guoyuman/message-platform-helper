"""Decision engine composition."""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..agents import AgentRegistry
from ..llm import LLMClient
from ..models import AssistantRequest, ConversationMemory, DecisionResult, ReasoningDecision
from ..tools import ToolRegistry
from ..workflow import WorkflowRegistry, build_default_workflow_registry
from .classifier import IntentClassifier, LLMIntentClassifier
from .policies import KnowledgePolicy
from .routers import AgentSelector, ToolRouter, WorkflowRouter
from .taxonomy import LEGACY_INTENT_AXES, legacy_intent_for, normalize_axes


@dataclass
class DecisionEngine:
    classifier: IntentClassifier
    knowledge_policy: KnowledgePolicy
    workflow_router: WorkflowRouter
    agent_selector: AgentSelector
    tool_router: ToolRouter
    workflow_registry: WorkflowRegistry

    def decide(self, request: AssistantRequest, memory: ConversationMemory) -> DecisionResult:
        intent = self.classifier.classify(request, memory)
        axes_payload = dict(intent.metadata)
        axes_payload["intent"] = intent.intent
        if intent.request_type:
            axes_payload["request_type"] = intent.request_type
        if intent.domain:
            axes_payload["domain"] = intent.domain
        if intent.operation:
            axes_payload["operation"] = intent.operation
        axes = normalize_axes(axes_payload, request.text)
        legacy_intent = (
            legacy_intent_for(axes["request_type"], axes["domain"], axes["operation"])
            if intent.intent in LEGACY_INTENT_AXES
            else intent.intent
        )
        metadata = dict(intent.metadata)
        metadata.update(axes)
        intent = replace(
            intent,
            intent=legacy_intent,
            request_type=axes["request_type"],
            domain=axes["domain"],
            operation=axes["operation"],
            metadata=metadata,
        )
        workflow = self.workflow_router.route(intent, request, memory)
        selected_agents = self.agent_selector.select(workflow, intent, self.workflow_registry)
        selected_tools = self.tool_router.select(workflow, intent, self.workflow_registry)
        need_rag = self.knowledge_policy.needs_rag(intent, request, memory)
        search_query = str(intent.metadata.get("searchQuery") or intent.metadata.get("search_query") or request.text)
        missing_slots = list(intent.metadata.get("missingSlots") or intent.metadata.get("missing_slots") or [])
        return DecisionResult(
            intent=intent.intent,
            request_type=intent.request_type,
            domain=intent.domain,
            operation=intent.operation,
            workflow=workflow,
            need_rag=need_rag,
            need_tools=bool(selected_tools),
            selected_agents=selected_agents,
            selected_tools=selected_tools,
            confidence=intent.confidence,
            search_query=search_query,
            rationale=intent.rationale,
            missing_slots=missing_slots,
            metadata=dict(intent.metadata),
        )


def build_default_decision_engine(
    llm: LLMClient,
    *,
    agent_registry: AgentRegistry | None = None,
    tool_registry: ToolRegistry | None = None,
    workflow_registry: WorkflowRegistry | None = None,
    knowledge_policy: KnowledgePolicy | None = None,
    workflow_router: WorkflowRouter | None = None,
    intent_prompt: str = "",
) -> DecisionEngine:
    workflows = workflow_registry or build_default_workflow_registry()
    return DecisionEngine(
        classifier=LLMIntentClassifier(llm, system_prompt=intent_prompt),
        knowledge_policy=knowledge_policy or KnowledgePolicy(),
        workflow_router=workflow_router or WorkflowRouter(),
        agent_selector=AgentSelector(registry=agent_registry),
        tool_router=ToolRouter(registry=tool_registry),
        workflow_registry=workflows,
    )


def to_reasoning_decision(decision: DecisionResult) -> ReasoningDecision:
    return ReasoningDecision(
        need_retrieval=decision.need_rag,
        selected_agents=decision.selected_agents,
        confidence=decision.confidence,
        rationale=decision.rationale,
        request_type=decision.request_type,
        domain=decision.domain,
        operation=decision.operation,
        missing_slots=decision.missing_slots,
        search_query=decision.search_query,
        intent=decision.intent,
        workflow=decision.workflow,
        need_tools=decision.need_tools,
        selected_tools=decision.selected_tools,
        metadata=decision.metadata,
    )


__all__ = ["DecisionEngine", "build_default_decision_engine", "to_reasoning_decision"]
