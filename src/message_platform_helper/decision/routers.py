"""Decision routers for workflow, agents, and tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

from ..agents import AgentRegistry
from ..models import AssistantRequest, ConversationMemory, IntentResult
from ..tools import ToolRegistry
from ..workflow import WorkflowRegistry
from .taxonomy import agent_hints_for, tool_hints_for, workflow_for


DEFAULT_INTENT_WORKFLOWS = {
    "knowledge_query": "knowledge_answer",
    "template_config": "template_workflow",
    "business_config": "implementation_workflow",
    "translation": "template_workflow",
    "template_translation_sync": "template_workflow",
    "implementation": "implementation_workflow",
    "error_code": "knowledge_answer",
    "workflow": "message_platform_workflow",
    "summary": "general_chat",
    "casual_chat": "general_chat",
}


DEFAULT_AGENT_ALIASES = {
    "template_agent": "template",
    "business": "business_config",
    "business_message": "business_config",
    "business_message_config": "business_config",
    "channel": "channel_config",
    "email_channel": "channel_config",
    "strategy": "send_strategy",
    "send_policy": "send_strategy",
    "manual": "knowledge",
    "help": "knowledge",
    "guide": "knowledge",
    "rag": "knowledge",
    "knowledge_base": "knowledge",
}


@dataclass
class WorkflowRouter:
    intent_workflows: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_INTENT_WORKFLOWS))

    def route(self, intent: IntentResult, request: AssistantRequest, memory: ConversationMemory) -> str:
        configured = intent.metadata.get("workflow") or intent.metadata.get("selectedWorkflow") or intent.metadata.get("selected_workflow")
        if configured:
            return str(configured)
        if intent.request_type or intent.domain or intent.operation:
            return workflow_for(intent.request_type, intent.domain, intent.operation)
        return self.intent_workflows.get(intent.intent, "general_chat")


@dataclass
class AgentSelector:
    registry: AgentRegistry | None = None
    aliases: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_AGENT_ALIASES))
    default_agents: List[str] = field(default_factory=lambda: ["business_config"])

    def select(self, workflow_name: str, intent: IntentResult, workflows: WorkflowRegistry) -> List[str]:
        if intent.request_type == "query":
            return self._normalize(["knowledge"])
        if intent.request_type == "chat":
            return []
        hinted = self._normalize(intent.metadata.get("selectedAgents") or intent.metadata.get("selected_agents") or [])
        if hinted:
            return hinted
        taxonomy_agents = self._normalize(agent_hints_for(intent.request_type, intent.domain))
        if taxonomy_agents:
            return taxonomy_agents
        workflow_agents = self._normalize(workflows.resolve_agents(workflow_name))
        if workflow_agents:
            return workflow_agents
        if intent.intent == "casual_chat":
            return []
        return self._normalize(self.default_agents)

    def _normalize(self, names: Iterable[str]) -> List[str]:
        available_names = self.registry.names() if self.registry else []
        available = set(available_names) if available_names else None
        result: List[str] = []
        for item in names:
            name = self.aliases.get(str(item), str(item))
            if available is not None and name not in available:
                continue
            if name not in result:
                result.append(name)
        return result


@dataclass
class ToolRouter:
    registry: ToolRegistry | None = None

    def select(self, workflow_name: str, intent: IntentResult, workflows: WorkflowRegistry) -> List[str]:
        if intent.request_type in {"chat", "query"}:
            selected = tool_hints_for(intent.request_type, intent.domain)
        else:
            selected = intent.metadata.get("selectedTools") or intent.metadata.get("selected_tools") or tool_hints_for(intent.request_type, intent.domain) or workflows.resolve_tools(workflow_name)
        result: List[str] = []
        available = set(self.registry.names()) if self.registry else None
        for item in selected or []:
            name = str(item)
            if available is not None and name not in available:
                continue
            if name not in result:
                result.append(name)
        return result


__all__ = ["AgentSelector", "ToolRouter", "WorkflowRouter", "DEFAULT_AGENT_ALIASES", "DEFAULT_INTENT_WORKFLOWS"]
