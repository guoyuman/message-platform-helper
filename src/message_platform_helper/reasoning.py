"""Backward-compatible reasoning adapter backed by the Decision Engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .decision import AgentSelector, LLMIntentClassifier, ToolRouter, WorkflowRouter, to_reasoning_decision
from .decision.engine import DecisionEngine
from .decision.policies import KnowledgePolicy
from .decision.routers import DEFAULT_AGENT_ALIASES
from .llm import LLMClient, RuleBasedLLMClient
from .models import AssistantRequest, ConversationMemory, ReasoningDecision
from .workflow import build_default_workflow_registry


KNOWN_AGENTS = {"template", "channel_config", "knowledge"}


@dataclass
class ReasoningLayer:
    llm: LLMClient

    def decide(self, request: AssistantRequest, memory: ConversationMemory) -> ReasoningDecision:
        engine = DecisionEngine(
            classifier=LLMIntentClassifier(self.llm),
            knowledge_policy=KnowledgePolicy(),
            workflow_router=WorkflowRouter(),
            agent_selector=AgentSelector(registry=None),
            tool_router=ToolRouter(registry=None),
            workflow_registry=build_default_workflow_registry(),
        )
        decision = engine.decide(request, memory)
        return to_reasoning_decision(decision)


def normalize_agents(agents: List[str]) -> List[str]:
    result: List[str] = []
    for item in agents:
        name = DEFAULT_AGENT_ALIASES.get(str(item), str(item))
        if name in KNOWN_AGENTS and name not in result:
            result.append(name)
    return result


def looks_like_knowledge_request(text: str) -> bool:
    classifier = LLMIntentClassifier(RuleBasedLLMClient())
    decision = classifier.classify(AssistantRequest(text=text), ConversationMemory(session_id="probe"))
    return decision.intent in {"knowledge_query", "error_code"}
