"""Reusable ReAct-style agent runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .application.security import TenantContext
from .llm import LLMClient
from .models import AgentResult, AgentStep, AssistantRequest, ConversationMemory, JsonDict, KnowledgeChunk, Tool
from .platform import PlatformGateway
from .tools import ToolRegistry


@dataclass
class AgentContext:
    request: AssistantRequest
    memory: ConversationMemory
    retrieved: List[KnowledgeChunk]
    llm: LLMClient
    platform: PlatformGateway
    tool_registry: ToolRegistry | None = None
    rag_service: Any | None = None
    tenant_context: TenantContext | None = None
    scratch: JsonDict = field(default_factory=dict)


class ReActAgent:
    name = "base"

    def tools(self, context: AgentContext) -> Dict[str, Tool]:
        raise NotImplementedError

    def plan(self, context: AgentContext) -> List[JsonDict]:
        raise NotImplementedError

    def finalize(self, context: AgentContext, observations: List[JsonDict], steps: List[AgentStep]) -> AgentResult:
        output = observations[-1] if observations else {}
        return AgentResult(agent=self.name, ok=True, output=output, steps=steps)

    def run(self, context: AgentContext) -> AgentResult:
        tools = self._available_tools(context)
        observations: List[JsonDict] = []
        steps: List[AgentStep] = []
        try:
            for planned in self.plan(context):
                tool_name = str(planned.get("tool") or "")
                tool = tools[tool_name]
                tool_payload = planned.get("input") or {}
                if context.tool_registry is not None:
                    context.tool_registry.authorize(tool_name, context.tenant_context)
                observation = tool.run(tool_payload, context)
                observations.append(observation)
                steps.append(
                    AgentStep(
                        agent=self.name,
                        thought=str(planned.get("thought") or "Need a tool observation before continuing."),
                        action=tool_name,
                        observation=str(observation.get("summary") or observation.get("status") or "ok"),
                        status="ok" if observation.get("ok", True) else "error",
                        data=observation,
                    )
                )
            return self.finalize(context, observations, steps)
        except Exception as exc:
            steps.append(
                AgentStep(
                    agent=self.name,
                    thought="Agent stopped because a tool raised an exception.",
                    action="error",
                    observation=str(exc),
                    status="error",
                )
            )
            return AgentResult(
                agent=self.name,
                ok=False,
                issues=[{"code": f"{self.name}.error", "message": str(exc), "severity": "error"}],
                steps=steps,
            )

    def _available_tools(self, context: AgentContext) -> Dict[str, Tool]:
        local_tools = self.tools(context)
        if context.tool_registry is None:
            return local_tools
        for tool in local_tools.values():
            context.tool_registry.register_tool(tool, replace_existing=True)
        registry_tools = {tool.name: tool for tool in context.tool_registry.resolve(context.tool_registry.names())}
        return {**local_tools, **registry_tools}
