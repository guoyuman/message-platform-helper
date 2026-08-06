"""Reusable plan-and-execute agent runner with failure replanning.

The agent separates planning from execution: ``plan()`` produces the initial
step list ({thought, tool, input}), ``run()`` executes it deterministically
and records every step as an ``AgentStep`` audit record. When a tool
observation reports failure (``ok`` falsy), ``replan()`` is consulted once
per failed step so the agent can recover (retry with different parameters,
switch to a fallback tool, or correct course) before the remaining plan
continues. ``max_steps`` bounds the total step budget so a persistently
failing recovery cannot loop forever.
"""

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

    # Hard cap on executed steps per run. Defaults to ``2 * len(plan) + 1``
    # so there is always room for recovery steps while a persistently
    # failing replan can never loop forever.
    max_steps: int | None = None

    def tools(self, context: AgentContext) -> Dict[str, Tool]:
        raise NotImplementedError

    def plan(self, context: AgentContext) -> List[JsonDict]:
        raise NotImplementedError

    def replan(self, context: AgentContext, observations: List[JsonDict], steps: List[AgentStep]) -> JsonDict | None:
        """Recovery hook called after a tool observation reports failure.

        Receives the accumulated observations and audit steps so the agent
        can inspect what failed (``steps[-1]`` is the failed step) and decide
        the next action: retry with different parameters, switch to a
        fallback tool, or return ``None`` to continue with the remaining
        plan (the historical behavior). Recovery steps are executed
        immediately and stamped with ``replanned: True`` in the audit trail.
        """
        return None

    def finalize(self, context: AgentContext, observations: List[JsonDict], steps: List[AgentStep]) -> AgentResult:
        output = observations[-1] if observations else {}
        return AgentResult(agent=self.name, ok=True, output=output, steps=steps)

    def run(self, context: AgentContext) -> AgentResult:
        tools = self._available_tools(context)
        observations: List[JsonDict] = []
        steps: List[AgentStep] = []
        plan = list(self.plan(context))
        cap = self.max_steps if self.max_steps is not None else len(plan) * 2 + 1
        idx = 0
        try:
            while len(steps) < cap:
                # Recovery step first: a failed observation gives replan() a
                # chance to correct course before the remaining plan runs.
                planned: JsonDict | None = None
                recovering = False
                if observations and not observations[-1].get("ok", True):
                    planned = self.replan(context, observations, steps)
                    recovering = planned is not None
                if planned is None:
                    if idx >= len(plan):
                        break
                    planned = plan[idx]
                    idx += 1
                tool_name = str(planned.get("tool") or "")
                tool = tools.get(tool_name)
                if tool is None:
                    raise KeyError(f"Unknown tool {tool_name!r} planned by agent {self.name}.")
                tool_payload = planned.get("input") or {}
                if context.tool_registry is not None:
                    context.tool_registry.authorize(tool_name, context.tenant_context)
                observation = tool.run(tool_payload, context)
                observations.append(observation)
                step_data = dict(observation)
                if recovering:
                    step_data["replanned"] = True
                steps.append(
                    AgentStep(
                        agent=self.name,
                        thought=str(planned.get("thought") or "Need a tool observation before continuing."),
                        action=tool_name,
                        observation=str(observation.get("summary") or observation.get("status") or "ok"),
                        status="ok" if observation.get("ok", True) else "error",
                        data=step_data,
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
