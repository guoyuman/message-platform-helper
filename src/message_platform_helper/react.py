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

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .application.security import TenantContext
from .llm import LLMClient, OpenAICompatibleLLMClient
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
    # Live step callback for streaming observability: invoked with every
    # AgentStep as it is produced, before the run completes.
    on_step: Any | None = None


class RuleBasedAgent:
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
                if context.on_step is not None:
                    context.on_step(steps[-1])
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
            if context.on_step is not None:
                context.on_step(steps[-1])
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


class FunctionCallingAgent(RuleBasedAgent):
    """RuleBasedAgent variant driven by native LLM tool calls instead of a rule plan.

    ``run()`` hands the available tools (as OpenAI function schemas) to the
    model, executes each ``tool_calls`` the model returns, feeds the
    observations back as ``tool`` messages, and repeats until the model
    answers without calling a tool. Every executed call is recorded as an
    ``AgentStep`` audit record, and tool errors are returned to the model as
    observations (standard function-calling recovery) rather than aborting
    the run. Requires an online provider that implements ``complete_chat``;
    falls back to the rule-based ``plan()`` otherwise.
    """

    def system_prompt_for(self, context: AgentContext) -> str:
        return (
            "You are an enterprise message platform assistant. "
            "Call the available tools when you need data or need to perform an action; "
            "otherwise answer the user directly. Do not invent tool outputs."
        )

    def run(self, context: AgentContext) -> AgentResult:
        if not isinstance(context.llm, OpenAICompatibleLLMClient):
            return super().run(context)
        tools = self._available_tools(context)
        tool_names = {_openai_tool_name(name): name for name in tools}
        schemas = [_openai_tool_schema(tools[name], function_name=function_name) for function_name, name in tool_names.items()]
        messages: List[JsonDict] = [
            {"role": "system", "content": self.system_prompt_for(context)},
            {
                "role": "user",
                "content": json.dumps(
                    {"text": context.request.text, "payload": context.request.payload},
                    ensure_ascii=False,
                ),
            },
        ]
        steps: List[AgentStep] = []
        issues: List[JsonDict] = []
        cap = self.max_steps if self.max_steps is not None else 10
        try:
            while len(steps) < cap:
                message = context.llm.complete_chat(messages, tools=schemas or None)
                tool_calls = message.get("tool_calls") or []
                if not tool_calls:
                    ok = not any(issue.get("severity") == "error" for issue in issues)
                    return AgentResult(
                        agent=self.name,
                        ok=ok,
                        output={"answer": message.get("content") or ""},
                        issues=issues,
                        steps=steps,
                    )
                messages.append(
                    {"role": "assistant", "content": message.get("content") or "", "tool_calls": tool_calls}
                )
                for call in tool_calls:
                    function = call.get("function") or {}
                    function_name = str(function.get("name") or "")
                    name = tool_names.get(function_name, function_name)
                    try:
                        arguments = json.loads(str(function.get("arguments") or "{}"))
                    except json.JSONDecodeError:
                        arguments = {}
                    if not isinstance(arguments, dict):
                        arguments = {}
                    tool = tools.get(name)
                    if tool is None:
                        observation: JsonDict = {"ok": False, "error": f"Unknown tool {name!r}."}
                    else:
                        try:
                            if context.tool_registry is not None:
                                context.tool_registry.authorize(name, context.tenant_context)
                            observation = tool.run(arguments, context)
                        except Exception as exc:
                            observation = {"ok": False, "error": str(exc)}
                    if not observation.get("ok", True):
                        issues.extend(observation.get("issues") or [{"code": f"{self.name}.tool_error", "message": str(observation.get("error") or "tool error"), "severity": "error"}])
                    steps.append(
                        AgentStep(
                            agent=self.name,
                            thought=str(function.get("arguments") or ""),
                            action=name,
                            observation=str(
                                observation.get("summary") or observation.get("status") or observation.get("error") or "ok"
                            ),
                            status="ok" if observation.get("ok", True) else "error",
                            data=dict(observation),
                        )
                    )
                    if context.on_step is not None:
                        context.on_step(steps[-1])
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": str(call.get("id") or ""),
                            "content": json.dumps(observation, ensure_ascii=False),
                        }
                    )
            return AgentResult(
                agent=self.name,
                ok=True,
                output={"answer": "Reached the maximum number of tool rounds."},
                issues=issues,
                steps=steps,
            )
        except Exception as exc:
            steps.append(
                AgentStep(
                    agent=self.name,
                    thought="Function calling stopped because of an exception.",
                    action="error",
                    observation=str(exc),
                    status="error",
                )
            )
            if context.on_step is not None:
                context.on_step(steps[-1])
            return AgentResult(
                agent=self.name,
                ok=False,
                issues=[{"code": f"{self.name}.error", "message": str(exc), "severity": "error"}],
                steps=steps,
            )


def _openai_tool_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name).strip("_") or "tool"


def _openai_tool_schema(tool: Tool, function_name: str | None = None) -> JsonDict:
    return {
        "type": "function",
        "function": {
            "name": function_name or _openai_tool_name(tool.name),
            "description": tool.description,
            "parameters": tool.parameters or {"type": "object", "properties": {}},
        },
    }
