"""Tests for ReActAgent failure replanning (plan-and-execute + replanner).

Covers the recovery contract:
- a failed observation (ok=False) triggers ``replan()`` before the remaining
  plan continues;
- recovery steps are stamped ``replanned: True`` in the audit trail;
- the default ``replan()`` (None) preserves the historical continue-on-error
  behavior;
- ``max_steps`` bounds the step budget so a persistently failing replan
  cannot loop forever;
- tool exceptions still produce a failed result with an error step.
"""

from __future__ import annotations

from typing import Dict, List

from message_platform_helper.models import AgentStep, AssistantRequest, ConversationMemory, JsonDict, Tool
from message_platform_helper.react import AgentContext, ReActAgent


def _tool(name: str, handler) -> Tool:
    return Tool(name=name, description=name, handler=handler)


def _context() -> AgentContext:
    return AgentContext(
        request=AssistantRequest(text="sync template languages", session_id="s1"),
        memory=ConversationMemory(session_id="s1"),
        retrieved=[],
        llm=None,  # type: ignore[arg-type]
        platform=object(),  # type: ignore[arg-type]
    )


class _FlakyAgent(ReActAgent):
    """Primary tool fails once; replan() switches to a fallback tool."""

    name = "flaky"

    def __init__(self) -> None:
        self.calls = {"primary": 0, "fallback": 0}
        self.seen_failed_step: AgentStep | None = None
        self.seen_observation: JsonDict | None = None

    def tools(self, context: AgentContext) -> Dict[str, Tool]:
        return {
            "primary": _tool("primary", self._primary),
            "fallback": _tool("fallback", self._fallback),
        }

    def plan(self, context: AgentContext) -> List[JsonDict]:
        return [{"thought": "try primary", "tool": "primary", "input": {}}]

    def replan(self, context: AgentContext, observations: List[JsonDict], steps: List[AgentStep]) -> JsonDict | None:
        # Record the feedback contract: the failed step and its observation
        # must be visible here.
        self.seen_failed_step = steps[-1] if steps else None
        self.seen_observation = observations[-1] if observations else None
        return {"thought": "primary failed, switch to fallback", "tool": "fallback", "input": {}}

    def _primary(self, payload: JsonDict) -> JsonDict:
        self.calls["primary"] += 1
        return {"ok": False, "status": "error", "summary": "primary failed"}

    def _fallback(self, payload: JsonDict) -> JsonDict:
        self.calls["fallback"] += 1
        return {"ok": True, "status": "ok", "summary": "fallback worked"}


class _NoReplanAgent(ReActAgent):
    """Legacy shape: plan of two steps, first fails, no replan override."""

    name = "legacy"

    def __init__(self) -> None:
        self.calls: List[str] = []

    def tools(self, context: AgentContext) -> Dict[str, Tool]:
        return {
            "first": _tool("first", self._first),
            "second": _tool("second", self._second),
        }

    def plan(self, context: AgentContext) -> List[JsonDict]:
        return [
            {"thought": "first", "tool": "first", "input": {}},
            {"thought": "second", "tool": "second", "input": {}},
        ]

    def _first(self, payload: JsonDict) -> JsonDict:
        self.calls.append("first")
        return {"ok": False, "status": "error", "summary": "first failed"}

    def _second(self, payload: JsonDict) -> JsonDict:
        self.calls.append("second")
        return {"ok": True, "status": "ok", "summary": "second ok"}


class _StuckReplanAgent(_NoReplanAgent):
    """replan() keeps returning a tool that always fails — the step budget
    must terminate the run."""

    name = "stuck"

    def replan(self, context: AgentContext, observations: List[JsonDict], steps: List[AgentStep]) -> JsonDict | None:
        return {"thought": "retry failing tool", "tool": "first", "input": {}}


class _ExplodingToolAgent(ReActAgent):
    name = "exploder"

    def tools(self, context: AgentContext) -> Dict[str, Tool]:
        return {"boom": _tool("boom", self._boom)}

    def plan(self, context: AgentContext) -> List[JsonDict]:
        return [{"thought": "boom", "tool": "boom", "input": {}}]

    def _boom(self, payload: JsonDict) -> JsonDict:
        raise RuntimeError("tool exploded")


def test_replan_on_failed_observation_recovers() -> None:
    agent = _FlakyAgent()
    result = agent.run(_context())
    assert result.ok
    assert agent.calls == {"primary": 1, "fallback": 1}
    assert len(result.steps) == 2
    # Audit trail distinguishes the recovery step.
    assert result.steps[0].action == "primary"
    assert result.steps[0].status == "error"
    assert result.steps[1].action == "fallback"
    assert result.steps[1].status == "ok"
    assert result.steps[1].data.get("replanned") is True


def test_replan_receives_failed_step_and_observation() -> None:
    agent = _FlakyAgent()
    agent.run(_context())
    assert agent.seen_failed_step is not None
    assert agent.seen_failed_step.action == "primary"
    assert agent.seen_failed_step.status == "error"
    assert agent.seen_observation is not None
    assert agent.seen_observation.get("ok") is False


def test_default_replan_preserves_continue_on_error() -> None:
    agent = _NoReplanAgent()
    result = agent.run(_context())
    # Historical behavior: a failed observation does not abort the run; the
    # remaining planned steps still execute.
    assert agent.calls == ["first", "second"]
    assert len(result.steps) == 2
    assert result.steps[0].status == "error"
    assert result.steps[1].status == "ok"
    assert result.steps[1].data.get("replanned") is None


def test_max_steps_caps_recovery_loop() -> None:
    agent = _StuckReplanAgent()
    result = agent.run(_context())
    # plan has 2 steps -> cap = 2*2 + 1 = 5; the stuck replan must not
    # exceed it and the run must terminate.
    assert len(result.steps) == 5
    assert len(agent.calls) == 5


def test_custom_max_steps_honored() -> None:
    agent = _StuckReplanAgent()
    agent.max_steps = 3
    result = agent.run(_context())
    assert len(result.steps) == 3


def test_tool_exception_still_fails_with_error_step() -> None:
    agent = _ExplodingToolAgent()
    result = agent.run(_context())
    assert not result.ok
    assert result.issues and result.issues[0]["code"] == "exploder.error"
    assert result.steps[-1].status == "error"
    assert result.steps[-1].action == "error"
