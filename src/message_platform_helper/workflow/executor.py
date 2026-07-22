"""Workflow execution over registered agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from ..agents import AgentRegistry
from ..models import AgentResult
from ..react import AgentContext
from .registry import WorkflowRegistry


@dataclass
class WorkflowExecutor:
    agent_registry: AgentRegistry
    workflow_registry: WorkflowRegistry

    def execute(self, workflow_name: str, selected_agents: Iterable[str], context: AgentContext) -> List[AgentResult]:
        agent_names = self.resolve_execution_order(workflow_name, selected_agents)
        results: List[AgentResult] = []
        for agent in self.agent_registry.resolve(agent_names):
            results.append(agent.run(context))
        return results

    def resolve_execution_order(self, workflow_name: str, selected_agents: Iterable[str] = ()) -> List[str]:
        selected = _dedupe(str(name) for name in selected_agents if str(name))
        workflow = self.workflow_registry.get(workflow_name)
        if not workflow:
            return selected

        workflow_order = [step.agent for step in workflow.steps if step.agent]
        if not selected:
            return _dedupe(workflow_order)

        selected_set = set(selected)
        ordered = [agent for agent in workflow_order if agent in selected_set]
        ordered.extend(agent for agent in selected if agent not in ordered)
        return ordered


def _dedupe(names: Iterable[str]) -> List[str]:
    result: List[str] = []
    for name in names:
        if name not in result:
            result.append(name)
    return result


__all__ = ["WorkflowExecutor"]
