"""Workflow registry and default workflow definitions."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List

from ..models import WorkflowDefinition, WorkflowStep


@dataclass
class WorkflowRegistry:
    _workflows: Dict[str, WorkflowDefinition] = field(default_factory=dict)

    def register(self, workflow: WorkflowDefinition, *, replace_existing: bool = False) -> "WorkflowRegistry":
        name = workflow.name.strip()
        if not name:
            raise ValueError("Workflow name is required.")
        if name in self._workflows and not replace_existing:
            raise ValueError(f"Workflow '{name}' is already registered.")
        self._workflows[name] = replace(workflow, name=name)
        return self

    def get(self, name: str) -> WorkflowDefinition | None:
        workflow = self._workflows.get(name)
        if workflow and workflow.enabled:
            return workflow
        return None

    def require(self, name: str) -> WorkflowDefinition:
        workflow = self.get(name)
        if not workflow:
            raise KeyError(f"Workflow '{name}' is not registered.")
        return workflow

    def names(self) -> List[str]:
        return sorted(self._workflows)

    def definitions(self) -> List[WorkflowDefinition]:
        return [self._workflows[name] for name in self.names() if self._workflows[name].enabled]

    def resolve_agents(self, workflow_name: str, fallback: Iterable[str] = ()) -> List[str]:
        workflow = self.get(workflow_name)
        if not workflow:
            return list(fallback)
        return [step.agent for step in workflow.steps if step.agent]

    def resolve_tools(self, workflow_name: str, fallback: Iterable[str] = ()) -> List[str]:
        workflow = self.get(workflow_name)
        if not workflow:
            return list(fallback)
        tools: List[str] = []
        for step in workflow.steps:
            for tool in step.tools:
                if tool not in tools:
                    tools.append(tool)
        return tools or list(fallback)


def build_default_workflow_registry() -> WorkflowRegistry:
    registry = WorkflowRegistry()
    registry.register(
        WorkflowDefinition(
            name="knowledge_answer",
            description="Answer product usage or troubleshooting questions from enterprise knowledge.",
            steps=[WorkflowStep(name="answer", agent="knowledge", tools=["knowledge.search", "knowledge.answer"])],
        )
    )
    registry.register(
        WorkflowDefinition(
            name="template_workflow",
            description="Translate or sync message templates and validate platform payloads.",
            steps=[WorkflowStep(name="template", agent="template", tools=["template.translate", "platform.template.save"])],
        )
    )
    registry.register(
        WorkflowDefinition(
            name="implementation_workflow",
            description="Configure message platform entities through selected implementation agents.",
            steps=[
                WorkflowStep(name="channel", agent="channel_config", tools=["channel.infer_email", "platform.channel.save"]),
            ],
        )
    )
    registry.register(
        WorkflowDefinition(
            name="message_platform_workflow",
            description="Run a multi-agent message platform workflow.",
            steps=[
                WorkflowStep(name="knowledge", agent="knowledge", tools=["knowledge.search"]),
                WorkflowStep(name="template", agent="template", tools=["template.translate"]),
                WorkflowStep(name="channel", agent="channel_config", tools=["channel.infer_email"]),
            ],
        )
    )
    registry.register(
        WorkflowDefinition(
            name="general_chat",
            description="Handle requests that do not need enterprise actions.",
            steps=[],
        )
    )
    return registry


def build_workflow_registry_from_config(config: dict) -> WorkflowRegistry:
    registry = WorkflowRegistry()
    workflows = config.get("workflows") if isinstance(config.get("workflows"), list) else []
    for item in workflows:
        if not isinstance(item, dict):
            continue
        steps = []
        for step in item.get("steps") or []:
            if not isinstance(step, dict):
                continue
            steps.append(
                WorkflowStep(
                    name=str(step.get("name") or step.get("agent") or ""),
                    agent=str(step.get("agent") or ""),
                    tools=[str(tool) for tool in step.get("tools") or []],
                    required=bool(step.get("required", True)),
                )
            )
        registry.register(
            WorkflowDefinition(
                name=str(item.get("name") or ""),
                description=str(item.get("description") or ""),
                steps=steps,
                enabled=bool(item.get("enabled", True)),
            )
        )
    return registry if registry.names() else build_default_workflow_registry()


__all__ = ["WorkflowRegistry", "build_default_workflow_registry", "build_workflow_registry_from_config"]
