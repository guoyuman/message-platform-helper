"""Agent registry and default agent composition."""

from __future__ import annotations

from dataclasses import dataclass, field, replace as replace_dataclass
from typing import Callable, Dict, Iterable, List

from ..models import AgentSpec
from ..rate_limit import CounterStore
from ..react import ReActAgent


AgentFactory = Callable[[], ReActAgent]


@dataclass(frozen=True)
class RegisteredAgent:
    spec: AgentSpec
    factory: AgentFactory

    def create(self) -> ReActAgent:
        return self.factory()


@dataclass
class AgentRegistry:
    _agents: Dict[str, RegisteredAgent] = field(default_factory=dict)

    def register(self, spec: AgentSpec, factory: AgentFactory, *, replace: bool = False) -> "AgentRegistry":
        name = spec.name.strip()
        if not name:
            raise ValueError("Agent name is required.")
        if name in self._agents and not replace:
            raise ValueError(f"Agent '{name}' is already registered.")
        self._agents[name] = RegisteredAgent(spec=replace_dataclass(spec, name=name), factory=factory)
        return self

    def get(self, name: str) -> RegisteredAgent:
        try:
            return self._agents[name]
        except KeyError as exc:
            raise KeyError(f"Agent '{name}' is not registered.") from exc

    def create(self, name: str) -> ReActAgent:
        registered = self.get(name)
        if not registered.spec.enabled:
            raise ValueError(f"Agent '{name}' is disabled.")
        return registered.create()

    def resolve(self, names: Iterable[str], *, strict: bool = False) -> List[ReActAgent]:
        agents: List[ReActAgent] = []
        for name in names:
            registered = self._agents.get(str(name))
            if not registered:
                if strict:
                    raise KeyError(f"Agent '{name}' is not registered.")
                continue
            if not registered.spec.enabled:
                if strict:
                    raise ValueError(f"Agent '{name}' is disabled.")
                continue
            agents.append(registered.create())
        return agents

    def select_by_capability(self, capability: str, *, limit: int | None = None) -> List[AgentSpec]:
        specs = [
            registered.spec
            for registered in self._agents.values()
            if registered.spec.enabled and capability in registered.spec.capabilities
        ]
        ordered = sorted(specs, key=lambda spec: (spec.priority, spec.name))
        return ordered if limit is None else ordered[:limit]

    def names(self) -> List[str]:
        return [spec.name for spec in self.specs()]

    def specs(self) -> List[AgentSpec]:
        return sorted((registered.spec for registered in self._agents.values()), key=lambda spec: (spec.priority, spec.name))


def build_default_agent_registry(counter_store: CounterStore) -> AgentRegistry:
    from .business_config import BusinessMessageConfigAgent
    from .channel_config import ChannelConfigAgent
    from .knowledge import KnowledgeAgent
    from .send_strategy import SendStrategyAgent
    from .template import TemplateAgent
    from ..strategy import SendStrategyEvaluator

    registry = AgentRegistry()
    registry.register(
        AgentSpec(
            name="template",
            description="Translate existing message templates and sync missing language versions.",
            capabilities=["template.translation", "template.language_sync"],
            priority=20,
        ),
        TemplateAgent,
    )
    registry.register(
        AgentSpec(
            name="business_config",
            description="Normalize and preview or publish business message configuration.",
            capabilities=["business_message.configuration"],
            priority=30,
        ),
        BusinessMessageConfigAgent,
    )
    registry.register(
        AgentSpec(
            name="channel_config",
            description="Infer, save, and test message channel configuration.",
            capabilities=["channel.configuration", "email.smtp"],
            priority=40,
        ),
        ChannelConfigAgent,
    )
    registry.register(
        AgentSpec(
            name="knowledge",
            description="Answer product usage and troubleshooting questions from retrieved knowledge.",
            capabilities=["knowledge.answer", "rag.answer"],
            priority=10,
        ),
        KnowledgeAgent,
    )
    registry.register(
        AgentSpec(
            name="send_strategy",
            description="Build, evaluate, and optionally save send strategy rules.",
            capabilities=["send_strategy.configuration", "rate_limit.evaluation"],
            priority=50,
        ),
        lambda: SendStrategyAgent(SendStrategyEvaluator(counter_store)),
    )
    return registry


__all__ = ["AgentFactory", "AgentRegistry", "RegisteredAgent", "build_default_agent_registry"]
