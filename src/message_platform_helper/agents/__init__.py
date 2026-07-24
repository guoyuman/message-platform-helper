"""Agent registry."""

from .channel_config import ChannelConfigAgent
from .knowledge import KnowledgeAgent
from .registry import AgentFactory, AgentRegistry, RegisteredAgent, build_default_agent_registry
from .template import TemplateAgent

__all__ = [
    "AgentFactory",
    "AgentRegistry",
    "ChannelConfigAgent",
    "KnowledgeAgent",
    "RegisteredAgent",
    "TemplateAgent",
    "build_default_agent_registry",
]
