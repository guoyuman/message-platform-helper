"""Tool registry for platform, RAG, MCP, and function tools."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Iterable, List

from ..application.security import PermissionPolicy, TenantContext
from ..models import Tool, ToolSpec


ToolFactory = Callable[[], Tool]


@dataclass(frozen=True)
class RegisteredTool:
    spec: ToolSpec
    factory: ToolFactory

    def create(self) -> Tool:
        return self.factory()


@dataclass
class ToolRegistry:
    _tools: Dict[str, RegisteredTool] = field(default_factory=dict)
    permission_policy: PermissionPolicy = field(default_factory=PermissionPolicy)

    def register(self, spec: ToolSpec, factory: ToolFactory, *, replace_existing: bool = False) -> "ToolRegistry":
        name = spec.name.strip()
        if not name:
            raise ValueError("Tool name is required.")
        if name in self._tools and not replace_existing:
            raise ValueError(f"Tool '{name}' is already registered.")
        self._tools[name] = RegisteredTool(spec=replace(spec, name=name), factory=factory)
        return self

    def register_tool(self, tool: Tool, spec: ToolSpec | None = None, *, replace_existing: bool = False) -> "ToolRegistry":
        tool_spec = spec or ToolSpec(name=tool.name, description=tool.description)
        return self.register(tool_spec, lambda: tool, replace_existing=replace_existing)

    def get(self, name: str) -> RegisteredTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Tool '{name}' is not registered.") from exc

    def create(self, name: str) -> Tool:
        registered = self.get(name)
        if not registered.spec.enabled:
            raise ValueError(f"Tool '{name}' is disabled.")
        return registered.create()

    def authorize(self, name: str, tenant: TenantContext | None = None) -> None:
        self.permission_policy.authorize_tool(name, tenant)

    def resolve(self, names: Iterable[str], *, strict: bool = False) -> List[Tool]:
        tools: List[Tool] = []
        for name in names:
            registered = self._tools.get(str(name))
            if not registered:
                if strict:
                    raise KeyError(f"Tool '{name}' is not registered.")
                continue
            if not registered.spec.enabled:
                if strict:
                    raise ValueError(f"Tool '{name}' is disabled.")
                continue
            tools.append(registered.create())
        return tools

    def select_by_capability(self, capability: str, *, limit: int | None = None) -> List[ToolSpec]:
        specs = [
            registered.spec
            for registered in self._tools.values()
            if registered.spec.enabled and capability in registered.spec.capabilities
        ]
        ordered = sorted(specs, key=lambda spec: (spec.priority, spec.name))
        return ordered if limit is None else ordered[:limit]

    def names(self) -> List[str]:
        return [spec.name for spec in self.specs()]

    def specs(self) -> List[ToolSpec]:
        return sorted((registered.spec for registered in self._tools.values()), key=lambda spec: (spec.priority, spec.name))


__all__ = ["RegisteredTool", "ToolFactory", "ToolRegistry"]
