"""Tenant and permission primitives."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import AssistantRequest


class PermissionDenied(RuntimeError):
    pass


@dataclass(frozen=True)
class TenantContext:
    tenant_id: str = ""
    user_id: str = "anonymous"
    roles: list[str] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)
    knowledge_namespace: str = "global"

    @classmethod
    def from_request(cls, request: AssistantRequest) -> "TenantContext":
        auth = request.payload.get("auth") or request.payload.get("_auth") or {}
        roles = [str(role) for role in auth.get("roles") or []] if isinstance(auth, dict) else []
        permissions = [str(item) for item in auth.get("permissions") or []] if isinstance(auth, dict) else []
        tenant_id = request.tenant_id or (str(auth.get("tenantId") or auth.get("tenant_id") or "") if isinstance(auth, dict) else "")
        namespace = f"tenant:{tenant_id}" if tenant_id else "global"
        return cls(
            tenant_id=tenant_id,
            user_id=request.user_id,
            roles=roles,
            permissions=permissions,
            knowledge_namespace=namespace,
        )


@dataclass
class PermissionPolicy:
    tool_permissions: dict[str, list[str]] = field(default_factory=dict)
    allow_unknown_tools: bool = True

    def authorize_tool(self, tool_name: str, tenant: TenantContext | None) -> None:
        required = self.tool_permissions.get(tool_name)
        if not required:
            if self.allow_unknown_tools:
                return
            required = [tool_name]
        tenant = tenant or TenantContext()
        if "admin" in tenant.roles or "*" in tenant.permissions:
            return
        missing = [permission for permission in required if permission not in tenant.permissions]
        if missing:
            raise PermissionDenied(f"Missing permission(s) for tool '{tool_name}': {', '.join(missing)}")


def permission_policy_from_config(config: dict | None) -> PermissionPolicy:
    raw = (config or {}).get("tool_permissions") or {}
    return PermissionPolicy(tool_permissions={str(key): [str(item) for item in value] for key, value in raw.items()})


__all__ = ["PermissionDenied", "PermissionPolicy", "TenantContext", "permission_policy_from_config"]
