"""Request taxonomy helpers for the decision engine."""

from __future__ import annotations

from typing import Any

from ..models import JsonDict


REQUEST_TYPES = {"query", "action", "tool", "chat"}
DOMAINS = {"business_message", "template", "error_code", "implementation", "channel", "send_strategy", "general"}
OPERATIONS = {"query", "explain", "create", "update", "delete", "execute"}

LEGACY_INTENT_AXES: dict[str, tuple[str, str, str]] = {
    "knowledge_query": ("query", "general", "query"),
    "template_config": ("action", "template", "update"),
    "business_config": ("action", "business_message", "create"),
    "implementation": ("action", "implementation", "execute"),
    "error_code": ("query", "error_code", "explain"),
    "workflow": ("action", "implementation", "execute"),
    "translation": ("action", "template", "execute"),
    "template_translation_sync": ("action", "template", "execute"),
    "summary": ("chat", "general", "query"),
    "casual_chat": ("chat", "general", "query"),
}

DOMAIN_AGENT_HINTS = {
    "business_message": ["business_config"],
    "template": ["template"],
    "channel": ["channel_config"],
    "send_strategy": ["send_strategy"],
    "error_code": ["knowledge"],
}

DOMAIN_TOOL_HINTS = {
    "business_message": ["business.build_payload", "platform.business.preview"],
    "template": ["domain.tree.get", "business_object.resolve", "template.sync_language", "platform.template.save"],
    "channel": ["channel.infer_email", "platform.channel.save"],
    "send_strategy": ["strategy.build", "strategy.evaluate"],
    "error_code": ["knowledge.search", "knowledge.answer"],
}

DOMAIN_WORKFLOWS = {
    ("query", "business_message"): "knowledge_answer",
    ("query", "template"): "knowledge_answer",
    ("query", "channel"): "knowledge_answer",
    ("query", "send_strategy"): "knowledge_answer",
    ("query", "error_code"): "knowledge_answer",
    ("query", "implementation"): "knowledge_answer",
    ("query", "general"): "knowledge_answer",
    ("action", "business_message"): "implementation_workflow",
    ("action", "template"): "template_workflow",
    ("action", "channel"): "implementation_workflow",
    ("action", "send_strategy"): "implementation_workflow",
    ("action", "implementation"): "implementation_workflow",
    ("tool", "general"): "general_chat",
    ("chat", "general"): "general_chat",
}


def normalize_axes(raw: JsonDict, text: str = "") -> JsonDict:
    intent = _string(raw, "intent") or "casual_chat"
    legacy_request_type, legacy_domain, legacy_operation = LEGACY_INTENT_AXES.get(intent, ("action", "implementation", "execute"))
    domain = _normalize_domain(_string(raw, "domain") or _string(raw, "businessDomain") or _string(raw, "business_domain") or _infer_domain(text) or legacy_domain)
    operation = _normalize_operation(
        _string(raw, "operation") or _string(raw, "op") or _infer_operation(text) or legacy_operation
    )
    request_type = _normalize_request_type(
        _string(raw, "requestType") or _string(raw, "request_type") or _infer_request_type(text, operation) or legacy_request_type
    )
    if request_type == "query" and operation in {"create", "update", "delete", "execute"}:
        operation = "explain"
    if request_type == "action" and operation in {"query", "explain"}:
        operation = "execute"
    return {"request_type": request_type, "domain": domain, "operation": operation}


def legacy_intent_for(request_type: str, domain: str, operation: str) -> str:
    if request_type == "chat":
        return "casual_chat"
    if request_type == "query":
        return "error_code" if domain == "error_code" else "knowledge_query"
    if domain == "template":
        return "translation" if operation == "execute" else "template_config"
    if request_type == "tool":
        return "workflow"
    return "implementation"


def workflow_for(request_type: str, domain: str, operation: str) -> str:
    if request_type == "chat":
        return "general_chat"
    if request_type == "query":
        return "knowledge_answer"
    return DOMAIN_WORKFLOWS.get((request_type, domain), DOMAIN_WORKFLOWS.get((request_type, "general"), "general_chat"))


def agent_hints_for(request_type: str, domain: str) -> list[str]:
    if request_type == "query":
        return ["knowledge"]
    if request_type == "chat":
        return []
    return list(DOMAIN_AGENT_HINTS.get(domain, []))


def tool_hints_for(request_type: str, domain: str) -> list[str]:
    if request_type == "query":
        return ["knowledge.search", "knowledge.answer"]
    if request_type == "chat":
        return []
    return list(DOMAIN_TOOL_HINTS.get(domain, []))


def _string(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    return str(value).strip() if value is not None else ""


def _normalize_request_type(value: str) -> str:
    normalized = value.strip().lower()
    aliases = {
        "question": "query",
        "knowledge": "query",
        "rag": "query",
        "command": "action",
        "execute": "action",
        "operation": "action",
        "function": "tool",
        "general_chat": "chat",
        "casual_chat": "chat",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in REQUEST_TYPES else "action"


def _normalize_domain(value: str) -> str:
    normalized = value.strip().lower()
    aliases = {
        "business": "business_message",
        "business_config": "business_message",
        "message": "business_message",
        "mail": "channel",
        "email": "channel",
        "strategy": "send_strategy",
        "send_policy": "send_strategy",
        "error": "error_code",
        "errorcode": "error_code",
        "knowledge": "general",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in DOMAINS else "implementation"


def _normalize_operation(value: str) -> str:
    normalized = value.strip().lower()
    aliases = {
        "read": "query",
        "search": "query",
        "answer": "explain",
        "configure": "create",
        "config": "create",
        "save": "create",
        "sync": "execute",
        "translate": "execute",
        "run": "execute",
        "remove": "delete",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in OPERATIONS else "execute"


def _infer_request_type(text: str, operation: str) -> str:
    lowered = text.lower()
    query_markers = (
        "how to",
        "how do",
        "what is",
        "why",
        "explain",
        "guide",
        "manual",
        "troubleshoot",
        "\u5982\u4f55",
        "\u600e\u4e48",
        "\u600e\u6837",
        "\u662f\u4ec0\u4e48",
        "\u4ec0\u4e48\u662f",
        "\u4e3a\u4ec0\u4e48",
        "\u8bf4\u660e",
        "\u6307\u5357",
        "\u6587\u6863",
        "\u67e5\u8be2",
        "\u539f\u56e0",
        "\u6392\u67e5",
    )
    if any(marker in lowered for marker in query_markers) or any(marker in text for marker in query_markers):
        return "query"
    action_markers = (
        "create",
        "update",
        "delete",
        "execute",
        "configure",
        "help me",
        "\u5e2e\u6211",
        "\u8bf7\u7ed9",
        "\u521b\u5efa",
        "\u65b0\u589e",
        "\u914d\u7f6e",
        "\u4fee\u6539",
        "\u66f4\u65b0",
        "\u5220\u9664",
        "\u6267\u884c",
        "\u540c\u6b65",
        "\u7ffb\u8bd1",
    )
    if any(marker in lowered for marker in action_markers) or any(marker in text for marker in action_markers):
        return "action"
    return "chat" if not text.strip() else ""


def _infer_domain(text: str) -> str:
    lowered = text.lower()
    if any(marker in lowered for marker in ("smtp", "email", "mail")) or any(marker in text for marker in ("\u90ae\u7bb1", "\u90ae\u4ef6", "\u901a\u9053")):
        return "channel"
    if any(marker in lowered for marker in ("template", "translation", "translate")) or any(marker in text for marker in ("\u6a21\u677f", "\u7ffb\u8bd1", "\u591a\u8bed")):
        return "template"
    if any(marker in lowered for marker in ("error", "failed", "failure")) or any(marker in text for marker in ("\u9519\u8bef\u7801", "\u62a5\u9519", "\u5931\u8d25", "\u5f02\u5e38")):
        return "error_code"
    if any(marker in lowered for marker in ("strategy", "schedule", "rate limit")) or any(marker in text for marker in ("\u7b56\u7565", "\u9891\u63a7", "\u9650\u6d41", "\u5b9a\u65f6")):
        return "send_strategy"
    if "business" in lowered or any(marker in text for marker in ("\u4e1a\u52a1\u6d88\u606f", "\u5355\u636e", "\u89e6\u53d1", "\u63a5\u6536\u4eba", "\u5ba1\u6279")):
        return "business_message"
    return ""


def _infer_operation(text: str) -> str:
    lowered = text.lower()
    if any(marker in lowered for marker in ("what is", "why", "explain", "how to")) or any(
        marker in text
        for marker in (
            "\u5982\u4f55",
            "\u600e\u4e48",
            "\u662f\u4ec0\u4e48",
            "\u4ec0\u4e48\u662f",
            "\u4e3a\u4ec0\u4e48",
            "\u8bf4\u660e",
            "\u539f\u56e0",
            "\u6392\u67e5",
        )
    ):
        return "explain"
    if any(marker in lowered for marker in ("delete", "remove")) or "\u5220\u9664" in text:
        return "delete"
    if any(marker in lowered for marker in ("update", "change", "modify")) or any(marker in text for marker in ("\u4fee\u6539", "\u66f4\u65b0")):
        return "update"
    if any(marker in lowered for marker in ("create", "configure", "save")) or any(marker in text for marker in ("\u521b\u5efa", "\u65b0\u589e", "\u914d\u7f6e", "\u4fdd\u5b58")):
        return "create"
    if any(marker in lowered for marker in ("execute", "run", "sync", "translate")) or any(marker in text for marker in ("\u6267\u884c", "\u540c\u6b65", "\u7ffb\u8bd1")):
        return "execute"
    return ""
