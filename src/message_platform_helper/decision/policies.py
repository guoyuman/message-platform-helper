"""Decision policies that do not depend on prompt wording."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import AssistantRequest, ConversationMemory, IntentResult


DEFAULT_RAG_INTENTS = {
    "knowledge_query": True,
    "template_config": True,
    "implementation": True,
    "error_code": True,
    "workflow": True,
    "translation": False,
    "summary": False,
    "casual_chat": False,
}

DEFAULT_RAG_REQUEST_TYPES = {
    "query": True,
    "action": False,
    "tool": False,
    "chat": False,
}


@dataclass
class KnowledgePolicy:
    intent_rules: dict[str, bool] = field(default_factory=lambda: dict(DEFAULT_RAG_INTENTS))
    request_type_rules: dict[str, bool] = field(default_factory=lambda: dict(DEFAULT_RAG_REQUEST_TYPES))

    def needs_rag(self, intent: IntentResult, request: AssistantRequest, memory: ConversationMemory) -> bool:
        configured = self.intent_rules.get(intent.intent)
        if configured is not None and intent.intent not in DEFAULT_RAG_INTENTS:
            return configured
        configured_request_type = self.request_type_rules.get(intent.request_type)
        if configured_request_type is not None:
            return configured_request_type
        if configured is not None:
            return configured
        return bool(intent.metadata.get("needRag") or intent.metadata.get("need_rag"))


__all__ = ["KnowledgePolicy", "DEFAULT_RAG_INTENTS", "DEFAULT_RAG_REQUEST_TYPES"]
