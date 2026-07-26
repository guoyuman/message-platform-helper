"""Intent classification boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..llm import LLMClient
from ..models import AssistantRequest, ConversationMemory, IntentResult, JsonDict, to_jsonable
from .taxonomy import legacy_intent_for, normalize_axes


class IntentClassifier(Protocol):
    def classify(self, request: AssistantRequest, memory: ConversationMemory) -> IntentResult:
        raise NotImplementedError


@dataclass
class LLMIntentClassifier:
    llm: LLMClient

    def classify(self, request: AssistantRequest, memory: ConversationMemory) -> IntentResult:
        raw = self.llm.complete_json(
            "Classify user intent and request taxonomy for the message platform helper. Return request_type, domain, operation, and legacy intent.",
            {
                "text": request.text,
                "payload": request.payload,
                "memory": {
                    "summary": memory.summary,
                    "facts": memory.facts,
                    "profile": to_jsonable(memory.profile),
                },
                "allowedIntents": [
                    "knowledge_query",
                    "template_config",
                    "implementation",
                    "error_code",
                    "workflow",
                    "translation",
                    "template_translation_sync",
                    "summary",
                    "casual_chat",
                ],
                "allowedRequestTypes": ["query", "action", "tool", "chat"],
                "allowedDomains": ["business_message", "template", "error_code", "implementation", "channel", "general"],
                "allowedOperations": ["query", "explain", "create", "update", "delete", "execute"],
            },
        )
        axes = normalize_axes(raw, request.text)
        intent = str(raw.get("intent") or "").strip() or legacy_intent_for(
            axes["request_type"], axes["domain"], axes["operation"]
        )
        if intent in {"implementation", "workflow", "knowledge_query", "error_code", "template_config", "translation", "summary", "casual_chat"}:
            intent = legacy_intent_for(axes["request_type"], axes["domain"], axes["operation"])
        metadata = {
            key: value
            for key, value in raw.items()
            if key
            not in {
                "intent",
                "confidence",
                "rationale",
                "labels",
                "requestType",
                "request_type",
                "domain",
                "businessDomain",
                "business_domain",
                "operation",
                "op",
            }
        }
        metadata.update(axes)
        return IntentResult(
            intent=intent,
            confidence=float(raw.get("confidence") or 0.5),
            request_type=axes["request_type"],
            domain=axes["domain"],
            operation=axes["operation"],
            rationale=str(raw.get("rationale") or "Intent classified by configured classifier."),
            labels=list(raw.get("labels") or []),
            metadata=metadata,
        )


__all__ = ["IntentClassifier", "LLMIntentClassifier"]
