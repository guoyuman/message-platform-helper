"""Intent classification boundary."""

from __future__ import annotations

import json
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
    system_prompt: str = ""

    def classify(self, request: AssistantRequest, memory: ConversationMemory) -> IntentResult:
        raw = self._complete_intent(request, memory)
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


    def _complete_intent(self, request: AssistantRequest, memory: ConversationMemory) -> JsonDict:
        """Classify via the function-calling protocol when the provider
        supports it (structured output through a forced tool call), falling
        back to JSON-mode completion otherwise."""
        system = self.system_prompt or _default_intent_prompt()
        payload: JsonDict = {
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
        }
        try:
            message = self.llm.complete_chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                tools=[_INTENT_TOOL_SCHEMA],
                # "auto" instead of a forced function choice: several
                # providers (e.g. DeepSeek thinking mode) reject a specific
                # tool_choice with 400. The unique tool is still called in
                # practice; content is parsed as JSON when it is not.
                tool_choice="auto",
            )
        except NotImplementedError:
            # Offline rule-based client: no tools protocol, use JSON mode.
            return self.llm.complete_json(system, payload)
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            try:
                arguments = json.loads(str((tool_calls[0].get("function") or {}).get("arguments") or "{}"))
            except json.JSONDecodeError:
                arguments = {}
            if isinstance(arguments, dict):
                return arguments
        content = message.get("content") or ""
        if content:
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        return {}


_INTENT_TOOL_SCHEMA: JsonDict = {
    "type": "function",
    "function": {
        "name": "classify_intent",
        "description": "Classify user intent and request taxonomy for the message platform helper.",
        "parameters": {
            "type": "object",
            "properties": {
                "request_type": {"type": "string", "enum": ["query", "action", "tool", "chat"]},
                "domain": {
                    "type": "string",
                    "enum": ["business_message", "template", "error_code", "implementation", "channel", "general"],
                },
                "operation": {"type": "string", "enum": ["query", "explain", "create", "update", "delete", "execute"]},
                "intent": {"type": "string"},
                "confidence": {"type": "number"},
                "rationale": {"type": "string"},
                "selectedAgents": {"type": "array", "items": {"type": "string"}},
                "selectedTools": {"type": "array", "items": {"type": "string"}},
                "workflow": {"type": "string"},
                "searchQuery": {"type": "string"},
                "missingSlots": {"type": "array", "items": {"type": "string"}},
                "taskSlices": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["request_type", "domain", "operation"],
        },
    },
}


def _default_intent_prompt() -> str:
    return (
        "Classify user intent and request taxonomy for the message platform helper. "
        "If the user message contains multiple independent tasks, split them before choosing agents. "
        "Return strict JSON with request_type, domain, operation, intent, confidence, rationale, "
        "selectedAgents, selectedTools, workflow, searchQuery, missingSlots, and taskSlices. "
        "taskSlices must be an array; each item has text, request_type, domain, operation, intent, "
        "workflow, selectedAgents, selectedTools, searchQuery, needRag, and payloadDomain. "
        "Use one taskSlice for knowledge-base questions, one for template translation/sync, "
        "and one for channel configuration when they appear in the same user input. "
        "Do not merge query slices with action slices. Do not invent payload values."
    )


__all__ = ["IntentClassifier", "LLMIntentClassifier"]
