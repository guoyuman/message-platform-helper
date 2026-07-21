"""LLM boundary with deterministic fallback."""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .config import Settings


JsonDict = Dict[str, Any]


AGENT_TOOLS = {
    "knowledge": ["knowledge.search", "knowledge.answer"],
    "template": ["template.translate", "platform.template.save"],
    "business_config": ["business.build_payload", "platform.business.preview"],
    "channel_config": ["channel.infer_email", "platform.channel.save"],
    "send_strategy": ["strategy.build", "strategy.evaluate"],
}

DOMAIN_AGENTS = {
    "business_message": ["business_config"],
    "template": ["template"],
    "channel": ["channel_config"],
    "send_strategy": ["send_strategy"],
    "error_code": ["knowledge"],
}

QUERY_PATTERNS = (
    "how to",
    "how do",
    "what is",
    "why",
    "explain",
    "guide",
    "manual",
    "troubleshoot",
    "doc",
    "document",
    "reference",
    "\u5982\u4f55",
    "\u600e\u4e48",
    "\u600e\u6837",
    "\u662f\u4ec0\u4e48",
    "\u4ec0\u4e48\u662f",
    "\u4e3a\u4ec0\u4e48",
    "\u8bf4\u660e",
    "\u6307\u5357",
    "\u6587\u6863",
    "\u53c2\u8003",
    "\u6392\u67e5",
    "\u539f\u56e0",
)

ACTION_PATTERNS = (
    "create",
    "update",
    "delete",
    "execute",
    "configure",
    "save",
    "sync",
    "translate",
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
    "\u4fdd\u5b58",
    "\u540c\u6b65",
    "\u7ffb\u8bd1",
)

KNOWLEDGE_PATTERNS = (
    "knowledge",
    "knowledge base",
    "manual",
    "guide",
    "troubleshoot",
    "failure",
    "failed",
    "error",
    "schema",
    "endpoint",
    "api",
    "\u77e5\u8bc6\u5e93",
    "\u8bf4\u660e\u4e66",
    "\u4f7f\u7528\u8bf4\u660e",
    "\u6545\u969c",
    "\u5931\u8d25",
    "\u62a5\u9519",
    "\u9519\u8bef\u7801",
    "\u63a5\u53e3",
    "\u5b57\u6bb5",
    "\u89c4\u5219",
)

DOMAIN_PATTERNS = {
    "business_message": (
        "business",
        "business message",
        "\u4e1a\u52a1\u6d88\u606f",
        "\u4e1a\u52a1\u914d\u7f6e",
        "\u5355\u636e",
        "\u89e6\u53d1",
        "\u63a5\u6536\u4eba",
        "\u5ba1\u6279",
    ),
    "template": (
        "template",
        "translation",
        "translate",
        "multi-language",
        "\u6a21\u677f",
        "\u7ffb\u8bd1",
        "\u672c\u5730\u5316",
        "\u8bed\u79cd",
        "\u591a\u8bed",
        "\u591a\u8bed\u8a00",
    ),
    "channel": (
        "smtp",
        "email",
        "mail",
        "channel",
        "\u90ae\u7bb1",
        "\u90ae\u4ef6",
        "\u90ae\u4ef6\u901a\u9053",
        "\u901a\u9053",
    ),
    "send_strategy": (
        "strategy",
        "schedule",
        "rate limit",
        "channel limit",
        "\u7b56\u7565",
        "\u9891\u6b21",
        "\u9891\u63a7",
        "\u9650\u6d41",
        "\u5b9a\u65f6",
        "\u62e6\u622a",
        "\u6bcf\u5c0f\u65f6",
        "\u6bcf\u5929",
        "\u6bcf\u65e5",
    ),
    "error_code": (
        "error",
        "error code",
        "failed",
        "failure",
        "exception",
        "troubleshoot",
        "\u9519\u8bef\u7801",
        "\u5931\u8d25",
        "\u62a5\u9519",
        "\u5f02\u5e38",
        "\u6392\u67e5",
    ),
}

OPERATION_PATTERNS = {
    "create": ("create", "configure", "save", "\u521b\u5efa", "\u65b0\u589e", "\u914d\u7f6e", "\u4fdd\u5b58"),
    "update": ("update", "change", "modify", "\u4fee\u6539", "\u66f4\u65b0"),
    "delete": ("delete", "remove", "\u5220\u9664"),
    "execute": ("execute", "run", "sync", "translate", "\u6267\u884c", "\u540c\u6b65", "\u7ffb\u8bd1"),
}


@dataclass
class RoutingContext:
    """Structured routing result for the offline LLM fallback."""

    request_type: str
    domain: str
    operation: str
    intent: str
    workflow: str
    selected_agents: List[str] = field(default_factory=list)
    selected_tools: List[str] = field(default_factory=list)
    need_retrieval: bool = False
    confidence: float = 0.55
    labels: List[str] = field(default_factory=list)
    rationale: str = ""


class LLMClient:
    def complete_json(self, system: str, payload: JsonDict) -> JsonDict:
        raise NotImplementedError

    def decide(self, text: str, hints: JsonDict) -> JsonDict:
        return self.complete_json(
            "Decide whether RAG retrieval is needed and which agents should run.",
            {"text": text, "hints": hints},
        )

    def translate_template(self, template: JsonDict, target_language: str, source_language: str = "") -> JsonDict:
        return self.complete_json(
            "Translate a message template. Preserve variables exactly.",
            {"template": template, "targetLanguage": target_language, "sourceLanguage": source_language},
        )

    def summarize(self, text: str, memory: JsonDict) -> JsonDict:
        return self.complete_json(
            "Update conversation summary and profile facts. Return JSON only.",
            {"text": text, "memory": memory},
        )


@dataclass
class RuleBasedLLMClient(LLMClient):
    """Offline model used for tests and local demos."""

    def complete_json(self, system: str, payload: JsonDict) -> JsonDict:
        text = str(payload.get("text") or "")
        lowered = text.lower()
        if "Classify user intent" in system:
            decision = self._route(text, payload.get("payload") or {})
            return {
                "intent": decision.intent,
                "request_type": decision.request_type,
                "domain": decision.domain,
                "operation": decision.operation,
                "workflow": decision.workflow,
                "selectedAgents": decision.selected_agents,
                "selectedTools": decision.selected_tools,
                "needRag": decision.need_retrieval,
                "confidence": decision.confidence,
                "rationale": decision.rationale,
                "missingSlots": [],
                "searchQuery": text,
                "labels": decision.labels,
            }
        if "Decide whether RAG" in system:
            hints = payload.get("hints") or {}
            decision = self._route(text, hints.get("payload") or {})
            return {
                "needRetrieval": decision.need_retrieval,
                "selectedAgents": decision.selected_agents,
                "request_type": decision.request_type,
                "domain": decision.domain,
                "operation": decision.operation,
                "confidence": decision.confidence,
                "rationale": decision.rationale,
                "missingSlots": [],
                "searchQuery": text,
            }
        if "Translate a message template" in system:
            template = payload.get("template") or {}
            target = str(payload.get("targetLanguage") or "en_US")
            return self._translate(template, target)
        if "Update conversation summary" in system:
            memory = payload.get("memory") or {}
            summary = str(memory.get("summary") or "")
            short = text[:120]
            return {
                "summary": (summary + " | " + short).strip(" |")[-800:],
                "profilePatch": self._profile_patch(text),
                "factsPatch": self._facts_patch(text),
            }
        return {"ok": True, "echo": payload, "note": lowered[:20]}

    def _route(self, text: str, payload: JsonDict) -> RoutingContext:
        signals = self._semantic_signals(text, payload)
        request_type = self._classify_request_type(signals, payload)
        domain = self._classify_domain(signals, payload)
        operation = self._classify_operation(signals, request_type)
        intent = self._intent_for(request_type, domain, operation)
        workflow = self._workflow_for(request_type, domain)
        agents = self._agents_for(request_type, domain, signals)
        tools = self._tools_for_agents(agents)
        confidence = 0.88 if signals["payload_domains"] else 0.76 if signals["domain_candidates"] else 0.58
        if request_type == "chat":
            confidence = 0.62
        return RoutingContext(
            request_type=request_type,
            domain=domain,
            operation=operation,
            intent=intent,
            workflow=workflow,
            selected_agents=agents,
            selected_tools=tools,
            need_retrieval=request_type == "query",
            confidence=confidence,
            labels=[request_type, domain, operation],
            rationale=(
                "Offline routing used a staged decision chain: payload/schema signals, "
                "request type, business domain, operation, then capability-to-agent mapping."
            ),
        )

    def _semantic_signals(self, text: str, payload: JsonDict) -> JsonDict:
        forms = self._text_forms(text)
        lowered = " ".join(forms).lower()
        payload_domains = self._domains_from_payload(payload)
        query_score = self._score(forms, lowered, QUERY_PATTERNS)
        action_score = self._score(forms, lowered, ACTION_PATTERNS)
        operation_scores = {name: self._score(forms, lowered, patterns) for name, patterns in OPERATION_PATTERNS.items()}
        domain_scores = {name: self._score(forms, lowered, patterns) for name, patterns in DOMAIN_PATTERNS.items()}
        if "email" in payload and "channel" not in payload_domains:
            payload_domains.append("channel")
        domain_candidates = [domain for domain, score in domain_scores.items() if score > 0]
        return {
            "forms": forms,
            "lowered": lowered,
            "query_score": query_score,
            "action_score": action_score,
            "operation_scores": operation_scores,
            "domain_scores": domain_scores,
            "payload_domains": payload_domains,
            "domain_candidates": domain_candidates,
            "is_empty": not text.strip() and not payload,
            "explicit_knowledge": bool(payload.get("knowledge")) or self._score(forms, lowered, KNOWLEDGE_PATTERNS) > 0,
        }

    def _classify_request_type(self, signals: JsonDict, payload: JsonDict) -> str:
        if signals["is_empty"]:
            return "chat"
        if signals["explicit_knowledge"]:
            return "query"
        if signals["query_score"] and signals["query_score"] >= signals["action_score"]:
            return "query"
        if signals["payload_domains"] or signals["action_score"]:
            return "action"
        return "chat"

    def _classify_domain(self, signals: JsonDict, payload: JsonDict) -> str:
        payload_domains = list(signals["payload_domains"])
        if payload_domains:
            if len(payload_domains) > 1:
                return "implementation"
            return payload_domains[0]
        domain_scores = signals["domain_scores"]
        if domain_scores.get("error_code", 0) > 0 and signals["query_score"]:
            return "error_code"
        ranked = sorted(domain_scores.items(), key=lambda item: (-item[1], item[0]))
        if ranked and ranked[0][1] > 0:
            return ranked[0][0]
        return "general" if signals["query_score"] or not payload else "implementation"

    def _classify_operation(self, signals: JsonDict, request_type: str) -> str:
        if request_type == "chat":
            return "query"
        if request_type == "query":
            return "explain"
        ranked = sorted(signals["operation_scores"].items(), key=lambda item: (-item[1], item[0]))
        if ranked and ranked[0][1] > 0:
            return ranked[0][0]
        return "execute"

    def _agents_for(self, request_type: str, domain: str, signals: JsonDict) -> List[str]:
        if request_type == "chat":
            return []
        if request_type == "query":
            return ["knowledge"]
        if domain == "implementation":
            domains = signals["payload_domains"] or signals["domain_candidates"]
            agents: List[str] = []
            for item in domains:
                for agent in DOMAIN_AGENTS.get(item, []):
                    if agent != "knowledge" and agent not in agents:
                        agents.append(agent)
            return agents or ["business_config"]
        return list(DOMAIN_AGENTS.get(domain, ["business_config"]))

    def _intent_for(self, request_type: str, domain: str, operation: str) -> str:
        if request_type == "chat":
            return "casual_chat"
        if request_type == "query":
            return "error_code" if domain == "error_code" else "knowledge_query"
        if domain == "template":
            return "translation" if operation == "execute" else "template_config"
        return "workflow" if domain == "implementation" else "implementation"

    def _workflow_for(self, request_type: str, domain: str) -> str:
        if request_type == "chat":
            return "general_chat"
        if request_type == "query":
            return "knowledge_answer"
        if domain == "template":
            return "template_workflow"
        if domain == "implementation":
            return "message_platform_workflow"
        return "implementation_workflow"

    def _domains_from_payload(self, payload: JsonDict) -> List[str]:
        checks = [
            ("template", ("template", "templates", "messageTemplate", "message_template")),
            ("business_message", ("businessConfig", "business_config", "businessMessage", "business_message")),
            ("channel", ("channel", "channelConfig", "channel_config", "email")),
            ("send_strategy", ("strategy", "sendStrategy", "send_strategy")),
        ]
        result: List[str] = []
        for domain, keys in checks:
            if any(key in payload and payload.get(key) is not None for key in keys):
                result.append(domain)
        return result

    def _score(self, forms: List[str], lowered: str, patterns: tuple[str, ...]) -> int:
        score = 0
        for pattern in patterns:
            probe = pattern.lower()
            if probe in lowered or any(pattern in form for form in forms):
                score += 1
        return score

    def _text_forms(self, text: str) -> List[str]:
        forms = [text]
        try:
            repaired = text.encode("gbk").decode("utf-8")
        except UnicodeError:
            repaired = ""
        if repaired and repaired not in forms:
            forms.append(repaired)
        return forms

    def _channels(self, text: str) -> List[str]:
        mapping = [("短信", "sms"), ("邮件", "mail"), ("邮箱", "mail"), ("企业微信", "enterprise_wechat"), ("微信", "weixin"), ("消息中心", "uspace")]
        channels: List[str] = []
        for label, code in mapping:
            if label in text and code not in channels:
                channels.append(code)
        return channels

    def _tools_for_agents(self, agents: List[str]) -> List[str]:
        tools: List[str] = []
        for agent in agents:
            for tool in AGENT_TOOLS.get(agent, []):
                if tool not in tools:
                    tools.append(tool)
        return tools

    def _translate(self, template: JsonDict, target: str) -> JsonDict:
        token_pattern = re.compile(r"(\{\{.*?\}\}|\$\{.*?\}|\[[a-zA-Z0-9_#:. -]+\])")

        def protect(value: str) -> tuple[str, List[str]]:
            tokens = token_pattern.findall(value)
            protected = value
            for index, token in enumerate(tokens):
                protected = protected.replace(token, f"__VAR_{index}__")
            return protected, tokens

        def restore(value: str, tokens: List[str]) -> str:
            restored = value
            for index, token in enumerate(tokens):
                restored = restored.replace(f"__VAR_{index}__", token)
            return restored

        dictionary = {
            "zh_CN": {
                "Business approval notification": "业务审批通知",
                "Hello, you have a business message to process.": "您好，您有一条业务消息需要处理。",
            },
            "en_US": {
                "业务审批通知": "Business approval notification",
                "业务逾期提醒": "Business overdue reminder",
                "业务消息通知": "Business message notification",
                "您好，您有一条业务消息需要处理，请及时查看。": "Hello, you have a business message to process. Please check it in time.",
                "您好，您有一条业务消息需要处理。": "Hello, you have a business message to process. ",
            },
            "ar_SA": {
                "Business approval notification": "إشعار الموافقة على الأعمال",
                "Business overdue reminder": "تذكير بتأخر الأعمال",
                "Business message notification": "إشعار رسالة الأعمال",
                "Hello, you have a business message to process. Please check it in time.": "مرحبًا، لديك رسالة أعمال تحتاج إلى معالجة. يرجى التحقق منها في الوقت المناسب.",
                "Hello, you have a business message to process.": "مرحبًا، لديك رسالة أعمال تحتاج إلى معالجة.",
                "Purchase requisition": "طلب الشراء",
                "Purchase amount": "مبلغ الشراء",
                "Dear": "عزيزي",
                "Title": "العنوان",
                "Content": "المحتوى",
                "Approved": "تمت الموافقة",
            },
        }
        target_dict = dictionary.get(target, {})

        def translate_text(value: str) -> str:
            protected, tokens = protect(value)
            translated = protected
            for src, dst in target_dict.items():
                translated = translated.replace(src, dst)
            if translated == protected and target and not target.startswith("zh"):
                translated = f"[{target}] {protected}"
            return restore(translated, tokens)

        return {
            **template,
            "language": target,
            "title": translate_text(str(template.get("title") or template.get("templateName") or "")),
            "content": translate_text(str(template.get("content") or "")),
        }

    def _profile_patch(self, text: str) -> JsonDict:
        channels = self._channels(text)
        patch: JsonDict = {}
        if channels:
            patch["preferred_channels"] = channels
        domains = [word for word in ("采购", "销售", "财务", "库存", "审批") if word in text]
        if domains:
            patch["business_domains"] = domains
        return patch

    def _facts_patch(self, text: str) -> JsonDict:
        facts: JsonDict = {}
        email_match = re.search(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}", text)
        if email_match:
            facts["lastEmail"] = email_match.group(0)
        return facts


@dataclass
class OpenAICompatibleLLMClient(LLMClient):
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.2
    response_format: str = "json_object"

    def complete_json(self, system: str, payload: JsonDict) -> JsonDict:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system + " Return strict JSON."},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "temperature": self.temperature,
        }
        if self.response_format:
            body["response_format"] = {"type": self.response_format}
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
        return json.loads(result["choices"][0]["message"]["content"])


def build_llm(settings: Settings) -> LLMClient:
    if settings.llm_provider == "openai_compatible" and settings.llm_base_url and settings.llm_api_key:
        return OpenAICompatibleLLMClient(
            settings.llm_base_url,
            settings.llm_api_key,
            settings.llm_model,
            temperature=settings.llm_temperature,
            response_format=settings.llm_response_format,
        )
    return RuleBasedLLMClient()


def describe_llm(client: LLMClient, settings: Settings) -> JsonDict:
    online = isinstance(client, OpenAICompatibleLLMClient)
    return {
        "mode": "online" if online else "offline",
        "client": client.__class__.__name__,
        "provider": settings.llm_provider if online else "offline_rule_based",
        "configuredProvider": settings.llm_provider,
        "model": settings.llm_model,
        "baseUrlConfigured": bool(settings.llm_base_url),
        "apiKeyConfigured": bool(settings.llm_api_key),
        "temperature": settings.llm_temperature,
        "responseFormat": settings.llm_response_format,
    }
