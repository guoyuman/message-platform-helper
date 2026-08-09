"""LLM boundary with deterministic fallback."""

from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

import requests

from .config import Settings


JsonDict = Dict[str, Any]


AGENT_TOOLS = {
    "knowledge": ["knowledge.search", "knowledge.answer"],
    "template": ["domain.tree.get", "business_object.resolve", "template.sync_language", "platform.template.save"],
    "channel_config": ["channel.infer_email", "platform.channel.save"],
}

DOMAIN_AGENTS = {
    "template": ["template"],
    "channel": ["channel_config"],
    "error_code": ["knowledge"],
}

PROMPT_CACHE = {
    "template.translate": (
        "Translate a message template. "
        "你是企业消息平台的模板本地化引擎。请把输入模板翻译为 targetLanguage，并返回严格 JSON。"
        "只翻译面向最终用户的人类可读文本；不要翻译字段名、变量名、枚举值、URL、代码、占位符或平台配置。"
        "必须逐字保留变量占位符，例如 ${x}、{{x}}、#{x}、%s、{0}，数量、顺序和拼写都不能改变。"
        "必须保留 HTML/XML 标签、属性、表格结构、换行、富文本结构和按钮/链接结构。"
        "如果某段文本无法可靠翻译，保留原文并在 issues 中说明。"
        "不要新增、删除或改写模板结构；不要编造源模板中不存在的内容。"
        "返回字段至少包含 translatedTemplate、targetLanguage、issues。"
    ),
    "channel.configure": (
        "Infer email channel configuration from the user's email/channel hints. "
        "你是企业消息平台的邮件通道配置抽取器。请根据用户文本和 payload 推断邮件通道配置，并返回严格 JSON。"
        "只能使用用户或 payload 已提供的信息，以及常见邮箱域名的 SMTP 默认值；不要编造密码、授权码、账号归属或验证结果。"
        "如果缺少 email、password/mailPwd、sender、verifyUser 等执行必需字段，请写入 missingFields。"
        "SMTP host、port、TLS/SSL 可由邮箱域名默认规则推断；无法确定时保守返回 missingFields 或 issues。"
        "返回字段至少包含 email、mailHost、mailPort、username、sender、verifyUser、smtpSSL、smtpTLS、missingFields、issues。"
    ),
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

    def complete_chat(
        self,
        messages: List[JsonDict],
        tools: List[JsonDict] | None = None,
        tool_choice: JsonDict | str | None = None,
    ) -> JsonDict:
        """Native chat completions with optional function-calling tools.

        Returns the assistant message dict (``content`` plus ``tool_calls``
        when the model chose to call tools). ``tool_choice`` is either the
        string ``"auto"``/``"none"`` or a forced tool object; not every
        provider accepts a forced tool (DeepSeek thinking mode returns 400).
        Only providers that implement the tools protocol override this; the
        offline rule-based client intentionally does not.
        """
        raise NotImplementedError

    def answer(self, text: str, system_prompt: str = "") -> JsonDict:
        system = system_prompt or (
            "Answer the user's chat message. "
            "你是企业消息平台助手。请直接回答用户当前消息；需要引用记忆时只使用提供的 memory，"
            "不要声称拥有未提供的历史记录。返回 JSON，包含 answer 字段。"
        )
        return self.complete_json(
            f"{system} Return JSON with an answer field.",
            {"text": text},
        )

    def decide(self, text: str, hints: JsonDict) -> JsonDict:
        return self.complete_json(
            (
                "Decide whether RAG retrieval is needed and which agents should run. "
                "请根据 request_type、domain、operation 和 hints 判断是否需要知识库检索，并选择 knowledge、template、channel_config。"
                "query 通常需要 knowledge/RAG；模板同步和通道配置 action 通常不需要 RAG。返回严格 JSON。"
            ),
            {"text": text, "hints": hints},
        )

    def translate_template(self, template: JsonDict, target_language: str, source_language: str = "") -> JsonDict:
        return self.complete_json(
            PROMPT_CACHE["template.translate"],
            {"template": template, "targetLanguage": target_language, "sourceLanguage": source_language},
        )

    def summarize(self, text: str, memory: JsonDict) -> JsonDict:
        return self.complete_json(
            (
                "Update conversation summary and profile facts. Return JSON only. "
                "请只从当前 text 中提取稳定偏好、邮箱、业务域、最近操作等长期有用信息；"
                "不要记录一次性闲聊或不确定推断。返回 summary、profilePatch、factsPatch。"
            ),
            {"text": text, "memory": memory},
        )

    def consolidate_memory(self, memory: JsonDict) -> JsonDict:
        return self.complete_json(
            (
                "Consolidate enterprise conversation memory. Deduplicate repeated facts, "
                "resolve contradictions by keeping the latest current fact, remove obsolete low-value memory, "
                "and keep a concise operationHistory. "
                "请保留对后续消息平台任务有用的事实，删除重复、过期和低价值内容。Return JSON with summary and facts."
            ),
            {"memory": memory},
        )


@dataclass
class RuleBasedLLMClient(LLMClient):
    """Offline model used for tests and local demos."""

    def complete_json(self, system: str, payload: JsonDict) -> JsonDict:
        text = str(payload.get("text") or "")
        lowered = text.lower()
        if "Classify user intent" in system:
            decision = self._route(text, payload.get("payload") or {})
            task_slices = self._task_slices(text, payload.get("payload") or {})
            if len(task_slices) > 1:
                selected_agents = _dedupe_strings(
                    agent
                    for task in task_slices
                    for agent in task.get("selectedAgents") or []
                )
                selected_tools = _dedupe_strings(
                    tool
                    for task in task_slices
                    for tool in task.get("selectedTools") or []
                )
                decision = RoutingContext(
                    request_type="action" if any(task.get("request_type") != "query" for task in task_slices) else "query",
                    domain="implementation",
                    operation="execute",
                    intent="workflow",
                    workflow="message_platform_workflow",
                    selected_agents=selected_agents,
                    selected_tools=selected_tools,
                    need_retrieval=any(bool(task.get("needRag")) for task in task_slices),
                    confidence=decision.confidence,
                    labels=["multi_task", *selected_agents],
                    rationale="Offline routing split the request into independent taskSlices before selecting agents.",
                )
            result = {
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
                "taskSlices": task_slices,
            }
            result.update(self._template_sync_metadata(text, payload.get("payload") or {}))
            return result
        if "Resolve a business object" in system:
            return self._resolve_business_object(payload)
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
        if "Consolidate enterprise conversation memory" in system:
            return self._consolidate_memory(payload.get("memory") or {})
        if "answer field" in system or "Answer user chat" in system or "Answer the user's chat message" in system:
            memory_answer = self._memory_answer(system)
            if memory_answer:
                return {"ok": True, "answer": memory_answer, "source": "offline_rule_based"}
            return {"ok": True, "answer": text, "source": "offline_rule_based"}
        return {"ok": True, "echo": payload, "note": lowered[:20]}

    def _route(self, text: str, payload: JsonDict) -> RoutingContext:
        signals = self._semantic_signals(text, payload)
        request_type = self._classify_request_type(signals, payload)
        domain = self._classify_domain(signals, payload)
        operation = self._classify_operation(signals, request_type)
        intent = self._intent_for(request_type, domain, operation)
        if request_type == "action" and domain == "template" and operation == "execute" and (
            "sync" in signals["lowered"] or "\u540c\u6b65" in text
        ):
            intent = "template_translation_sync"
        agents = self._agents_for(request_type, domain, signals)
        workflow = self._workflow_for(request_type, domain, agents)
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
            "explicit_knowledge": bool(payload.get("../../knowledge")) or self._score(forms, lowered, KNOWLEDGE_PATTERNS) > 0,
        }

    def _task_slices(self, text: str, payload: JsonDict) -> List[JsonDict]:
        clauses = _split_task_clauses(text)
        tasks: List[JsonDict] = []
        if any(_clause_has_knowledge_signal(clause) for clause in clauses) or payload.get("knowledgeQuery") or payload.get("knowledge_query"):
            task_text = _task_text_for("knowledge", clauses, text)
            tasks.append(
                {
                    "text": task_text,
                    "request_type": "query",
                    "domain": "general",
                    "operation": "explain",
                    "intent": "knowledge_query",
                    "workflow": "knowledge_answer",
                    "selectedAgents": ["knowledge"],
                    "selectedTools": ["knowledge.search", "knowledge.answer"],
                    "searchQuery": task_text,
                    "needRag": True,
                    "payloadDomain": "knowledge",
                }
            )
        if _has_template_task(text, payload, clauses):
            task_text = _task_text_for("template", clauses, text)
            tasks.append(
                {
                    "text": task_text,
                    "request_type": "action",
                    "domain": "template",
                    "operation": "execute",
                    "intent": "template_translation_sync",
                    "workflow": "template_workflow",
                    "selectedAgents": ["template"],
                    "selectedTools": AGENT_TOOLS["template"],
                    "searchQuery": "",
                    "needRag": False,
                    "payloadDomain": "template",
                }
            )
        if _has_channel_task(text, payload, clauses):
            task_text = _task_text_for("channel_config", clauses, text)
            tasks.append(
                {
                    "text": task_text,
                    "request_type": "action",
                    "domain": "channel",
                    "operation": "create",
                    "intent": "implementation",
                    "workflow": "implementation_workflow",
                    "selectedAgents": ["channel_config"],
                    "selectedTools": AGENT_TOOLS["channel_config"],
                    "searchQuery": "",
                    "needRag": False,
                    "payloadDomain": "channel",
                }
            )
        return tasks

    def _memory_answer(self, system: str) -> str:
        if "operationHistory" not in system:
            return ""
        names = re.findall(r"'templateName': '([^']+)'", system)
        codes = re.findall(r"'templateCode': '([^']+)'", system)
        labels = names or codes
        if not labels:
            return ""
        return "\u4f60\u540c\u6b65\u8fc7\u8fd9\u4e9b\u5355\u636e\u7684\u6a21\u677f\u7ffb\u8bd1\uff1a" + "\u3001".join(dict.fromkeys(labels))

    def _consolidate_memory(self, memory: JsonDict) -> JsonDict:
        facts = dict(memory.get("facts") or {})
        history = list(facts.get("operationHistory") or [])
        consolidated_history = _dedupe_operations(history)[-20:]
        facts = _dedupe_fact_lists(facts)
        if consolidated_history:
            facts["operationHistory"] = consolidated_history
        else:
            facts.pop("operationHistory", None)
        facts["memoryMeta"] = {
            "consolidated": True,
            "operationHistoryCount": len(consolidated_history),
        }
        return {
            "summary": str(memory.get("summary") or "")[-800:],
            "facts": facts,
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
            return agents
        return list(DOMAIN_AGENTS.get(domain, []))

    def _intent_for(self, request_type: str, domain: str, operation: str) -> str:
        if request_type == "chat":
            return "casual_chat"
        if request_type == "query":
            return "error_code" if domain == "error_code" else "knowledge_query"
        if domain == "template":
            return "translation" if operation == "execute" else "template_config"
        return "workflow" if domain == "implementation" else "implementation"

    def _workflow_for(self, request_type: str, domain: str, agents: List[str] | None = None) -> str:
        if request_type == "chat":
            return "general_chat"
        if request_type == "query":
            return "knowledge_answer"
        if domain == "template":
            return "template_workflow"
        if domain == "implementation":
            return "message_platform_workflow" if agents else "general_chat"
        if domain == "business_message":
            return "general_chat"
        return "implementation_workflow"

    def _domains_from_payload(self, payload: JsonDict) -> List[str]:
        checks = [
            ("template", ("template", "templates", "messageTemplate", "message_template")),
            ("channel", ("channel", "channelConfig", "channel_config", "email")),
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
        token_pattern = re.compile(r"(<[^>]+>|\{\{.*?\}\}|\$\{.*?\}|#\{.*?\}|\[[a-zA-Z0-9_#:. -]+\])")

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

    def _template_sync_metadata(self, text: str, payload: JsonDict) -> JsonDict:
        if "template" not in text.lower() and "\u6a21\u677f" not in text:
            return {}
        if "sync" not in text.lower() and "\u540c\u6b65" not in text:
            return {}
        business_object = _extract_between(text, "\u5c06", "\u4e0b") or _extract_before(text, "\u4e0b")
        target_language = (
            "\u5370\u5c3c\u8bed"
            if "\u5370\u5c3c" in text
            else "\u963f\u62c9\u4f2f\u8bed"
            if "\u963f\u62c9\u4f2f" in text
            else ""
        )
        return {
            "businessObject": business_object,
            "targetLanguage": target_language,
            "operation": "sync",
        }

    def _resolve_business_object(self, payload: JsonDict) -> JsonDict:
        name = str(payload.get("name") or "")
        candidates = payload.get("candidates") or []
        if not name or not isinstance(candidates, list):
            return {"ok": False}
        normalized = re.sub(r"[\s_\-:：/\\]+", "", name.lower())
        for candidate in candidates:
            candidate_name = str(candidate.get("businessObjectName") or "")
            if re.sub(r"[\s_\-:：/\\]+", "", candidate_name.lower()) == normalized:
                return {**candidate, "confidence": 1.0}
        return {"ok": False}

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
        if "Translate a message template" in system:
            system = PROMPT_CACHE["template.translate"]
        elif "Infer email channel configuration" in system:
            system = PROMPT_CACHE["channel.configure"]
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
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=150,
            )
            response.raise_for_status()
            result = response.json()
        except requests.Timeout as exc:
            raise TimeoutError(f"LLM request timed out after 150s: model={self.model}, base_url={self.base_url}") from exc
        except requests.RequestException as exc:
            raise _llm_request_error(self.model, self.base_url, exc) from exc
        return json.loads(result["choices"][0]["message"]["content"])

    def complete_chat(
        self,
        messages: List[JsonDict],
        tools: List[JsonDict] | None = None,
        tool_choice: JsonDict | str | None = None,
    ) -> JsonDict:
        body: JsonDict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            body["tools"] = tools
        if tool_choice:
            body["tool_choice"] = tool_choice
        # No response_format here: json_object mode is incompatible with
        # the tools protocol on several providers. Tool-call JSON is the
        # structured-output mechanism in this path.
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=150,
            )
            response.raise_for_status()
            result = response.json()
        except requests.Timeout as exc:
            raise TimeoutError(f"LLM request timed out after 150s: model={self.model}, base_url={self.base_url}") from exc
        except requests.RequestException as exc:
            raise _llm_request_error(self.model, self.base_url, exc) from exc
        return result["choices"][0]["message"]


def _llm_request_error(model: str, base_url: str, exc: requests.RequestException) -> RuntimeError:
    """Build the RuntimeError for a failed LLM request, including the
    provider's response body when one was returned (400s carry the real
    reason, e.g. unsupported tool_choice)."""
    detail = ""
    if exc.response is not None:
        detail = f", body={exc.response.text[:300]}"
    return RuntimeError(f"LLM request failed: model={model}, base_url={base_url}, error={exc}{detail}")


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


def _extract_between(text: str, start: str, end: str) -> str:
    if start not in text or end not in text:
        return ""
    value = text.split(start, 1)[1].split(end, 1)[0]
    return value.strip()


def _extract_before(text: str, marker: str) -> str:
    if marker not in text:
        return ""
    value = text.split(marker, 1)[0]
    return value.strip()


def _dedupe_operations(history: List[JsonDict]) -> List[JsonDict]:
    result: List[JsonDict] = []
    seen: set[str] = set()
    for item in reversed(history):
        key = json.dumps(
            {
                "agent": item.get("agent"),
                "workflow": item.get("workflow"),
                "summary": item.get("summary"),
                "details": item.get("details"),
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return list(reversed(result))


def _split_task_clauses(text: str) -> List[str]:
    clauses = [item.strip() for item in re.split(r"[，,。；;、\n]+|(?:\s+and\s+)|(?:\s+then\s+)", text or "", flags=re.IGNORECASE)]
    return [item for item in clauses if item]


def _task_text_for(agent: str, clauses: List[str], fallback: str) -> str:
    matchers = {
        "knowledge": _clause_has_knowledge_signal,
        "template": _clause_has_template_signal,
        "channel_config": _clause_has_channel_signal,
    }
    matcher = matchers[agent]
    selected = [clause for clause in clauses if matcher(clause)]
    return "，".join(selected) if selected else fallback


def _has_template_task(text: str, payload: JsonDict, clauses: List[str]) -> bool:
    if any(key in payload for key in ("template", "templates", "templateId", "templateIds")):
        return True
    return any(
        _clause_has_template_signal(clause) and _clause_has_action_signal(clause) and not _clause_has_query_guard(clause)
        for clause in clauses
    )


def _has_channel_task(text: str, payload: JsonDict, clauses: List[str]) -> bool:
    if any(key in payload for key in ("channel", "channelConfig", "channel_config", "email")):
        return not clauses or any(_clause_has_action_signal(clause) and not _clause_has_query_guard(clause) for clause in clauses)
    return any(
        _clause_has_channel_signal(clause) and _clause_has_action_signal(clause) and not _clause_has_query_guard(clause)
        for clause in clauses
    )


def _clause_has_knowledge_signal(clause: str) -> bool:
    lowered = clause.lower()
    return any(
        item in lowered or item in clause
        for item in (
            "knowledge",
            "knowledge base",
            "how to",
            "why",
            "troubleshoot",
            "failure",
            "failed",
            "manual",
            "doc",
            "知识库",
            "查询",
            "如何",
            "怎么",
            "为什么",
            "排查",
            "失败",
            "报错",
            "说明",
            "文档",
        )
    )


def _clause_has_template_signal(clause: str) -> bool:
    lowered = clause.lower()
    return any(item in lowered or item in clause for item in ("template", "translate", "sync", "translation", "模板", "翻译", "同步", "语种", "多语"))


def _clause_has_channel_signal(clause: str) -> bool:
    lowered = clause.lower()
    return any(item in lowered or item in clause for item in ("smtp", "email", "mail", "channel", "邮箱", "邮件", "通道"))


def _clause_has_action_signal(clause: str) -> bool:
    lowered = clause.lower()
    return any(
        item in lowered or item in clause
        for item in ("configure", "config", "setup", "create", "save", "sync", "translate", "test", "配置", "创建", "新增", "保存", "同步", "翻译", "测试")
    )


def _clause_has_query_guard(clause: str) -> bool:
    lowered = clause.lower()
    return any(
        item in lowered or item in clause
        for item in ("how to", "how do", "what is", "why", "troubleshoot", "manual", "doc", "knowledge", "如何", "怎么", "怎样", "为什么", "排查", "说明", "文档", "知识库", "查询")
    )


def _dedupe_strings(values) -> List[str]:
    result: List[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in result:
            result.append(text)
    return result


def _dedupe_fact_lists(facts: JsonDict) -> JsonDict:
    result: JsonDict = {}
    for key, value in facts.items():
        if isinstance(value, list):
            deduped = []
            seen: set[str] = set()
            for item in value:
                marker = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
                if marker not in seen:
                    seen.add(marker)
                    deduped.append(item)
            result[key] = deduped
        else:
            result[key] = value
    return result


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
