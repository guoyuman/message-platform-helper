"""Query analysis for RAG routing and filtering."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class QueryAnalysis:
    intent: str = "manual"
    filters: dict[str, list[str]] = field(default_factory=dict)
    terms: list[str] = field(default_factory=list)
    confidence: float = 0.0


class QueryAnalyzer:
    def analyze(self, query: str) -> QueryAnalysis:
        normalized = " ".join(str(query or "").split())
        lowered = normalized.lower()
        tags: list[str] = []
        intent = "manual"
        confidence = 0.2

        if _has_any(lowered, normalized, ("fail", "failure", "error", "troubleshoot"), ("失败", "报错", "错误", "异常", "排查", "原因")):
            intent = "troubleshooting"
            tags.append("failure")
            confidence = max(confidence, 0.75)
        if _has_any(lowered, normalized, ("template", "variable"), ("模板", "变量")):
            intent = "template"
            tags.append("template")
            confidence = max(confidence, 0.7)
        if _has_any(lowered, normalized, ("api", "endpoint"), ("接口", "字段")):
            intent = "api"
            tags.append("api")
            confidence = max(confidence, 0.65)
        if _has_any(lowered, normalized, ("mail", "email", "smtp"), ("邮件", "邮箱", "邮件通道")):
            tags.append("mail")
            confidence = max(confidence, 0.65)
        if _has_any(lowered, normalized, ("strategy", "rate limit", "schedule"), ("策略", "频控", "限流", "定时")):
            tags.append("strategy")
            confidence = max(confidence, 0.65)
        if _has_any(lowered, normalized, ("business message", "business"), ("业务消息", "单据", "触发", "接收人", "审批")):
            tags.append("business")
            confidence = max(confidence, 0.6)

        filters = {"tags": _dedupe(tags)} if tags and confidence >= 0.6 else {}
        return QueryAnalysis(intent=intent, filters=filters, terms=_query_terms(normalized), confidence=confidence)


def _has_any(lowered: str, raw: str, english: tuple[str, ...], chinese: tuple[str, ...]) -> bool:
    return any(keyword in lowered for keyword in english) or any(keyword in raw for keyword in chinese)


def _query_terms(query: str) -> list[str]:
    return _dedupe([term.lower() for term in re.findall(r"[a-zA-Z0-9_.#/-]+|[\u4e00-\u9fff]{2,}", query or "")])


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result
