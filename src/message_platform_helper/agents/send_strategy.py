"""Send strategy agent."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

from ..models import AgentResult, AgentStep, JsonDict, SendStrategy, StrategyRule, Tool, to_jsonable
from ..react import AgentContext, ReActAgent
from ..strategy import SendStrategyEvaluator


@dataclass
class SendStrategyAgent(ReActAgent):
    evaluator: SendStrategyEvaluator
    name = "send_strategy"

    def tools(self, context: AgentContext) -> Dict[str, Tool]:
        return {
            "strategy.build": Tool("strategy.build", "Build channel/schedule/frequency strategy.", lambda payload: self._build(context, payload)),
            "strategy.evaluate": Tool("strategy.evaluate", "Evaluate strategy against a sample message.", lambda payload: self._evaluate(context, payload)),
            "strategy.save": Tool("strategy.save", "Save or dry-run save strategy.", lambda payload: self._save(context, payload)),
        }

    def plan(self, context: AgentContext) -> List[JsonDict]:
        payload = dict(context.request.payload.get("strategy") or context.request.payload)
        steps = [
            {
                "thought": "The strategy request should be normalized into explicit interceptor rules.",
                "tool": "strategy.build",
                "input": payload,
            }
        ]
        if payload.get("sampleMessage") or payload.get("evaluate", True):
            steps.append(
                {
                    "thought": "A local strategy evaluation catches obvious channel, schedule, and frequency blocks.",
                    "tool": "strategy.evaluate",
                    "input": payload,
                }
            )
        if payload.get("save") or payload.get("publish"):
            steps.append(
                {
                    "thought": "The user asked to persist strategy configuration, so call the platform boundary.",
                    "tool": "strategy.save",
                    "input": payload,
                }
            )
        return steps

    def finalize(self, context: AgentContext, observations: List[JsonDict], steps: List[AgentStep]) -> AgentResult:
        output: JsonDict = {}
        issues: List[JsonDict] = []
        for item in observations:
            output.update(item)
            issues.extend(item.get("issues") or [])
        return AgentResult(agent=self.name, ok=not any(i.get("severity") == "error" for i in issues), output=output, issues=issues, steps=steps)

    def _build(self, context: AgentContext, payload: JsonDict) -> JsonDict:
        strategy = build_strategy(context.request.text, payload)
        platform_payload = strategy_platform_payload(strategy, payload)
        context.scratch["sendStrategy"] = strategy
        context.scratch["sendStrategyPayload"] = platform_payload
        return {"ok": True, "summary": "send strategy built", "strategy": to_jsonable(strategy), "strategyPayload": platform_payload}

    def _evaluate(self, context: AgentContext, payload: JsonDict) -> JsonDict:
        strategy = context.scratch.get("sendStrategy") or build_strategy(context.request.text, payload)
        sample = payload.get("sampleMessage") or {
            "channels": payload.get("channels") or ["mail"],
            "receivers": payload.get("receivers") or ["demo-user"],
            "templateCode": payload.get("templateCode") or "demo-template",
        }
        decision = self.evaluator.evaluate(strategy, sample)
        return {"ok": True, "summary": "send strategy evaluated", "strategyDecision": to_jsonable(decision)}

    def _save(self, context: AgentContext, payload: JsonDict) -> JsonDict:
        platform_payload = context.scratch.get("sendStrategyPayload") or strategy_platform_payload(build_strategy(context.request.text, payload), payload)
        response = context.platform.save_strategy(platform_payload, dry_run=context.request.dry_run)
        return {"ok": True, "summary": "strategy save boundary called", "strategySaveResponse": response}


def build_strategy(text: str, payload: JsonDict) -> SendStrategy:
    if payload.get("rules"):
        rules = [StrategyRule(kind=str(item["kind"]), config=dict(item.get("config") or {}), enabled=bool(item.get("enabled", True))) for item in payload["rules"]]
        return SendStrategy(name=str(payload.get("name") or "AI send strategy"), rules=rules, enabled=bool(payload.get("enabled", True)))
    rules: List[StrategyRule] = []
    channel_rule = parse_channel_rule(text, payload)
    if channel_rule:
        rules.append(channel_rule)
    schedule_rule = parse_schedule_rule(text, payload)
    if schedule_rule:
        rules.append(schedule_rule)
    frequency_rule = parse_frequency_rule(text, payload)
    if frequency_rule:
        rules.append(frequency_rule)
    return SendStrategy(
        name=str(payload.get("name") or "AI generated send strategy"),
        rules=rules,
        enabled=bool(payload.get("enabled", True)),
        description=text[:300],
    )


def parse_channel_rule(text: str, payload: JsonDict) -> StrategyRule | None:
    allowed = payload.get("allowedChannels")
    blocked = payload.get("blockedChannels")
    channels = channel_keywords(text)
    if not allowed and ("只" in text or "仅" in text or "only" in text.lower()) and channels:
        allowed = channels
    if not blocked and ("不发" in text or "禁止" in text or "拦截" in text or "block" in text.lower()) and channels:
        blocked = channels
    if allowed or blocked:
        return StrategyRule("channel_limit", {"allowedChannels": allowed or [], "blockedChannels": blocked or []})
    return None


def parse_schedule_rule(text: str, payload: JsonDict) -> StrategyRule | None:
    config: JsonDict = {}
    if payload.get("sendAt"):
        config["sendAt"] = payload["sendAt"]
    if payload.get("windowStart") and payload.get("windowEnd"):
        config["windowStart"] = payload["windowStart"]
        config["windowEnd"] = payload["windowEnd"]
    match = re.search(r"(\d{1,2}:\d{2})\s*[-~到至]\s*(\d{1,2}:\d{2})", text)
    if match:
        config["windowStart"] = match.group(1)
        config["windowEnd"] = match.group(2)
    if "定时" in text and payload.get("schedule"):
        config.update(payload["schedule"])
    return StrategyRule("schedule", config) if config else None


def parse_frequency_rule(text: str, payload: JsonDict) -> StrategyRule | None:
    if payload.get("frequency"):
        config = dict(payload["frequency"])
        config.setdefault("name", payload.get("name") or "frequency")
        return StrategyRule("frequency", config)
    match = re.search(r"每\s*(分钟|小时|天|日)\s*(?:最多|不超过)?\s*(?:发送)?\s*(\d+)\s*次", text)
    if not match:
        match = re.search(r"(\d+)\s*次\s*/\s*(分钟|小时|天|日)", text)
    if match:
        unit, limit = (match.group(1), match.group(2)) if match.group(1) in {"分钟", "小时", "天", "日"} else (match.group(2), match.group(1))
        ttl = {"分钟": 60, "小时": 3600, "天": 86400, "日": 86400}[unit]
        return StrategyRule("frequency", {"name": "frequency", "scope": "recipient", "limit": int(limit), "windowSeconds": ttl})
    if any(word in text for word in ("频次", "频控", "限流")):
        return StrategyRule("frequency", {"name": "frequency", "scope": "recipient", "limit": 1, "windowSeconds": 3600})
    return None


def channel_keywords(text: str) -> List[str]:
    mapping = [("短信", "sms"), ("邮件", "mail"), ("邮箱", "mail"), ("企业微信", "enterprise_wechat"), ("微信", "weixin"), ("消息中心", "uspace")]
    result: List[str] = []
    for label, code in mapping:
        if label in text and code not in result:
            result.append(code)
    return result


def strategy_platform_payload(strategy: SendStrategy, hints: JsonDict) -> JsonDict:
    return {
        "id": hints.get("id") or "",
        "name": strategy.name,
        "enabled": strategy.enabled,
        "description": strategy.description,
        "rules": [to_jsonable(rule) for rule in strategy.rules],
        "runtime": {
            "requiresRedis": any(rule.kind == "frequency" for rule in strategy.rules),
            "interceptorOrder": ["channel_limit", "schedule", "frequency"],
        },
    }
