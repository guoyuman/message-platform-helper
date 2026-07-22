"""Send strategy construction and local evaluation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List

from .models import JsonDict, SendStrategy, StrategyDecision, StrategyRule
from .rate_limit import CounterStore


@dataclass
class SendStrategyEvaluator:
    counter_store: CounterStore

    def evaluate(self, strategy: SendStrategy, message: JsonDict, now: float | None = None) -> StrategyDecision:
        if not strategy.enabled:
            return StrategyDecision(allowed=True, reasons=["strategy disabled"])
        now = now or time.time()
        reasons: List[str] = []
        matched: List[str] = []
        delay_until = None
        for rule in strategy.rules:
            if not rule.enabled:
                continue
            if rule.kind == "channel_limit":
                allowed_channels = set(rule.config.get("allowedChannels") or [])
                blocked_channels = set(rule.config.get("blockedChannels") or [])
                channels = set(message.get("channels") or [])
                if blocked_channels.intersection(channels):
                    reasons.append("blocked by channel_limit.blockedChannels")
                    matched.append(rule.kind)
                if allowed_channels and not channels.issubset(allowed_channels):
                    reasons.append("blocked by channel_limit.allowedChannels")
                    matched.append(rule.kind)
            elif rule.kind == "schedule":
                decision = _evaluate_schedule(rule.config, now)
                if not decision.allowed:
                    reasons.extend(decision.reasons)
                    matched.append(rule.kind)
                    delay_until = decision.delay_until or delay_until
            elif rule.kind == "frequency":
                key = frequency_key(rule.config, message)
                ttl = int(rule.config.get("windowSeconds") or 3600)
                limit = int(rule.config.get("limit") or 1)
                count = self.counter_store.increment(key, ttl)
                matched.append(rule.kind)
                if count > limit:
                    reasons.append(f"blocked by frequency limit {limit}/{ttl}s")
        return StrategyDecision(
            allowed=not reasons,
            reasons=reasons or ["allowed"],
            delay_until=delay_until,
            matched_rules=matched,
        )


def _evaluate_schedule(config: JsonDict, now: float) -> StrategyDecision:
    send_at = config.get("sendAt")
    if send_at:
        target = parse_time(str(send_at))
        if target and now < target:
            return StrategyDecision(allowed=False, reasons=["scheduled for future"], delay_until=target)
    window_start = config.get("windowStart")
    window_end = config.get("windowEnd")
    if window_start and window_end:
        current = datetime.fromtimestamp(now).strftime("%H:%M")
        if not (str(window_start) <= current <= str(window_end)):
            return StrategyDecision(allowed=False, reasons=["outside send window"])
    return StrategyDecision(allowed=True, reasons=["schedule matched"])


def parse_time(value: str) -> float | None:
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            parsed = datetime.strptime(value, fmt)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.timestamp()
        except ValueError:
            continue
    return None


def frequency_key(config: JsonDict, message: JsonDict) -> str:
    scope = str(config.get("scope") or "recipient")
    if scope == "channel":
        value = ",".join(message.get("channels") or [])
    elif scope == "template":
        value = str(message.get("templateCode") or message.get("templateId") or "unknown")
    else:
        value = ",".join(message.get("receivers") or message.get("receiverIds") or ["unknown"])
    return f"{scope}:{value}:{config.get('name') or 'default'}"
