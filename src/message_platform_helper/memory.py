"""Long-term memory stores and memory manager."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .llm import LLMClient
from .models import ConversationMemory, JsonDict, UserProfile, deep_merge, now_ts, to_jsonable


@dataclass(frozen=True)
class MemoryPolicy:
    recent_message_limit: int = 20
    summary_trigger_messages: int = 24
    operation_history_limit: int = 50
    fact_list_limit: int = 50
    summary_char_limit: int = 4000
    message_char_limit: int = 1200
    fact_string_char_limit: int = 2000
    consolidation_operation_threshold: int = 40
    consolidation_fact_size_threshold: int = 24000
    consolidation_summary_threshold: int = 3500
    consolidation_fact_count_threshold: int = 120


FACT_RECORDS_KEY = "factRecords"
OPERATION_HISTORY_KEY = "operationHistory"
MEMORY_META_KEY = "memoryMeta"
FACT_STATUSES = {"active", "superseded", "expired", "deleted"}
FACT_SOURCES = {"system", "admin", "user_explicit", "tool", "inferred"}
SOURCE_PRIORITY = {"system": 50, "admin": 50, "user_explicit": 40, "tool": 30, "inferred": 10}
CANONICAL_FACT_KEYS = {
    "preferredchannel": "preferred_channel",
    "preferred_channel": "preferred_channel",
    "defaultchannel": "preferred_channel",
    "messagechannelpreference": "preferred_channel",
    "lastemail": "last_email",
    "last_email": "last_email",
    "currentemail": "current_email",
    "current_email": "current_email",
    "lasttemplaterequest": "last_template_request",
    "last_template_request": "last_template_request",
    "lasttemplatetargetlanguage": "last_template_target_language",
    "last_template_target_language": "last_template_target_language",
    "lasttemplatescope": "last_template_scope",
    "last_template_scope": "last_template_scope",
    "laststrategyrequest": "last_strategy_request",
    "last_strategy_request": "last_strategy_request",
    "lastagents": "last_agents",
    "last_agents": "last_agents",
    "lastok": "last_ok",
    "last_ok": "last_ok",
}
LEGACY_FACT_KEYS = {
    "preferred_channel": "preferredChannel",
    "last_email": "lastEmail",
    "current_email": "currentEmail",
    "last_template_request": "lastTemplateRequest",
    "last_template_target_language": "lastTemplateTargetLanguage",
    "last_template_scope": "lastTemplateScope",
    "last_strategy_request": "lastStrategyRequest",
    "last_agents": "lastAgents",
    "last_ok": "lastOk",
}
RESERVED_FACT_KEYS = {FACT_RECORDS_KEY, OPERATION_HISTORY_KEY, MEMORY_META_KEY}


class MemoryStore(ABC):
    @abstractmethod
    def load(self, session_id: str) -> ConversationMemory:
        raise NotImplementedError

    @abstractmethod
    def save(self, memory: ConversationMemory) -> ConversationMemory:
        raise NotImplementedError

    @abstractmethod
    def clear(self, session_id: str) -> None:
        raise NotImplementedError


@dataclass
class PostgresMemoryStore(MemoryStore):
    database_url: str

    def __post_init__(self) -> None:
        self._helper_run_record, self._session_memory_record, build_session_factory = _load_postgres_memory_store()
        self._factory = build_session_factory(self.database_url)

    def load(self, session_id: str) -> ConversationMemory:
        with self._factory() as session:
            record = session.get(self._session_memory_record, session_id)
            if record is None:
                return ConversationMemory(session_id=session_id)
            return memory_from_dict(dict(record.memory_json or {}))

    def save(self, memory: ConversationMemory) -> ConversationMemory:
        memory.updated_at = now_ts()
        payload = to_jsonable(memory)
        with self._factory() as session:
            record = session.get(self._session_memory_record, memory.session_id)
            if record is None:
                record = self._session_memory_record(session_id=memory.session_id)
                session.add(record)
            record.memory_json = payload
            session.commit()
        return memory

    def clear(self, session_id: str) -> None:
        with self._factory() as session:
            record = session.get(self._session_memory_record, session_id)
            if record is not None:
                session.delete(record)
                session.commit()

    def save_run(self, session_id: str, request: JsonDict, response: JsonDict) -> int:
        with self._factory() as session:
            record = self._helper_run_record(session_id=session_id, request_json=request, response_json=response)
            session.add(record)
            session.commit()
            return int(record.id)

    def list_runs(self, session_id: str, limit: int = 20) -> list[JsonDict]:
        from sqlalchemy import select

        with self._factory() as session:
            rows = session.execute(
                select(self._helper_run_record.request_json, self._helper_run_record.response_json, self._helper_run_record.created_at)
                .where(self._helper_run_record.session_id == session_id)
                .order_by(self._helper_run_record.created_at.desc())
                .limit(limit)
            ).all()
        return [{"request": dict(row[0] or {}), "response": dict(row[1] or {}), "created_at": str(row[2] or "")} for row in rows]

    def list_sessions(self) -> list[JsonDict]:
        from sqlalchemy import select

        with self._factory() as session:
            rows = session.execute(
                select(self._session_memory_record.session_id, self._session_memory_record.memory_json, self._session_memory_record.updated_at)
                .order_by(self._session_memory_record.updated_at.desc())
                .limit(50)
            ).all()
        return [_session_summary(row[0], dict(row[1] or {}), row[2]) for row in rows]


@dataclass
class RedisMemoryStore(MemoryStore):
    url: str
    prefix: str = "message-platform-helper:memory:"

    def __post_init__(self) -> None:
        try:
            import redis  # type: ignore
        except ImportError as exc:
            raise RuntimeError("redis package is not installed") from exc
        self._client = redis.Redis.from_url(self.url, decode_responses=True)
        self._client.ping()

    def load(self, session_id: str) -> ConversationMemory:
        raw = self._client.get(self.prefix + session_id)
        if not raw:
            return ConversationMemory(session_id=session_id)
        return memory_from_dict(json.loads(raw))

    def save(self, memory: ConversationMemory) -> ConversationMemory:
        memory.updated_at = now_ts()
        self._client.set(self.prefix + memory.session_id, json.dumps(to_jsonable(memory), ensure_ascii=False))
        return memory

    def clear(self, session_id: str) -> None:
        self._client.delete(self.prefix + session_id)

    def list_sessions(self) -> list[JsonDict]:
        sessions: list[JsonDict] = []
        for key in self._client.scan_iter(self.prefix + "*", count=50):
            raw = self._client.get(key)
            if not raw:
                continue
            session_id = str(key).removeprefix(self.prefix)
            sessions.append(_session_summary(session_id, json.loads(raw), None))
        return sorted(sessions, key=lambda item: float(item.get("updatedAt") or 0), reverse=True)[:50]


@dataclass
class InMemoryMemoryStore(MemoryStore):
    values: Dict[str, ConversationMemory]
    runs: List[JsonDict] | None = None

    def load(self, session_id: str) -> ConversationMemory:
        return self.values.get(session_id, ConversationMemory(session_id=session_id))

    def save(self, memory: ConversationMemory) -> ConversationMemory:
        memory.updated_at = now_ts()
        self.values[memory.session_id] = memory_from_dict(to_jsonable(memory))
        return memory

    def clear(self, session_id: str) -> None:
        self.values.pop(session_id, None)

    def save_run(self, session_id: str, request: JsonDict, response: JsonDict) -> int:
        if self.runs is None:
            self.runs = []
        self.runs.append({"session_id": session_id, "request": request, "response": response, "created_at": now_ts()})
        return len(self.runs)

    def list_runs(self, session_id: str, limit: int = 20) -> list[JsonDict]:
        runs = [run for run in self.runs or [] if run.get("session_id") == session_id]
        return list(reversed(runs))[:limit]

    def list_sessions(self) -> list[JsonDict]:
        return sorted(
            [_session_summary(session_id, to_jsonable(memory), None) for session_id, memory in self.values.items()],
            key=lambda item: float(item.get("updatedAt") or 0),
            reverse=True,
        )[:50]


@dataclass
class MemoryManager:
    store: MemoryStore
    llm: LLMClient
    policy: MemoryPolicy = field(default_factory=MemoryPolicy)

    def load(self, session_id: str) -> ConversationMemory:
        return self.store.load(session_id)

    def remember_request(self, memory: ConversationMemory, text: str, request_payload: JsonDict) -> ConversationMemory:
        memory.recent_messages = memory.recent_messages + [{"role": "user", "text": _trim_text(text, self.policy.message_char_limit), "at": now_ts()}]
        memory.facts = _merge_facts(memory.facts, _facts_from_payload(request_payload), self.policy)
        memory.summary = _request_summary(memory.summary, text, self.policy.summary_char_limit)
        memory.profile = merge_profile(memory.profile, _profile_patch_from_text(text))
        return self.store.save(self._compress(memory))

    def remember_response(self, memory: ConversationMemory, response_summary: str, facts_patch: JsonDict | None = None) -> ConversationMemory:
        memory.recent_messages = memory.recent_messages + [{"role": "assistant", "text": _trim_text(response_summary, self.policy.message_char_limit), "at": now_ts()}]
        if facts_patch:
            memory.facts = _merge_facts(memory.facts, facts_patch, self.policy)
        return self.store.save(self._compress(memory))

    def _compress(self, memory: ConversationMemory) -> ConversationMemory:
        overflow = len(memory.recent_messages) - self.policy.summary_trigger_messages
        if overflow > 0:
            archived = memory.recent_messages[:overflow]
            archived_text = _messages_to_text(archived)
            if archived_text:
                summary_update = self.llm.summarize(
                    f"Compress older conversation turns into durable enterprise memory:\n{archived_text}",
                    to_jsonable(memory),
                )
                memory.summary = str(summary_update.get("summary") or memory.summary)
                memory.facts = _merge_facts(memory.facts, summary_update.get("factsPatch") or {}, self.policy)
            memory.recent_messages = memory.recent_messages[overflow:]
        memory.recent_messages = memory.recent_messages[-self.policy.recent_message_limit :]
        memory.summary = _compact_summary(memory.summary, self.policy.summary_char_limit)
        memory.operation_history = _compact_operation_history(_operation_history(memory), self.policy)
        memory.facts = _compact_facts(memory.facts, self.policy)
        if _should_consolidate(memory, self.policy):
            memory = self._consolidate(memory)
        return memory

    def _consolidate(self, memory: ConversationMemory) -> ConversationMemory:
        try:
            proposal = _validate_consolidation_proposal(self.llm.consolidate_memory(to_jsonable(memory)))
        except Exception:
            return memory
        if proposal.get("summary") is not None:
            memory.summary = _compact_summary(str(proposal.get("summary") or memory.summary), self.policy.summary_char_limit)
        memory.facts = _apply_consolidation_proposal(memory.facts, proposal, self.policy)
        memory.operation_history = _compact_operation_history(_operation_history(memory), self.policy)
        memory.facts.setdefault(MEMORY_META_KEY, {})
        if isinstance(memory.facts[MEMORY_META_KEY], dict):
            memory.facts[MEMORY_META_KEY]["lastConsolidatedAt"] = now_ts()
            memory.facts[MEMORY_META_KEY]["consolidated"] = True
            memory.facts[MEMORY_META_KEY]["operationHistoryCount"] = len(memory.operation_history)
        return memory


def merge_profile(profile: UserProfile, patch: JsonDict) -> UserProfile:
    for key in ("preferred_channels", "business_domains"):
        values = list(getattr(profile, key))
        for item in patch.get(key) or []:
            if item not in values:
                values.append(item)
        setattr(profile, key, values[-20:])
    if patch.get("traits"):
        profile.traits = deep_merge(profile.traits, patch["traits"])
    profile.updated_at = now_ts()
    return profile


def memory_from_dict(payload: JsonDict) -> ConversationMemory:
    profile_payload = payload.get("profile") or {}
    profile = UserProfile(
        user_id=str(profile_payload.get("user_id") or profile_payload.get("userId") or "anonymous"),
        tenant_id=str(profile_payload.get("tenant_id") or profile_payload.get("tenantId") or ""),
        locale=str(profile_payload.get("locale") or "zh_CN"),
        preferred_channels=list(profile_payload.get("preferred_channels") or profile_payload.get("preferredChannels") or []),
        business_domains=list(profile_payload.get("business_domains") or profile_payload.get("businessDomains") or []),
        traits=dict(profile_payload.get("traits") or {}),
        updated_at=float(profile_payload.get("updated_at") or profile_payload.get("updatedAt") or now_ts()),
    )
    facts = _compact_facts(dict(payload.get("facts") or {}), MemoryPolicy())
    operation_history = list(payload.get("operation_history") or payload.get("operationHistory") or facts.get(OPERATION_HISTORY_KEY) or [])
    operation_history = _compact_operation_history(operation_history, MemoryPolicy())
    facts[OPERATION_HISTORY_KEY] = operation_history
    return ConversationMemory(
        session_id=str(payload.get("session_id") or payload.get("sessionId") or "default"),
        profile=profile,
        summary=str(payload.get("summary") or ""),
        facts=facts,
        operation_history=operation_history,
        recent_messages=list(payload.get("recent_messages") or payload.get("recentMessages") or []),
        updated_at=float(payload.get("updated_at") or payload.get("updatedAt") or now_ts()),
    )


def _facts_from_payload(payload: JsonDict) -> JsonDict:
    facts: JsonDict = {}
    if payload.get("email"):
        facts["lastEmail"] = payload["email"]
    if payload.get("template"):
        facts["lastTemplateRequest"] = payload["template"]
        if isinstance(payload["template"], dict):
            target_language = payload["template"].get("targetLanguage") or payload["template"].get("language")
            if target_language:
                facts["lastTemplateTargetLanguage"] = target_language
            scope = {key: payload["template"].get(key) for key in ("documentId", "msgDocumentId", "groupCode", "appCode", "orgId") if payload["template"].get(key)}
            if scope:
                facts["lastTemplateScope"] = scope
    if payload.get("strategy"):
        facts["lastStrategyRequest"] = payload["strategy"]
    return facts


def _request_summary(existing: str, text: str, limit: int) -> str:
    short = _trim_text(" ".join(str(text or "").split()), 160)
    if not short:
        return existing
    if not existing:
        return short
    if short in existing:
        return _trim_text(existing, limit)
    return _trim_text(f"{existing} | {short}", limit)


def _profile_patch_from_text(text: str) -> JsonDict:
    channels = []
    lowered = str(text or "").lower()
    channel_patterns = {
        "mail": ("email", "mail", "smtp", "邮箱", "邮件", "閭", "閭欢"),
        "sms": ("sms", "短信", "鐭俊"),
        "enterprise_wechat": ("企业微信", "企微", "浼佷笟寰俊"),
        "weixin": ("微信", "寰俊"),
        "uspace": ("消息中心", "娑堟伅涓績"),
    }
    for channel, patterns in channel_patterns.items():
        if any(pattern in lowered or pattern in text for pattern in patterns):
            channels.append(channel)
    return {"preferred_channels": channels} if channels else {}


def _messages_to_text(messages: List[JsonDict]) -> str:
    lines = []
    for item in messages:
        role = str(item.get("role") or "")
        text = str(item.get("text") or "")
        if role or text:
            lines.append(f"{role}: {text}".strip())
    return "\n".join(lines)


def _merge_facts(base: JsonDict, patch: JsonDict, policy: MemoryPolicy) -> JsonDict:
    if not patch:
        return _compact_facts(base, policy)
    facts = _compact_facts(base, policy)
    operation_patch = patch.get(OPERATION_HISTORY_KEY)
    if isinstance(operation_patch, list):
        combined_history = list(facts.get(OPERATION_HISTORY_KEY) or []) + operation_patch
        facts[OPERATION_HISTORY_KEY] = _compact_operation_history(combined_history, policy)
        facts.setdefault(MEMORY_META_KEY, {})
        if isinstance(facts[MEMORY_META_KEY], dict):
            facts[MEMORY_META_KEY]["operationHistoryRawCount"] = len(combined_history)
    for key, value in patch.items():
        if key in RESERVED_FACT_KEYS:
            continue
        legacy_key = str(key)
        facts[legacy_key] = _compact_fact_value(value, policy)
        facts = _upsert_fact_record(facts, {"key": _canonical_fact_key(legacy_key), "value": value, "source": "user_explicit", "confidence": 1.0}, policy)
    return _compact_facts(facts, policy)


def _compact_facts(facts: JsonDict, policy: MemoryPolicy) -> JsonDict:
    compacted: JsonDict = {}
    records = _fact_records_from_facts(facts, policy)
    if records:
        compacted[FACT_RECORDS_KEY] = records
    for key, value in facts.items():
        if key == OPERATION_HISTORY_KEY and isinstance(value, list):
            compacted[key] = _compact_operation_history(value, policy)
        elif key == FACT_RECORDS_KEY:
            continue
        else:
            compacted[key] = _compact_fact_value(value, policy)
    return compacted


def _should_consolidate(memory: ConversationMemory, policy: MemoryPolicy) -> bool:
    operation_count = len(_operation_history(memory))
    meta = memory.facts.get(MEMORY_META_KEY) if isinstance(memory.facts.get(MEMORY_META_KEY), dict) else {}
    operation_count = max(operation_count, int(meta.get("operationHistoryRawCount") or 0))
    if operation_count > policy.consolidation_operation_threshold:
        return True
    if len(memory.summary) > policy.consolidation_summary_threshold:
        return True
    if len(_fact_records_from_facts(memory.facts, policy)) > policy.consolidation_fact_count_threshold:
        return True
    return _json_size(memory.facts) > policy.consolidation_fact_size_threshold


def _json_size(value: object) -> int:
    return len(json.dumps(to_jsonable(value), ensure_ascii=False, default=str))


def _compact_fact_value(value: Any, policy: MemoryPolicy, depth: int = 0) -> Any:
    if depth >= 5:
        return _trim_text(str(value), policy.fact_string_char_limit)
    if isinstance(value, dict):
        return {str(key): _compact_fact_value(item, policy, depth + 1) for key, item in value.items()}
    if isinstance(value, list):
        compacted = [_compact_fact_value(item, policy, depth + 1) for item in value]
        unique: list[Any] = []
        seen: set[str] = set()
        for item in compacted:
            marker = _value_marker(item)
            if marker in seen:
                continue
            seen.add(marker)
            unique.append(item)
        return unique[-policy.fact_list_limit :]
    if isinstance(value, str):
        return _trim_text(value, policy.fact_string_char_limit)
    return value


def _trim_text(value: str, limit: int) -> str:
    if limit <= 0 or len(value) <= limit:
        return value
    return value[:limit]


def _compact_summary(summary: str, limit: int) -> str:
    parts = [part.strip() for part in str(summary or "").split("|") if part.strip()]
    deduped = list(dict.fromkeys(parts))
    compacted = " | ".join(deduped)
    if len(compacted) <= limit:
        return compacted
    kept: list[str] = []
    size = 0
    for part in reversed(deduped):
        next_size = size + len(part) + (3 if kept else 0)
        if next_size > limit:
            break
        kept.append(part)
        size = next_size
    return " | ".join(reversed(kept)) or _trim_text(compacted, limit)


def _operation_history(memory: ConversationMemory) -> list[JsonDict]:
    history = list(getattr(memory, "operation_history", []) or [])
    legacy = memory.facts.get(OPERATION_HISTORY_KEY)
    if isinstance(legacy, list):
        history.extend(item for item in legacy if isinstance(item, dict))
    return _dedupe_operations(history)


def _compact_operation_history(history: list[Any], policy: MemoryPolicy) -> list[JsonDict]:
    items = [_compact_fact_value(item, policy) for item in history if isinstance(item, dict)]
    return _dedupe_operations(items)[-policy.operation_history_limit :]


def _dedupe_operations(history: list[JsonDict]) -> list[JsonDict]:
    deduped: list[JsonDict] = []
    seen: set[str] = set()
    for item in history:
        marker = _value_marker(item)
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(item)
    return deduped


def _canonical_fact_key(key: str) -> str:
    text = str(key or "").strip()
    normalized = "".join(ch for ch in text if ch.isalnum() or ch == "_").lower()
    return CANONICAL_FACT_KEYS.get(normalized, _camel_to_snake(text))


def _camel_to_snake(value: str) -> str:
    chars: list[str] = []
    previous_lower = False
    for char in value:
        if char in "- .":
            chars.append("_")
            previous_lower = False
        elif char.isupper() and previous_lower:
            chars.extend(["_", char.lower()])
            previous_lower = False
        else:
            chars.append(char.lower())
            previous_lower = char.isalnum() and not char.isupper()
    return "_".join(part for part in "".join(chars).split("_") if part)


def _fact_records_from_facts(facts: JsonDict, policy: MemoryPolicy) -> list[JsonDict]:
    records: list[JsonDict] = []
    for item in facts.get(FACT_RECORDS_KEY) or []:
        record = _normalize_fact_record(item, policy)
        if record:
            records.append(record)
    for key, value in facts.items():
        if key in RESERVED_FACT_KEYS:
            continue
        canonical = _canonical_fact_key(key)
        if not _has_record(records, canonical, value):
            records.append(_new_fact_record(canonical, value, source="user_explicit", confidence=1.0, policy=policy))
    return _resolve_fact_records(records, policy)


def _normalize_fact_record(item: Any, policy: MemoryPolicy) -> JsonDict | None:
    if not isinstance(item, dict):
        return None
    key = _canonical_fact_key(str(item.get("key") or ""))
    if not key:
        return None
    status = str(item.get("status") or "active")
    source = str(item.get("source") or "inferred")
    if status not in FACT_STATUSES or source not in FACT_SOURCES:
        return None
    now = now_ts()
    confidence = _float_between(item.get("confidence"), 0.0, 1.0, 0.5 if source == "inferred" else 1.0)
    if source == "inferred":
        confidence = min(confidence, 0.7)
    observed = _float_or(item.get("observedAt") or item.get("observed_at"), now)
    updated = _float_or(item.get("updatedAt") or item.get("updated_at"), observed)
    return {
        "key": key,
        "value": _compact_fact_value(item.get("value"), policy),
        "status": status,
        "source": source,
        "confidence": confidence,
        "observedAt": observed,
        "updatedAt": updated,
        "sourceMessageId": str(item.get("sourceMessageId") or item.get("source_message_id") or ""),
    }


def _new_fact_record(key: str, value: Any, *, source: str, confidence: float, policy: MemoryPolicy) -> JsonDict:
    now = now_ts()
    source = source if source in FACT_SOURCES else "inferred"
    return {
        "key": key,
        "value": _compact_fact_value(value, policy),
        "status": "active",
        "source": source,
        "confidence": min(float(confidence), 0.7) if source == "inferred" else float(confidence),
        "observedAt": now,
        "updatedAt": now,
        "sourceMessageId": "",
    }


def _has_record(records: list[JsonDict], key: str, value: Any) -> bool:
    marker = _value_marker(value)
    return any(record.get("key") == key and _value_marker(record.get("value")) == marker for record in records)


def _resolve_fact_records(records: list[JsonDict], policy: MemoryPolicy) -> list[JsonDict]:
    active_by_key: dict[str, JsonDict] = {}
    resolved: list[JsonDict] = []
    for record in sorted(records, key=lambda item: float(item.get("updatedAt") or 0)):
        if record.get("status") != "active":
            resolved.append(record)
            continue
        key = str(record.get("key") or "")
        current = active_by_key.get(key)
        if current is None:
            active_by_key[key] = record
            resolved.append(record)
            continue
        if _value_marker(current.get("value")) == _value_marker(record.get("value")):
            current["updatedAt"] = max(float(current.get("updatedAt") or 0), float(record.get("updatedAt") or 0))
            current["confidence"] = max(float(current.get("confidence") or 0), float(record.get("confidence") or 0))
            continue
        winner, loser = _choose_fact_winner(current, record)
        loser["status"] = "superseded"
        loser["updatedAt"] = max(float(loser.get("updatedAt") or 0), float(winner.get("updatedAt") or 0))
        if winner is record:
            active_by_key[key] = record
            resolved.append(record)
    active = [item for item in resolved if item.get("status") == "active"]
    inactive = [item for item in resolved if item.get("status") != "active"][-policy.fact_list_limit :]
    return inactive + active


def _choose_fact_winner(left: JsonDict, right: JsonDict) -> tuple[JsonDict, JsonDict]:
    left_score = (
        SOURCE_PRIORITY.get(str(left.get("source")), 0),
        float(left.get("confidence") or 0),
        float(left.get("observedAt") or 0),
        float(left.get("updatedAt") or 0),
    )
    right_score = (
        SOURCE_PRIORITY.get(str(right.get("source")), 0),
        float(right.get("confidence") or 0),
        float(right.get("observedAt") or 0),
        float(right.get("updatedAt") or 0),
    )
    return (right, left) if right_score >= left_score else (left, right)


def _upsert_fact_record(facts: JsonDict, proposal: JsonDict, policy: MemoryPolicy) -> JsonDict:
    now = now_ts()
    record = _normalize_fact_record({**proposal, "status": proposal.get("status") or "active", "observedAt": proposal.get("observedAt") or now, "updatedAt": proposal.get("updatedAt") or now}, policy)
    if not record:
        return facts
    records = _fact_records_from_facts(facts, policy)
    if not _has_record(records, str(record["key"]), record.get("value")):
        records.append(record)
    facts[FACT_RECORDS_KEY] = _resolve_fact_records(records, policy)
    active = _active_fact(facts[FACT_RECORDS_KEY], str(record["key"]))
    legacy_key = LEGACY_FACT_KEYS.get(str(record["key"]))
    if legacy_key and active:
        facts[legacy_key] = active.get("value")
    if str(record["key"]) == "preferred_channel" and active:
        facts["preferredChannel"] = active.get("value")
    return facts


def _active_fact(records: list[JsonDict], key: str) -> JsonDict | None:
    for record in reversed(records):
        if record.get("key") == key and record.get("status") == "active":
            return record
    return None


def _apply_consolidation_proposal(facts: JsonDict, proposal: JsonDict, policy: MemoryPolicy) -> JsonDict:
    next_facts = _compact_facts(facts, policy)
    for item in proposal.get("upserts") or []:
        next_facts = _upsert_fact_record(next_facts, item, policy)
    records = _fact_records_from_facts(next_facts, policy)
    for item in proposal.get("superseded") or []:
        _mark_records(records, item, "superseded")
    for item in proposal.get("deleted") or []:
        _mark_records(records, item, "deleted")
    next_facts[FACT_RECORDS_KEY] = _resolve_fact_records(records, policy)
    next_facts[OPERATION_HISTORY_KEY] = _compact_operation_history(list(facts.get(OPERATION_HISTORY_KEY) or []), policy)
    return _compact_facts(next_facts, policy)


def _mark_records(records: list[JsonDict], selector: JsonDict, status: str) -> None:
    key = _canonical_fact_key(str(selector.get("key") or ""))
    marker = _value_marker(selector.get("value")) if "value" in selector else ""
    for record in records:
        if record.get("key") != key:
            continue
        if marker and _value_marker(record.get("value")) != marker:
            continue
        record["status"] = status
        record["updatedAt"] = now_ts()


def _validate_consolidation_proposal(result: Any) -> JsonDict:
    if not isinstance(result, dict):
        raise ValueError("consolidation proposal must be a dict")
    allowed = {"summary", "upserts", "superseded", "deleted", "warnings"}
    if any(key not in allowed for key in result):
        raise ValueError("unexpected consolidation proposal field")
    proposal: JsonDict = {"summary": result.get("summary"), "upserts": [], "superseded": [], "deleted": [], "warnings": []}
    for list_key in ("upserts", "superseded", "deleted", "warnings"):
        value = result.get(list_key) or []
        if not isinstance(value, list):
            raise ValueError(f"{list_key} must be a list")
        proposal[list_key] = value
    for item in proposal["upserts"]:
        if not isinstance(item, dict) or not item.get("key") or "value" not in item:
            raise ValueError("invalid upsert")
        source = str(item.get("source") or "inferred")
        confidence = _float_between(item.get("confidence"), 0.0, 1.0, 0.5)
        if source not in FACT_SOURCES:
            raise ValueError("invalid source")
        if source == "inferred" and confidence > 0.7:
            raise ValueError("inferred fact confidence too high")
    for list_key in ("superseded", "deleted"):
        for item in proposal[list_key]:
            if not isinstance(item, dict) or not item.get("key"):
                raise ValueError(f"invalid {list_key} selector")
    return proposal


def _float_between(value: Any, minimum: float, maximum: float, default: float) -> float:
    return min(max(_float_or(value, default), minimum), maximum)


def _float_or(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _value_marker(value: Any) -> str:
    return json.dumps(to_jsonable(value), ensure_ascii=False, sort_keys=True, default=str)


def _session_summary(session_id: str, memory: JsonDict, updated_at: object) -> JsonDict:
    return {
        "sessionId": session_id,
        "summary": str(memory.get("summary") or ""),
        "updatedAt": memory.get("updated_at") or memory.get("updatedAt") or str(updated_at or ""),
        "recentMessageCount": len(memory.get("recent_messages") or memory.get("recentMessages") or []),
    }


def build_memory_store(database_url: str, redis_url: str = "") -> MemoryStore:
    if redis_url:
        try:
            return RedisMemoryStore(redis_url)
        except Exception:
            pass
    if database_url and _has_sqlalchemy():
        return PostgresMemoryStore(database_url)
    return InMemoryMemoryStore({})


def _has_sqlalchemy() -> bool:
    try:
        import sqlalchemy  # noqa: F401
    except ImportError:
        return False
    return True


def _load_postgres_memory_store():
    from .rag.db import HelperRunRecord, SessionMemoryRecord, build_session_factory

    return HelperRunRecord, SessionMemoryRecord, build_session_factory
