"""Long-term memory stores and memory manager."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

from .llm import LLMClient
from .models import ConversationMemory, JsonDict, UserProfile, deep_merge, now_ts, to_jsonable


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

    def load(self, session_id: str) -> ConversationMemory:
        return self.store.load(session_id)

    def remember_request(self, memory: ConversationMemory, text: str, request_payload: JsonDict) -> ConversationMemory:
        memory.recent_messages = (memory.recent_messages + [{"role": "user", "text": text, "at": now_ts()}])[-30:]
        summary_update = self.llm.summarize(text, to_jsonable(memory))
        memory.summary = str(summary_update.get("summary") or memory.summary)
        memory.facts = deep_merge(memory.facts, summary_update.get("factsPatch") or {})
        memory.facts = deep_merge(memory.facts, _facts_from_payload(request_payload))
        profile_patch = summary_update.get("profilePatch") or {}
        memory.profile = merge_profile(memory.profile, profile_patch)
        return self.store.save(memory)

    def remember_response(self, memory: ConversationMemory, response_summary: str, facts_patch: JsonDict | None = None) -> ConversationMemory:
        memory.recent_messages = (memory.recent_messages + [{"role": "assistant", "text": response_summary, "at": now_ts()}])[-30:]
        if facts_patch:
            memory.facts = deep_merge(memory.facts, facts_patch)
        return self.store.save(memory)


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
    return ConversationMemory(
        session_id=str(payload.get("session_id") or payload.get("sessionId") or "default"),
        profile=profile,
        summary=str(payload.get("summary") or ""),
        facts=dict(payload.get("facts") or {}),
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
    if payload.get("businessConfig"):
        facts["lastBusinessConfig"] = payload["businessConfig"]
    return facts


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
