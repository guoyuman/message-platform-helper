"""Long-term memory stores and memory manager."""

from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
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
class SQLiteMemoryStore(MemoryStore):
    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as conn:
            conn.executescript(
                """
                create table if not exists session_memory (
                  session_id text primary key,
                  memory_json text not null,
                  updated_at real not null
                );
                create table if not exists helper_runs (
                  id integer primary key autoincrement,
                  session_id text not null,
                  request_json text not null,
                  response_json text not null,
                  created_at real not null
                );
                create index if not exists idx_helper_runs_session on helper_runs(session_id, created_at desc);
                """
            )
            conn.commit()

    def load(self, session_id: str) -> ConversationMemory:
        with closing(sqlite3.connect(self.path)) as conn:
            row = conn.execute("select memory_json from session_memory where session_id=?", (session_id,)).fetchone()
        if not row:
            return ConversationMemory(session_id=session_id)
        return memory_from_dict(json.loads(row[0]))

    def save(self, memory: ConversationMemory) -> ConversationMemory:
        memory.updated_at = now_ts()
        payload = json.dumps(to_jsonable(memory), ensure_ascii=False)
        with closing(sqlite3.connect(self.path)) as conn:
            conn.execute(
                """
                insert into session_memory(session_id, memory_json, updated_at)
                values(?, ?, ?)
                on conflict(session_id) do update set
                  memory_json=excluded.memory_json,
                  updated_at=excluded.updated_at
                """,
                (memory.session_id, payload, memory.updated_at),
            )
            conn.commit()
        return memory

    def clear(self, session_id: str) -> None:
        with closing(sqlite3.connect(self.path)) as conn:
            conn.execute("delete from session_memory where session_id=?", (session_id,))
            conn.commit()

    def save_run(self, session_id: str, request: JsonDict, response: JsonDict) -> int:
        with closing(sqlite3.connect(self.path)) as conn:
            cursor = conn.execute(
                "insert into helper_runs(session_id, request_json, response_json, created_at) values(?, ?, ?, ?)",
                (
                    session_id,
                    json.dumps(request, ensure_ascii=False),
                    json.dumps(response, ensure_ascii=False),
                    now_ts(),
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)


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


def build_memory_store(data_dir: Path, redis_url: str = "") -> MemoryStore:
    if redis_url:
        try:
            return RedisMemoryStore(redis_url)
        except Exception:
            pass
    return SQLiteMemoryStore(data_dir / "memory.sqlite3")
