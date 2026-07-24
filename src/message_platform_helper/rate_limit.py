"""Counter stores for frequency-control strategy rules."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict

from sqlalchemy.dialects.postgresql import insert

from .rag.db import CounterRecord, build_session_factory


class CounterStore(ABC):
    @abstractmethod
    def increment(self, key: str, ttl_seconds: int) -> int:
        raise NotImplementedError


@dataclass
class PostgresCounterStore(CounterStore):
    database_url: str

    def __post_init__(self) -> None:
        self._factory = build_session_factory(self.database_url)

    def increment(self, key: str, ttl_seconds: int) -> int:
        now = time.time()
        expires_at = now + ttl_seconds
        with self._factory() as session:
            record = session.get(CounterRecord, key)
            if record is None or record.expires_at < now:
                value = 1
            else:
                value = int(record.value) + 1
                expires_at = float(record.expires_at)
            stmt = (
                insert(CounterRecord)
                .values(key=key, value=value, expires_at=expires_at)
                .on_conflict_do_update(
                    index_elements=[CounterRecord.key],
                    set_={"value": value, "expires_at": expires_at},
                )
            )
            session.execute(stmt)
            session.commit()
        return value


@dataclass
class RedisCounterStore(CounterStore):
    url: str
    prefix: str = "message-platform-helper:counter:"

    def __post_init__(self) -> None:
        try:
            import redis  # type: ignore
        except ImportError as exc:
            raise RuntimeError("redis package is not installed") from exc
        self._client = redis.Redis.from_url(self.url, decode_responses=True)
        self._client.ping()

    def increment(self, key: str, ttl_seconds: int) -> int:
        full_key = self.prefix + key
        value = int(self._client.incr(full_key))
        if value == 1:
            self._client.expire(full_key, ttl_seconds)
        return value


@dataclass
class MemoryCounterStore(CounterStore):
    values: Dict[str, tuple[int, float]]

    def increment(self, key: str, ttl_seconds: int) -> int:
        now = time.time()
        value, expires_at = self.values.get(key, (0, 0.0))
        if expires_at < now:
            value = 0
            expires_at = now + ttl_seconds
        value += 1
        self.values[key] = (value, expires_at)
        return value


def build_counter_store(database_url: str, redis_url: str = "") -> CounterStore:
    if redis_url:
        try:
            return RedisCounterStore(redis_url)
        except Exception:
            pass
    if database_url and insert is not None:
        return PostgresCounterStore(database_url)
    return MemoryCounterStore({})
