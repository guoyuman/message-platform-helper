"""Counter stores for frequency-control strategy rules."""

from __future__ import annotations

import sqlite3
import time
from abc import ABC, abstractmethod
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


class CounterStore(ABC):
    @abstractmethod
    def increment(self, key: str, ttl_seconds: int) -> int:
        raise NotImplementedError


@dataclass
class SQLiteCounterStore(CounterStore):
    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as conn:
            conn.execute(
                """
                create table if not exists counters (
                  key text primary key,
                  value integer not null,
                  expires_at real not null
                )
                """
            )
            conn.commit()

    def increment(self, key: str, ttl_seconds: int) -> int:
        now = time.time()
        with closing(sqlite3.connect(self.path)) as conn:
            row = conn.execute("select value, expires_at from counters where key=?", (key,)).fetchone()
            if not row or row[1] < now:
                value = 1
                expires_at = now + ttl_seconds
            else:
                value = int(row[0]) + 1
                expires_at = float(row[1])
            conn.execute(
                """
                insert into counters(key, value, expires_at) values(?, ?, ?)
                on conflict(key) do update set value=excluded.value, expires_at=excluded.expires_at
                """,
                (key, value, expires_at),
            )
            conn.commit()
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


def build_counter_store(data_dir: Path, redis_url: str = "") -> CounterStore:
    if redis_url:
        try:
            return RedisCounterStore(redis_url)
        except Exception:
            pass
    return SQLiteCounterStore(data_dir / "counters.sqlite3")
