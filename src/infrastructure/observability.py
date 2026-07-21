"""Structured tracing and metrics helpers."""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field

from ..models import AssistantRequest, JsonDict, now_ts


logger = logging.getLogger("message_platform_helper")


@dataclass(frozen=True)
class TraceContext:
    trace_id: str
    request_id: str
    conversation_id: str
    tenant_id: str = ""
    user_id: str = ""

    @classmethod
    def from_request(cls, request: AssistantRequest) -> "TraceContext":
        return cls(
            trace_id=request.trace_id or _new_id("trace"),
            request_id=request.request_id or _new_id("req"),
            conversation_id=request.session_id,
            tenant_id=request.tenant_id,
            user_id=request.user_id,
        )


@dataclass
class RequestMetrics:
    started_at: float = field(default_factory=now_ts)
    latency_ms: float = 0.0
    rag_time_ms: float = 0.0
    tool_time_ms: float = 0.0
    model_time_ms: float = 0.0
    token_usage: JsonDict = field(default_factory=dict)

    def finish(self) -> "RequestMetrics":
        self.latency_ms = round((time.perf_counter() - self._perf_started) * 1000, 3)
        return self

    def __post_init__(self) -> None:
        self._perf_started = time.perf_counter()

    def as_dict(self) -> JsonDict:
        return {
            "started_at": self.started_at,
            "latency_ms": self.latency_ms,
            "rag_time_ms": round(self.rag_time_ms, 3),
            "tool_time_ms": round(self.tool_time_ms, 3),
            "model_time_ms": round(self.model_time_ms, 3),
            "token_usage": self.token_usage,
        }


def elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def log_request_completed(trace: TraceContext, metrics: RequestMetrics, payload: JsonDict) -> None:
    logger.info(
        json.dumps(
            {
                "event": "request.completed",
                "trace_id": trace.trace_id,
                "request_id": trace.request_id,
                "conversation_id": trace.conversation_id,
                "tenant_id": trace.tenant_id,
                "user_id": trace.user_id,
                "metrics": metrics.as_dict(),
                **payload,
            },
            ensure_ascii=False,
        )
    )


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


__all__ = ["RequestMetrics", "TraceContext", "elapsed_ms", "log_request_completed"]
