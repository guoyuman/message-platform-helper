"""Structured tracing and metrics helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, ContextManager

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


class NoopObservation:
    def update(self, **kwargs: Any) -> None:
        return None


class NoopObserver:
    enabled = False

    def start_observation(
        self,
        name: str,
        *,
        as_type: str = "span",
        input: Any | None = None,
        metadata: JsonDict | None = None,
        model: str | None = None,
        model_parameters: JsonDict | None = None,
        trace_context: JsonDict | None = None,
    ) -> ContextManager[NoopObservation]:
        return nullcontext(NoopObservation())

    def start_request(self, trace: TraceContext, request: AssistantRequest) -> ContextManager[NoopObservation]:
        return self.start_observation("message_platform_helper.request")

    def propagate(self, trace: TraceContext, request: AssistantRequest) -> ContextManager[None]:
        return nullcontext()

    def score_current_trace(self, name: str, value: float | str | bool, **kwargs: Any) -> None:
        return None

    def flush(self) -> None:
        return None

    def status(self) -> JsonDict:
        return {"enabled": False, "provider": "langfuse", "configured": False}


class LangfuseObserver(NoopObserver):
    enabled = True

    def __init__(self, client: Any, propagate_attributes: Any | None = None, environment: str = "") -> None:
        self.client = client
        self._propagate_attributes = propagate_attributes
        self.environment = environment

    @classmethod
    def from_env(cls) -> NoopObserver:
        configured = bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))
        enabled_raw = os.environ.get("MESSAGE_HELPER_LANGFUSE_ENABLED")
        enabled = configured if enabled_raw is None else _truthy(enabled_raw)
        if not enabled and not configured:
            return NoopObserver()
        if not enabled:
            return NoopObserver()
        try:
            from langfuse import get_client, propagate_attributes
        except Exception as exc:  # pragma: no cover - depends on optional extra
            logger.warning("Langfuse requested but unavailable: %s", exc)
            return NoopObserver()
        try:
            return cls(get_client(), propagate_attributes, os.environ.get("LANGFUSE_TRACING_ENVIRONMENT", ""))
        except Exception as exc:  # pragma: no cover - defensive against SDK setup errors
            logger.warning("Langfuse initialization failed: %s", exc)
            return NoopObserver()

    def start_observation(
        self,
        name: str,
        *,
        as_type: str = "span",
        input: Any | None = None,
        metadata: JsonDict | None = None,
        model: str | None = None,
        model_parameters: JsonDict | None = None,
        trace_context: JsonDict | None = None,
    ) -> ContextManager[Any]:
        return _LangfuseObservationContext(
            self.client,
            {
                "name": name,
                "as_type": as_type,
                "input": sanitize_for_observability(input),
                "metadata": sanitize_for_observability(metadata or {}),
                "model": model,
                "model_parameters": sanitize_for_observability(model_parameters or {}),
                "trace_context": trace_context,
            },
        )

    def start_request(self, trace: TraceContext, request: AssistantRequest) -> ContextManager[Any]:
        langfuse_trace_id = self._trace_id(trace.trace_id or trace.request_id)
        return self.start_observation(
            "message_platform_helper.request",
            as_type="agent",
            input={"text": request.text, "payload": request.payload, "dry_run": request.dry_run},
            metadata={
                "external_trace_id": trace.trace_id,
                "request_id": trace.request_id,
                "conversation_id": trace.conversation_id,
                "tenant_id_hash": stable_hash(trace.tenant_id),
                "user_id_hash": stable_hash(trace.user_id),
                "locale": request.locale,
                "dry_run": str(request.dry_run).lower(),
            },
            trace_context={"trace_id": langfuse_trace_id},
        )

    def propagate(self, trace: TraceContext, request: AssistantRequest) -> ContextManager[Any]:
        if self._propagate_attributes is None:
            return nullcontext()
        return self._propagate_attributes(
            user_id=stable_hash(trace.user_id) or "anonymous",
            session_id=trace.conversation_id,
            metadata={
                "external_trace_id": trace.trace_id,
                "request_id": trace.request_id,
                "tenant_id_hash": stable_hash(trace.tenant_id),
                "dry_run": str(request.dry_run).lower(),
            },
            environment=self.environment or None,
            trace_name="message_platform_helper.request",
        )

    def score_current_trace(self, name: str, value: float | str | bool, **kwargs: Any) -> None:
        try:
            if isinstance(value, bool):
                value = 1.0 if value else 0.0
            self.client.score_current_trace(
                name=name,
                value=value,
                **{key: sanitize_for_observability(item) for key, item in kwargs.items() if item is not None},
            )
        except Exception as exc:  # pragma: no cover - SDK/network failure should not break requests
            logger.warning("Langfuse score failed: %s", exc)

    def flush(self) -> None:
        try:
            self.client.flush()
        except Exception as exc:  # pragma: no cover - SDK/network failure should not break requests
            logger.warning("Langfuse flush failed: %s", exc)

    def status(self) -> JsonDict:
        return {
            "enabled": True,
            "provider": "langfuse",
            "configured": True,
            "environment": self.environment,
        }

    def _trace_id(self, seed: str) -> str:
        create_trace_id = getattr(self.client, "create_trace_id", None)
        if callable(create_trace_id):
            return str(create_trace_id(seed=seed or uuid.uuid4().hex))
        return hashlib.sha256(str(seed).encode("utf-8")).hexdigest()[:32]


class _LangfuseObservationContext:
    def __init__(self, client: Any, kwargs: JsonDict) -> None:
        self.client = client
        self.kwargs = {key: value for key, value in kwargs.items() if value not in (None, {}, [])}
        self._context: Any | None = None
        self._observation: Any = NoopObservation()

    def __enter__(self) -> Any:
        try:
            self._context = self.client.start_as_current_observation(**self.kwargs)
            self._observation = self._context.__enter__()
            return self._observation
        except Exception as exc:  # pragma: no cover - SDK failure should not break requests
            logger.warning("Langfuse observation failed: %s", exc)
            self._context = None
            self._observation = NoopObservation()
            return self._observation

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if exc is not None:
            _update_observation(
                self._observation,
                level="ERROR",
                status_message=str(exc),
                metadata={"error_type": getattr(exc_type, "__name__", str(exc_type))},
            )
        if self._context is None:
            return False
        return bool(self._context.__exit__(exc_type, exc, tb))


def get_observer() -> NoopObserver:
    global _OBSERVER
    if _OBSERVER is None:
        _OBSERVER = LangfuseObserver.from_env()
    return _OBSERVER


def set_observer_for_testing(observer: NoopObserver | None) -> None:
    global _OBSERVER
    _OBSERVER = observer


def start_span(name: str, **kwargs: Any) -> ContextManager[Any]:
    return get_observer().start_observation(name, **kwargs)


def start_generation(name: str, *, model: str = "", input: Any | None = None, metadata: JsonDict | None = None, model_parameters: JsonDict | None = None) -> ContextManager[Any]:
    return get_observer().start_observation(
        name,
        as_type="generation",
        input=input,
        metadata=metadata,
        model=model or None,
        model_parameters=model_parameters,
    )


def update_observation(observation: Any, **kwargs: Any) -> None:
    _update_observation(observation, **kwargs)


def _update_observation(observation: Any, **kwargs: Any) -> None:
    update = getattr(observation, "update", None)
    if not callable(update):
        return
    try:
        update(**{key: sanitize_for_observability(value) for key, value in kwargs.items() if value is not None})
    except Exception as exc:  # pragma: no cover - SDK failure should not break requests
        logger.warning("Langfuse observation update failed: %s", exc)


SENSITIVE_KEY_RE = re.compile(
    r"^(password|passwd|pwd|secret|api[_-]?key|authorization|cookie|"
    r"access[_-]?token|refresh[_-]?token|auth[_-]?token|bearer[_-]?token)$",
    re.IGNORECASE,
)
SENSITIVE_VALUE_RE = re.compile(
    r"(?P<key>password|pwd|secret|token|api[_-]?key|authorization|cookie)\s*[:=]\s*(?P<value>[^\s,;&]+)",
    re.IGNORECASE,
)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}")
MAX_OBSERVABILITY_STRING = 2000


def sanitize_for_observability(value: Any, depth: int = 0) -> Any:
    if depth >= 6:
        return _brief(value)
    if isinstance(value, dict):
        result: JsonDict = {}
        for key, item in value.items():
            key_text = str(key)
            result[key_text] = "[REDACTED]" if SENSITIVE_KEY_RE.fullmatch(key_text) else sanitize_for_observability(item, depth + 1)
        return result
    if isinstance(value, list):
        return [sanitize_for_observability(item, depth + 1) for item in value[:50]]
    if isinstance(value, tuple):
        return [sanitize_for_observability(item, depth + 1) for item in value[:50]]
    if isinstance(value, str):
        redacted = SENSITIVE_VALUE_RE.sub(lambda match: f"{match.group('key')}=[REDACTED]", value)
        return _brief(EMAIL_RE.sub(lambda match: f"[EMAIL:{stable_hash(match.group(0))}]", redacted))
    return value


def stable_hash(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _brief(value: Any) -> Any:
    if isinstance(value, str) and len(value) > MAX_OBSERVABILITY_STRING:
        return value[:MAX_OBSERVABILITY_STRING] + "...[truncated]"
    return value


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


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


_OBSERVER: NoopObserver | None = None


__all__ = [
    "LangfuseObserver",
    "NoopObserver",
    "RequestMetrics",
    "TraceContext",
    "elapsed_ms",
    "get_observer",
    "log_request_completed",
    "sanitize_for_observability",
    "set_observer_for_testing",
    "stable_hash",
    "start_generation",
    "start_span",
    "update_observation",
]
