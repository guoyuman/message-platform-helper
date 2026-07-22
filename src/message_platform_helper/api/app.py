"""Optional FastAPI presentation layer."""

from __future__ import annotations

import json
from typing import Callable

from ..application import response_to_streaming_events
from ..models import to_jsonable
from ..orchestrator import MessagePlatformHelper, request_from_payload


HelperFactory = Callable[[], MessagePlatformHelper]


def create_app(helper_factory: HelperFactory = MessagePlatformHelper.from_env):
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError as exc:
        raise RuntimeError("FastAPI support requires installing the 'server' extra dependencies.") from exc

    app = FastAPI(title="Message Platform Helper", version="0.1.0")
    helper = helper_factory()

    @app.middleware("http")
    async def trace_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Request-ID", request.headers.get("X-Request-ID", ""))
        response.headers.setdefault("X-Trace-ID", request.headers.get("X-Trace-ID", ""))
        return response

    @app.get("/api/health")
    async def health():
        return helper.health()

    @app.post("/api/chat")
    async def chat(payload: dict, request: Request):
        enriched = _with_trace_headers(payload, request)
        response = helper.handle(request_from_payload(enriched))
        return JSONResponse(to_jsonable(response), status_code=200 if response.ok else 422)

    @app.post("/api/chat/stream")
    async def chat_stream(payload: dict, request: Request):
        enriched = _with_trace_headers(payload, request)
        response = helper.handle(request_from_payload(enriched))

        def events():
            for event in response_to_streaming_events(response):
                yield "data: " + json.dumps(to_jsonable(event), ensure_ascii=False) + "\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/api/knowledge/stats")
    async def knowledge_stats():
        return helper.knowledge_stats()

    @app.post("/api/knowledge/ingest")
    async def ingest_knowledge(payload: dict):
        if payload.get("path"):
            return helper.ingest_knowledge_file(
                path=str(payload.get("path")),
                tags=list(payload.get("tags") or []),
                replace=bool(payload.get("replace", True)),
            )
        return helper.ingest_knowledge(
            title=str(payload.get("title") or "Untitled"),
            content=str(payload.get("content") or ""),
            source=str(payload.get("source") or "api"),
            tags=list(payload.get("tags") or []),
            replace=bool(payload.get("replace", True)),
        )

    return app


def _with_trace_headers(payload: dict, request) -> dict:
    enriched = dict(payload)
    enriched.setdefault("requestId", request.headers.get("X-Request-ID", ""))
    enriched.setdefault("traceId", request.headers.get("X-Trace-ID", ""))
    return enriched


__all__ = ["create_app"]
