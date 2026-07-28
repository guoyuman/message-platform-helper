"""Optional FastAPI presentation layer."""

import json
import queue
import threading
import time
from typing import Callable

from ..application import response_to_streaming_events
from ..models import to_jsonable
from ..orchestrator import MessagePlatformHelper, request_from_payload

HelperFactory = Callable[[], MessagePlatformHelper]
STREAM_KEEPALIVE_SECONDS = 10


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

    @app.get("/api/memory")
    async def memory(sessionId: str = "default"):
        load_for_display = getattr(helper, "load_memory_for_display", None)
        memory_payload = load_for_display(sessionId) if callable(load_for_display) else helper.memory_manager.load(sessionId)
        return {"ok": True, "memory": to_jsonable(memory_payload)}

    @app.get("/api/memory/sessions")
    async def memory_sessions():
        list_sessions = getattr(helper.memory_manager.store, "list_sessions", None)
        sessions = list_sessions() if callable(list_sessions) else []
        return {"ok": True, "sessions": to_jsonable(sessions)}

    @app.post("/api/chat")
    async def chat(payload: dict, request: Request):
        enriched = _with_trace_headers(payload, request)
        response = helper.handle(request_from_payload(enriched))
        return JSONResponse(to_jsonable(response), status_code=200 if response.ok else 422)

    @app.post("/api/chat/stream")
    async def chat_stream(payload: dict, request: Request):
        enriched = _with_trace_headers(payload, request)

        def events():
            sequence = 1
            started_at = time.perf_counter()
            yield _stream_event("Started", {"message": "request accepted"}, enriched, sequence)
            try:
                sequence += 1
                yield _stream_event("Running", {"message": "executing workflow"}, enriched, sequence)
                result_queue: queue.Queue = queue.Queue(maxsize=1)

                def run_helper():
                    try:
                        result_queue.put(("response", helper.handle(request_from_payload(enriched))))
                    except Exception as exc:
                        result_queue.put(("error", exc))

                threading.Thread(target=run_helper, daemon=True).start()
                while True:
                    try:
                        result_type, result = result_queue.get(timeout=STREAM_KEEPALIVE_SECONDS)
                        break
                    except queue.Empty:
                        sequence += 1
                        yield _stream_event(
                            "Running",
                            {"message": "executing workflow", "elapsedSeconds": round(time.perf_counter() - started_at, 1)},
                            enriched,
                            sequence,
                        )
                if result_type == "error":
                    raise result

                response = result
                for event in response_to_streaming_events(response):
                    serialized = to_jsonable(event)
                    if serialized.get("type") in {"Done", "Error"}:
                        serialized.setdefault("data", {})["response"] = to_jsonable(response)
                    yield "data: " + json.dumps(serialized, ensure_ascii=False) + "\n\n"
            except Exception as exc:
                sequence += 1
                yield _stream_event(
                    "Error",
                    {
                        "ok": False,
                        "error": str(exc),
                        "errorType": type(exc).__name__,
                        "elapsedSeconds": round(time.perf_counter() - started_at, 1),
                    },
                    enriched,
                    sequence,
                )

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/api/knowledge/stats")
    async def knowledge_stats():
        return helper.knowledge_stats()

    @app.get("/api/knowledge")
    async def knowledge():
        return helper.knowledge_stats()

    @app.post("/api/knowledge/ingest")
    async def ingest_knowledge(payload: dict):
        if payload.get("path"):
            return helper.ingest_knowledge_file(
                path=str(payload.get("path")),
                tags=list(payload.get("tags") or []),
                replace=bool(payload.get("replace", True)),
                document_key=str(payload.get("documentKey") or payload.get("document_key") or payload.get("documentId") or payload.get("document_id") or ""),
            )
        return helper.ingest_knowledge(
            title=str(payload.get("title") or "Untitled"),
            content=str(payload.get("content") or ""),
            source=str(payload.get("source") or "api"),
            tags=list(payload.get("tags") or []),
            replace=bool(payload.get("replace", True)),
        )

    @app.get("/api/knowledge/search")
    async def search_knowledge(q: str = "", query: str = "", limit: int = 5, tag: list[str] | None = None, tags: list[str] | None = None):
        return helper.search_knowledge(
            q or query,
            limit=_positive_int(limit, default=5, maximum=20),
            tags=_tags_from_values([*(tag or []), *(tags or [])]),
        )

    @app.post("/api/knowledge/search")
    async def search_knowledge_post(payload: dict):
        return helper.search_knowledge(
            query=str(payload.get("query") or payload.get("q") or ""),
            limit=_positive_int(payload.get("limit", 5), default=5, maximum=20),
            tags=list(payload.get("tags") or []),
        )


    return app


def _with_trace_headers(payload: dict, request) -> dict:
    enriched = dict(payload)
    enriched.setdefault("requestId", request.headers.get("X-Request-ID", ""))
    enriched.setdefault("traceId", request.headers.get("X-Trace-ID", ""))
    return enriched


def _stream_event(event_type: str, data: dict, payload: dict, sequence: int) -> str:
    event = {
        "type": event_type,
        "data": data,
        "trace_id": str(payload.get("traceId") or payload.get("trace_id") or ""),
        "request_id": str(payload.get("requestId") or payload.get("request_id") or ""),
        "conversation_id": str(payload.get("sessionId") or payload.get("session_id") or "default"),
        "sequence": sequence,
    }
    return "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"


def _positive_int(value, *, default: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(parsed, maximum))


def _tags_from_values(values: list[str]) -> list[str]:
    tags: list[str] = []
    for value in values:
        for item in str(value).split(","):
            tag = item.strip()
            if tag and tag not in tags:
                tags.append(tag)
    return tags


__all__ = ["create_app"]
