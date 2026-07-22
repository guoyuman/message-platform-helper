"""Small stdlib HTTP server for local use."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse

from .models import SendStrategy, StrategyRule, to_jsonable
from .orchestrator import MessagePlatformHelper, request_from_payload
from .strategy import SendStrategyEvaluator


JsonDict = Dict[str, Any]
APP_DIR = Path(__file__).resolve().parents[2]
WEB_DIR = APP_DIR / "web"


class HelperHandler(BaseHTTPRequestHandler):
    helper: MessagePlatformHelper

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/health":
            self._json(self.helper.health())
            return
        if path == "/api/memory":
            query = parse_qs(urlparse(self.path).query)
            session_id = (query.get("sessionId") or ["default"])[0]
            self._json({"ok": True, "memory": to_jsonable(self.helper.memory_manager.load(session_id))})
            return
        if path in {"/api/knowledge", "/api/knowledge/stats"}:
            self._json(self.helper.knowledge_stats())
            return
        if path == "/api/knowledge/search":
            query = parse_qs(urlparse(self.path).query)
            text = (query.get("q") or query.get("query") or [""])[0]
            limit = _positive_int((query.get("limit") or ["5"])[0], default=5, maximum=20)
            tags = _tags_from_query(query)
            self._json(self.helper.search_knowledge(text, limit=limit, tags=tags))
            return
        if path in {"/", "/index.html"}:
            self._static(WEB_DIR / "index.html", "text/html; charset=utf-8")
            return
        if path == "/app.js":
            self._static(WEB_DIR / "app.js", "text/javascript; charset=utf-8")
            return
        if path == "/styles.css":
            self._static(WEB_DIR / "styles.css", "text/css; charset=utf-8")
            return
        self.send_error(404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
            if path == "/api/chat":
                response = self.helper.handle(request_from_payload(payload))
                self._json(to_jsonable(response), status=200 if response.ok else 422)
                return
            if path == "/api/knowledge/ingest":
                result = self.helper.ingest_knowledge(
                    title=str(payload.get("title") or "Untitled"),
                    content=str(payload.get("content") or ""),
                    source=str(payload.get("source") or "api"),
                    tags=list(payload.get("tags") or []),
                    replace=bool(payload.get("replace", True)),
                )
                self._json(result)
                return
            if path == "/api/knowledge/search":
                result = self.helper.search_knowledge(
                    query=str(payload.get("query") or payload.get("q") or ""),
                    limit=_positive_int(payload.get("limit", 5), default=5, maximum=20),
                    tags=list(payload.get("tags") or []),
                )
                self._json(result)
                return
            if path == "/api/strategy/evaluate":
                self._handle_strategy_evaluate(payload)
                return
        except Exception as exc:
            self._json({"ok": False, "error": str(exc)}, status=500)
            return
        self.send_error(404)

    def _handle_strategy_evaluate(self, payload: JsonDict) -> None:
        strategy_payload = payload.get("strategy") or {}
        strategy = SendStrategy(
            name=str(strategy_payload.get("name") or "strategy"),
            enabled=bool(strategy_payload.get("enabled", True)),
            rules=[
                StrategyRule(kind=str(rule["kind"]), config=dict(rule.get("config") or {}), enabled=bool(rule.get("enabled", True)))
                for rule in strategy_payload.get("rules", [])
            ],
        )
        decision = SendStrategyEvaluator(self.helper.counter_store).evaluate(strategy, payload.get("message") or {})
        self._json({"ok": True, "decision": to_jsonable(decision)})

    def _read_json(self) -> JsonDict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw or "{}")

    def _json(self, payload: JsonDict, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _static(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self.send_error(404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        return


def _positive_int(value: Any, *, default: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(parsed, maximum))


def _tags_from_query(query: Dict[str, list[str]]) -> list[str]:
    values = query.get("tag") or query.get("tags") or []
    tags: list[str] = []
    for value in values:
        for item in str(value).split(","):
            tag = item.strip()
            if tag and tag not in tags:
                tags.append(tag)
    return tags


def run_server(host: str = "127.0.0.1", port: int = 8790) -> None:
    HelperHandler.helper = MessagePlatformHelper.from_env()
    server = ThreadingHTTPServer((host, port), HelperHandler)
    print(f"message-platform-helper listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    args = parser.parse_args()
    run_server(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
