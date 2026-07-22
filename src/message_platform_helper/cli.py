"""CLI entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .models import AssistantRequest, to_jsonable
from .orchestrator import MessagePlatformHelper
from .server import run_server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="message-platform-helper")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8790)

    serve_fastapi = sub.add_parser("serve-fastapi")
    serve_fastapi.add_argument("--host", default="127.0.0.1")
    serve_fastapi.add_argument("--port", type=int, default=8790)

    demo = sub.add_parser("demo")
    demo.add_argument("--text", default="帮我给 ops@example.com 配置邮件通道并测试，同步采购订单下所有模板到阿拉伯语，每小时最多发送 2 次")

    ingest = sub.add_parser("ingest")
    ingest.add_argument("path")
    ingest.add_argument("--tag", action="append", default=[])
    ingest.add_argument("--source", default="")
    ingest.add_argument("--append", action="store_true")

    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=5)
    search.add_argument("--tag", action="append", default=[])

    sub.add_parser("stats")

    args = parser.parse_args(argv)
    if args.command == "serve":
        run_server(args.host, args.port)
        return 0
    if args.command == "serve-fastapi":
        try:
            import uvicorn

            from .api import create_app
        except ImportError as exc:
            raise SystemExit("FastAPI server requires optional dependency group: message-platform-helper[server]") from exc
        uvicorn.run(create_app(), host=args.host, port=args.port)
        return 0
    helper = MessagePlatformHelper.from_env()
    if args.command == "demo":
        response = helper.handle(AssistantRequest(text=args.text, session_id="cli-demo", payload=demo_payload(), dry_run=True))
        print(json.dumps(to_jsonable(response), ensure_ascii=False, indent=2))
        return 0
    if args.command == "ingest":
        path = Path(args.path)
        source = args.source or str(path)
        chunks = helper.knowledge_base.ingest_text(path.stem, path.read_text(encoding="utf-8"), source=source, tags=args.tag, replace=not args.append)
        print(
            json.dumps(
                {"ok": True, "db_path": str(helper.knowledge_base.path), "chunks": [to_jsonable(chunk) for chunk in chunks], "stats": helper.knowledge_base.stats()},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "search":
        response = helper.search_knowledge(args.query, limit=args.limit, tags=args.tag)
        print(json.dumps(response, ensure_ascii=False, indent=2))
        return 0
    if args.command == "stats":
        print(json.dumps(helper.knowledge_stats(), ensure_ascii=False, indent=2))
        return 0
    return 1


def demo_payload() -> dict:
    return {
        "email": "ops@example.com",
        "template": {
            "sourceLanguage": "en_US",
            "targetLanguage": "ar_SA",
            "documentId": "purchase_order",
            "groupCode": "businessprocess",
            "templates": [
                {
                    "id": "tpl-001",
                    "templateName": "Purchase order notice",
                    "templateCode": "purchase_order_notice",
                    "language": "en_US",
                    "channels": ["mail"],
                    "title": "Business approval notification",
                    "content": "Hello, you have a business message to process. {{orderNo}}",
                }
            ],
        },
        "strategy": {"evaluate": True},
    }


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
