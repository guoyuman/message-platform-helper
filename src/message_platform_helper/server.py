"""FastAPI server entrypoint for local use."""

from __future__ import annotations

import argparse


def run_server(host: str = "127.0.0.1", port: int = 8790) -> None:
    try:
        import uvicorn

        from .api import create_app
    except ImportError as exc:
        raise RuntimeError("FastAPI server requires optional dependency group: message-platform-helper[server]") from exc
    uvicorn.run(create_app(), host=host, port=port)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    args = parser.parse_args()
    run_server(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
