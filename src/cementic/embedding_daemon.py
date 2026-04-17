"""Persistent local embedding daemon for llama.cpp search queries."""

from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from cementic.embedding_providers.llama_cpp import LlamaCppEmbeddingProvider
from cementic.embedding_runtime import llama_cpp_runtime_fingerprint


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="cementic llama.cpp embedding daemon")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--n-ctx", required=True, type=int)
    parser.add_argument("--n-gpu-layers", required=True, type=int)
    parser.add_argument("--embedding-dim", required=True, type=int)
    parser.add_argument("--pid-file", required=True)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _write_pid_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(os.getpid()), encoding="utf-8")


def _cleanup_pid_file(path: Path) -> None:
    if path.exists():
        path.unlink()


def _json_response(handler: BaseHTTPRequestHandler, payload: dict[str, Any], status: int) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _build_handler(
    provider: LlamaCppEmbeddingProvider,
    fingerprint: str,
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                _json_response(self, {"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            _json_response(
                self,
                {
                    "status": "ok",
                    "fingerprint": fingerprint,
                    "embedding_dim": provider.embedding_dim,
                },
                HTTPStatus.OK,
            )

        def do_POST(self) -> None:  # noqa: N802
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            try:
                payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
            except json.JSONDecodeError:
                _json_response(self, {"error": "invalid json"}, HTTPStatus.BAD_REQUEST)
                return

            if self.path == "/embed":
                text = str(payload.get("text", ""))
                embedding = provider.embed(text)
                _json_response(self, {"embedding": embedding}, HTTPStatus.OK)
                return

            if self.path == "/embed-batch":
                texts = payload.get("texts", [])
                if not isinstance(texts, list):
                    _json_response(self, {"error": "texts must be a list"}, HTTPStatus.BAD_REQUEST)
                    return
                embeddings = provider.embed_batch([str(text) for text in texts])
                _json_response(self, {"embeddings": embeddings}, HTTPStatus.OK)
                return

            _json_response(self, {"error": "not found"}, HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

    return Handler


def main() -> None:
    args = _parse_args()
    pid_file = Path(args.pid_file)
    _write_pid_file(pid_file)
    atexit.register(_cleanup_pid_file, pid_file)

    provider = LlamaCppEmbeddingProvider(
        model_path=args.model_path,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        embedding_dim=args.embedding_dim,
        verbose=args.verbose,
    )
    fingerprint = llama_cpp_runtime_fingerprint(
        model_path=args.model_path,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        embedding_dim=args.embedding_dim,
        verbose=args.verbose,
    )
    server = ThreadingHTTPServer((args.host, args.port), _build_handler(provider, fingerprint))

    def _shutdown(signum: int, frame: object) -> None:
        del signum, frame
        server.shutdown()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    server.serve_forever()


if __name__ == "__main__":
    main()
