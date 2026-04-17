"""Persistent runtime client helpers for llama.cpp embeddings."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import sys
import time
from typing import Any

import requests

from cementic.config import Config
from cementic.embedding_providers.base import EmbeddingProvider
from cementic.supervisor import is_pid_running, spawn_detached, wait_for_exit


def llama_cpp_runtime_fingerprint(
    *,
    model_path: str,
    n_ctx: int,
    n_gpu_layers: int,
    embedding_dim: int,
    verbose: bool,
) -> str:
    """Return a stable fingerprint for one llama.cpp runtime config."""
    payload = json.dumps(
        {
            "embedding_dim": embedding_dim,
            "model_path": model_path,
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "verbose": verbose,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LlamaCppDaemonClient(EmbeddingProvider):
    """Embedding provider client backed by a persistent local daemon."""

    def __init__(
        self,
        host: str,
        port: int,
        embedding_dim: int,
        expected_fingerprint: str,
        timeout: float = 30.0,
    ) -> None:
        self.host = host
        self.port = port
        self._embedding_dim = embedding_dim
        self.expected_fingerprint = expected_fingerprint
        self.timeout = timeout

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _get_health(self) -> dict[str, Any] | None:
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)
            response.raise_for_status()
            return dict(response.json())
        except (requests.RequestException, ValueError):
            return None

    def matches_expected_runtime(self) -> bool:
        payload = self._get_health()
        return bool(payload and payload.get("fingerprint") == self.expected_fingerprint)

    def health_check(self) -> bool:
        return self.matches_expected_runtime()

    def embed(self, text: str) -> list[float]:
        response = requests.post(
            f"{self.base_url}/embed",
            json={"text": text},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return [float(value) for value in payload["embedding"]]

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        response = requests.post(
            f"{self.base_url}/embed-batch",
            json={"texts": texts},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        results: list[list[float] | None] = []
        for embedding in payload["embeddings"]:
            if embedding is None:
                results.append(None)
            else:
                results.append([float(value) for value in embedding])
        return results

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim


def get_llama_cpp_runtime_client(config: Config) -> LlamaCppDaemonClient:
    """Return a daemon-backed llama.cpp client, starting the daemon if needed."""
    fingerprint = llama_cpp_runtime_fingerprint(
        model_path=config.llama_cpp.model_path,
        n_ctx=config.llama_cpp.n_ctx,
        n_gpu_layers=config.llama_cpp.n_gpu_layers,
        embedding_dim=config.llama_cpp.embedding_dim,
        verbose=config.llama_cpp.verbose,
    )
    client = LlamaCppDaemonClient(
        host=config.llama_cpp.daemon_host,
        port=config.llama_cpp.daemon_port,
        embedding_dim=config.llama_cpp.embedding_dim,
        expected_fingerprint=fingerprint,
        timeout=float(config.llama_cpp.daemon_start_timeout_seconds),
    )
    if client.matches_expected_runtime():
        return client

    _restart_llama_cpp_daemon_if_needed(config)
    _start_llama_cpp_daemon(config)
    _wait_for_daemon_ready(client, timeout_seconds=config.llama_cpp.daemon_start_timeout_seconds)
    return client


def _restart_llama_cpp_daemon_if_needed(config: Config) -> None:
    pid_file = config.llama_cpp.daemon_pid_file
    if pid_file is None or not pid_file.exists():
        return

    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except ValueError:
        pid_file.unlink(missing_ok=True)
        return

    if not is_pid_running(pid):
        pid_file.unlink(missing_ok=True)
        return

    os.kill(pid, signal.SIGTERM)
    remaining = wait_for_exit([pid], timeout_seconds=5.0)
    if remaining:
        os.kill(pid, signal.SIGKILL)
        wait_for_exit([pid], timeout_seconds=2.0)
    pid_file.unlink(missing_ok=True)


def _start_llama_cpp_daemon(config: Config) -> None:
    log_file = config.llama_cpp.daemon_log_file
    pid_file = config.llama_cpp.daemon_pid_file
    if log_file is None or pid_file is None:
        raise RuntimeError("llama.cpp daemon paths are not configured")

    command = [
        sys.executable,
        "-m",
        "cementic.embedding_daemon",
        "--host",
        config.llama_cpp.daemon_host,
        "--port",
        str(config.llama_cpp.daemon_port),
        "--model-path",
        config.llama_cpp.model_path,
        "--n-ctx",
        str(config.llama_cpp.n_ctx),
        "--n-gpu-layers",
        str(config.llama_cpp.n_gpu_layers),
        "--embedding-dim",
        str(config.llama_cpp.embedding_dim),
        "--pid-file",
        str(pid_file),
    ]
    if config.llama_cpp.verbose:
        command.append("--verbose")
    spawn_detached(command, log_file)


def _wait_for_daemon_ready(client: LlamaCppDaemonClient, timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if client.matches_expected_runtime():
            return
        time.sleep(0.2)
    raise RuntimeError("llama.cpp embedding daemon did not become ready in time")
