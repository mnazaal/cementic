"""Persistent runtime client helpers for llama.cpp embeddings."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import requests

from cementic.config import Config, resolve_llama_model_path
from cementic.embedding_provider import EmbeddingFacts, EmbeddingProvider
from cementic.embedding_text import (
    format_document_text_for_model,
    format_query_text_for_model,
)
from cementic.supervisor import (
    is_managed_process_alive,
    process_start_token,
    spawn_detached,
    wait_for_exit,
)


@dataclass(frozen=True)
class EmbeddingRuntimeSpec:
    """Embedding runtime identity used by indexing and search."""

    provider: str
    model_identifier: str
    embedding_dim: int
    distance_metric: str = "cosine"
    n_ctx: int | None = None
    n_gpu_layers: int | None = None
    verbose: bool = False


def runtime_spec_from_config(config: Config) -> EmbeddingRuntimeSpec:
    """Build an embedding runtime spec from current configuration."""
    provider = config.pipeline.embedding_provider
    if provider == "llama-cpp":
        return EmbeddingRuntimeSpec(
            provider="llama-cpp",
            model_identifier=config.llama_cpp.model_path,
            embedding_dim=config.llama_cpp.embedding_dim,
            n_ctx=config.llama_cpp.n_ctx,
            n_gpu_layers=config.llama_cpp.n_gpu_layers,
            verbose=config.llama_cpp.verbose,
        )
    raise ValueError(f"Unknown embedding provider: {provider}")


def runtime_spec_from_profile_json(config_json: str) -> EmbeddingRuntimeSpec:
    """Build an embedding runtime spec from stored embedding profile JSON.

    Parsed generically: the spec carries every runtime field and a provider
    simply leaves unused ones empty. No per-provider branching.
    """
    payload = json.loads(config_json)
    n_ctx = payload.get("n_ctx")
    n_gpu_layers = payload.get("n_gpu_layers")
    return EmbeddingRuntimeSpec(
        provider=str(payload["provider"]),
        model_identifier=str(payload["model_identifier"]),
        embedding_dim=int(payload["embedding_dim"]),
        distance_metric=str(payload.get("distance_metric", "cosine")),
        n_ctx=int(n_ctx) if n_ctx is not None else None,
        n_gpu_layers=int(n_gpu_layers) if n_gpu_layers is not None else None,
        verbose=bool(payload.get("verbose", False)),
    )


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


class RemoteEmbeddingClient(EmbeddingProvider):
    """Client for a local OpenAI-compatible embedding server (``llama_cpp.server``).

    Embeds over ``/v1/embeddings`` (an input array is batched natively) and
    verifies identity through the server's own ``/v1/models`` endpoint: the
    served model's id is the runtime fingerprint, so any config change forces a
    restart rather than silently reusing a mismatched model.
    """

    name = "llama-cpp"

    def __init__(
        self,
        host: str,
        port: int,
        embedding_dim: int,
        expected_fingerprint: str,
        model_identifier: str = "",
        timeout: float = 30.0,
    ) -> None:
        self.host = host
        self.port = port
        self._embedding_dim = embedding_dim
        self._probed_dim: int | None = None
        self.expected_fingerprint = expected_fingerprint
        self.model_identifier = model_identifier
        self.timeout = timeout

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def format_document(self, text: str) -> str:
        """Apply the served model's document prefix before embedding."""
        return format_document_text_for_model(text, self.model_identifier)

    def format_query(self, text: str) -> str:
        """Apply the served model's query prefix before embedding."""
        return format_query_text_for_model(text, self.model_identifier)

    def describe(self) -> EmbeddingFacts:
        """Report facts, probing the live server for the true embedding dim once.

        Falls back to the configured dimension if the probe fails, so profile
        resolution never depends on a transient network error.
        """
        if self._probed_dim is None:
            try:
                probed = len(self.embed("dimension probe"))
                self._probed_dim = probed if probed > 0 else self._embedding_dim
            except Exception:
                self._probed_dim = self._embedding_dim
        return EmbeddingFacts(
            name=self.name,
            embedding_dim=self._probed_dim,
            distance_metric=self.distance_metric,
        )

    def _list_models(self) -> list[dict[str, Any]] | None:
        # One retry for transient blips only. llama_cpp.server serializes all
        # requests behind a single model lock (see app.py's llama_outer_lock),
        # so /v1/models can legitimately block for the *entire* duration of an
        # in-flight embedding batch -- seconds to tens of seconds, not
        # something a short retry budget can wait out. Callers that need to
        # tell "busy" apart from "down" should fall back to a process-level
        # liveness check (see status_service.check_health) instead of
        # widening the timeout/retry count here.
        for attempt in range(2):
            try:
                response = requests.get(f"{self.base_url}/v1/models", timeout=2)
                response.raise_for_status()
                payload = response.json()
                return list(payload.get("data", []))
            except (requests.RequestException, ValueError):
                if attempt == 0:
                    time.sleep(0.3)
        return None

    def matches_expected_runtime(self) -> bool:
        models = self._list_models()
        if not models:
            return False
        return any(str(model.get("id")) == self.expected_fingerprint for model in models)

    def health_check(self) -> bool:
        return self.matches_expected_runtime()

    def _embed_inputs(self, inputs: str | list[str]) -> list[list[float]]:
        response = requests.post(
            f"{self.base_url}/v1/embeddings",
            json={"model": self.expected_fingerprint, "input": inputs},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        rows = sorted(payload["data"], key=lambda row: row.get("index", 0))
        return [[float(value) for value in row["embedding"]] for row in rows]

    def embed(self, text: str) -> list[float]:
        return self._embed_inputs(text)[0]

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        if not texts:
            return []
        embeddings = self._embed_inputs(texts)
        if len(embeddings) != len(texts):
            raise ValueError(
                f"embedding count {len(embeddings)} does not match input count {len(texts)}"
            )
        # The optional element type is part of the EmbeddingProvider contract
        # (a provider may report per-item failure); this one either returns a
        # full batch or raises.
        return list(embeddings)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim


def get_llama_cpp_runtime_client(
    config: Config,
    spec: EmbeddingRuntimeSpec | None = None,
    autostart: bool | None = None,
) -> RemoteEmbeddingClient:
    """Return a client for the llama.cpp embedding server for one runtime spec."""
    runtime_spec = spec or runtime_spec_from_config(config)
    if runtime_spec.provider != "llama-cpp":
        raise ValueError(f"Expected llama-cpp runtime spec, got {runtime_spec.provider}")

    n_ctx = runtime_spec.n_ctx or config.llama_cpp.n_ctx
    n_gpu_layers = (
        runtime_spec.n_gpu_layers
        if runtime_spec.n_gpu_layers is not None
        else config.llama_cpp.n_gpu_layers
    )
    fingerprint = llama_cpp_runtime_fingerprint(
        model_path=runtime_spec.model_identifier,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        embedding_dim=runtime_spec.embedding_dim,
        verbose=runtime_spec.verbose,
    )
    client = RemoteEmbeddingClient(
        host=config.llama_cpp.daemon_host,
        port=config.llama_cpp.daemon_port,
        embedding_dim=runtime_spec.embedding_dim,
        expected_fingerprint=fingerprint,
        model_identifier=runtime_spec.model_identifier,
        timeout=float(config.llama_cpp.llama_embed_timeout_seconds),
    )

    models = client._list_models()
    if models is not None:
        if any(str(model.get("id")) == fingerprint for model in models):
            return client
        # Reachable and reporting a real model list, but not ours -- a genuine
        # config change, not busyness. Fall through to restart below.
    elif _daemon_pid_alive(config):
        # /v1/models didn't respond, but the daemon process is confirmed alive
        # (PID + start-token). llama_cpp.server serializes every request behind
        # one lock, so a daemon mid-embedding-batch looks identical to a dead
        # one over that probe alone. Wait the batch out instead of killing a
        # healthy process.
        extended_timeout = max(
            config.llama_cpp.daemon_start_timeout_seconds,
            config.llama_cpp.llama_embed_timeout_seconds,
        )
        if _poll_until_ready(client, extended_timeout):
            return client
        raise RuntimeError(
            "llama.cpp embedding daemon appears busy (in-flight request) and did "
            "not become available in time; try again shortly"
        )

    should_autostart = config.llama_cpp.daemon_autostart if autostart is None else autostart
    if not should_autostart:
        raise RuntimeError(
            "llama.cpp embedding daemon is not running for the requested model; "
            "run `cementic embedding start` or set CEMENTIC_LLAMA_DAEMON_AUTOSTART=true"
        )

    print(
        "starting embedding daemon (first search after a restart can take 30s+)...",
        file=sys.stderr,
    )
    _stop_mismatched_llama_cpp_daemon(config)
    _start_llama_cpp_daemon(config, spec=runtime_spec)
    _wait_for_daemon_ready(client, timeout_seconds=config.llama_cpp.daemon_start_timeout_seconds)
    return client


def client_is_healthy_or_busy(client: EmbeddingProvider, config: Config) -> bool:
    """Health check that tells a busy llama.cpp daemon apart from a dead one.

    A single ``health_check()`` probe can't distinguish "mid-batch, lock
    held" from "down" (see ``RemoteEmbeddingClient._list_models``). For a
    ``RemoteEmbeddingClient``, fall back to PID+start-token liveness and a
    longer poll before reporting unhealthy.
    """
    if client.health_check():
        return True
    if not isinstance(client, RemoteEmbeddingClient):
        return False
    if not _daemon_pid_alive(config):
        return False
    extended_timeout = max(
        config.llama_cpp.daemon_start_timeout_seconds,
        config.llama_cpp.llama_embed_timeout_seconds,
    )
    return _poll_until_ready(client, extended_timeout)


def _create_llama_cpp_provider(
    spec: EmbeddingRuntimeSpec, config: Config, autostart: bool | None
) -> EmbeddingProvider:
    return get_llama_cpp_runtime_client(config=config, spec=spec, autostart=autostart)


# The single embedding-provider seam: maps a runtime spec to a ready provider that
# talks to the warm shared server (the path indexing and search take). Each factory
# owns everything about its backend; adding one is a single entry here.
_PROVIDER_FACTORIES: dict[
    str, Callable[[EmbeddingRuntimeSpec, Config, bool | None], EmbeddingProvider]
] = {
    "llama-cpp": _create_llama_cpp_provider,
}


def create_provider(
    spec: EmbeddingRuntimeSpec,
    config: Config,
    *,
    autostart: bool | None = None,
) -> EmbeddingProvider:
    """Resolve an embedding runtime spec to a ready provider.

    The single point that maps embedding-method *data* to a provider object;
    callers never branch on the provider name. Adding a backend is one new entry
    in ``_PROVIDER_FACTORIES``, nothing else.
    """
    try:
        factory = _PROVIDER_FACTORIES[spec.provider]
    except KeyError:
        raise ValueError(f"Unknown embedding provider: {spec.provider}") from None
    return factory(spec, config, autostart)


def _write_daemon_pid_file(pid_file: Path, pid: int) -> None:
    """Record the daemon PID plus a start-token so a recycled PID isn't mistaken
    for the daemon later (mirrors the supervisor's managed-process records)."""
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    record = {"pid": pid, "start_token": process_start_token(pid)}
    tmp_path = pid_file.with_name(f".{pid_file.name}.tmp")
    tmp_path.write_text(json.dumps(record), encoding="utf-8")
    os.replace(tmp_path, pid_file)


def _read_daemon_pid_file(pid_file: Path) -> tuple[int, str | None] | None:
    """Read ``(pid, start_token)`` from the daemon pid-file.

    Accepts the current JSON record and the legacy bare-integer format (token
    unknown -> PID-only liveness, like ``is_managed_process_alive``).
    """
    try:
        raw = pid_file.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except ValueError:
        return None
    if isinstance(payload, dict):
        pid = payload.get("pid")
        if not isinstance(pid, int):
            return None
        token = payload.get("start_token")
        return pid, token if isinstance(token, str) else None
    if isinstance(payload, int):
        return payload, None
    return None


def _live_daemon_pid(config: Config) -> int | None:
    """Return the daemon's PID if its pid-file names a still-alive process."""
    pid_file = config.llama_cpp.daemon_pid_file
    if pid_file is None or not pid_file.exists():
        return None
    record = _read_daemon_pid_file(pid_file)
    if record is None:
        return None
    pid, token = record
    return pid if is_managed_process_alive(pid, token) else None


def _daemon_pid_alive(config: Config) -> bool:
    return _live_daemon_pid(config) is not None


def llama_daemon_status(config: Config) -> str:
    """Human-readable status of the llama.cpp daemon from its pid-file."""
    pid = _live_daemon_pid(config)
    return f"running, pid={pid}" if pid is not None else "stopped"


def _stop_mismatched_llama_cpp_daemon(config: Config) -> None:
    """Stop any existing llama.cpp daemon so a matching runtime can be started.

    Callers reach here only after establishing that no daemon serving the wanted
    runtime is available, so the stop is unconditional.
    """
    stop_llama_cpp_runtime(config)


def stop_llama_cpp_runtime(config: Config) -> bool:
    """Stop the configured llama.cpp daemon if a live PID file exists."""
    pid_file = config.llama_cpp.daemon_pid_file
    if pid_file is None or not pid_file.exists():
        return False

    record = _read_daemon_pid_file(pid_file)
    if record is None:
        pid_file.unlink(missing_ok=True)
        return False

    pid, token = record
    # Only signal a process we can confirm is still the daemon we started; a
    # recycled PID (token mismatch) belongs to someone else.
    if not is_managed_process_alive(pid, token):
        pid_file.unlink(missing_ok=True)
        return False

    os.kill(pid, signal.SIGTERM)
    remaining = wait_for_exit([pid], timeout_seconds=5.0)
    if remaining:
        os.kill(pid, signal.SIGKILL)
        wait_for_exit([pid], timeout_seconds=2.0)
    pid_file.unlink(missing_ok=True)
    return True


def _start_llama_cpp_daemon(
    config: Config,
    spec: EmbeddingRuntimeSpec | None = None,
) -> None:
    log_file = config.llama_cpp.daemon_log_file
    pid_file = config.llama_cpp.daemon_pid_file
    if log_file is None or pid_file is None:
        raise RuntimeError("llama.cpp daemon paths are not configured")

    runtime_spec = spec or runtime_spec_from_config(config)
    if runtime_spec.provider != "llama-cpp":
        raise ValueError(f"Expected llama-cpp runtime spec, got {runtime_spec.provider}")

    n_ctx = runtime_spec.n_ctx or config.llama_cpp.n_ctx
    n_gpu_layers = (
        runtime_spec.n_gpu_layers
        if runtime_spec.n_gpu_layers is not None
        else config.llama_cpp.n_gpu_layers
    )
    fingerprint = llama_cpp_runtime_fingerprint(
        model_path=runtime_spec.model_identifier,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        embedding_dim=runtime_spec.embedding_dim,
        verbose=runtime_spec.verbose,
    )
    # Run llama.cpp's own OpenAI-compatible server rather than a hand-rolled
    # daemon. The runtime fingerprint is the served model's alias, so /v1/models
    # reports exactly which config is loaded.
    command = [
        sys.executable,
        "-m",
        "llama_cpp.server",
        "--host",
        config.llama_cpp.daemon_host,
        "--port",
        str(config.llama_cpp.daemon_port),
        "--model",
        str(resolve_llama_model_path(runtime_spec.model_identifier)),
        "--model_alias",
        fingerprint,
        "--n_ctx",
        str(n_ctx),
        "--n_gpu_layers",
        str(n_gpu_layers),
        "--embedding",
        "true",
        "--verbose",
        "true" if runtime_spec.verbose else "false",
    ]
    pid = spawn_detached(command, log_file)
    # llama_cpp.server does not write its own PID file; record it for teardown.
    _write_daemon_pid_file(pid_file, pid)


def _poll_until_ready(client: RemoteEmbeddingClient, timeout_seconds: float) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if client.matches_expected_runtime():
            return True
        time.sleep(0.2)
    return False


def _wait_for_daemon_ready(client: RemoteEmbeddingClient, timeout_seconds: int) -> None:
    if not _poll_until_ready(client, timeout_seconds):
        raise RuntimeError("llama.cpp embedding daemon did not become ready in time")
