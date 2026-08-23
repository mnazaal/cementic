"""Persistent runtime client helpers for llama.cpp embeddings."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import requests
import tiktoken

from cementic.chunk import TOKENIZER
from cementic.config import Config, resolve_llama_model_path
from cementic.embedding_provider import EmbeddingFacts, EmbeddingProvider
from cementic.embedding_text import (
    format_document_text_for_model,
    format_query_text_for_model,
)
from cementic.filelock import file_lock
from cementic.supervisor import (
    is_managed_process_alive,
    process_start_token,
    spawn_detached,
    wait_for_exit,
)

#: Upper bound on how many of the model's own tokens one ``chunk.TOKENIZER``
#: token can become, used only as a cheap pre-filter: below this, a text
#: provably fits and no round trip is needed. Measured over real indexed chunks
#: against the default Nomic model (median 1.14, p95 1.24, max 1.33); 1.45
#: leaves headroom. Over-estimating only costs an extra exact count, so err high.
#:
#: ``pipeline.chunk_size`` must be small enough that a full chunk *plus its task
#: prefix* clears this bound, or the pre-filter fires on every chunk and the
#: exact check stops being the rare path it is designed to be. The invariant is
#: pinned by a test; see config.PipelineConfig.chunk_size.
_TOKEN_RATIO_UPPER_BOUND = 1.45

#: Arithmetic allowance for the task prefix (e.g. "search_document: ") the
#: client prepends before embedding, in ``chunk.TOKENIZER`` tokens. The real
#: prefix is 3-4 tokens; 8 keeps the config-time invariant conservative without
#: putting a tokenizer in the config load path.
_TASK_PREFIX_TOKEN_ALLOWANCE = 8

#: Budget for a tokenize round trip. Tokenizing is trivial once the model is
#: loaded, so a long timeout here buys nothing and costs the embedding request
#: that follows its own budget when the daemon is cold.
_TOKENIZE_TIMEOUT_SECONDS = 30.0

#: Attempts (and the gap between them) when probing the model's true embedding
#: dimension. Short: the daemon is already known reachable by this point, so
#: this only rides out a blip, not a cold start.
_PROBE_ATTEMPTS = 3
_PROBE_RETRY_DELAY_SECONDS = 1.0


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
    n_batch: int,
    n_gpu_layers: int,
    verbose: bool,
) -> str:
    """Return a stable fingerprint for one llama.cpp runtime config.

    Only fields that change what the server loads belong here -- they become the
    served model's alias, so a difference forces a daemon restart.

    ``embedding_dim`` is deliberately excluded. It is not a llama.cpp launch
    argument (the model file determines it), and it reaches callers from two
    different sources: indexing derives it from config while search reads the
    probed value stored on the profile. Including it meant that for any model
    whose true dimension differed from the configured one, the two computed
    different aliases and restarted the daemon from under each other on every
    operation. The dimension still participates in the *profile* fingerprint,
    where it does identify the vectors.
    """
    payload = json.dumps(
        {
            "model_path": model_path,
            "n_ctx": n_ctx,
            # The batch size caps how many tokens the server will embed per
            # input, so it changes what the daemon does. It also has to be in
            # the alias for a second reason: a daemon left running by a version
            # that did not pass --n_batch otherwise fingerprints identically and
            # gets reused, keeping its 512-token cap until a manual restart.
            "n_batch": n_batch,
            "n_gpu_layers": n_gpu_layers,
            "verbose": verbose,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


#: Long enough to wait out another process's cold model load rather than
#: giving up and spawning a competing daemon.
_DAEMON_LOCK_TIMEOUT_SECONDS = 180.0


def _daemon_lock_path(config: Config) -> Path:
    """Lock file guarding daemon stop/spawn, beside the pid file."""
    pid_file = config.llama_cpp.daemon_pid_file
    if pid_file is None:
        raise RuntimeError("llama.cpp daemon paths are not configured")
    return pid_file.with_name(f"{pid_file.name}.lock")



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
        n_ctx: int = 0,
    ) -> None:
        self.host = host
        self.port = port
        self._embedding_dim = embedding_dim
        self._probed_dim: int | None = None
        self.expected_fingerprint = expected_fingerprint
        self.model_identifier = model_identifier
        self.timeout = timeout
        # 0 disables the budget guard, for callers that genuinely do not know
        # the window. Every production construction site passes the real value.
        self.n_ctx = n_ctx
        self._tokenize_endpoint_available: bool | None = None

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

        Raises if the dimension cannot be established. It used to fall back to
        the configured value so that profile resolution never depended on a
        transient error -- but the result is not transient: the dimension is
        written into an *immutable* embedding profile. One blip would mint a
        second profile keyed on a guess, which forks the revision, silently
        re-embeds the whole corpus into a new vector table, and makes
        multi-collection search refuse with "different active embedding models".
        If the guess is also wrong, every vector insert fails. Failing loudly is
        the lesser harm.
        """
        if self._probed_dim is None:
            self._probed_dim = self._probe_embedding_dim()
        return EmbeddingFacts(
            name=self.name,
            embedding_dim=self._probed_dim,
            distance_metric=self.distance_metric,
        )

    def _probe_embedding_dim(self) -> int:
        """Ask the running model for its embedding dimension, retrying briefly."""
        last_error: Exception | None = None
        for attempt in range(_PROBE_ATTEMPTS):
            try:
                probed = len(self.embed("dimension probe"))
                if probed > 0:
                    return probed
                last_error = ValueError("server returned an empty embedding")
            except Exception as error:  # noqa: BLE001 - reported below
                last_error = error
            if attempt < _PROBE_ATTEMPTS - 1:
                time.sleep(_PROBE_RETRY_DELAY_SECONDS)
        raise RuntimeError(
            "Could not determine the embedding dimension from the running model at "
            f"{self.base_url} after {_PROBE_ATTEMPTS} attempts: {last_error}. "
            "Refusing to record an embedding profile from the configured fallback, "
            "which would re-embed the collection under a second profile."
        ) from last_error

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

    def probe_served_runtime(self) -> bool | None:
        """Tri-state: True serving ours, False serving another, None no answer.

        Collapsing "no answer" into False is what let three callers disagree
        about what healthy means: a daemon serving the *wrong* model and one
        that is merely mid-batch became indistinguishable, so a fallback meant
        to cover busyness also excused a genuine mismatch.
        """
        models = self._list_models()
        if models is None:
            return None
        return any(str(model.get("id")) == self.expected_fingerprint for model in models)

    def matches_expected_runtime(self) -> bool:
        return self.probe_served_runtime() is True

    def probe_embedding(self, timeout_seconds: float) -> bool:
        """Whether the embedding endpoint answers a one-token request in time.

        ``/v1/models`` is served without the model lock, so it stays chatty
        while the embedding path is dead -- which is how a wedged daemon held
        its port for 21 hours while ``status`` said healthy. Only an actual
        embedding round trip exercises the path users depend on.
        """
        try:
            response = requests.post(
                f"{self.base_url}/v1/embeddings",
                json={"model": self.expected_fingerprint, "input": "ping"},
                timeout=timeout_seconds,
            )
            response.raise_for_status()
        except Exception:
            return False
        return True

    def health_check(self) -> bool:
        return self.matches_expected_runtime()

    def count_model_tokens(self, text: str) -> int | None:
        """Tokens in ``text`` per the *model's own* tokenizer, or None if unsupported.

        ``llama_cpp.server`` exposes the loaded model's tokenizer over
        ``/extras/tokenize/count``. That is the only way to answer the budget
        question exactly: ``chunk.TOKENIZER`` is a different tokenizer and
        disagrees by up to a third on ordinary English and source code.

        None means this server does not offer the endpoint -- a permanent,
        structural fact, cached after the first 404. A *transient* failure
        (daemon down, cold, mid-restart) is not swallowed: it propagates, so the
        caller's existing retry path treats it as the temporary problem it is
        rather than recording chunks as permanently unembeddable.
        """
        if self._tokenize_endpoint_available is False:
            return None
        response = requests.post(
            f"{self.base_url}/extras/tokenize/count",
            json={"input": text},
            # Deliberately not self.timeout: tokenizing is trivial work, and a
            # long budget here would be spent waiting out a cold model load and
            # then leave nothing for the embedding request it precedes.
            timeout=min(self.timeout, _TOKENIZE_TIMEOUT_SECONDS),
        )
        if response.status_code == 404:
            self._tokenize_endpoint_available = False
            return None
        response.raise_for_status()
        count = int(response.json()["count"])
        self._tokenize_endpoint_available = True
        return count

    def over_budget_tokens(self, text: str) -> int | None:
        """Exact model-token count if ``text`` exceeds the window, else None.

        The server truncates over-long input silently, so an over-budget text
        embeds "successfully" into a vector that represents only its head. This
        is the check that turns that into a visible failure.

        Costs nothing in the common case: ``chunk.TOKENIZER`` counting is local,
        and only a text near enough to the limit to be in doubt pays for the
        exact round trip.
        """
        if self.n_ctx <= 0:
            return None
        approx = len(tiktoken.get_encoding(TOKENIZER).encode(text)) * _TOKEN_RATIO_UPPER_BOUND
        if approx <= self.n_ctx:
            return None
        exact = self.count_model_tokens(text)
        if exact is None:
            # No exact tokenizer to appeal to. Report the estimate rather than
            # letting the text through: a silently truncated vector is the
            # failure this exists to prevent, and at the shipped chunk_size
            # this branch is unreachable anyway.
            return int(approx)
        return exact if exact > self.n_ctx else None

    def _over_budget_message(self, tokens: int) -> str:
        # No advice sentence here: embed() also serves search queries, which
        # were told to "Lower pipeline.chunk_size" for a query that was simply
        # too long. Chunk-context advice is appended by over_budget_reason.
        return (
            f"text is about {tokens} tokens in the embedding model's own tokenizer, "
            f"over its {self.n_ctx}-token context window; the server would embed only "
            "the head and drop the rest"
        )

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
        over = self.over_budget_tokens(text)
        if over is not None:
            raise ValueError(self._over_budget_message(over))
        return self._embed_inputs(text)[0]

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        """Embed a batch, reporting per-item None for anything over the window.

        A single over-long chunk must not fail the whole batch: the caller
        records it as a failed chunk and keeps going, which is how every other
        per-document failure behaves.
        """
        if not texts:
            return []
        results: list[list[float] | None] = [None] * len(texts)
        sendable = [
            (i, text) for i, text in enumerate(texts) if self.over_budget_tokens(text) is None
        ]
        if not sendable:
            return results
        embeddings = self._embed_inputs([text for _, text in sendable])
        if len(embeddings) != len(sendable):
            raise ValueError(
                f"embedding count {len(embeddings)} does not match input count {len(sendable)}"
            )
        for (index, _), embedding in zip(sendable, embeddings):
            results[index] = embedding
        return results

    def over_budget_reason(self, text: str) -> str | None:
        """Why ``text`` cannot be embedded, or None if it can.

        Lets a caller that got a None back from ``embed_batch`` say *why* rather
        than reporting a bare failure. Callers of this method hold chunk text,
        so the chunk-sizing advice belongs here, not in the shared message.
        """
        over = self.over_budget_tokens(text)
        if over is None:
            return None
        return f"{self._over_budget_message(over)} -- lower pipeline.chunk_size"

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim


def build_llama_cpp_client(
    config: Config, spec: EmbeddingRuntimeSpec | None = None
) -> RemoteEmbeddingClient:
    """Construct a client for the configured runtime, performing no I/O.

    Separated from ``get_llama_cpp_runtime_client`` so a caller that only wants
    to *ask about* the daemon does not also inherit its start/restart machinery.
    `cementic status` used to build its client through that path, which polls
    for up to two minutes on a busy daemon before returning an answer the pid
    file already had.
    """
    runtime_spec = spec or runtime_spec_from_config(config)
    if runtime_spec.provider != "llama-cpp":
        raise ValueError(f"Expected llama-cpp runtime spec, got {runtime_spec.provider}")
    n_ctx = runtime_spec.n_ctx or config.llama_cpp.n_ctx
    n_gpu_layers = (
        runtime_spec.n_gpu_layers
        if runtime_spec.n_gpu_layers is not None
        else config.llama_cpp.n_gpu_layers
    )
    return RemoteEmbeddingClient(
        host=config.llama_cpp.daemon_host,
        port=config.llama_cpp.daemon_port,
        embedding_dim=runtime_spec.embedding_dim,
        expected_fingerprint=llama_cpp_runtime_fingerprint(
            model_path=runtime_spec.model_identifier,
            n_ctx=n_ctx,
            n_batch=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=runtime_spec.verbose,
        ),
        model_identifier=runtime_spec.model_identifier,
        timeout=float(config.llama_cpp.llama_embed_timeout_seconds),
        # The window the daemon is actually launched with (n_ctx above), not
        # current config: they diverge once config changes after indexing, and
        # the budget that matters is the one the served model enforces.
        n_ctx=n_ctx,
    )


def get_llama_cpp_runtime_client(
    config: Config,
    spec: EmbeddingRuntimeSpec | None = None,
    autostart: bool | None = None,
) -> RemoteEmbeddingClient:
    """Return a client for the llama.cpp embedding server for one runtime spec."""
    runtime_spec = spec or runtime_spec_from_config(config)
    client = build_llama_cpp_client(config, runtime_spec)
    # The client already carries the fingerprint it was built from; recomputing
    # it here is a second chance for the two to disagree.
    fingerprint = client.expected_fingerprint

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

    # Check the model is actually there before spawning a server around it.
    # `search` and `embed` reach this path without ever running the bootstrapper,
    # so a missing or mistyped model produced `llama_cpp.server --model
    # /does/not/exist`, a child that died instantly, and a generic startup
    # failure -- instead of the accurate "model not found at ..." message that
    # already exists. Imported locally to keep this module's import graph free of
    # the database layer that bootstrap pulls in.
    from cementic.bootstrap import Bootstrapper

    Bootstrapper(config).ensure_embedding_runtime()

    print(
        "starting embedding daemon (a cold start loads the model; this can take 30s+)...",
        file=sys.stderr,
    )
    # Serialise stop-then-spawn across processes. Two racers that both saw no
    # daemon would both spawn on the same port: one wins the bind, the loser
    # exits, and whichever wrote the pid file last could record the *loser's*
    # PID -- leaving a live daemon holding the port and a multi-GB model that
    # `embedding stop` then reports as "already stopped".
    with file_lock(_daemon_lock_path(config), timeout=_DAEMON_LOCK_TIMEOUT_SECONDS):
        # Another process may have started a matching daemon while we waited.
        if client.matches_expected_runtime():
            return client
        _stop_mismatched_llama_cpp_daemon(config)
        daemon_pid = _start_llama_cpp_daemon(config, spec=runtime_spec)
        _wait_for_daemon_ready(
            client,
            timeout_seconds=config.llama_cpp.daemon_start_timeout_seconds,
            config=config,
            pid=daemon_pid,
            start_token=process_start_token(daemon_pid),
        )
    return client


class DaemonHealth(str, Enum):
    """What a probe of the embedding daemon established."""

    #: Answered, and is serving the runtime we expect.
    HEALTHY = "healthy"
    #: Did not answer, but the process is confirmed alive: mid-batch, not down.
    BUSY = "busy"
    #: Answered, serving something else. A config change, not busyness.
    WRONG_MODEL = "wrong_model"
    #: No answer and no live process.
    DOWN = "down"
    #: Answers listings but not embeddings, with no worker load to explain the
    #: silence: the process is up, the port is held, and the one path users
    #: depend on is dead.
    WEDGED = "wedged"


#: Budget for the optional embedding-path probe. A warm daemon answers a
#: one-token embedding well inside a second; five leaves room for a concurrent
#: search query's embedding to clear the model lock first.
EMBED_PROBE_SECONDS = 5.0


def probe_daemon(
    client: EmbeddingProvider,
    config: Config,
    *,
    wait_seconds: float = 0.0,
    embed_probe_seconds: float = 0.0,
) -> DaemonHealth:
    """Classify the embedding daemon, spending at most ``wait_seconds`` waiting.

    One helper with an explicit budget, replacing three implementations that
    used three different probes and three different timeouts -- which is why
    `status`, `doctor` and `search` could each report something different
    about the same daemon.

    ``wait_seconds=0`` costs one ``/v1/models`` round (a couple of seconds at
    most) and never polls, which is what a status read wants. Only a caller that
    genuinely needs the daemon *now* should pay to wait out a batch.

    ``embed_probe_seconds>0`` adds a second stage after the model list answers:
    a one-token embedding under that budget. ``/v1/models`` is metadata, served
    without the model lock, so it cannot see a dead embedding path -- the
    failure that let a wedged daemon read as healthy for 21 hours. A timed-out
    probe is only ``WEDGED`` when no live pipeline worker is mid-work;
    a worker's batch legitimately holds the model lock for tens of seconds,
    and misreporting that as a wedge would page the user during every index.
    """
    if not isinstance(client, RemoteEmbeddingClient):
        return DaemonHealth.HEALTHY if client.health_check() else DaemonHealth.DOWN

    served = client.probe_served_runtime()
    if served is True:
        if embed_probe_seconds <= 0 or client.probe_embedding(embed_probe_seconds):
            return DaemonHealth.HEALTHY
        if _worker_load_explains_slow_embeddings(config):
            return DaemonHealth.BUSY
        return DaemonHealth.WEDGED
    if served is False:
        # It answered. Whatever is loaded is not what this config asks for, and
        # no amount of waiting changes that.
        return DaemonHealth.WRONG_MODEL
    if not _daemon_pid_alive(config):
        return DaemonHealth.DOWN
    # No answer, but the process is alive. llama_cpp.server serializes every
    # request behind one lock, so a daemon mid-batch is indistinguishable from a
    # dead one over HTTP alone.
    if wait_seconds > 0 and _poll_until_ready(client, wait_seconds):
        return DaemonHealth.HEALTHY
    return DaemonHealth.BUSY


def _worker_load_explains_slow_embeddings(config: Config) -> bool:
    """Whether a live pipeline worker is mid-work, explaining a slow embed path.

    Reads the worker's own state file: a *live* worker (PID token-checked, so a
    crashed worker's leftover state does not count) reporting a current file or
    activity is saturating the daemon legitimately. Never the reason a probe
    fails -- any error reading state means "no explanation", not "wedged for
    sure", and the caller still reports the probe's own result.
    """
    from cementic.state import StateManager

    try:
        state = StateManager(config.pipeline_worker.state_path).load()
    except Exception:
        return False
    if not (state.current_file or state.current_activity):
        return False
    if not state.pid:
        return False
    return is_managed_process_alive(state.pid, state.start_token)


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


def supported_embedding_providers() -> tuple[str, ...]:
    """Return the registered embedding provider names."""
    return tuple(_PROVIDER_FACTORIES)


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

    Runs the *locked body* directly: its only caller already holds the daemon
    lock, and ``flock`` is not reentrant even within one process, so calling the
    public ``stop_llama_cpp_runtime`` here deadlocked the restart path against
    itself -- every runtime-config change waited out the full lock timeout and
    then failed with "another cementic process holds ...", naming this one.
    """
    pid_file = config.llama_cpp.daemon_pid_file
    if pid_file is None or not pid_file.exists():
        return
    _stop_llama_cpp_runtime_locked(config, pid_file)


def stop_llama_cpp_runtime(config: Config) -> bool:
    """Stop the configured llama.cpp daemon if a live PID file exists.

    Runs under the same daemon file lock as start/autostart: unlocked, a stop
    racing a concurrent autostart could kill the freshly started daemon or
    unlink the pid file it had just written -- recreating exactly the orphan
    (daemon holding the port with no record) the lock exists to prevent.
    """
    pid_file = config.llama_cpp.daemon_pid_file
    if pid_file is None or not pid_file.exists():
        return False
    with file_lock(_daemon_lock_path(config), timeout=_DAEMON_LOCK_TIMEOUT_SECONDS):
        return _stop_llama_cpp_runtime_locked(config, pid_file)


def _stop_llama_cpp_runtime_locked(config: Config, pid_file: Path) -> bool:
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

    # Classic TOCTOU: the liveness check above can pass and the daemon exit
    # before the signal lands (crash, OOM, a concurrent `embedding stop`).
    # Unguarded, that turned a normal race into a traceback out of
    # `cementic embedding stop` -- and out of any search or index that happened
    # to be restarting the daemon. supervisor.force_kill already handles this.
    signalled = _signal_daemon(pid, signal.SIGTERM)
    if signalled is _SignalResult.GONE:
        pid_file.unlink(missing_ok=True)
        return False
    if signalled is _SignalResult.NOT_OURS:
        # Alive, and not ours to signal. Deleting the pid file here would lose
        # the only record of a daemon that still holds the port, so `embedding
        # status` would report stopped forever with no way back to it.
        raise RuntimeError(
            f"llama.cpp daemon (pid {pid}) is running but cannot be signalled by this user"
        )
    remaining = wait_for_exit([pid], timeout_seconds=5.0)
    if remaining:
        _signal_daemon(pid, signal.SIGKILL)
        remaining = wait_for_exit([pid], timeout_seconds=2.0)
    if remaining:
        # It survived SIGKILL (uninterruptible sleep, e.g. unmapping a
        # multi-GB model over network storage). Reporting success and dropping
        # the pid file would strand it holding the port.
        raise RuntimeError(
            f"llama.cpp daemon (pid {pid}) did not exit after SIGKILL; it still holds "
            f"port {config.llama_cpp.daemon_port}"
        )
    pid_file.unlink(missing_ok=True)
    return True


class _SignalResult(str, Enum):
    """Outcome of signalling the daemon."""

    SENT = "sent"
    GONE = "gone"
    NOT_OURS = "not_ours"


def _signal_daemon(pid: int, signal_number: int) -> _SignalResult:
    """Send a signal to the daemon, distinguishing "already gone" from "not ours".

    Collapsing the two meant a daemon owned by another user was treated as
    already exited: the pid file was deleted and the stop reported as a no-op,
    leaving a live process nothing could find again.
    """
    try:
        os.kill(pid, signal_number)
    except ProcessLookupError:
        return _SignalResult.GONE
    except PermissionError:
        return _SignalResult.NOT_OURS
    return _SignalResult.SENT


def _start_llama_cpp_daemon(
    config: Config,
    spec: EmbeddingRuntimeSpec | None = None,
) -> int:
    """Spawn the llama.cpp server and return its PID."""
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
        n_batch=n_ctx,
        n_gpu_layers=n_gpu_layers,
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
        # The server truncates every embedding input at n_batch tokens, whose
        # own default is 512 -- so without these, raising n_ctx silently did
        # nothing and inputs were still cut at 512. Both are derived from
        # n_ctx, which is already in the runtime fingerprint, so pinning them
        # here does not re-version anything.
        "--n_batch",
        str(n_ctx),
        "--n_ubatch",
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
    try:
        _write_daemon_pid_file(pid_file, pid)
    except OSError:
        # The daemon is already running -- a multi-GB model resident -- with
        # nothing recording its PID (full/read-only data dir, EACCES). Leaving
        # it would orphan it: `embedding status` would report "stopped" and
        # `embedding stop` would return False forever with no way back to it.
        # Killing it here is safer than a live daemon nothing can track.
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:  # pragma: no cover - best effort; the write failure wins
            pass
        raise
    return pid


def _poll_until_ready(client: RemoteEmbeddingClient, timeout_seconds: float) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if client.matches_expected_runtime():
            return True
        time.sleep(0.2)
    return False


def _log_tail(log_file: Path | None, lines: int = 12) -> str:
    """Return the last few lines of the daemon log, or '' if unavailable."""
    if log_file is None:
        return ""
    try:
        content = log_file.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    if not content:
        return ""
    return "\n".join(content.splitlines()[-lines:])


def _daemon_failure_message(reason: str, config: Config) -> str:
    """Build a daemon startup error that points at the evidence."""
    log_file = config.llama_cpp.daemon_log_file
    message = reason
    if log_file is not None:
        message += f"\nDaemon log: {log_file}"
    tail = _log_tail(log_file)
    if tail:
        message += f"\n--- last lines of the daemon log ---\n{tail}"
    return message


def _wait_for_daemon_ready(
    client: RemoteEmbeddingClient,
    timeout_seconds: int,
    *,
    config: Config,
    pid: int | None = None,
    start_token: str | None = None,
) -> None:
    """Wait for the freshly spawned daemon, failing fast if it dies.

    Watches the child process as well as the endpoint. Every distinct startup
    failure -- corrupt GGUF, missing model, port already bound, out of memory,
    n_gpu_layers too high -- used to surface as the same sentence after the full
    timeout, with no hint that a log existed. The child usually dies within a
    second, so noticing that turns a blind 30s wait into an immediate, specific
    error carrying the log that explains it.
    """
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        models = client._list_models()
        if models:
            served = [str(model.get("id")) for model in models]
            if client.expected_fingerprint in served:
                return
            # Answering, with a real (non-empty) model list that is not ours.
            # One daemon serves one model fixed at launch, so waiting out the
            # rest of the timeout cannot change this answer -- it used to burn
            # the full budget and then report the generic "did not become
            # ready" for a definitively wrong model.
            raise RuntimeError(
                _daemon_failure_message(
                    "llama.cpp embedding daemon started but is serving a different "
                    f"model/runtime than this config expects ({', '.join(served)}).",
                    config,
                )
            )
        if pid is not None and not is_managed_process_alive(pid, start_token):
            raise RuntimeError(
                _daemon_failure_message(
                    f"llama.cpp embedding daemon (PID {pid}) exited during startup.",
                    config,
                )
            )
        time.sleep(0.2)
    raise RuntimeError(
        _daemon_failure_message(
            f"llama.cpp embedding daemon did not become ready within {timeout_seconds}s.",
            config,
        )
    )
