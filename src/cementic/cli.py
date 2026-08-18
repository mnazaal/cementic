"""CLI interface for cementic using Typer."""

import inspect
import json
import os
import shutil
import sys
import time
from collections.abc import Callable
from importlib import resources
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, TypeVar, cast

import click
import typer
from pydantic import ValidationError
from pydantic_settings import SettingsError
from rich.console import Console
from rich.markup import escape
from sqlalchemy.exc import InterfaceError, OperationalError, ProgrammingError
from typer.core import TyperCommand, TyperGroup

from cementic.bootstrap import Bootstrapper
from cementic.chunk import chunk_text
from cementic.collections import (
    collection_exists,
    delete_collection_records,
    drop_orphan_vector_tables,
    list_collection_revisions,
    list_collections,
    promote_ready_revision,
    reindex_collection,
    remove_artifacts,
)
from cementic.config import (
    Config,
    ConfigError,
    config_path_error,
    default_config_path,
    format_config_error,
    get_config,
    resolve_config_path,
)
from cementic.db import get_engine, get_session_factory
from cementic.doctor import collect_doctor_report
from cementic.embedding_runtime import (
    create_provider,
    get_llama_cpp_runtime_client,
    llama_daemon_status,
    runtime_spec_from_config,
    stop_llama_cpp_runtime,
)
from cementic.embedding_text import describe_text_policy
from cementic.extract import extract_document
from cementic.filelock import LockUnavailableError, file_lock
from cementic.search import MAX_SEARCH_RESULTS, Searcher
from cementic.state import DaemonState, StateManager
from cementic.status_service import (
    build_supervisor_status,
    check_health,
    load_file_progress,
    load_pipeline_status,
    load_pipeline_status_bulk,
    load_worker_statuses,
)
from cementic.supervisor import (
    ManagedProcess,
    force_kill,
    is_managed_process_alive,
    is_pid_running,
    load_supervisor_state,
    managed_process_pid,
    managed_process_start_token,
    process_start_token,
    save_supervisor_state,
    spawn_detached,
    wait_for_exit,
)
from cementic.validation import validate_collection_name

_CommandFn = TypeVar("_CommandFn", bound=Callable[..., Any])


class _UpperFormatter(click.HelpFormatter):
    """Plain help formatter that uppercases section headings (USAGE, OPTIONS, …).

    Keeps the headings consistent with Click's uppercase usage metavars
    (``[OPTIONS] COMMAND [ARGS]``).
    """

    def write_heading(self, heading: str) -> None:
        super().write_heading(heading.upper())

    def write_usage(self, prog: str, args: str = "", prefix: str | None = None) -> None:
        super().write_usage(prog, args, prefix=(prefix or "Usage: ").upper())


class _UpperContext(click.Context):
    def make_formatter(self) -> click.HelpFormatter:
        return _UpperFormatter(width=self.terminal_width, max_width=self.max_content_width)


class _PlainEpilogMixin:
    """Render the epilog at the base indent so its own headers line up with
    USAGE/OPTIONS/COMMANDS (Click otherwise indents the whole epilog)."""

    epilog: str | None

    def format_epilog(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        if self.epilog:
            formatter.write_paragraph()
            formatter.write_text(inspect.cleandoc(self.epilog))


class _UpperGroup(_PlainEpilogMixin, TyperGroup):
    context_class = _UpperContext


class _UpperCommand(_PlainEpilogMixin, TyperCommand):
    context_class = _UpperContext


class CementicTyper(typer.Typer):
    """Typer app with plain, case-consistent help formatting."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("rich_markup_mode", None)
        kwargs.setdefault("add_completion", False)
        kwargs.setdefault("no_args_is_help", True)
        kwargs.setdefault("cls", _UpperGroup)
        context_settings = cast(dict[str, Any], dict(kwargs.get("context_settings") or {}))
        context_settings.setdefault("help_option_names", ["-h", "--help"])
        kwargs["context_settings"] = context_settings
        super().__init__(*args, **kwargs)

    def command(self, *args: Any, **kwargs: Any) -> Callable[[_CommandFn], _CommandFn]:
        kwargs.setdefault("cls", _UpperCommand)
        return super().command(*args, **kwargs)


app = CementicTyper(help="Index and semantically search document collections")
collection_app = CementicTyper(help="Inspect and manage collections")
embedding_app = CementicTyper(help="Manage embedding runtime service")
config_app = CementicTyper(help="View and manage the config file")
init_app = CementicTyper(help="Initialize local setup files")
app.add_typer(collection_app, name="collection")
app.add_typer(embedding_app, name="embedding")
app.add_typer(config_app, name="config")
app.add_typer(init_app, name="init")
# soft_wrap: off a TTY rich falls back to an 80-column hard wrap, which split
# paths and aligned rows mid-word in piped or redirected output. Line breaking
# belongs to the terminal or the consuming program, not to us.
console = Console(soft_wrap=True)
# Errors and diagnostics go here so a failure never pollutes the data on
# stdout (which would otherwise be piped on as content, or break `--json | jq`).
err_console = Console(stderr=True, soft_wrap=True)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(_get_cli_version())
        raise typer.Exit()


# A leading "\b" marks a paragraph as preformatted so Click does not re-wrap it.
_ROOT_EPILOG = """\
\b
EXAMPLES:
  cementic start ~/research-papers -c research
  cementic search "vector databases" -c research
  cementic status

\b
Run 'cementic COMMAND --help' for details on a command.
Docs and issues: https://github.com/mnazaal/cementic
"""

_START_EPILOG = """\
\b
EXAMPLES:
  cementic start ~/research-papers
  cementic start ~/papers ~/notes --collection research
"""

_SEARCH_EPILOG = """\
\b
EXAMPLES:
  cementic search "transformer inference"
  cementic search "graph theory" -n 5 -c math papers
  cementic search "graph theory" --json | jq .source_path
"""


@app.callback(epilog=_ROOT_EPILOG)
def _root(
    version: bool = typer.Option(
        None,
        "--version",
        "-V",
        help="Show the cementic version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Index and semantically search document collections.

    cementic watches directories of documents (PDF, Markdown, plain text) and
    builds a versioned extract -> chunk -> embed pipeline in Postgres (pgvector),
    then serves fast semantic search over the indexed chunks.
    """

_config: Config | None = None


def _get_config() -> Config:
    """Lazy-load the config singleton, reporting config problems in one line.

    Every command routes through here, so this is where a broken config stops
    being a multi-screen pydantic traceback. A failed load is deliberately not
    cached: fixing the file and re-running must work.
    """
    global _config
    if _config is None:
        try:
            _config = get_config()
        # escape(): these messages quote section names like "[llama_cpp]", which
        # rich would otherwise consume as markup -- dropping the one detail the
        # message exists to convey.
        except ConfigError as error:
            err_console.print(f"[red]config error: {escape(str(error))}[/red]")
            raise typer.Exit(1)
        except ValidationError as error:
            detail = format_config_error(error, resolve_config_path())
            err_console.print(f"[red]config error: {escape(detail)}[/red]")
            raise typer.Exit(1)
        except SettingsError as error:
            # pydantic-settings JSON-parses complex-typed fields from the
            # environment and raises SettingsError -- a ValueError, *not* a
            # ValidationError -- so a plausible spelling like
            # CEMENTIC_EXTRACT_BACKENDS=pdf=pymupdf4llm reached the user as a
            # multi-screen traceback from every command.
            err_console.print(f"[red]config error: {escape(str(error))}[/red]")
            err_console.print(
                "hint: settings that take a list or table are read from the "
                "environment as JSON, e.g. CEMENTIC_EXTRACT_BACKENDS='{\"pdf\": "
                "\"pymupdf4llm\"}'"
            )
            raise typer.Exit(1)
    return _config


_DEFAULT_CONFIG_TOML = """\
# cementic configuration
#
# Precedence (low -> high): built-in defaults < this file < CEMENTIC_* env vars
# < command-line flags. Every value below is optional; delete what you don't need.

[database]
# Postgres with the pgvector and pgvectorscale extensions. Generate a local
# setup with `cementic init postgres ./cementic-postgres`, then start it with
# `docker compose up -d` (or `podman compose up -d`).
host = "localhost"
port = 5432
name = "cementic"
user = "cementic"
# password = "cementic"   # override outside local development

[pipeline]
embedding_provider = "llama-cpp"
# Counted with tiktoken, while llama_cpp.n_ctx (512) counts the model's own
# tokens -- for the default model one of these is up to 1.33 of the other, so
# chunk_size must stay well under n_ctx or chunks embed truncated. Re-measure
# with scripts/measure_chunk_context_fit.py before raising it.
chunk_size = 320
chunk_overlap = 80

[index]
# ANN index: "hnsw" (lower latency, more RAM) or "diskann" (disk-resident, low RAM)
method = "hnsw"
# Keep scanning until top_k rows survive the filter, rather than stopping after
# ef_search candidates. Leave on unless you are on pgvector older than 0.8,
# where it is ignored anyway.
hnsw_iterative_scan = "relaxed_order"
# maintenance_work_mem for index builds only. PostgreSQL's 64MB default makes an
# HNSW build spill to disk and slow sharply; lower this on a small server.
build_memory = "2GB"

[llama_cpp]
# Spelled exactly as the built-in default: the model path string is part of the
# embedding profile fingerprint, so writing "./models/..." here instead would
# mint a second profile for the same file and re-embed the whole corpus.
model_path = "models/nomic-embed-text-v2-moe.Q8_0.gguf"

[extraction]
use_ocr = false

# Optional: choose a specific extractor per file type. Unset types use the
# registry default. Keys are bare file types; values are registered extractor
# names (currently: "pymupdf4llm" for pdf, "plaintext" for txt/md/markdown).
[extraction.backends]
# pdf = "pymupdf4llm"
"""


@config_app.command("path", short_help="Print the active (or default) config path")
def config_path() -> None:
    """Print the active config file path, or the default location if none exists."""
    # The same guard every other command hits via get_config(): without it this
    # command printed the fallback path for an unusable CEMENTIC_CONFIG -- the
    # one symptom it exists to diagnose.
    problem = config_path_error()
    if problem is not None:
        err_console.print(f"config error: {problem}")
        raise typer.Exit(1)
    active = resolve_config_path()
    typer.echo(str(active if active is not None else default_config_path()))


@config_app.command("init", short_help="Write an annotated default config file")
def config_init(
    force: bool = typer.Option(False, "--force", help="Overwrite an existing config file"),
) -> None:
    """Write an annotated default config to the user config directory."""
    path = default_config_path()
    if path.exists() and not force:
        console.print(f"config already exists at {path} (use --force to overwrite)")
        raise typer.Exit(1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_DEFAULT_CONFIG_TOML, encoding="utf-8")
    console.print(f"wrote {path}")


@config_app.command("show", short_help="Print the effective merged config as JSON")
def config_show() -> None:
    """Print the effective configuration (defaults + file + env) as JSON to stdout."""
    typer.echo(json.dumps(_get_config().model_dump(mode="json"), indent=2, sort_keys=True))


@init_app.command("postgres", short_help="Write a local Postgres setup directory")
def init_postgres(
    directory: Path = typer.Argument(..., help="Directory to write setup files into"),
    force: bool = typer.Option(
        False, "--force", help="Overwrite setup files in an existing non-empty directory"
    ),
) -> None:
    """Copy static Docker/Podman Postgres setup files for cementic."""
    if directory.exists() and not directory.is_dir():
        err_console.print(f"{directory} exists and is not a directory")
        raise typer.Exit(1)
    try:
        non_empty = directory.is_dir() and any(directory.iterdir())
    except OSError as error:
        err_console.print(f"cannot read {directory}: {error}")
        raise typer.Exit(1)
    if non_empty and not force:
        err_console.print(f"{directory} already exists and is not empty (use --force to overwrite)")
        raise typer.Exit(1)

    template_root = resources.files("cementic") / "templates" / "postgres"
    with resources.as_file(template_root) as source:
        if not source.is_dir():
            err_console.print("Postgres setup templates are missing from this installation")
            raise typer.Exit(1)
        # Overwrite the template files in place rather than clearing the
        # directory first: `--force` used to `rmtree` whatever it was pointed
        # at, so `cementic init postgres ~ --force` deleted the user's home
        # directory before writing five files into it.
        try:
            shutil.copytree(source, directory, dirs_exist_ok=True)
        except OSError as error:
            err_console.print(f"failed to write setup files to {directory}: {error}")
            raise typer.Exit(1)

    console.print(f"Wrote Postgres setup to {directory}")
    console.print("")
    console.print("Start the persistent local database once:")
    console.print(f"  cd {directory}")
    console.print("  docker compose up -d")
    console.print("  # or: podman compose up -d")
    console.print("")
    console.print("Then check readiness:")
    console.print("  cementic status --doctor")


def _load_supervisor_state() -> dict[str, object]:
    return load_supervisor_state(_get_supervisor_state_path())


def _save_supervisor_state(state: dict[str, object]) -> None:
    save_supervisor_state(_get_supervisor_state_path(), state)


def _supervisor_processes(state: dict[str, object]) -> list[dict[str, object]]:
    processes = state.get("processes", [])
    if not isinstance(processes, list):
        return []
    return [proc for proc in processes if isinstance(proc, dict)]


def _is_managed_proc_alive(process: dict[str, object]) -> bool:
    return is_managed_process_alive(
        managed_process_pid(process), managed_process_start_token(process)
    )


_STARTUP_GRACE_SECONDS = 2.0


def _pipeline_worker_activity() -> str | None:
    """What the pipeline worker is busy with, if it published anything.

    Read from its state file rather than inferred, and tolerant of every way
    that can fail -- this only ever adds an explanation to a stop message, so it
    must never be the reason a stop fails.
    """
    try:
        state = StateManager(_get_config().pipeline_worker.state_path).load()
    except Exception:
        return None
    return state.current_activity or None


def _worker_processes_from_state_files() -> list[dict[str, object]]:
    """Recover live worker records when the supervisor file is missing.

    ``supervisor.json`` is the normal source of PIDs, but it is a single file
    that every stop deletes and every start overwrites. Losing it left
    `cementic stop` unable to stop anything while reporting that nothing was
    running -- the user's only recourse being `ps` and `kill`. Each worker
    already persists its own pid and start-token, so the information was never
    actually lost; this just looks where it still is.
    """
    config = _get_config()
    records: list[dict[str, object]] = []
    for name, state_path in (
        ("source-watcher", config.source_watcher.state_path),
        ("pipeline-worker", config.pipeline_worker.state_path),
    ):
        if state_path is None:
            continue
        worker_state = StateManager(state_path).load()
        pid = worker_state.pid
        if pid and is_managed_process_alive(pid, worker_state.start_token):
            records.append(
                {"name": name, "pid": pid, "start_token": worker_state.start_token}
            )
    return records


def _terminate_managed(processes: list[ManagedProcess]) -> None:
    """Stop spawned workers, escalating to SIGKILL for anything that lingers."""
    pids = [
        proc.pid
        for proc in processes
        if is_managed_process_alive(proc.pid, proc.start_token)
    ]
    for pid in pids:
        try:
            os.kill(pid, 15)
        except OSError:
            continue
    remaining = wait_for_exit(pids, timeout_seconds=5.0)
    if remaining:
        force_kill(remaining)


def _clear_worker_state_files() -> None:
    """Mark both workers stopped after they were killed.

    A worker writes `stopped` from its own shutdown path, which never runs under
    SIGKILL. Without this, `status --verbose` reported "stopped, state=running,
    pid=<dead pid>" indefinitely -- a contradiction in the same line. The
    headline worker state was already correct (it checks liveness), so this
    aligns the detail with it.
    """
    config = _get_config()
    for state_path in (
        config.source_watcher.state_path,
        config.pipeline_worker.state_path,
    ):
        if state_path is None:
            continue
        try:
            StateManager(state_path).update(
                daemon_state=DaemonState.STOPPED, pid=None, start_token=None, current_file=None
            )
        except OSError:
            continue



def _wait_for_worker_startup(
    processes: list[ManagedProcess], grace_seconds: float = _STARTUP_GRACE_SECONDS
) -> list[ManagedProcess]:
    """Return the processes that died within the startup grace period.

    A worker that fails at startup (bad config, unreachable runtime, lock held)
    exits within a fraction of a second, so a short watch catches it; a healthy
    worker outlives the grace period and this returns empty.
    """
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        dead = [
            proc
            for proc in processes
            if not is_managed_process_alive(proc.pid, proc.start_token)
        ]
        if dead:
            return dead
        time.sleep(0.1)
    return []


def _get_cli_version() -> str:
    """Return installed cementic version, or unknown."""
    try:
        return version("cementic")
    except PackageNotFoundError:
        return "unknown"


def _get_data_dir() -> Path:
    """Return cementic data directory path."""
    state_path = _get_config().source_watcher.state_path or _get_config().pipeline_worker.state_path
    if state_path is None:
        raise RuntimeError("State path is not configured")
    return state_path.parent


def _get_start_lock_path() -> Path:
    """Lock file serialising `cementic start`'s check-then-spawn sequence."""
    return _get_data_dir() / "start.lock"



def _get_supervisor_state_path() -> Path:
    """Lazy supervisor state path."""
    return _get_data_dir() / "supervisor.json"


@collection_app.callback(invoke_without_command=True)
def collection_callback(ctx: typer.Context) -> None:
    """Show collection subcommand help when no subcommand is provided."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


def _build_collection_filters(
    option_collections: list[str] | None,
    trailing_collections: list[str] | None,
) -> list[str] | None:
    """Build collection filters from option and trailing values."""
    option_values = list(option_collections or [])
    trailing_values = list(trailing_collections or [])

    if trailing_values and not option_values:
        joined = " ".join(trailing_values)
        raise typer.BadParameter(
            f"Unexpected argument(s): {joined}. Use --collection/-c before collection names."
        )

    merged = [*option_values, *trailing_values]
    return merged or None


def _is_database_unavailable(error: Exception) -> bool:
    """Return whether the error indicates an unreachable database."""
    return isinstance(error, (OperationalError, InterfaceError))


def _is_schema_missing(error: Exception) -> bool:
    """Return whether the error means cementic's tables don't exist yet.

    The schema is created by the pipeline worker on first `cementic start`, so a
    reachable-but-empty database is the normal pre-first-run state, not a fault.
    """
    return isinstance(error, ProgrammingError) and "does not exist" in str(error.orig)


_NO_SCHEMA_HINT = "nothing indexed yet — run `cementic start DIRECTORY -c COLLECTION` first"


def _require_known_collection(session: Any, collection: str) -> None:
    """Exit 1 with a clear message when a named collection does not exist.

    "Does not exist" and "exists but has nothing to show" used to be
    indistinguishable -- a zero-filled status report, an empty revision list, or
    "no ready revision", each at exit 0 -- so a typo'd collection name read as a
    real but idle one, and no script could tell the difference.
    """
    if collection_exists(session, collection):
        return
    err_console.print(f"collection: {collection}")
    err_console.print("status: unknown collection (check `cementic collection list`)")
    raise typer.Exit(1)


def _report_db_error(error: Exception, action: str) -> None:
    """Print the right message for a failed database operation."""
    if _is_database_unavailable(error):
        _print_database_unavailable(action)
    elif _is_schema_missing(error):
        err_console.print(_NO_SCHEMA_HINT)
    else:
        err_console.print(f"{action} failed: {error}")


_DB_HINT = (
    "hint: run `cementic init postgres ./cementic-postgres` once and follow its README, "
    "or set CEMENTIC_DB_URL to an existing Postgres with pgvector and pgvectorscale"
)


def _state(ok: bool, ok_word: str, bad_word: str) -> str:
    """A status word, colored sparingly (rich drops color off-TTY / NO_COLOR)."""
    return f"[green]{ok_word}[/green]" if ok else f"[red]{bad_word}[/red]"


def _print_database_unavailable(action: str) -> None:
    """Print a concise database-unavailable message with a recovery hint."""
    err_console.print(f"{action}: database not reachable")
    err_console.print(_DB_HINT)


def _llama_daemon_runtime_status() -> str:
    """Return llama.cpp daemon runtime status."""
    return llama_daemon_status(_get_config())


def _print_doctor_report(report: dict[str, Any]) -> None:
    """Print read-only doctor diagnostics in a compact human format."""
    doctor_status = "[green]ok[/green]" if report["ok"] else "[red]failed[/red]"
    console.print(f"cementic doctor: {doctor_status}")
    checks = report["checks"]
    for name, payload in checks.items():
        if name == "extensions":
            console.print("extensions:")
            for extension, extension_payload in payload.items():
                console.print(
                    f"  - {extension}: {extension_payload['status']} "
                    f"({extension_payload['message']})"
                )
            continue
        status_text = payload.get("status", "unknown")
        message = payload.get("message")
        console.print(f"{name}: {status_text}" + (f" — {message}" if message else ""))


def _print_status_summary(
    supervisor_collection: str,
    directories: list[str],
    source_watcher_status: Any,
    pipeline_worker_status: Any,
    health: Any,
    verbose: bool,
) -> None:
    """Print the concise worker + health summary; full detail behind --verbose."""
    source_running = source_watcher_status.process == "running"
    pipeline_running = pipeline_worker_status.process == "running"
    running = int(source_running) + int(pipeline_running)
    if running == 2:
        workers = "[green]running[/green]"
    elif running == 0:
        workers = "[red]stopped[/red]"
    else:
        workers = "[yellow]partial[/yellow]"
    console.print(f"{'workers':<11} {workers}")
    if health is not None:
        console.print(f"{'database':<11} {_state(health.db_reachable, 'reachable', 'unreachable')}")
        if health.embedding_healthy:
            embedding_text = "[green]healthy[/green]"
        elif _get_config().llama_cpp.daemon_autostart:
            # Same state `status --doctor` calls a warning: not running now, but
            # cementic starts it on demand. Not an error.
            embedding_text = "[yellow]stopped (autostarts when needed)[/yellow]"
        else:
            embedding_text = "[red]unhealthy[/red]"
        console.print(f"{'embedding':<11} {embedding_text}")

    # A worker looping on a permanent failure otherwise looks exactly like a
    # healthy idle one, so this is headline information rather than --verbose
    # detail: without it the only evidence is a log file the user must know about.
    for label, worker in (
        ("source watcher", source_watcher_status),
        ("pipeline worker", pipeline_worker_status),
    ):
        if worker.last_error:
            console.print(
                f"{'last error':<11} [red]{escape(f'{label}: {worker.last_error}')}[/red]"
            )

    # Headline, not --verbose detail: a skipped file never becomes a document,
    # so it is absent from every pipeline count. Without this a collection whose
    # watcher dropped a directory of symlinks still reported 100% complete and
    # promoted cleanly, with the only evidence a number behind --verbose and a
    # log file the user is never pointed at.
    if source_watcher_status.failed_count:
        console.print(
            f"{'skipped':<11} [yellow]{source_watcher_status.failed_count} file(s) not "
            "indexed[/yellow]" + ("" if verbose else " (run with --verbose for paths)")
        )

    if not verbose:
        return

    console.print()
    console.print(f"session collection: {supervisor_collection}")
    if directories:
        console.print("directories:")
        for directory in directories:
            console.print(f"- {directory}")
    console.print(
        f"source watcher: {source_watcher_status.process}, "
        f"state={source_watcher_status.state}, pid={source_watcher_status.pid}, "
        f"processed={source_watcher_status.processed_count}, "
        f"failed={source_watcher_status.failed_count}"
    )
    if source_watcher_status.current_file != "None":
        console.print(f"  current file: {source_watcher_status.current_file}")
    if source_watcher_status.skipped_files:
        console.print("  skipped files:")
        for entry in source_watcher_status.skipped_files:
            console.print(f"  - {escape(entry)}")
        recorded = len(source_watcher_status.skipped_files)
        if source_watcher_status.failed_count > recorded:
            console.print(
                f"  (showing the {recorded} most recent of "
                f"{source_watcher_status.failed_count})"
            )
    console.print(
        f"pipeline worker: {pipeline_worker_status.process}, "
        f"state={pipeline_worker_status.state}, pid={pipeline_worker_status.pid}"
    )
    if pipeline_worker_status.current_file != "None":
        console.print(f"  current file: {pipeline_worker_status.current_file}")
    if pipeline_worker_status.current_activity:
        console.print(f"  activity: {pipeline_worker_status.current_activity}")
        console.print("  (no other work happens until this finishes)")
    if health is not None and health.llama_daemon != "N/A":
        console.print(f"embedding daemon: {health.llama_daemon}")


def _in_flight_revision_text(ready_label: str | None, building_label: str | None) -> str:
    """Render the not-yet-active revision under the status it is actually in.

    Both summary builders bucket ready and building together, so a finished
    revision was reported as `building=...` -- hiding the one fact the promote
    workflow turns on, that there is something ready to promote.
    """
    # Labels are interpolated into a markup-enabled string, so escape them: a
    # label containing a closing tag would raise MarkupError mid-render, and one
    # containing an opening tag would be swallowed.
    if ready_label:
        return f"[green]ready={escape(ready_label)}[/green]"
    return f"building={escape(building_label) if building_label else '-'}"


def _print_collection_detail(
    collection: str,
    pipeline_status: Any,
    verbose: bool,
) -> None:
    """Print a concise per-collection summary; per-file detail behind --verbose."""
    ps = pipeline_status
    console.print(f"{'collection':<11} {collection}")
    console.print(f"{'documents':<11} {ps.documents:,}")
    console.print(
        f"{'extracted':<11} {ps.extracted_done:,}/{ps.documents:,} ({ps.extraction_pct}%)"
    )
    console.print(
        f"{'chunked':<11} {ps.chunked_done:,}/{ps.extracted_done:,} ({ps.chunking_pct}%)"
    )
    # The denominator is chunks that exist *so far*, so mid-build this can read
    # 100% while most documents have not been chunked yet. Say so rather than
    # implying the collection is finished.
    #
    # total_chunks is final only once extraction has finished *and* chunking has
    # caught up with it -- the same two clauses as the worker's own completeness
    # check (pipeline_worker.revision_is_complete), so status and the worker
    # cannot disagree about whether a collection is done. Comparing chunking to
    # `documents` instead meant one document that failed to extract could never
    # be chunked, pinning the caveat on a collection that was in fact finished;
    # comparing to extracted_done alone would drop the caveat mid-extraction,
    # while more chunks were still on the way.
    extraction_complete = ps.extracted_done + ps.extracted_failed >= ps.documents
    chunking_complete = extraction_complete and (
        ps.chunked_done + ps.chunked_failed >= ps.extracted_done
    )
    embedded_suffix = "" if chunking_complete else " of chunks created so far"
    console.print(
        f"{'embedded':<11} {ps.done_embeddings:,}/{ps.total_chunks:,} "
        f"({ps.embedding_pct}%{embedded_suffix})"
    )
    console.print(
        f"{'revision':<11} active={ps.active_revision_label or '-'}  "
        f"{_in_flight_revision_text(ps.ready_revision_label, ps.building_revision_label)}"
    )
    if ps.ready_revision_label:
        console.print(
            f"{'':<11} run `cementic collection promote {collection}` to serve it"
        )
    failures = []
    if ps.extracted_failed:
        failures.append(f"extract={ps.extracted_failed}")
    if ps.chunked_failed:
        failures.append(f"chunk={ps.chunked_failed}")
    if ps.failed_embeddings:
        failures.append(f"embed={ps.failed_embeddings}")
    if failures:
        console.print(f"{'failures':<11} {', '.join(failures)}")

    if verbose and collection:
        try:
            files = load_file_progress(_get_config(), collection)
            if not files:
                console.print("files: none")
                return
            console.print("files:")
            for f in files:
                status_line = (
                    f"- {f.source_path} | extract={f.extraction_status}"
                    f" | chunk={f.chunking_status}"
                    f" | embeddings={f.embeddings_done}/{f.embeddings_total}"
                )
                if f.embeddings_failed:
                    status_line += f" (failed: {f.embeddings_failed})"
                if f.error_message:
                    status_line += f" | error: {f.error_message[:80]}"
                console.print(status_line)
        except Exception as error:
            if not _is_database_unavailable(error):
                console.print(f"files: error - {error}")


def _print_status_json(
    supervisor_status: Any,
    source_watcher_status: Any,
    pipeline_worker_status: Any,
    directories: list[str],
    health: Any,
    collection: str | None,
    verbose: bool,
) -> bool:
    """Print full status as JSON. Returns whether the pipeline section failed."""
    output: dict[str, Any] = {
        "supervisor": {
            "state": supervisor_status.state,
            "collection": supervisor_status.collection,
            "directories": directories,
        },
        "source_watcher": {
            "process": source_watcher_status.process,
            "state": source_watcher_status.state,
            "pid": source_watcher_status.pid,
            "processed": source_watcher_status.processed_count,
            "failed": source_watcher_status.failed_count,
            "skipped_files": source_watcher_status.skipped_files,
            "last_error": source_watcher_status.last_error,
            "last_error_at": source_watcher_status.last_error_at,
        },
        "pipeline_worker": {
            "process": pipeline_worker_status.process,
            "state": pipeline_worker_status.state,
            "pid": pipeline_worker_status.pid,
            "last_error": pipeline_worker_status.last_error,
            "last_error_at": pipeline_worker_status.last_error_at,
            "current_activity": pipeline_worker_status.current_activity,
        },
    }

    if health is not None:
        output["health"] = {
            "db_reachable": health.db_reachable,
            "embedding_provider": health.embedding_provider,
            "embedding_healthy": health.embedding_healthy,
            "llama_daemon": health.llama_daemon,
        }

    try:
        engine = get_engine(_get_config().database.url)
        session_factory = get_session_factory(engine)
        with session_factory() as session:
            if collection is None:
                rows = list_collections(session)
                status_by_collection = load_pipeline_status_bulk(
                    _get_config(), [row.name for row in rows]
                )
                collections_data: dict[str, dict[str, Any]] = {}
                for row in rows:
                    ps = status_by_collection[row.name]
                    collections_data[row.name] = {
                        "documents": ps.documents,
                        "extracted_done": ps.extracted_done,
                        "extracted_failed": ps.extracted_failed,
                        "chunked_done": ps.chunked_done,
                        "chunked_failed": ps.chunked_failed,
                        "total_chunks": ps.total_chunks,
                        "pending_embeddings": ps.pending_embeddings,
                        "processing_embeddings": ps.processing_embeddings,
                        "done_embeddings": ps.done_embeddings,
                        "failed_embeddings": ps.failed_embeddings,
                        "extraction_pct": ps.extraction_pct,
                        "chunking_pct": ps.chunking_pct,
                        "embedding_pct": ps.embedding_pct,
                        "active_revision_label": ps.active_revision_label,
                        "ready_revision_label": ps.ready_revision_label,
                        "building_revision_label": ps.building_revision_label,
                    }
                output["collections"] = collections_data
            else:
                # Mirror _require_known_collection on the human path: a typo'd
                # name otherwise produced a full zero-filled pipeline block at
                # exit 0, indistinguishable from a real collection not started.
                if not collection_exists(session, collection):
                    raise ValueError(
                        f"unknown collection: {collection}"
                        " (check `cementic collection list`)"
                    )
                ps = load_pipeline_status(_get_config(), collection)
                output["pipeline"] = {
                    "collection": collection,
                    "documents": ps.documents,
                    "extracted_done": ps.extracted_done,
                    "extracted_failed": ps.extracted_failed,
                    "chunked_done": ps.chunked_done,
                    "chunked_failed": ps.chunked_failed,
                    "total_chunks": ps.total_chunks,
                    "pending_embeddings": ps.pending_embeddings,
                    "processing_embeddings": ps.processing_embeddings,
                    "done_embeddings": ps.done_embeddings,
                    "failed_embeddings": ps.failed_embeddings,
                    "extraction_pct": ps.extraction_pct,
                    "chunking_pct": ps.chunking_pct,
                    "embedding_pct": ps.embedding_pct,
                    "active_revision_label": ps.active_revision_label,
                    "ready_revision_label": ps.ready_revision_label,
                    "building_revision_label": ps.building_revision_label,
                }
                if verbose:
                    files = load_file_progress(_get_config(), collection)
                    output["files"] = [
                        {
                            "source_path": f.source_path,
                            "extraction_status": f.extraction_status,
                            "chunking_status": f.chunking_status,
                            "embeddings_done": f.embeddings_done,
                            "embeddings_failed": f.embeddings_failed,
                            "embeddings_total": f.embeddings_total,
                            "error_message": f.error_message,
                        }
                        for f in files
                    ]
        failed = False
    except Exception as error:
        output["error"] = _NO_SCHEMA_HINT if _is_schema_missing(error) else str(error)
        failed = True

    typer.echo(json.dumps(output, indent=2, default=str))
    return failed


@app.command(
    "start",
    short_help="Start background indexing of one or more directories",
    epilog=_START_EPILOG,
    no_args_is_help=True,
)
def start_background(
    directories: list[str] = typer.Argument(..., help="Directories to watch for documents"),
    collection: str = typer.Option(
        "default",
        "-c",
        "--collection",
        help="Collection name for indexed documents",
    ),
) -> None:
    """Start source watcher and pipeline worker in the background."""
    try:
        collection = validate_collection_name(collection)
    except ValueError as e:
        err_console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    missing = [d for d in directories if not Path(d).is_dir()]
    if missing:
        for d in missing:
            err_console.print(f"[red]Error: Directory does not exist: {d}[/red]")
        raise typer.Exit(1)

    # Absolute from here on: the detached workers and any later `cementic status`
    # can run from a different working directory, where a relative path would
    # name something else entirely.
    directories = [str(Path(d).resolve()) for d in directories]

    # Serialise the whole check-then-spawn-then-record sequence. Two concurrent
    # `cementic start` runs could both see nothing running, both spawn, and the
    # second's supervisor record replace the first's -- leaving the first pair
    # running and invisible to `cementic stop`.
    try:
        with file_lock(_get_start_lock_path(), timeout=0):
            _start_background_locked(directories, collection)
    except LockUnavailableError:
        console.print("[yellow]Another `cementic start` is already in progress[/yellow]")
        raise typer.Exit(1)


def _start_background_locked(directories: list[str], collection: str) -> None:
    """The body of `cementic start`, run while holding the start lock."""
    state = _load_supervisor_state()
    running = [proc for proc in _supervisor_processes(state) if _is_managed_proc_alive(proc)]

    if running:
        console.print("[yellow]Background cementic processes already running:[/yellow]")
        for proc in running:
            console.print(f"- {proc.get('name')}: PID {proc.get('pid')}")
        raise typer.Exit(1)

    try:
        bootstrapper = Bootstrapper(_get_config())
        bootstrapper.ensure_for_convert()
        bootstrapper.ensure_for_index()
    except RuntimeError as error:
        err_console.print(f"[red]Bootstrap failed before background start: {error}[/red]")
        raise typer.Exit(1)

    base_cmd = [sys.executable, "-m", "cementic.runner"]
    data_dir = _get_data_dir()
    source_watcher_log = data_dir / "source-watcher-background.log"
    pipeline_log = data_dir / "pipeline-background.log"

    spawned: list[ManagedProcess] = []
    try:
        # `--` ends option parsing so a directory whose name begins with "-" is
        # not read as a flag by the runner.
        source_watcher_pid = spawn_detached(
            [*base_cmd, "source-watcher", "--collection", collection, "--", *directories],
            source_watcher_log,
        )
        spawned.append(
            ManagedProcess(
                "source-watcher",
                source_watcher_pid,
                str(source_watcher_log),
                process_start_token(source_watcher_pid),
            )
        )
        pipeline_pid = spawn_detached(
            [*base_cmd, "pipeline-worker", "--collection", collection],
            pipeline_log,
        )
        spawned.append(
            ManagedProcess(
                "pipeline-worker",
                pipeline_pid,
                str(pipeline_log),
                process_start_token(pipeline_pid),
            )
        )
    except OSError as error:
        # The first spawn can succeed and the second fail (ENOMEM, EMFILE,
        # unwritable log dir). Without this the survivor keeps running with no
        # supervisor record, so `cementic stop` could never find it again.
        _terminate_managed(spawned)
        err_console.print(f"[red]cementic failed to start: {error}[/red]")
        raise typer.Exit(1)

    _save_supervisor_state(
        {
            "collection": collection,
            "directories": directories,
            "processes": [proc.__dict__ for proc in spawned],
        }
    )

    # A worker can exit immediately (bootstrap failure, embedding runtime down,
    # lock held) writing only to its log file. Reporting success without looking
    # would leave the user believing indexing started.
    dead = _wait_for_worker_startup(spawned)
    if dead:
        err_console.print("[red]cementic failed to start:[/red]")
        for managed in dead:
            err_console.print(f"- {managed.name} exited immediately; see {managed.log_file}")
        # Stop whatever did come up and clear the record. Leaving a survivor
        # running would hold the collection's advisory lock and make the next
        # `cementic start` refuse with "already running" -- contradicting the
        # failure just reported, with no hint that `cementic stop` is the way out.
        dead_pids = {managed.pid for managed in dead}
        survivors = [managed for managed in spawned if managed.pid not in dead_pids]
        if survivors:
            _terminate_managed(survivors)
            for managed in survivors:
                console.print(f"- stopped {managed.name} (PID {managed.pid})")
        _get_supervisor_state_path().unlink(missing_ok=True)
        raise typer.Exit(1)

    console.print("[green]Started cementic in background[/green]")
    console.print(f"- source watcher PID: {source_watcher_pid}")
    console.print(f"- pipeline worker PID: {pipeline_pid}")
    console.print(f"- collection: {collection}")
    if collection == "default":
        console.print(
            "- note: no collection was specified, so documents will be indexed into 'default'"
        )
    console.print("Use `cementic status` to check progress and `cementic stop` to stop both.")


@app.command(
    "status",
    short_help="Show worker status and indexing progress",
)
def status(
    collection: str | None = typer.Option(
        None,
        "-c",
        "--collection",
        help="Show detailed status for one collection",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show per-file pipeline progress",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output status as JSON",
    ),
    doctor: bool = typer.Option(
        False,
        "--doctor",
        help="Run read-only runtime readiness diagnostics",
    ),
) -> None:
    """Show background worker status and collection progress."""
    if doctor:
        report = collect_doctor_report(_get_config())
        if json_output:
            typer.echo(json.dumps(report, indent=2, sort_keys=True))
        else:
            _print_doctor_report(report)
        if not report["ok"]:
            raise typer.Exit(1)
        return

    if collection is not None:
        try:
            collection = validate_collection_name(collection)
        except ValueError as e:
            err_console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

    state = _load_supervisor_state()
    source_watcher_status, pipeline_worker_status = load_worker_statuses(_get_config())
    supervisor_status = build_supervisor_status(state)
    directories = supervisor_status.directories or source_watcher_status.watched_directories

    try:
        health = check_health(_get_config())
    except Exception:
        health = None

    if json_output:
        failed = _print_status_json(
            supervisor_status,
            source_watcher_status,
            pipeline_worker_status,
            directories,
            health,
            collection,
            verbose,
        )
        if failed:
            raise typer.Exit(1)
        return

    _print_status_summary(
        supervisor_status.collection,
        directories,
        source_watcher_status,
        pipeline_worker_status,
        health,
        verbose,
    )

    if health is not None and not health.db_reachable:
        err_console.print(_DB_HINT)
        # Non-zero so `cementic status && ...` cannot succeed against a database
        # cementic could not reach; every other database-backed command exits 1.
        raise typer.Exit(1)

    try:
        engine = get_engine(_get_config().database.url)
        session_factory = get_session_factory(engine)
        with session_factory() as session:
            if collection is None:
                rows = list_collections(session)
                console.print()
                console.print("collections")
                if not rows:
                    console.print("  (none)")
                    return
                status_by_collection = load_pipeline_status_bulk(
                    _get_config(), [row.name for row in rows]
                )
                items = [(row, status_by_collection[row.name]) for row in rows]
                name_w = max(len(row.name) for row, _ in items)
                doc_w = max(len(f"{ps.documents:,}") for _, ps in items)
                frac_w = max(
                    len(f"{ps.done_embeddings:,}/{ps.total_chunks:,}") for _, ps in items
                )
                for row, ps in items:
                    frac = f"{ps.done_embeddings:,}/{ps.total_chunks:,}"
                    console.print(
                        f"  {row.name:<{name_w}}   {ps.documents:>{doc_w},} docs   "
                        f"{frac:>{frac_w}} embedded ({ps.embedding_pct}%)"
                    )
                return
            # This session used to be opened and then discarded on the
            # named-collection path. It now earns its connection: a typo'd name
            # otherwise produced a full zero-filled report at exit 0, which
            # reads as a real collection that has not started yet.
            console.print()
            _require_known_collection(session, collection)

        pipeline_status = load_pipeline_status(_get_config(), collection)
        _print_collection_detail(collection, pipeline_status, verbose)
    except typer.Exit:
        raise
    except Exception as error:
        _report_db_error(error, "status")
        raise typer.Exit(1)


@app.command(
    "stop",
    short_help="Stop cementic's background processes",
)
def stop_background(
    force: bool = typer.Option(
        False,
        "--force",
        help="Force kill processes that don't stop gracefully",
    ),
) -> None:
    """Stop cementic's background processes (source watcher, pipeline worker).

    The shared embedding daemon is intentionally left running so other
    collections and `search` stay warm; stop it separately with
    `cementic embedding stop`.

    Postgres is not managed by cementic; stop it with your container engine
    (e.g. `docker compose down` or `podman compose down`).
    """
    state = _load_supervisor_state()
    processes = _supervisor_processes(state)

    recovered = False
    if not processes:
        processes = _worker_processes_from_state_files()
        recovered = bool(processes)

    if not processes:
        console.print("no background cementic processes found")
        return

    if recovered:
        console.print(
            "supervisor record missing; found running workers from their own state files"
        )

    signaled_pids: list[int] = []
    unsignalable_pids: list[int] = []
    for proc in processes:
        pid = managed_process_pid(proc)
        # Only signal a process we can confirm is still ours; a recycled PID
        # (token mismatch) belongs to someone else and must not be killed.
        if not _is_managed_proc_alive(proc):
            continue
        try:
            os.kill(pid, 15)
            signaled_pids.append(pid)
        except ProcessLookupError:
            continue  # exited between the liveness check and the signal
        except OSError:
            # EPERM: alive, just not ours to signal (sudo, a service account, a
            # user namespace) -- the same reasoning as is_pid_running. Folding
            # this into "already gone" made stop print "cleared stale state"
            # and delete the state files of workers that kept indexing.
            unsignalable_pids.append(pid)

    timeout_seconds = 10.0  # grace period before --force is required
    remaining = wait_for_exit(signaled_pids, timeout_seconds=timeout_seconds)
    # SIGTERM never reached the EPERM'd pids, so they are certainly still
    # running: they rejoin the not-stopped set so no path below clears their
    # state, and --force reports them as unkillable instead of stale.
    remaining += unsignalable_pids

    if not remaining:
        _get_supervisor_state_path().unlink(missing_ok=True)
        _clear_worker_state_files()
        if signaled_pids:
            console.print(f"stopped {len(signaled_pids)} process(es)")
        else:
            # Recorded processes had already exited; "stopped 0 process(es)" read
            # as though a stop had happened and hid that they died on their own.
            console.print("no running processes found; cleared stale state")
        return

    if force:
        activity = _pipeline_worker_activity()
        if activity is not None:
            console.print(f"discarding in-progress work: {activity}")
        killed = force_kill(remaining)
        time.sleep(0.5)
        still_alive = [pid for pid in killed if is_pid_running(pid)]
        _get_supervisor_state_path().unlink(missing_ok=True)
        if not still_alive:
            _clear_worker_state_files()
        if still_alive:
            console.print(
                f"force killed {len(remaining) - len(still_alive)} process(es); "
                f"could not kill: {', '.join(str(pid) for pid in still_alive)}"
            )
            # Exiting 0 here told `cementic stop && cementic start` that the
            # workers were gone; the start then refused with "already running".
            raise typer.Exit(1)
        console.print(f"force stopped {len(remaining)} process(es)")
        return

    still_running = [proc for proc in processes if managed_process_pid(proc) in remaining]
    _save_supervisor_state(
        {
            "collection": state.get("collection"),
            "directories": state.get("directories", []),
            "processes": still_running,
        }
    )

    timed_out = [pid for pid in remaining if pid not in unsignalable_pids]
    if timed_out:
        console.print(
            f"stop timed out after {timeout_seconds}s; "
            f"still running PID(s): {', '.join(str(pid) for pid in timed_out)}"
        )
    if unsignalable_pids:
        # No timeout elapsed for these -- the signal itself was refused.
        console.print(
            f"could not signal PID(s) {', '.join(str(pid) for pid in unsignalable_pids)}: "
            "permission denied; still running, likely started by another user"
        )
    activity = _pipeline_worker_activity()
    if activity is not None:
        # Otherwise this reads as a hung worker. It is not: the worker cannot
        # answer SIGTERM from inside CREATE INDEX, and the statement is not
        # resumable, so forcing now throws the whole build away.
        console.print(f"the pipeline worker is {activity}, which does not stop on request")
        console.print("--force will discard that work; it restarts from scratch next run")
    else:
        console.print("use --force to kill stubborn processes")
    # Same reason as the force path above: nothing was stopped, so a caller
    # chaining on success must not proceed.
    raise typer.Exit(1)


@embedding_app.command("start", short_help="Start embedding runtime")
def start_embedding_runtime() -> None:
    """Start the configured embedding runtime service."""
    config = _get_config()
    if config.pipeline.embedding_provider != "llama-cpp":
        console.print(
            f"embedding start failed: unsupported provider "
            f"{config.pipeline.embedding_provider}"
        )
        raise typer.Exit(1)
    try:
        Bootstrapper(config).ensure_embedding_runtime()
    except RuntimeError as error:
        err_console.print(f"embedding start failed: {error}")
        raise typer.Exit(1)
    try:
        client = get_llama_cpp_runtime_client(config=config, autostart=True)
    except Exception as error:
        err_console.print(f"embedding start failed: {error}")
        raise typer.Exit(1)
    console.print("embedding: running")
    # Report the probed dimension, not the configured fallback: this command's
    # whole job is to confirm what is actually loaded.
    try:
        dimension: object = client.describe().embedding_dim
    except Exception as error:
        dimension = f"unknown ({error})"
    console.print(f"provider: llama-cpp, dim={dimension}")
    # Asymmetric models need task prefixes and the policy is chosen from the
    # model filename, so a renamed file silently disables it and quietly
    # degrades retrieval. Showing the choice makes that visible.
    console.print(f"text policy: {describe_text_policy(config.llama_cpp.model_path)}")


@embedding_app.command("stop", short_help="Stop embedding runtime")
def stop_embedding_runtime() -> None:
    """Stop the configured embedding runtime service."""
    config = _get_config()
    try:
        stopped = stop_llama_cpp_runtime(config)
    except RuntimeError as error:
        # The daemon is still up. Saying "stopped" here used to come with
        # discarding its pid file, so nothing could find it again.
        err_console.print(f"embedding stop failed: {error}")
        raise typer.Exit(1)
    console.print("embedding: stopped" if stopped else "embedding: already stopped")


@embedding_app.command("status", short_help="Show embedding runtime status")
def embedding_runtime_status() -> None:
    """Show configured embedding runtime service status."""
    console.print(f"embedding: {_llama_daemon_runtime_status()}")


@collection_app.command(
    "remove",
    short_help="Delete a collection and its artifacts",
    no_args_is_help=True,
)
def remove_collection(
    collection: str = typer.Argument(..., help="Collection name to delete"),
    force: bool = typer.Option(False, "--force", help="Skip confirmation prompt"),
) -> None:
    """Delete all documents and chunks belonging to a collection."""
    try:
        collection = validate_collection_name(collection)
    except ValueError as e:
        err_console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    if not force:
        confirm = typer.confirm(f"Delete collection '{collection}' and all associated chunks?")
        if not confirm:
            raise typer.Abort()

    try:
        engine = get_engine(_get_config().database.url)
        session_factory = get_session_factory(engine)

        with session_factory() as session:
            result = delete_collection_records(session, collection)
            if result is None:
                console.print(f"collection: {collection}")
                console.print("status: not found")
                return
    except Exception as e:
        _report_db_error(e, f"collection remove '{collection}'")
        raise typer.Exit(1)

    # The rows are committed by here, so the collection *is* deleted. Leftover
    # artifacts/vector tables are reported as a warning rather than turning a
    # successful delete into a reported failure.
    console.print(f"collection: {collection}")
    console.print("status: deleted")
    console.print(f"documents: {result.deleted_docs}")
    console.print(f"chunks: {result.deleted_chunks}")
    # A running watcher re-registers the files it watches and resurrects the
    # collection; the pipeline worker notices the deleted revision and exits on
    # its next poll. Deleting is still allowed -- the rows cascade safely --
    # but silently racing the watcher is not.
    supervisor_state = _load_supervisor_state()
    if supervisor_state.get("collection") == collection and any(
        _is_managed_proc_alive(proc) for proc in _supervisor_processes(supervisor_state)
    ):
        console.print(
            "warning: background workers are still watching this collection; "
            "the watcher will re-register its files -- run `cementic stop` to stop them"
        )
    try:
        unremoved = remove_artifacts(result.artifact_paths, config=_get_config())
        drop_orphan_vector_tables(engine, result.vector_profile_ids)
    except Exception as e:
        # The delete is already committed, so this is a warning about leftovers
        # on disk, not a failed removal. Exiting non-zero here contradicted both
        # the comment above and the documented behaviour, and told scripts the
        # collection had not been removed when it had.
        console.print(f"warning: collection deleted but cleanup failed: {e}")
        return
    if unremoved:
        # remove_artifacts has always returned the paths it could not remove;
        # both callers threw the list away, so files left behind were reported
        # only to a log file nobody is told about -- and the rows naming them
        # are gone, so nothing can find them again.
        console.print(f"warning: {len(unremoved)} artifact file(s) could not be removed:")
        for path in unremoved[:5]:
            console.print(f"  {path}")
        if len(unremoved) > 5:
            console.print(f"  ... and {len(unremoved) - 5} more")
    console.print(f"vector_tables_dropped: {len(result.vector_profile_ids)}")


@collection_app.command("list", short_help="List known collections")
def list_collection_command() -> None:
    """Show known collections."""
    try:
        engine = get_engine(_get_config().database.url)
        session_factory = get_session_factory(engine)
        with session_factory() as session:
            rows = list_collections(session)

        console.print("collections")
        if not rows:
            console.print("  (none)")
            return

        name_w = max(len(row.name) for row in rows)
        doc_w = max(len(f"{row.documents:,}") for row in rows)
        for row in rows:
            console.print(
                f"  {row.name:<{name_w}}   {row.documents:>{doc_w},} docs   "
                f"active={row.active_revision_label or '-'}  "
                f"{_in_flight_revision_text(row.ready_revision_label, row.building_revision_label)}"
            )
    except Exception as error:
        _report_db_error(error, "collection list")
        raise typer.Exit(1)


@collection_app.command(
    "promote",
    short_help="Promote a collection's ready revision to active",
    no_args_is_help=True,
)
def promote_collection(
    collection: str = typer.Argument(..., help="Collection name to promote"),
    force: bool = typer.Option(
        False,
        "-f",
        "--force",
        help="Promote even if the ready revision built with failed documents or chunks",
    ),
) -> None:
    """Promote the ready pipeline revision for one collection."""
    try:
        collection = validate_collection_name(collection)
    except ValueError as e:
        err_console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    try:
        engine = get_engine(_get_config().database.url)
        session_factory = get_session_factory(engine)
        with session_factory() as session:
            _require_known_collection(session, collection)
            outcome = promote_ready_revision(
                session, collection, config=_get_config(), force=force
            )
            # Read everything we need while the session is open; ORM attributes
            # expire on commit and would raise once the session closes.
            status = outcome.status
            counts = outcome.counts
            unremoved = outcome.unremoved_artifacts
            cleanup_error = outcome.cleanup_error
            revision_label = (
                outcome.revision.label or outcome.revision.id
                if outcome.revision is not None
                else None
            )
    except typer.Exit:
        # typer.Exit subclasses RuntimeError, so the broad handler below
        # would otherwise swallow a deliberate exit and re-report it as
        # "failed: 1".
        raise
    except Exception as error:
        _report_db_error(error, "collection promote")
        raise typer.Exit(1)

    console.print(f"collection: {collection}")
    if status == "no_ready":
        console.print("status: no ready revision")
        return
    if status == "empty":
        console.print("status: nothing to promote (revision has no documents)")
        console.print(
            "promoting would retire the active revision and leave nothing searchable"
        )
        raise typer.Exit(1)
    if status == "incomplete":
        pending = []
        if counts is not None:
            not_extracted = counts.documents - counts.extracted_done - counts.extracted_failed
            not_chunked = counts.extracted_done - counts.chunked_done - counts.chunked_failed
            not_embedded = counts.total_chunks - counts.done_embeddings - counts.failed_embeddings
            if not_extracted > 0:
                pending.append(f"extract={not_extracted}")
            if not_chunked > 0:
                pending.append(f"chunk={not_chunked}")
            if not_embedded > 0:
                pending.append(f"embed={not_embedded}")
        console.print(f"status: incomplete ({', '.join(pending)} pending)")
        console.print(
            "the revision took on new work after it was marked ready; "
            "wait for `cementic status` to show it finished, or --force to publish it as-is"
        )
        raise typer.Exit(1)
    if status == "blocked_by_failures":
        parts = []
        if counts is not None:
            if counts.extracted_failed:
                parts.append(f"extract={counts.extracted_failed}")
            if counts.chunked_failed:
                parts.append(f"chunk={counts.chunked_failed}")
            if counts.failed_embeddings:
                parts.append(f"embed={counts.failed_embeddings}")
        console.print(f"status: blocked ({', '.join(parts)} failed)")
        console.print("re-run with --force to promote anyway")
        raise typer.Exit(1)
    console.print("status: promoted")
    console.print(f"revision: {revision_label}")
    # Same contract as `collection remove`: the promote is committed, so
    # leftover files are a warning, not a failure -- but they used to be
    # discarded entirely here while remove reported them.
    if cleanup_error is not None:
        console.print(f"warning: promoted but cleanup failed: {cleanup_error}")
    if unremoved:
        console.print(f"warning: {len(unremoved)} artifact file(s) could not be removed:")
        for path in unremoved[:5]:
            console.print(f"  {path}")
        if len(unremoved) > 5:
            console.print(f"  ... and {len(unremoved) - 5} more")


@collection_app.command(
    "reindex",
    short_help="Rebuild a collection's ANN index from current index config",
    no_args_is_help=True,
)
def reindex_collection_command(
    collection: str = typer.Argument(..., help="Collection name to reindex"),
    force: bool = typer.Option(
        False,
        "-f",
        "--force",
        help="Rebuild even if the index method is unchanged (picks up hnsw_m and "
        "ef_construction, which are fixed at build time)",
    ),
) -> None:
    """Reconcile the active revision's ANN index with the current `[index]` config.

    The index is built once, when a revision first completes, so editing
    `index.method` afterwards otherwise had no effect and no way to ask for one.
    """
    try:
        collection = validate_collection_name(collection)
    except ValueError as e:
        err_console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    try:
        engine = get_engine(_get_config().database.url)
        session_factory = get_session_factory(engine)
        with session_factory() as session:
            # Before the "this can take several minutes" line, so an unknown
            # collection does not first announce work that will never start.
            _require_known_collection(session, collection)
            console.print(f"collection: {collection}")
            console.print(
                "building the index — this can take several minutes on a large corpus"
            )
            outcome = reindex_collection(
                session, collection, config=_get_config(), force=force
            )
    except typer.Exit:
        # typer.Exit subclasses RuntimeError, so the broad handler below
        # would otherwise swallow a deliberate exit and re-report it as
        # "failed: 1".
        raise
    except Exception as error:
        _report_db_error(error, "collection reindex")
        raise typer.Exit(1)

    if outcome.status == "no_active":
        console.print("status: no active revision — nothing has been promoted yet")
        raise typer.Exit(1)
    if outcome.status == "no_vectors":
        console.print("status: no vectors to index")
        return
    if outcome.previous_method is None:
        console.print(f"status: built ({outcome.method})")
    elif outcome.previous_method == outcome.method:
        console.print(f"status: rebuilt ({outcome.method})" if force else "status: unchanged")
        if not force:
            console.print(
                f"the index is already {outcome.method}; --force rebuilds it anyway"
            )
    else:
        console.print(f"status: rebuilt ({outcome.previous_method} -> {outcome.method})")


@collection_app.command(
    "revisions",
    short_help="Show a collection's revision history",
    no_args_is_help=True,
)
def list_collection_revision_command(
    collection: str = typer.Argument(..., help="Collection name to inspect"),
) -> None:
    """Show revision history for one collection."""
    try:
        collection = validate_collection_name(collection)
    except ValueError as e:
        err_console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    try:
        engine = get_engine(_get_config().database.url)
        session_factory = get_session_factory(engine)
        with session_factory() as session:
            _require_known_collection(session, collection)
            rows = list_collection_revisions(session, collection)
            console.print(f"{'collection':<11} {collection}")
            console.print("revisions")
            if not rows:
                console.print("  (none)")
                return

            id_w = max(len(str(row.id)) for row in rows)
            status_w = max(len(row.status) for row in rows)
            label_w = max(len(row.label or "-") for row in rows)
            for row in rows:
                console.print(
                    f"  {row.id:>{id_w}}  {row.status:<{status_w}}  "
                    f"{(row.label or '-'):<{label_w}}  "
                    f"extract={row.extractor_profile.name} "
                    f"chunk={row.chunk_profile.fingerprint[:8]} "
                    f"embed={row.embedding_profile.provider}:{row.embedding_profile.fingerprint[:8]}"
                )
    except typer.Exit:
        # typer.Exit subclasses RuntimeError, so the broad handler below
        # would otherwise swallow a deliberate exit and re-report it as
        # "failed: 1".
        raise
    except Exception as error:
        _report_db_error(error, "collection revisions")
        raise typer.Exit(1)


@app.command(
    short_help="Semantic search over indexed documents",
    epilog=_SEARCH_EPILOG,
    no_args_is_help=True,
)
def search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(
        10, "-n", "--top-k", "--limit", min=1, max=MAX_SEARCH_RESULTS, help="Number of results"
    ),
    collections: list[str] | None = typer.Option(
        None,
        "-c",
        "--collection",
        help="Filter collections; supports '-c work personal' or repeated '-c'.",
    ),
    trailing_collections: list[str] | None = typer.Argument(
        None,
        help="Additional collections after --collection/-c",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output results as JSONL (one JSON object per line)",
    ),
) -> None:
    """Search indexed documents."""
    filters = _build_collection_filters(collections, trailing_collections)
    if filters is not None:
        try:
            filters = [validate_collection_name(c) for c in filters]
        except ValueError as e:
            err_console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

    searcher = Searcher(_get_config())

    try:
        results = searcher.search(query, top_k=top_k, collections=filters)

        # Distinguish "no matches" from "that collection isn't indexed" — the two
        # are otherwise identical (empty output, exit 0), in both output modes.
        # Checked whenever collections were named, not only when nothing matched:
        # gating on an empty result set meant `-c work -c persnal` said nothing
        # about the typo as long as `work` returned a hit, so half the query was
        # dropped invisibly — and in --json mode the stream was well-formed and
        # the exit code 0, leaving a script no way to notice.
        unknown = searcher.unsearchable_collections(filters) if filters else []

        if json_output:
            for result in results:
                typer.echo(json.dumps(result))
            if unknown:
                # stdout is the JSONL stream; diagnostics go to stderr, and the
                # exit code has to distinguish this from a genuine no-match.
                err_console.print(
                    f"search failed: no indexed revision for {', '.join(unknown)} "
                    "(check `cementic collection list`)"
                )
                raise typer.Exit(1)
            return

        if not results:
            console.print("no results")
        else:
            rank_w = len(str(len(results)))
            for i, result in enumerate(results, 1):
                preview = " ".join(result["content"].split())
                console.print(
                    f"{i:>{rank_w}}. {result['score']:.3f}  {escape(result['source_path'])}"
                )
                # Keep the preview to a single line (truncate to the terminal width).
                console.print(f"   {escape(preview)}", no_wrap=True, overflow="ellipsis")

        if unknown:
            # Same condition, same exit code as the --json branch above: the two
            # modes used to disagree (0 here, 1 there) for identical input, so
            # whether a script could detect a typo'd collection depended on the
            # output format it happened to ask for.
            console.print(
                f"search failed: no indexed revision for {', '.join(unknown)} "
                "(check `cementic collection list`)"
            )
            raise typer.Exit(1)

    except typer.Exit:
        # typer.Exit subclasses RuntimeError, so the broad handler below would
        # otherwise swallow a deliberate exit and report it as "search failed: 1".
        raise
    except Exception as e:
        if json_output:
            # Diagnostics must not land on stdout, which is the JSONL stream.
            if _is_database_unavailable(e):
                err_console.print(f"search failed: database unavailable: {e}")
            elif _is_schema_missing(e):
                err_console.print(f"search failed: {_NO_SCHEMA_HINT}")
            else:
                err_console.print(f"search failed: {e}")
        else:
            _report_db_error(e, "search")
        raise typer.Exit(1)


_FILTER_EPILOG = """\
\b
EXAMPLES:
  cementic extract paper.pdf
  cementic extract paper.pdf | cementic chunk | cementic embed
"""


@app.command(
    short_help="Extract a document to Markdown on stdout",
    epilog=_FILTER_EPILOG,
    no_args_is_help=True,
)
def extract(path: str = typer.Argument(..., help="Path to a document file")) -> None:
    """Extract one document to Markdown on stdout — no database, for piping/debugging."""
    cfg = _get_config()
    try:
        markdown = extract_document(path, cfg)
    except (OSError, ValueError, RuntimeError) as error:
        # OSError rather than FileNotFoundError: a directory named `notes.md`
        # raises IsADirectoryError, an unreadable file PermissionError, and a
        # symlink loop ELOOP -- all ordinary inputs that produced a traceback.
        err_console.print(f"extract failed: {error}")
        raise typer.Exit(1)
    typer.echo(markdown)


@app.command(short_help="Chunk text from a file or stdin to JSONL on stdout")
def chunk(
    path: str | None = typer.Argument(None, help="Text file to chunk (default: stdin)"),
    chunk_size: int | None = typer.Option(
        None, "--chunk-size", min=1, help="Tokens per chunk (default: config value)"
    ),
    chunk_overlap: int | None = typer.Option(
        None,
        "--chunk-overlap",
        min=0,
        help="Overlap tokens between chunks (default: config value)",
    ),
) -> None:
    """Chunk text into JSONL on stdout (one object per line) — no database.

    Reads from PATH or stdin, so it pipes after `cementic extract`.
    """
    cfg = _get_config()
    try:
        text = Path(path).read_text(encoding="utf-8") if path else sys.stdin.read()
    except (OSError, ValueError) as error:
        # ValueError covers UnicodeDecodeError: `chunk` takes text, and a binary
        # file should be a one-line error, not a traceback.
        err_console.print(f"chunk failed: {error}")
        raise typer.Exit(1)
    size = chunk_size if chunk_size is not None else cfg.pipeline.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else cfg.pipeline.chunk_overlap
    try:
        pieces = chunk_text(text, chunk_size=size, chunk_overlap=overlap)
    except ValueError as error:
        # Reachable from a plausible invocation: passing only --chunk-size leaves
        # the overlap at its (larger) configured default, so say where each value
        # came from rather than printing a traceback.
        err_console.print(f"chunk failed: {error} (chunk_size={size}, chunk_overlap={overlap})")
        if chunk_overlap is None:
            err_console.print(
                "hint: --chunk-overlap defaults to the config value; pass it explicitly"
            )
        raise typer.Exit(1)
    for piece in pieces:
        typer.echo(json.dumps({"index": piece.chunk_index, "content": piece.content}))


@app.command(short_help="Embed chunk JSONL from stdin to JSONL on stdout")
def embed() -> None:
    """Embed chunk JSONL from stdin, adding an "embedding" field to each line.

    Reads the JSONL produced by `cementic chunk` (objects with a "content" field);
    needs the embedding model/runtime.
    """
    cfg = _get_config()
    try:
        records = [json.loads(line) for line in sys.stdin if line.strip()]
    except json.JSONDecodeError as error:
        err_console.print(f"embed failed: invalid JSONL on stdin: {error}")
        raise typer.Exit(1)
    if not records:
        return
    # Validate the whole input before embedding any of it, so a malformed line
    # is not reported only after some output has already been written.
    contents: list[str] = []
    for line_number, rec in enumerate(records, 1):
        if not isinstance(rec, dict):
            err_console.print(
                f"embed failed: line {line_number} is a JSON {type(rec).__name__}, "
                "not an object with a \"content\" field"
            )
            raise typer.Exit(1)
        content = rec.get("content")
        # str(rec.get("content", "")) used to turn a missing field into the
        # empty string and a null into the literal "None" -- both of which
        # embed happily into a plausible-looking vector for text that was never
        # there.
        if not isinstance(content, str):
            missing = "is missing" if "content" not in rec else f"is {json.dumps(content)}"
            err_console.print(
                f"embed failed: line {line_number} has no text to embed: \"content\" {missing}"
            )
            raise typer.Exit(1)
        if not content.strip():
            # The same hole as missing/null, one layer down: empty and
            # whitespace-only strings embed into a plausible-looking vector for
            # text that was never there. The pipeline worker never embeds such
            # chunks either.
            err_console.print(
                f"embed failed: line {line_number} has no text to embed: "
                "\"content\" is empty or whitespace"
            )
            raise typer.Exit(1)
        contents.append(content)

    batch_size = cfg.pipeline_worker.batch_size
    try:
        provider = create_provider(runtime_spec_from_config(cfg), cfg)
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            texts = [
                provider.format_document(content)
                for content in contents[start : start + batch_size]
            ]
            vectors = provider.embed_batch(texts)
            for offset, (rec, vector) in enumerate(zip(batch, vectors)):
                if vector is None:
                    # embed_batch reports per-item failure as None. Emitting
                    # "embedding": null would look like a successful record.
                    reason = getattr(provider, "over_budget_reason", lambda _text: None)(
                        texts[offset]
                    )
                    err_console.print(
                        f"embed failed: line {start + offset + 1} could not be embedded"
                        + (f": {reason}" if reason else "")
                    )
                    raise typer.Exit(1)
                typer.echo(json.dumps({**rec, "embedding": vector}))
    except typer.Exit:
        raise
    except Exception as error:
        err_console.print(f"embed failed: {error}")
        raise typer.Exit(1)


def main() -> None:
    """Entry point — Typer handles help (-h/--help), --version, and dispatch."""
    app()


if __name__ == "__main__":
    main()
