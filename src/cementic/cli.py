"""CLI interface for cementic using Typer."""

import inspect
import json
import os
import sys
import time
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, TypeVar, cast

import click
import typer
from rich.console import Console
from rich.markup import escape
from sqlalchemy.exc import InterfaceError, OperationalError
from typer.core import TyperCommand, TyperGroup

from cementic.bootstrap import Bootstrapper
from cementic.chunk import chunk_text
from cementic.collections import (
    delete_collection_records,
    drop_orphan_vector_tables,
    list_collection_revisions,
    list_collections,
    promote_ready_revision,
    remove_artifacts,
)
from cementic.config import (
    Config,
    default_config_path,
    get_config,
    resolve_config_path,
)
from cementic.db import get_engine, get_session_factory
from cementic.embedding_runtime import (
    create_provider,
    get_llama_cpp_runtime_client,
    llama_daemon_status,
    runtime_spec_from_config,
    stop_llama_cpp_runtime,
)
from cementic.extract import extract_document
from cementic.search import Searcher
from cementic.status_service import (
    build_supervisor_status,
    check_health,
    load_file_progress,
    load_pipeline_status,
    load_worker_statuses,
)
from cementic.supervisor import (
    ManagedProcess,
    force_kill,
    is_managed_process_alive,
    is_pid_running,
    load_supervisor_state,
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
app.add_typer(collection_app, name="collection")
app.add_typer(embedding_app, name="embedding")
app.add_typer(config_app, name="config")
console = Console()
# Diagnostics for the stdin/stdout filter commands go here so a failure never
# pollutes the data on stdout (which would otherwise be piped on as content).
err_console = Console(stderr=True)


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
    """Lazy-load the config singleton."""
    global _config
    if _config is None:
        _config = get_config()
    return _config


_DEFAULT_CONFIG_TOML = """\
# cementic configuration
#
# Precedence (low -> high): built-in defaults < this file < CEMENTIC_* env vars
# < command-line flags. Every value below is optional; delete what you don't need.

[database]
# Postgres with the pgvector extension. Bring one up with the bundled compose
# file: `docker compose up -d`  (or `podman compose up -d`).
host = "localhost"
port = 5432
name = "cementic"
user = "cementic"
# password = "cementic"   # override outside local development

[pipeline]
embedding_provider = "llama-cpp"
chunk_size = 512
chunk_overlap = 128

[index]
# ANN index: "hnsw" (lower latency, more RAM) or "diskann" (disk-resident, low RAM)
method = "hnsw"

[llama_cpp]
model_path = "./models/nomic-embed-text-v2-moe.Q8_0.gguf"

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
    import json

    typer.echo(json.dumps(_get_config().model_dump(mode="json"), indent=2, sort_keys=True))


def _load_supervisor_state() -> dict[str, object]:
    return load_supervisor_state(_get_supervisor_state_path())


def _save_supervisor_state(state: dict[str, object]) -> None:
    save_supervisor_state(_get_supervisor_state_path(), state)


def _supervisor_processes(state: dict[str, object]) -> list[dict[str, object]]:
    processes = state.get("processes", [])
    if not isinstance(processes, list):
        return []
    return [proc for proc in processes if isinstance(proc, dict)]


def _process_pid(process: dict[str, object]) -> int:
    pid = process.get("pid", 0)
    return pid if isinstance(pid, int) else 0


def _process_start_token(process: dict[str, object]) -> str | None:
    token = process.get("start_token")
    return token if isinstance(token, str) else None


def _is_managed_proc_alive(process: dict[str, object]) -> bool:
    return is_managed_process_alive(_process_pid(process), _process_start_token(process))


def _spawn_detached(command: list[str], log_file: Path) -> int:
    return spawn_detached(command, log_file)


def _wait_for_exit(pids: list[int], timeout_seconds: float = 20.0) -> list[int]:
    return wait_for_exit(pids, timeout_seconds=timeout_seconds)


def _force_kill(pids: list[int]) -> list[int]:
    return force_kill(pids)


def _is_pid_running(pid: int) -> bool:
    return is_pid_running(pid)


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


_DB_HINT = "hint: start Postgres with `docker compose up -d` (or `podman compose up -d`)"


def _state(ok: bool, ok_word: str, bad_word: str) -> str:
    """A status word, colored sparingly (rich drops color off-TTY / NO_COLOR)."""
    return f"[green]{ok_word}[/green]" if ok else f"[red]{bad_word}[/red]"


def _print_database_unavailable(action: str) -> None:
    """Print a concise database-unavailable message with a recovery hint."""
    console.print(f"{action}: database not reachable")
    console.print(_DB_HINT)


def _llama_daemon_runtime_status() -> str:
    """Return llama.cpp daemon runtime status."""
    return llama_daemon_status(_get_config())


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
        console.print(
            f"{'embedding':<11} {_state(health.embedding_healthy, 'healthy', 'unhealthy')}"
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
    console.print(
        f"pipeline worker: {pipeline_worker_status.process}, "
        f"state={pipeline_worker_status.state}, pid={pipeline_worker_status.pid}"
    )
    if pipeline_worker_status.current_file != "None":
        console.print(f"  current file: {pipeline_worker_status.current_file}")
    if _get_config().pipeline.embedding_provider == "llama-cpp":
        console.print(f"search daemon: {_llama_daemon_runtime_status()}")
    if health is not None and health.llama_daemon != "N/A":
        console.print(f"llama daemon: {health.llama_daemon}")


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
    console.print(
        f"{'embedded':<11} {ps.done_embeddings:,}/{ps.total_chunks:,} ({ps.embedding_pct}%)"
    )
    console.print(
        f"{'revision':<11} active={ps.active_revision_label or '-'}  "
        f"building={ps.building_revision_label or '-'}"
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
) -> None:
    """Print full status as JSON."""
    import json as _json

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
        },
        "pipeline_worker": {
            "process": pipeline_worker_status.process,
            "state": pipeline_worker_status.state,
            "pid": pipeline_worker_status.pid,
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
                collections_data: dict[str, dict[str, Any]] = {}
                for row in rows:
                    ps = load_pipeline_status(_get_config(), row.name)
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
                        "building_revision_label": ps.building_revision_label,
                    }
                output["collections"] = collections_data
            else:
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
    except Exception as error:
        output["files_error"] = str(error)

    console.print(_json.dumps(output, indent=2, default=str))


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
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    missing = [d for d in directories if not Path(d).is_dir()]
    if missing:
        for d in missing:
            console.print(f"[red]Error: Directory does not exist: {d}[/red]")
        raise typer.Exit(1)

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
        console.print(f"[red]Bootstrap failed before background start: {error}[/red]")
        raise typer.Exit(1)

    base_cmd = [sys.executable, "-m", "cementic.runner"]
    data_dir = _get_data_dir()
    source_watcher_log = data_dir / "source-watcher-background.log"
    pipeline_log = data_dir / "pipeline-background.log"

    source_watcher_pid = _spawn_detached(
        [*base_cmd, "source-watcher", *directories, "--collection", collection],
        source_watcher_log,
    )
    pipeline_pid = _spawn_detached(
        [
            *base_cmd,
            "pipeline-worker",
            "--collection",
            collection,
        ],
        pipeline_log,
    )

    _save_supervisor_state(
        {
            "collection": collection,
            "directories": directories,
            "processes": [
                ManagedProcess(
                    "source-watcher",
                    source_watcher_pid,
                    str(source_watcher_log),
                    process_start_token(source_watcher_pid),
                ).__dict__,
                ManagedProcess(
                    "pipeline-worker",
                    pipeline_pid,
                    str(pipeline_log),
                    process_start_token(pipeline_pid),
                ).__dict__,
            ],
        }
    )

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
) -> None:
    """Show background worker status and collection progress."""
    if collection is not None:
        try:
            collection = validate_collection_name(collection)
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
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
        _print_status_json(
            supervisor_status,
            source_watcher_status,
            pipeline_worker_status,
            directories,
            health,
            collection,
            verbose,
        )
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
        console.print(_DB_HINT)
        return

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
                items = [(row, load_pipeline_status(_get_config(), row.name)) for row in rows]
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

        console.print()
        pipeline_status = load_pipeline_status(_get_config(), collection)
        _print_collection_detail(collection, pipeline_status, verbose)
    except Exception as error:
        if _is_database_unavailable(error):
            _print_database_unavailable("status")
        else:
            console.print(f"status failed: {error}")


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
    """Stop cementic's background processes (watcher, worker, embedding server).

    Postgres is not managed by cementic; stop it with your container engine
    (e.g. `docker compose down` or `podman compose down`).
    """
    state = _load_supervisor_state()
    processes = _supervisor_processes(state)

    if not processes:
        console.print("no background cementic processes found")
        return

    signaled_pids: list[int] = []
    for proc in processes:
        pid = _process_pid(proc)
        # Only signal a process we can confirm is still ours; a recycled PID
        # (token mismatch) belongs to someone else and must not be killed.
        if not _is_managed_proc_alive(proc):
            continue
        try:
            os.kill(pid, 15)
            signaled_pids.append(pid)
        except (ProcessLookupError, OSError):
            continue

    timeout_seconds = 10.0  # grace period before --force is required
    remaining = _wait_for_exit(signaled_pids, timeout_seconds=timeout_seconds)

    if not remaining:
        if _get_supervisor_state_path().exists():
            _get_supervisor_state_path().unlink()
        console.print(f"stopped {len(signaled_pids)} process(es)")
        return

    if force:
        killed = _force_kill(remaining)
        time.sleep(0.5)
        still_alive = [pid for pid in killed if _is_pid_running(pid)]
        if _get_supervisor_state_path().exists():
            _get_supervisor_state_path().unlink()
        if still_alive:
            console.print(
                f"force killed {len(remaining) - len(still_alive)} process(es); "
                f"could not kill: {', '.join(str(pid) for pid in still_alive)}"
            )
        else:
            console.print(f"force stopped {len(remaining)} process(es)")
        return

    still_running = [proc for proc in processes if _process_pid(proc) in remaining]
    _save_supervisor_state(
        {
            "collection": state.get("collection"),
            "directories": state.get("directories", []),
            "processes": still_running,
        }
    )

    console.print(
        f"stop timed out after {timeout_seconds}s; "
        f"still running PID(s): {', '.join(str(pid) for pid in remaining)}"
    )
    console.print("use --force to kill stubborn processes")


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
        client = get_llama_cpp_runtime_client(config=config, autostart=True)
    except Exception as error:
        console.print(f"embedding start failed: {error}")
        raise typer.Exit(1)
    console.print("embedding: running")
    console.print(f"provider: llama-cpp, dim={client.embedding_dim}")


@embedding_app.command("stop", short_help="Stop embedding runtime")
def stop_embedding_runtime() -> None:
    """Stop the configured embedding runtime service."""
    config = _get_config()
    stopped = stop_llama_cpp_runtime(config)
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
        console.print(f"[red]Error: {e}[/red]")
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

        remove_artifacts(result.artifact_paths, config=_get_config())
        drop_orphan_vector_tables(engine, result.vector_profile_ids)

        console.print(f"collection: {collection}")
        console.print("status: deleted")
        console.print(f"documents: {result.deleted_docs}")
        console.print(f"chunks: {result.deleted_chunks}")
        console.print(f"vector_tables_dropped: {len(result.vector_profile_ids)}")
    except Exception as e:
        if _is_database_unavailable(e):
            _print_database_unavailable("collection remove")
        else:
            console.print(f"failed to remove collection '{collection}': {e}")
        raise typer.Exit(1)


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
                f"building={row.building_revision_label or '-'}"
            )
    except Exception as error:
        if _is_database_unavailable(error):
            _print_database_unavailable("collection list")
        else:
            console.print(f"failed to list collections: {error}")
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
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    try:
        engine = get_engine(_get_config().database.url)
        session_factory = get_session_factory(engine)
        with session_factory() as session:
            outcome = promote_ready_revision(
                session, collection, config=_get_config(), force=force
            )
            # Read everything we need while the session is open; ORM attributes
            # expire on commit and would raise once the session closes.
            status = outcome.status
            failures = outcome.failures
            revision_label = (
                outcome.revision.label or outcome.revision.id
                if outcome.revision is not None
                else None
            )
    except Exception as error:
        if _is_database_unavailable(error):
            _print_database_unavailable("collection promote")
        else:
            console.print(f"promotion failed: {error}")
        raise typer.Exit(1)

    console.print(f"collection: {collection}")
    if status == "no_ready":
        console.print("status: no ready revision")
        return
    if status == "blocked_by_failures":
        parts = []
        if failures is not None:
            if failures.extracted_failed:
                parts.append(f"extract={failures.extracted_failed}")
            if failures.chunked_failed:
                parts.append(f"chunk={failures.chunked_failed}")
            if failures.failed_embeddings:
                parts.append(f"embed={failures.failed_embeddings}")
        console.print(f"status: blocked ({', '.join(parts)} failed)")
        console.print("re-run with --force to promote anyway")
        raise typer.Exit(1)
    console.print("status: promoted")
    console.print(f"revision: {revision_label}")


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
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    try:
        engine = get_engine(_get_config().database.url)
        session_factory = get_session_factory(engine)
        with session_factory() as session:
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
    except Exception as error:
        if _is_database_unavailable(error):
            _print_database_unavailable("collection revisions")
        else:
            console.print(f"failed to load revisions: {error}")
        raise typer.Exit(1)


@app.command(
    short_help="Semantic search over indexed documents",
    epilog=_SEARCH_EPILOG,
    no_args_is_help=True,
)
def search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(10, "-n", min=1, max=50, help="Number of results"),
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
) -> None:
    """Search indexed documents."""
    filters = _build_collection_filters(collections, trailing_collections)
    if filters is not None:
        try:
            filters = [validate_collection_name(c) for c in filters]
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

    searcher = Searcher(_get_config())

    try:
        results = searcher.search(query, top_k=top_k, collections=filters)

        if not results:
            console.print("no results")
            return

        rank_w = len(str(len(results)))
        for i, result in enumerate(results, 1):
            preview = " ".join(result["content"].split())
            console.print(
                f"{i:>{rank_w}}. {result['score']:.3f}  {escape(result['source_path'])}"
            )
            # Keep the preview to a single line (truncate to the terminal width).
            console.print(f"   {escape(preview)}", no_wrap=True, overflow="ellipsis")

    except Exception as e:
        if _is_database_unavailable(e):
            _print_database_unavailable("search")
        else:
            console.print(f"search failed: {e}")
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
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        err_console.print(f"extract failed: {error}")
        raise typer.Exit(1)
    typer.echo(markdown)


@app.command(short_help="Chunk text from a file or stdin to JSONL on stdout")
def chunk(
    path: str | None = typer.Argument(None, help="Text file to chunk (default: stdin)"),
) -> None:
    """Chunk text into JSONL on stdout (one object per line) — no database.

    Reads from PATH or stdin, so it pipes after `cementic extract`.
    """
    cfg = _get_config()
    try:
        text = Path(path).read_text(encoding="utf-8") if path else sys.stdin.read()
    except (FileNotFoundError, OSError) as error:
        err_console.print(f"chunk failed: {error}")
        raise typer.Exit(1)
    for piece in chunk_text(
        text,
        chunk_size=cfg.pipeline.chunk_size,
        chunk_overlap=cfg.pipeline.chunk_overlap,
    ):
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
    try:
        provider = create_provider(runtime_spec_from_config(cfg), cfg)
        texts = [provider.format_document(str(rec.get("content", ""))) for rec in records]
        vectors = provider.embed_batch(texts)
    except Exception as error:
        err_console.print(f"embed failed: {error}")
        raise typer.Exit(1)
    for rec, vector in zip(records, vectors):
        typer.echo(json.dumps({**rec, "embedding": vector}))


def main() -> None:
    """Entry point — Typer handles help (-h/--help), --version, and dispatch."""
    app()


if __name__ == "__main__":
    main()
