"""CLI interface for cementic using Typer."""

import os
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, List, Optional, cast

import typer
from rich.console import Console
from sqlalchemy.exc import InterfaceError, OperationalError

from cementic.bootstrap import Bootstrapper
from cementic.collections import (
    delete_collection_records,
    list_collection_revisions,
    list_collections,
    promote_ready_revision,
    remove_artifacts,
)
from cementic.config import Config, get_config
from cementic.db import get_engine, get_session_factory
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
    is_pid_running,
    load_supervisor_state,
    save_supervisor_state,
    spawn_detached,
    wait_for_exit,
)


class CementicTyper(typer.Typer):
    """Typer app with plain help formatting."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("rich_markup_mode", None)
        kwargs.setdefault("add_completion", False)
        kwargs.setdefault("no_args_is_help", True)
        context_settings = cast(dict[str, Any], dict(kwargs.get("context_settings") or {}))
        context_settings.setdefault("help_option_names", [])
        kwargs["context_settings"] = context_settings
        super().__init__(*args, **kwargs)


app = CementicTyper(help="Index and semantically search PDF collections")
collection_app = CementicTyper(help="Inspect and manage collections")
app.add_typer(collection_app, name="collection")
console = Console()

_config: Config | None = None


def _get_config() -> Config:
    """Lazy-load the config singleton."""
    global _config
    if _config is None:
        _config = get_config()
    return _config


ROOT_HELP_TEXT = """Index and semantically search PDF collections.

Usage: cementic <COMMAND> [ARGS]...

Arguments:
  <COMMAND>
          Command to run.
  [ARGS]...
          Arguments for the selected command.

Commands:
  start <DIR1> [<DIR2> ...] [-c, --collection <COLLECTION>]
          Start background indexing for one or more directories.
          -c, --collection <COLLECTION>
                  Collection name to index into. [default: default]

  status [-c, --collection <COLLECTION>]
          Show background worker status and either a global collection overview or one collection.

  stop [--include-infra|--no-infra] [--force]
          Stop background workers and, optionally, infrastructure containers.
          --include-infra / --no-infra
                  Also stop Postgres/Ollama containers. [default: include-infra]
          --force
                  Kill workers if graceful stop times out.

  collection <COMMAND> [ARGS]...
          Inspect and manage collections.
          list
                  Show known collections.
          promote <COLLECTION>
                  Promote the ready pipeline revision for one collection.
          revisions <COLLECTION>
                  Show revision history for one collection.
          remove <COLLECTION> [--force]
                  Remove one collection and its stored artifacts.

  search <QUERY> [-n <N>] [-c, --collection <COLLECTION1> [<COLLECTION2> ...]]
          Run semantic search over indexed chunks.
          -n <N>
                  Number of results to return. [default: 10]
          -c, --collection <COLLECTION>
                  Filter by one or more collections.
                  You can use one flag with multiple values: -c work personal.
                  You can also repeat the flag: -c work -c personal.

Examples:
  cementic start /path/to/dir1 /path/to/dir2 -c research
  cementic collection list
  cementic collection promote research
  cementic search "your query" -n 5 -c work personal

Options:
  -h, --help
          Print help

  -V, --version
          Print version
"""


COLLECTION_HELP_TEXT = """Inspect and manage collections.

Usage: cementic collection <COMMAND> [ARGS]...

Commands:
  list
          Show known collections.

  promote <COLLECTION>
          Promote the ready pipeline revision for one collection.

  revisions <COLLECTION>
          Show revision history for one collection.

  remove <COLLECTION> [--force]
          Remove one collection and its stored artifacts.

Options:
  -h, --help
          Print help
"""


START_HELP_TEXT = """Start background indexing.

Usage: cementic start <DIR1> [<DIR2> ...] [-c, --collection <COLLECTION>]

Arguments:
  <DIR1> [<DIR2> ...]
          One or more directories to watch for PDFs.

Options:
  -c, --collection <COLLECTION>
          Collection name to index into. [default: default]

Examples:
  cementic start ~/research-papers
  cementic start ~/research-papers --collection papers
"""


SEARCH_HELP_TEXT = """Run semantic search over indexed chunks.

Usage: cementic search <QUERY> [-n <N>] [-c, --collection <COLLECTION> ...]

Arguments:
  <QUERY>
          Natural-language search query.

Options:
  -n <N>
          Number of results to return. [default: 10]
  -c, --collection <COLLECTION>
          Filter by one or more collections.

Examples:
  cementic search "transformer inference"
  cementic search "transformer inference" --collection papers
"""


COLLECTION_PROMOTE_HELP_TEXT = """Promote a ready revision for one collection.

Usage: cementic collection promote <COLLECTION>

Arguments:
  <COLLECTION>
          Collection name to promote.
"""


COLLECTION_REVISIONS_HELP_TEXT = """Show revision history for one collection.

Usage: cementic collection revisions <COLLECTION>

Arguments:
  <COLLECTION>
          Collection name to inspect.
"""


COLLECTION_REMOVE_HELP_TEXT = """Remove one collection and its stored artifacts.

Usage: cementic collection remove <COLLECTION> [--force]

Arguments:
  <COLLECTION>
          Collection name to remove.

Options:
  --force
          Skip confirmation prompt.
"""


STATUS_HELP_TEXT = """Show background worker status and collection progress.

Usage: cementic status [-c, --collection <COLLECTION>]

Options:
  -c, --collection <COLLECTION>
          Show detailed status for one collection instead of a global overview.
"""


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


def _spawn_detached(command: List[str], log_file: Path) -> int:
    return spawn_detached(command, log_file)


def _wait_for_exit(pids: List[int], timeout_seconds: float = 20.0) -> List[int]:
    return wait_for_exit(pids, timeout_seconds=timeout_seconds)


def _force_kill(pids: List[int]) -> List[int]:
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
    option_collections: Optional[List[str]],
    trailing_collections: Optional[List[str]],
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


def _print_database_unavailable(action: str) -> None:
    """Print a helpful database-unavailable message."""
    console.print(f"{action} failed: database not reachable")
    console.print("hint: run `cementic start` to start infrastructure")


def _llama_daemon_runtime_status() -> str:
    """Return llama.cpp daemon runtime status."""
    pid_file = _get_config().llama_cpp.daemon_pid_file
    if pid_file is None or not pid_file.exists():
        return "stopped"

    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except ValueError:
        return "stopped"

    if not _is_pid_running(pid):
        return "stopped"

    return f"running, pid={pid}"


def _print_runtime_status(
    supervisor_collection: str,
    directories: list[str],
    source_watcher_status: Any,
    pipeline_worker_status: Any,
) -> None:
    """Print background runtime status shared by global and collection views."""
    console.print(f"session collection: {supervisor_collection}")
    source_running = source_watcher_status.process == "running"
    pipeline_running = pipeline_worker_status.process == "running"
    running_count = int(source_running) + int(pipeline_running)
    console.print(f"workers: {running_count}/2 running")

    if directories:
        console.print("directories:")
        for directory in directories:
            console.print(f"- {directory}")

    console.print("source watcher:")
    console.print(
        f"- {source_watcher_status.process}, state={source_watcher_status.state}, "
        f"pid={source_watcher_status.pid}"
    )
    console.print(
        f"- processed={source_watcher_status.processed_count}, "
        f"failed={source_watcher_status.failed_count}"
    )
    if source_watcher_status.current_file != "None":
        console.print(f"- current file: {source_watcher_status.current_file}")

    console.print("pipeline worker:")
    console.print(
        f"- {pipeline_worker_status.process}, state={pipeline_worker_status.state}, "
        f"pid={pipeline_worker_status.pid}"
    )
    if pipeline_worker_status.current_file != "None":
        console.print(f"- current file: {pipeline_worker_status.current_file}")

    if _get_config().pipeline.embedding_provider == "llama-cpp":
        console.print(f"search daemon: {_llama_daemon_runtime_status()}")


def _print_health_section(health: Any) -> None:
    """Print health check status."""
    if health is None:
        return
    console.print("health:")
    db_state = "[green]reachable[/green]" if health.db_reachable else "[red]unreachable[/red]"
    console.print(f"- database: {db_state}")
    embed_state = (
        "[green]healthy[/green]" if health.embedding_healthy else "[red]unhealthy[/red]"
    )
    console.print(f"- embedding ({health.embedding_provider}): {embed_state}")
    if health.llama_daemon != "N/A":
        console.print(f"- llama daemon: {health.llama_daemon}")


def _print_collection_detail(
    collection: str,
    pipeline_status: Any,
    verbose: bool,
) -> None:
    """Print detailed pipeline status for one collection."""
    console.print(f"collection: {collection}")
    console.print("pipeline:")
    console.print(f"- documents: {pipeline_status.documents}")
    console.print(
        f"- extraction: {pipeline_status.extracted_done}/{pipeline_status.documents}"
        f" ({pipeline_status.extraction_pct}%)"
    )
    if pipeline_status.extracted_failed:
        console.print(f"  [!] failed: {pipeline_status.extracted_failed}")
    console.print(
        f"- chunking: {pipeline_status.chunked_done}/{pipeline_status.extracted_done}"
        f" ({pipeline_status.chunking_pct}%)"
    )
    if pipeline_status.chunked_failed:
        console.print(f"  [!] failed: {pipeline_status.chunked_failed}")
    console.print(
        f"- embeddings: done={pipeline_status.done_embeddings}, "
        f"processing={pipeline_status.processing_embeddings}, "
        f"pending={pipeline_status.pending_embeddings}, "
        f"failed={pipeline_status.failed_embeddings} "
        f"(total chunks: {pipeline_status.total_chunks}, "
        f"{pipeline_status.embedding_pct}% complete)"
    )
    console.print(
        f"- revisions: active={pipeline_status.active_revision_label}, "
        f"building={pipeline_status.building_revision_label}"
    )

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
    except Exception:
        pass

    console.print(_json.dumps(output, indent=2, default=str))


@app.command(
    "start",
    short_help="DIRECTORY... [-c|--collection COLLECTION]",
)
def start_background(
    directories: List[str] = typer.Argument(..., help="Directories to watch for PDFs"),
    collection: str = typer.Option(
        "default",
        "-c",
        "--collection",
        help="Collection name for indexed documents",
    ),
) -> None:
    """Start source watcher and pipeline worker in the background."""
    missing = [d for d in directories if not Path(d).is_dir()]
    if missing:
        for d in missing:
            console.print(f"[red]Error: Directory does not exist: {d}[/red]")
        raise typer.Exit(1)

    state = _load_supervisor_state()
    running = [proc for proc in _supervisor_processes(state) if _is_pid_running(_process_pid(proc))]

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
                    "source-watcher", source_watcher_pid, str(source_watcher_log)
                ).__dict__,
                ManagedProcess("pipeline-worker", pipeline_pid, str(pipeline_log)).__dict__,
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
    short_help="[--collection COLLECTION] [--verbose] [--json]",
)
def status(
    collection: Optional[str] = typer.Option(
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

    _print_runtime_status(
        supervisor_status.collection,
        directories,
        source_watcher_status,
        pipeline_worker_status,
    )
    _print_health_section(health)

    try:
        engine = get_engine(_get_config().database.url)
        session_factory = get_session_factory(engine)
        with session_factory() as session:
            if collection is None:
                console.print("collections:")
                rows = list_collections(session)
                if not rows:
                    console.print("- none")
                    return
                for row in rows:
                    pipeline_status = load_pipeline_status(_get_config(), row.name)
                    console.print(
                        f"- name={row.name}, documents={pipeline_status.documents}, "
                        f"extraction={pipeline_status.extracted_done}/{pipeline_status.documents}"
                        f" ({pipeline_status.extraction_pct}%)"
                    )
                    if pipeline_status.extracted_failed:
                        console.print(
                            f"  [!] extraction failures: {pipeline_status.extracted_failed}"
                        )
                    console.print(
                        f"  chunks={pipeline_status.chunked_done}/{pipeline_status.extracted_done}"
                        f" ({pipeline_status.chunking_pct}%)"
                    )
                    if pipeline_status.chunked_failed:
                        console.print(f"  [!] chunking failures: {pipeline_status.chunked_failed}")
                    console.print(
                        f"  embeddings={pipeline_status.done_embeddings}/"
                        f"{pipeline_status.total_chunks}"
                        f" ({pipeline_status.embedding_pct}%)"
                    )
                    if pipeline_status.failed_embeddings:
                        console.print(
                            f"  [!] embedding failures: {pipeline_status.failed_embeddings}"
                        )
                    console.print(
                        f"  active={row.active_revision_label or '-'}, "
                        f"building={row.building_revision_label or '-'}"
                    )
                return

        pipeline_status = load_pipeline_status(_get_config(), collection)
        _print_collection_detail(collection, pipeline_status, verbose)
    except Exception as error:
        if _is_database_unavailable(error):
            _print_database_unavailable("status")
        else:
            console.print(f"status failed: {error}")


@app.command(
    "stop",
    short_help="[--include-infra|--no-infra] [--force]",
)
def stop_background(
    include_infra: bool = typer.Option(
        True,
        "--include-infra/--no-infra",
        help="Also stop infrastructure containers (Postgres/Ollama)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Force kill processes that don't stop gracefully",
    ),
) -> None:
    """Stop background processes and optionally infrastructure containers."""
    state = _load_supervisor_state()
    processes = _supervisor_processes(state)

    if not processes:
        console.print("no background cementic processes found")
        if include_infra:
            _stop_infra_containers()
        return

    signaled_pids: List[int] = []
    for proc in processes:
        pid = _process_pid(proc)
        if pid <= 0:
            continue
        try:
            os.kill(pid, 15)
            signaled_pids.append(pid)
        except (ProcessLookupError, OSError):
            continue

    timeout_seconds = float(_get_config().bootstrap.wait_timeout_seconds)
    remaining = _wait_for_exit(signaled_pids, timeout_seconds=timeout_seconds)

    if not remaining:
        if _get_supervisor_state_path().exists():
            _get_supervisor_state_path().unlink()
        console.print(f"stopped {len(signaled_pids)} process(es)")
        if include_infra:
            _stop_infra_containers()
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
        if include_infra:
            _stop_infra_containers()
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


def _stop_infra_containers() -> None:
    """Stop infrastructure containers helper."""
    from cementic.bootstrap import Bootstrapper

    bootstrapper = Bootstrapper(_get_config())
    try:
        include_ollama = _get_config().pipeline.embedding_provider == "ollama"
        bootstrapper.stop_containers(include_ollama=include_ollama)
        console.print("infrastructure containers stopped")
    except RuntimeError as e:
        console.print(f"could not stop containers: {e}")


@collection_app.command(
    "remove",
    short_help="COLLECTION [--force]",
)
def remove_collection(
    collection: str = typer.Argument(..., help="Collection name to delete"),
    force: bool = typer.Option(False, "--force", help="Skip confirmation prompt"),
) -> None:
    """Delete all documents and chunks belonging to a collection."""
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

        remove_artifacts(result.artifact_paths)

        console.print(f"collection: {collection}")
        console.print("status: deleted")
        console.print(f"documents: {result.deleted_docs}")
        console.print(f"chunks: {result.deleted_chunks}")
    except Exception as e:
        if _is_database_unavailable(e):
            _print_database_unavailable("collection remove")
        else:
            console.print(f"failed to remove collection '{collection}': {e}")
        raise typer.Exit(1)


@collection_app.command("list", short_help="(no flags)")
def list_collection_command() -> None:
    """Show known collections."""
    try:
        engine = get_engine(_get_config().database.url)
        session_factory = get_session_factory(engine)
        with session_factory() as session:
            rows = list_collections(session)

        if not rows:
            console.print("collections: none")
            return

        console.print("collections:")
        for row in rows:
            console.print(
                f"- name={row.name}, documents={row.documents}, "
                f"active={row.active_revision_label or '-'}, "
                f"building={row.building_revision_label or '-'}"
            )
    except Exception as error:
        if _is_database_unavailable(error):
            _print_database_unavailable("collection list")
        else:
            console.print(f"failed to list collections: {error}")
        raise typer.Exit(1)


@collection_app.command("promote", short_help="COLLECTION")
def promote_collection(
    collection: str = typer.Argument(..., help="Collection name to promote"),
) -> None:
    """Promote the ready pipeline revision for one collection."""
    try:
        engine = get_engine(_get_config().database.url)
        session_factory = get_session_factory(engine)
        with session_factory() as session:
            revision = promote_ready_revision(session, collection)
            if revision is None:
                console.print(f"collection: {collection}")
                console.print("status: no ready revision")
                return
            console.print(f"collection: {collection}")
            console.print("status: promoted")
            console.print(f"revision: {revision.label or revision.id}")
    except Exception as error:
        if _is_database_unavailable(error):
            _print_database_unavailable("collection promote")
        else:
            console.print(f"promotion failed: {error}")
        raise typer.Exit(1)


@collection_app.command("revisions", short_help="COLLECTION")
def list_collection_revision_command(
    collection: str = typer.Argument(..., help="Collection name to inspect"),
) -> None:
    """Show revision history for one collection."""
    try:
        engine = get_engine(_get_config().database.url)
        session_factory = get_session_factory(engine)
        with session_factory() as session:
            rows = list_collection_revisions(session, collection)
            if not rows:
                console.print(f"collection: {collection}")
                console.print("revisions: none")
                return

            console.print(f"collection: {collection}")
            console.print("revisions:")
            for row in rows:
                console.print(
                    f"- id={row.id}, status={row.status}, label={row.label or '-'}, "
                    f"extractor={row.extractor_profile.name}, "
                    f"chunk={row.chunk_profile.fingerprint[:8]}, "
                    f"embedding={row.embedding_profile.provider}:{row.embedding_profile.fingerprint[:8]}"
                )
    except Exception as error:
        if _is_database_unavailable(error):
            _print_database_unavailable("collection revisions")
        else:
            console.print(f"failed to load revisions: {error}")
        raise typer.Exit(1)


@app.command(
    short_help="QUERY [-n N] [-c|--collection COLLECTION ...]",
)
def search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(10, "-n", help="Number of results"),
    collections: Optional[List[str]] = typer.Option(
        None,
        "-c",
        "--collection",
        help="Filter collections; supports '-c work personal' or repeated '-c'.",
    ),
    trailing_collections: Optional[List[str]] = typer.Argument(
        None,
        help="Additional collections after --collection/-c",
    ),
) -> None:
    """Search indexed documents."""
    filters = _build_collection_filters(collections, trailing_collections)
    searcher = Searcher(_get_config())

    try:
        results = searcher.search(query, top_k=top_k, collections=filters)

        if not results:
            console.print("results: none")
            return

        console.print(f"query: {query}")
        console.print("results:")
        for i, result in enumerate(results, 1):
            preview = (
                result["content"][:100] + "..."
                if len(result["content"]) > 100
                else result["content"]
            )
            console.print(
                f"- rank={i}, score={result['score']:.3f}, source={result['source_path']}"
            )
            console.print(f"  {preview.replace(chr(10), ' ')}")

    except Exception as e:
        if _is_database_unavailable(e):
            _print_database_unavailable("search")
        else:
            console.print(f"search failed: {e}")
        raise typer.Exit(1)


def main() -> None:
    """Entry point."""
    argv = sys.argv[1:]
    if not argv or (len(argv) == 1 and argv[0] in {"-h", "--help"}):
        typer.echo(ROOT_HELP_TEXT.rstrip())
        return

    if len(argv) == 1 and argv[0] == "collection":
        typer.echo(COLLECTION_HELP_TEXT.rstrip())
        return

    if len(argv) == 1 and argv[0] == "start":
        typer.echo(START_HELP_TEXT.rstrip())
        return

    if len(argv) == 1 and argv[0] == "search":
        typer.echo(SEARCH_HELP_TEXT.rstrip())
        return

    if len(argv) == 2 and argv[:2] == ["collection", "promote"]:
        typer.echo(COLLECTION_PROMOTE_HELP_TEXT.rstrip())
        return

    if len(argv) == 2 and argv[:2] == ["collection", "revisions"]:
        typer.echo(COLLECTION_REVISIONS_HELP_TEXT.rstrip())
        return

    if len(argv) == 2 and argv[:2] == ["collection", "remove"]:
        typer.echo(COLLECTION_REMOVE_HELP_TEXT.rstrip())
        return

    if len(argv) == 1 and argv[0] in {"-V", "--version"}:
        typer.echo(_get_cli_version())
        return

    app()


if __name__ == "__main__":
    main()
