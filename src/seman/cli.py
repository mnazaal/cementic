"""CLI interface for seman using Typer."""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from seman.config import get_config
from seman.db import Chunk, Document, get_engine, get_session_factory
from seman.search import Searcher
from seman.state import StateManager


class SemanTyper(typer.Typer):
    """Typer app with plain help formatting."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("rich_markup_mode", None)
        kwargs.setdefault("add_completion", False)
        context_settings = dict(kwargs.get("context_settings") or {})
        context_settings.setdefault("help_option_names", [])
        kwargs["context_settings"] = context_settings
        super().__init__(*args, **kwargs)


app = SemanTyper(help="Semantic search CLI tool")
console = Console()

config = get_config()


ROOT_HELP_TEXT = """Semantic search CLI tool for indexing and searching PDF documents.

Usage: seman <COMMAND> [ARGS]...

Arguments:
  <COMMAND>
          Command to run.
  [ARGS]...
          Arguments for the selected command.

Commands:
  start <DIR1> [<DIR2> ...] [-c, --collection <COLLECTION>]
          Watch one or more directories and start converter + indexer in background.
          -c, --collection <COLLECTION>
                  Collection name to index into. [default: default]

  status
          Show converter, indexer, supervisor, and embedding queue status.

  stop [--include-infra|--no-infra] [--force]
          Stop background workers and optionally infrastructure containers.
          --include-infra / --no-infra
                  Also stop Postgres/Ollama containers. [default: include-infra]
          --force
                  Kill workers if graceful stop times out.

  delete-collection <COLLECTION> [--force]
          Delete all documents and chunks in one collection.
          --force
                  Skip confirmation prompt.

  search <QUERY> [-n <N>] [-c, --collection <COLLECTION1> [<COLLECTION2> ...]]
          Semantic search over indexed chunks.
          -n <N>
                  Number of results to return. [default: 10]
          -c, --collection <COLLECTION>
                  Filter by one or more collections.
                  You can use one flag with multiple values: -c work personal.
                  You can also repeat the flag: -c work -c personal.

Examples:
  seman start /path/to/dir1 /path/to/dir2 -c research
  seman search "your query" -n 5 -c work personal

Options:
  -h, --help
          Print help

  -V, --version
          Print version
"""


def _get_cli_version() -> str:
    """Return installed seman version, or unknown."""
    try:
        return version("seman")
    except PackageNotFoundError:
        return "unknown"


def _get_data_dir() -> Path:
    """Return seman data directory path."""
    state_path = config.converter.state_path or config.embedder.state_path
    if state_path is None:
        raise RuntimeError("State path is not configured")
    return state_path.parent


def _converter_state_manager() -> StateManager:
    """Create state manager for converter daemon."""
    return StateManager(config.converter.state_path)


def _embedder_state_manager() -> StateManager:
    """Create state manager for indexer daemon."""
    return StateManager(config.embedder.state_path)


supervisor_state_path = _get_data_dir() / "supervisor.json"


@dataclass
class ManagedProcess:
    """Metadata for a background seman process."""

    name: str
    pid: int
    log_file: str


def _is_pid_running(pid: int) -> bool:
    """Check whether a PID is currently running."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, OSError):
        return False


def _daemon_state_text(value: object) -> str:
    """Normalize daemon state for display."""
    if hasattr(value, "value"):
        return str(getattr(value, "value"))
    return str(value)


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


def _load_supervisor_state() -> dict:
    """Load background supervisor state from disk."""
    if not supervisor_state_path.exists():
        return {}

    try:
        return json.loads(supervisor_state_path.read_text())
    except json.JSONDecodeError:
        return {}


def _save_supervisor_state(state: dict) -> None:
    """Persist background supervisor state."""
    supervisor_state_path.parent.mkdir(parents=True, exist_ok=True)
    supervisor_state_path.write_text(json.dumps(state, indent=2))


def _spawn_detached(command: List[str], log_file: Path) -> int:
    """Spawn a detached background process and return its PID."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return process.pid


def _wait_for_exit(pids: List[int], timeout_seconds: float = 20.0) -> List[int]:
    """Wait for PIDs to exit and return any still running."""
    if not pids:
        return []

    deadline = time.time() + timeout_seconds
    remaining = [pid for pid in pids if _is_pid_running(pid)]
    while remaining and time.time() < deadline:
        time.sleep(0.2)
        remaining = [pid for pid in remaining if _is_pid_running(pid)]
    return remaining


def _force_kill(pids: List[int]) -> List[int]:
    """Send SIGKILL to PIDs and return any that couldn't be killed."""
    remaining = []
    for pid in pids:
        try:
            os.kill(pid, 9)
        except (ProcessLookupError, OSError):
            pass
        else:
            if _is_pid_running(pid):
                remaining.append(pid)
    return remaining


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
    """Start converter and indexer in the background."""
    missing = [d for d in directories if not Path(d).is_dir()]
    if missing:
        for d in missing:
            console.print(f"[red]Error: Directory does not exist: {d}[/red]")
        raise typer.Exit(1)

    state = _load_supervisor_state()
    running = [
        proc
        for proc in state.get("processes", [])
        if isinstance(proc, dict) and _is_pid_running(int(proc.get("pid", 0)))
    ]

    if running:
        console.print("[yellow]Background seman processes already running:[/yellow]")
        for proc in running:
            console.print(f"- {proc.get('name')}: PID {proc.get('pid')}")
        raise typer.Exit(1)

    base_cmd = [sys.executable, "-m", "seman.runner"]
    data_dir = _get_data_dir()
    convert_log = data_dir / "convert-background.log"
    index_log = data_dir / "index-background.log"

    convert_pid = _spawn_detached(
        [*base_cmd, "converter", *directories, "--collection", collection],
        convert_log,
    )
    index_pid = _spawn_detached(
        [
            *base_cmd,
            "indexer",
        ],
        index_log,
    )

    _save_supervisor_state(
        {
            "collection": collection,
            "directories": directories,
            "processes": [
                ManagedProcess("converter", convert_pid, str(convert_log)).__dict__,
                ManagedProcess("indexer", index_pid, str(index_log)).__dict__,
            ],
        }
    )

    console.print("[green]Started seman in background[/green]")
    console.print(f"- converter PID: {convert_pid}")
    console.print(f"- indexer PID: {index_pid}")
    console.print(f"- collection: {collection}")
    console.print("Use `seman status` to check progress and `seman stop` to stop both.")


@app.command(
    "status",
    short_help="(no flags)",
)
def status() -> None:
    """Show detailed converter/indexer status and embedding queue."""
    state = _load_supervisor_state()
    processes = state.get("processes", [])

    converter_state = _converter_state_manager().load()
    indexer_state = _embedder_state_manager().load()

    watched_directories = getattr(converter_state, "watched_directories", []) or []
    if not isinstance(watched_directories, list):
        watched_directories = [str(watched_directories)]

    converter_running = bool(converter_state.pid and _is_pid_running(converter_state.pid))
    indexer_running = bool(indexer_state.pid and _is_pid_running(indexer_state.pid))

    converter_table = Table(title="Converter Status")
    converter_table.add_column("Field", style="cyan")
    converter_table.add_column("Value", style="magenta")
    converter_table.add_row("State", _daemon_state_text(converter_state.daemon_state))
    converter_table.add_row("PID", str(converter_state.pid) if converter_state.pid else "N/A")
    converter_table.add_row("Process", "running" if converter_running else "stopped")
    converter_table.add_row("Watched Directories", "\n".join(watched_directories) or "None")
    converter_table.add_row(
        "Current File", str(getattr(converter_state, "current_file", None) or "None")
    )
    converter_table.add_row("Processed", str(converter_state.processed_count))
    converter_table.add_row("Failed", str(converter_state.failed_count))
    console.print(converter_table)

    indexer_table = Table(title="Indexer Status")
    indexer_table.add_column("Field", style="cyan")
    indexer_table.add_column("Value", style="magenta")
    indexer_table.add_row("State", _daemon_state_text(indexer_state.daemon_state))
    indexer_table.add_row("PID", str(indexer_state.pid) if indexer_state.pid else "N/A")
    indexer_table.add_row("Process", "running" if indexer_running else "stopped")
    indexer_table.add_row(
        "Current File", str(getattr(indexer_state, "current_file", None) or "None")
    )
    console.print(indexer_table)

    supervisor_table = Table(title="Supervisor")
    supervisor_table.add_column("Field", style="cyan")
    supervisor_table.add_column("Value", style="magenta")
    if not processes:
        supervisor_table.add_row("State", "not started")
    else:
        running_count = sum(
            1
            for proc in processes
            if isinstance(proc, dict) and _is_pid_running(int(proc.get("pid", 0)))
        )
        supervisor_table.add_row("State", f"{running_count}/{len(processes)} running")
        supervisor_table.add_row("Collection", str(state.get("collection", "N/A")))
        supervisor_table.add_row("Directories", "\n".join(state.get("directories", [])) or "None")
    console.print(supervisor_table)

    try:
        engine = get_engine(config.database.url)
        session_factory = get_session_factory(engine)
        with session_factory() as session:
            pending = session.query(Chunk).filter_by(embedding_status="pending").count()
            processing = session.query(Chunk).filter_by(embedding_status="processing").count()
            done = session.query(Chunk).filter_by(embedding_status="done").count()
            failed = session.query(Chunk).filter_by(embedding_status="failed").count()
            documents = session.query(Document).count()
            processing_docs = session.query(Document).filter_by(status="processing").count()
            collections = session.query(Document.collection).distinct().count()

        queue_table = Table(title="Embedding Queue")
        queue_table.add_column("Field", style="cyan")
        queue_table.add_column("Value", style="magenta")
        queue_table.add_row("Documents", str(documents))
        queue_table.add_row("Documents Processing", str(processing_docs))
        queue_table.add_row("Collections", str(collections))
        queue_table.add_row("Pending Chunks", str(pending))
        queue_table.add_row("Processing Chunks", str(processing))
        queue_table.add_row("Completed Chunks", str(done))
        queue_table.add_row("Failed Chunks", str(failed))
        console.print(queue_table)
    except Exception:
        queue_table = Table(title="Embedding Queue")
        queue_table.add_column("Field", style="cyan")
        queue_table.add_column("Value", style="magenta")
        queue_table.add_row("Status", "[yellow]Database not reachable[/yellow]")
        queue_table.add_row("Hint", "Run 'seman start' to start infrastructure")
        console.print(queue_table)


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
    processes = state.get("processes", [])

    if not processes:
        console.print("[yellow]No background seman processes found[/yellow]")
        if include_infra:
            _stop_infra_containers()
        return

    signaled_pids: List[int] = []
    for proc in processes:
        pid = int(proc.get("pid", 0))
        if pid <= 0:
            continue
        try:
            os.kill(pid, 15)
            signaled_pids.append(pid)
        except (ProcessLookupError, OSError):
            continue

    timeout_seconds = float(config.bootstrap.wait_timeout_seconds)
    remaining = _wait_for_exit(signaled_pids, timeout_seconds=timeout_seconds)

    if not remaining:
        if supervisor_state_path.exists():
            supervisor_state_path.unlink()
        console.print(f"[green]Stopped {len(signaled_pids)} process(es)[/green]")
        if include_infra:
            _stop_infra_containers()
        return

    if force:
        killed = _force_kill(remaining)
        time.sleep(0.5)
        still_alive = [pid for pid in killed if _is_pid_running(pid)]
        if supervisor_state_path.exists():
            supervisor_state_path.unlink()
        if still_alive:
            console.print(
                f"[red]Force killed {len(remaining) - len(still_alive)} process(es); "
                f"could not kill: {', '.join(str(pid) for pid in still_alive)}[/red]"
            )
        else:
            console.print(f"[green]Force stopped {len(remaining)} process(es)[/green]")
        if include_infra:
            _stop_infra_containers()
        return

    still_running = [
        proc
        for proc in processes
        if isinstance(proc, dict) and int(proc.get("pid", 0)) in remaining
    ]
    _save_supervisor_state(
        {
            "collection": state.get("collection"),
            "directories": state.get("directories", []),
            "processes": still_running,
        }
    )

    console.print(
        f"[yellow]Stop timed out after {timeout_seconds}s; "
        f"still running PID(s): {', '.join(str(pid) for pid in remaining)}[/yellow]"
    )
    console.print("[yellow]Use --force to kill stubborn processes[/yellow]")


def _stop_infra_containers() -> None:
    """Stop infrastructure containers helper."""
    from seman.bootstrap import Bootstrapper

    bootstrapper = Bootstrapper(config)
    try:
        include_ollama = config.indexing.embedder == "ollama"
        bootstrapper.stop_containers(include_ollama=include_ollama)
        console.print("[green]Infrastructure containers stopped[/green]")
    except RuntimeError as e:
        console.print(f"[yellow]Could not stop containers: {e}[/yellow]")


@app.command(
    "delete-collection",
    short_help="COLLECTION [--force]",
)
def delete_collection(
    collection: str = typer.Argument(..., help="Collection name to delete"),
    force: bool = typer.Option(False, "--force", help="Skip confirmation prompt"),
) -> None:
    """Delete all documents and chunks belonging to a collection."""
    if not force:
        confirm = typer.confirm(f"Delete collection '{collection}' and all associated chunks?")
        if not confirm:
            raise typer.Abort()

    try:
        engine = get_engine(config.database.url)
        session_factory = get_session_factory(engine)

        with session_factory() as session:
            docs = session.query(Document).filter_by(collection=collection).all()
            if not docs:
                console.print(f"[yellow]Collection '{collection}' not found[/yellow]")
                return

            doc_ids = [doc.id for doc in docs]
            deleted_chunks = (
                session.query(Chunk)
                .filter(Chunk.document_id.in_(doc_ids))
                .delete(synchronize_session=False)
            )
            deleted_docs = (
                session.query(Document)
                .filter(Document.id.in_(doc_ids))
                .delete(synchronize_session=False)
            )
            session.commit()

        console.print(
            "[green]Deleted collection "
            f"'{collection}' ({deleted_docs} docs, {deleted_chunks} chunks)[/green]"
        )
    except Exception as e:
        console.print(f"[red]Failed to delete collection '{collection}': {e}[/red]")
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
    searcher = Searcher(config)

    try:
        results = searcher.search(query, top_k=top_k, collections=filters)

        if not results:
            console.print("[yellow]No results found[/yellow]")
            return

        table = Table(title=f"Search Results for: {query}")
        table.add_column("Rank", style="cyan", justify="right")
        table.add_column("Source", style="green")
        table.add_column("Score", style="magenta")
        table.add_column("Preview", style="white")

        for i, result in enumerate(results, 1):
            preview = (
                result["content"][:100] + "..."
                if len(result["content"]) > 100
                else result["content"]
            )
            table.add_row(
                str(i),
                result["source_path"],
                f"{result['score']:.3f}",
                preview.replace("\n", " "),
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Search failed: {e}[/red]")
        raise typer.Exit(1)


def main() -> None:
    """Entry point."""
    argv = sys.argv[1:]
    if not argv or (len(argv) == 1 and argv[0] in {"-h", "--help"}):
        typer.echo(ROOT_HELP_TEXT.rstrip())
        return

    if len(argv) == 1 and argv[0] in {"-V", "--version"}:
        typer.echo(_get_cli_version())
        return

    app()


if __name__ == "__main__":
    main()
