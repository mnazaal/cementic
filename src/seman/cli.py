"""CLI interface for seman using Typer."""

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from seman.bootstrap import Bootstrapper
from seman.config import get_config
from seman.converter import ConverterDaemon
from seman.db import Chunk, Document, get_engine, get_session_factory
from seman.embedder import EmbedderDaemon
from seman.search import Searcher
from seman.state import DaemonState, StateManager

app = typer.Typer(help="Semantic search CLI tool")
console = Console()

config = get_config()


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

# Create sub-commands for convert and index
convert_app = typer.Typer(help="PDF conversion commands")
index_app = typer.Typer(help="Indexing commands (compute embeddings)")
app.add_typer(convert_app, name="convert", hidden=True)
app.add_typer(index_app, name="index", hidden=True)


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


@app.command("start")
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

    base_cmd = [sys.executable, "-m", "seman.cli"]
    data_dir = _get_data_dir()
    convert_log = data_dir / "convert-background.log"
    index_log = data_dir / "index-background.log"

    convert_pid = _spawn_detached(
        [*base_cmd, "convert", "start", *directories, "--collection", collection],
        convert_log,
    )
    index_pid = _spawn_detached(
        [
            *base_cmd,
            "index",
            "start",
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


@app.command("status")
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
            collections = session.query(Document.collection).distinct().count()

        queue_table = Table(title="Embedding Queue")
        queue_table.add_column("Field", style="cyan")
        queue_table.add_column("Value", style="magenta")
        queue_table.add_row("Documents", str(documents))
        queue_table.add_row("Collections", str(collections))
        queue_table.add_row("Pending Chunks", str(pending))
        queue_table.add_row("Processing Chunks", str(processing))
        queue_table.add_row("Completed Chunks", str(done))
        queue_table.add_row("Failed Chunks", str(failed))
        console.print(queue_table)
    except Exception as e:
        console.print(f"[yellow]Queue status unavailable: {e}[/yellow]")


@app.command("stop")
def stop_background() -> None:
    """Stop background converter and indexer processes."""
    state = _load_supervisor_state()
    processes = state.get("processes", [])

    if not processes:
        console.print("[yellow]No background seman processes found[/yellow]")
        return

    stopped = 0
    for proc in processes:
        pid = int(proc.get("pid", 0))
        if pid <= 0:
            continue
        try:
            os.kill(pid, 15)
            stopped += 1
        except (ProcessLookupError, OSError):
            continue

    if supervisor_state_path.exists():
        supervisor_state_path.unlink()

    console.print(f"[green]Sent stop signal to {stopped} process(es)[/green]")


@convert_app.command("start")
def convert_start(
    directories: List[str] = typer.Argument(..., help="Directories to watch for PDFs"),
    collection: str = typer.Option(
        "default",
        "-c",
        "--collection",
        help="Collection name for indexed documents",
    ),
) -> None:
    """Start the converter daemon (watches PDFs and converts to chunks)."""
    daemon = ConverterDaemon(config)
    bootstrapper = Bootstrapper(config)

    try:
        bootstrapper.ensure_for_convert()
        daemon.start(directories, collection=collection)
    except RuntimeError as e:
        console.print(f"[red]Bootstrap failed: {e}[/red]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down converter...[/yellow]")
        daemon.stop()


@convert_app.command("stop")
def convert_stop() -> None:
    """Stop the converter daemon."""
    state_manager = _converter_state_manager()
    state = state_manager.load()

    if state.pid:
        try:
            os.kill(state.pid, 15)  # SIGTERM
            console.print(f"[green]Sent stop signal to converter (PID {state.pid})[/green]")
        except (OSError, ProcessLookupError):
            console.print("[yellow]Converter process not found, cleaning up state[/yellow]")
            state_manager.update(daemon_state=DaemonState.STOPPED, pid=None)
    else:
        console.print("[yellow]No converter PID found[/yellow]")


@index_app.command("start")
def index_start() -> None:
    """Start the indexing daemon (generates embeddings for chunks)."""
    daemon = EmbedderDaemon(config)
    bootstrapper = Bootstrapper(config)

    try:
        bootstrapper.ensure_for_index()
        daemon.start()
    except RuntimeError as e:
        console.print(f"[red]Bootstrap failed: {e}[/red]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down indexer...[/yellow]")
        daemon.stop()


@index_app.command("stop")
def index_stop() -> None:
    """Stop the indexer daemon."""
    state_manager = _embedder_state_manager()
    state = state_manager.load()

    if state.pid:
        try:
            os.kill(state.pid, 15)  # SIGTERM
            console.print(f"[green]Sent stop signal to indexer (PID {state.pid})[/green]")
        except (OSError, ProcessLookupError):
            console.print("[yellow]Indexer process not found, cleaning up state[/yellow]")
            state_manager.update(daemon_state=DaemonState.STOPPED, pid=None)
    else:
        console.print("[yellow]No indexer PID found[/yellow]")


@app.command()
def reset(
    force: bool = typer.Option(False, "--force", help="Skip confirmation prompt"),
) -> None:
    """Reset all indexing data (DANGEROUS!)."""
    if not force:
        confirm = typer.confirm("This will delete all indexed data. Are you sure?")
        if not confirm:
            raise typer.Abort()

    # Reset state
    converter_state_manager = _converter_state_manager()
    embedder_state_manager = _embedder_state_manager()
    converter_state_manager.reset()
    embedder_state_manager.reset()
    console.print("[yellow]Reset converter/indexer state files[/yellow]")

    console.print(
        "[green]State reset. Use database tools to clear PostgreSQL data if needed.[/green]"
    )


@app.command("delete-collection")
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


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(10, "-n", "--top-k", help="Number of results"),
    collections: Optional[List[str]] = typer.Option(
        None,
        "-c",
        "--collection",
        help="Filter to one or more collections (defaults to all)",
    ),
) -> None:
    """Search indexed documents."""
    searcher = Searcher(config)

    try:
        results = searcher.search(query, top_k=top_k, collections=collections or None)

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
    app()


if __name__ == "__main__":
    main()
