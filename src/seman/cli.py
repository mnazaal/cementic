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
from seman.db import Chunk, get_engine, get_session_factory
from seman.embedder import EmbedderDaemon
from seman.search import Searcher
from seman.state import DaemonState, StateManager

app = typer.Typer(help="Semantic search CLI tool")
console = Console()

config = get_config()


def _get_data_dir() -> Path:
    """Return seman data directory path."""
    state_path = config.indexing.state_path
    if state_path is None:
        raise RuntimeError("State path is not configured")
    return state_path.parent


supervisor_state_path = _get_data_dir() / "supervisor.json"

# Create sub-commands for convert and index
convert_app = typer.Typer(help="PDF conversion commands")
index_app = typer.Typer(help="Indexing commands (compute embeddings)")
app.add_typer(convert_app, name="convert")
app.add_typer(index_app, name="index")


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
    console.print("Use `seman ps` to check processes and `seman stop` to stop both.")


@app.command("ps")
def process_status() -> None:
    """Show background seman process status."""
    state = _load_supervisor_state()
    processes = state.get("processes", [])

    if not processes:
        console.print("[yellow]No background seman processes found[/yellow]")
        return

    table = Table(title="Seman Background Processes")
    table.add_column("Name", style="cyan")
    table.add_column("PID", style="magenta")
    table.add_column("Status", style="green")
    table.add_column("Log", style="white")

    for proc in processes:
        pid = int(proc.get("pid", 0))
        status = "running" if _is_pid_running(pid) else "stopped"
        table.add_row(
            str(proc.get("name", "unknown")),
            str(pid),
            status,
            str(proc.get("log_file", "")),
        )

    console.print(table)


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


@app.command()
def infra_up() -> None:
    """Start PostgreSQL and Ollama containers."""
    try:
        subprocess.run(
            ["podman-compose", "up", "-d"],
            check=True,
            cwd=Path(__file__).parent.parent,
        )
        console.print("[green]Infrastructure started successfully[/green]")
        console.print(f"PostgreSQL: localhost:{config.database.port}")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Failed to start infrastructure: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def infra_down() -> None:
    """Stop PostgreSQL and Ollama containers."""
    try:
        subprocess.run(
            ["podman-compose", "down"],
            check=True,
            cwd=Path(__file__).parent.parent,
        )
        console.print("[green]Infrastructure stopped successfully[/green]")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Failed to stop infrastructure: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def infra_status() -> None:
    """Check infrastructure status."""
    try:
        result = subprocess.run(
            [
                "podman",
                "ps",
                "--filter",
                "name=seman-",
                "--format",
                "table {{.Names}}\t{{.Status}}\t{{.Ports}}",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        console.print(result.stdout)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Failed to get status: {e}[/red]")


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


@convert_app.command("pause")
def convert_pause() -> None:
    """Pause PDF conversion (keep watching)."""
    state_manager = StateManager(config.indexing.state_path)
    state = state_manager.load()

    if state.daemon_state != DaemonState.RUNNING:
        console.print("[yellow]Converter is not running[/yellow]")
        return

    state_manager.update(daemon_state=DaemonState.PAUSED)
    console.print("[green]Converter paused[/green]")


@convert_app.command("resume")
def convert_resume() -> None:
    """Resume PDF conversion."""
    state_manager = StateManager(config.indexing.state_path)
    state = state_manager.load()

    if state.daemon_state != DaemonState.PAUSED:
        console.print("[yellow]Converter is not paused[/yellow]")
        return

    state_manager.update(daemon_state=DaemonState.RUNNING)
    console.print("[green]Converter resumed[/green]")


@convert_app.command("status")
def convert_status() -> None:
    """Show converter status."""
    state_manager = StateManager(config.indexing.state_path)
    state = state_manager.load()

    table = Table(title="Converter Status")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("State", _daemon_state_text(state.daemon_state))
    table.add_row("PID", str(state.pid) if state.pid else "N/A")
    table.add_row("Watched Directories", "\n".join(state.watched_directories) or "None")
    table.add_row("Current File", state.current_file or "None")
    table.add_row("Processed", str(state.processed_count))
    table.add_row("Failed", str(state.failed_count))

    console.print(table)


@convert_app.command("stop")
def convert_stop() -> None:
    """Stop the converter daemon."""
    state_manager = StateManager(config.indexing.state_path)
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


@index_app.command("pause")
def index_pause() -> None:
    """Pause indexing (embedding generation)."""
    state_manager = StateManager(config.indexing.state_path)
    state = state_manager.load()

    if state.daemon_state != DaemonState.RUNNING:
        console.print("[yellow]Indexer is not running[/yellow]")
        return

    state_manager.update(daemon_state=DaemonState.PAUSED)
    console.print("[green]Indexer paused[/green]")


@index_app.command("resume")
def index_resume() -> None:
    """Resume indexing."""
    state_manager = StateManager(config.indexing.state_path)
    state = state_manager.load()

    if state.daemon_state != DaemonState.PAUSED:
        console.print("[yellow]Indexer is not paused[/yellow]")
        return

    state_manager.update(daemon_state=DaemonState.RUNNING)
    console.print("[green]Indexer resumed[/green]")


@index_app.command("status")
def index_status() -> None:
    """Show indexer status and queue."""
    state_manager = StateManager(config.indexing.state_path)
    state = state_manager.load()

    # Get chunk statistics from database
    engine = get_engine(config.database.url)
    session_factory = get_session_factory(engine)

    with session_factory() as session:
        pending = session.query(Chunk).filter_by(embedding_status="pending").count()
        processing = session.query(Chunk).filter_by(embedding_status="processing").count()
        done = session.query(Chunk).filter_by(embedding_status="done").count()
        failed = session.query(Chunk).filter_by(embedding_status="failed").count()

    table = Table(title="Indexer Status")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("State", _daemon_state_text(state.daemon_state))
    table.add_row("PID", str(state.pid) if state.pid else "N/A")
    table.add_row("Current File", state.current_file or "None")
    table.add_row("Pending Chunks", str(pending))
    table.add_row("Processing Chunks", str(processing))
    table.add_row("Completed Chunks", str(done))
    table.add_row("Failed Chunks", str(failed))

    console.print(table)


@index_app.command("stop")
def index_stop() -> None:
    """Stop the indexer daemon."""
    state_manager = StateManager(config.indexing.state_path)
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
    state_manager = StateManager(config.indexing.state_path)
    state_manager.reset()
    console.print("[yellow]Reset state file[/yellow]")

    console.print(
        "[green]State reset. Use database tools to clear PostgreSQL data if needed.[/green]"
    )


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
