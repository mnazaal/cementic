"""CLI interface for seman using Typer."""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from seman.config import get_config
from seman.converter import ConverterDaemon
from seman.db import Chunk, Document, get_engine, get_session_factory
from seman.embedder import EmbedderDaemon
from seman.search import Searcher
from seman.state import DaemonState, StateManager

app = typer.Typer(help="Semantic search CLI tool")
console = Console()

config = get_config()

# Create sub-commands for convert and embed
convert_app = typer.Typer(help="PDF conversion commands")
embed_app = typer.Typer(help="Embedding generation commands")
app.add_typer(convert_app, name="convert")
app.add_typer(embed_app, name="embed")


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
) -> None:
    """Start the converter daemon (watches PDFs and converts to chunks)."""
    daemon = ConverterDaemon(config)

    try:
        daemon.start(directories)
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

    table.add_row("State", state.daemon_state.value)
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
            import os

            os.kill(state.pid, 15)  # SIGTERM
            console.print(f"[green]Sent stop signal to converter (PID {state.pid})[/green]")
        except (OSError, ProcessLookupError):
            console.print("[yellow]Converter process not found, cleaning up state[/yellow]")
            state_manager.update(daemon_state=DaemonState.STOPPED, pid=None)
    else:
        console.print("[yellow]No converter PID found[/yellow]")


@embed_app.command("start")
def embed_start() -> None:
    """Start the embedder daemon (generates embeddings for chunks)."""
    daemon = EmbedderDaemon(config)

    try:
        daemon.start()
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down embedder...[/yellow]")
        daemon.stop()


@embed_app.command("pause")
def embed_pause() -> None:
    """Pause embedding generation."""
    state_manager = StateManager(config.indexing.state_path)
    state = state_manager.load()

    if state.daemon_state != DaemonState.RUNNING:
        console.print("[yellow]Embedder is not running[/yellow]")
        return

    state_manager.update(daemon_state=DaemonState.PAUSED)
    console.print("[green]Embedder paused[/green]")


@embed_app.command("resume")
def embed_resume() -> None:
    """Resume embedding generation."""
    state_manager = StateManager(config.indexing.state_path)
    state = state_manager.load()

    if state.daemon_state != DaemonState.PAUSED:
        console.print("[yellow]Embedder is not paused[/yellow]")
        return

    state_manager.update(daemon_state=DaemonState.RUNNING)
    console.print("[green]Embedder resumed[/green]")


@embed_app.command("status")
def embed_status() -> None:
    """Show embedder status and queue."""
    state_manager = StateManager(config.indexing.state_path)
    state = state_manager.load()

    # Get chunk statistics from database
    engine = get_engine(config.database.url)
    Session = get_session_factory(engine)

    with Session() as session:
        pending = session.query(Chunk).filter_by(embedding_status="pending").count()
        processing = session.query(Chunk).filter_by(embedding_status="processing").count()
        done = session.query(Chunk).filter_by(embedding_status="done").count()
        failed = session.query(Chunk).filter_by(embedding_status="failed").count()

    table = Table(title="Embedder Status")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("State", state.daemon_state.value)
    table.add_row("PID", str(state.pid) if state.pid else "N/A")
    table.add_row("Current File", state.current_file or "None")
    table.add_row("Pending Chunks", str(pending))
    table.add_row("Processing Chunks", str(processing))
    table.add_row("Completed Chunks", str(done))
    table.add_row("Failed Chunks", str(failed))

    console.print(table)


@embed_app.command("stop")
def embed_stop() -> None:
    """Stop the embedder daemon."""
    state_manager = StateManager(config.indexing.state_path)
    state = state_manager.load()

    if state.pid:
        try:
            import os

            os.kill(state.pid, 15)  # SIGTERM
            console.print(f"[green]Sent stop signal to embedder (PID {state.pid})[/green]")
        except (OSError, ProcessLookupError):
            console.print("[yellow]Embedder process not found, cleaning up state[/yellow]")
            state_manager.update(daemon_state=DaemonState.STOPPED, pid=None)
    else:
        console.print("[yellow]No embedder PID found[/yellow]")


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
) -> None:
    """Search indexed documents."""
    searcher = Searcher(config)

    try:
        results = searcher.search(query, top_k=top_k)

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
