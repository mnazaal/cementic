"""Internal background runner for source and pipeline workers."""

from __future__ import annotations

import typer
from rich.console import Console

from cementic.bootstrap import Bootstrapper
from cementic.config import get_config
from cementic.pipeline_worker import PipelineWorker
from cementic.source_watcher import SourceWatcher
from cementic.validation import validate_collection_name

app = typer.Typer(help="Internal cementic runner")
console = Console()


@app.command("source-watcher")
def run_source_watcher(
    directories: list[str] = typer.Argument(..., help="Directories to watch for documents"),
    collection: str = typer.Option(
        "default",
        "-c",
        "--collection",
        help="Collection name for indexed documents",
    ),
) -> None:
    """Run source watcher in foreground."""
    try:
        collection = validate_collection_name(collection)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    config = get_config()
    watcher = SourceWatcher(config)
    bootstrapper = Bootstrapper(config)

    try:
        bootstrapper.ensure_for_convert()
    except RuntimeError as e:
        console.print(f"[red]Bootstrap failed: {e}[/red]")
        raise typer.Exit(1)

    try:
        # start() installs SIGINT/SIGTERM handlers and returns on shutdown; this
        # catches a Ctrl-C landing in the window before they are installed.
        watcher.start(directories, collection=collection)
    except KeyboardInterrupt:
        watcher.stop()
    except RuntimeError as e:
        # e.g. every watch directory vanished between `cementic start`'s check
        # and here. A one-line reason on stderr lands in the background log the
        # CLI points at; a traceback would not explain anything.
        console.print(f"[red]Source watcher failed: {e}[/red]")
        raise typer.Exit(1)


@app.command("pipeline-worker")
def run_pipeline_worker(
    collection: str = typer.Option(
        "default",
        "-c",
        "--collection",
        help="Collection name for indexed documents",
    ),
) -> None:
    """Run pipeline worker in foreground."""
    try:
        collection = validate_collection_name(collection)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    config = get_config()
    worker = PipelineWorker(config)
    bootstrapper = Bootstrapper(config)

    try:
        bootstrapper.ensure_for_index()
    except RuntimeError as e:
        console.print(f"[red]Bootstrap failed: {e}[/red]")
        raise typer.Exit(1)

    try:
        # start() installs SIGINT/SIGTERM handlers and returns on shutdown; this
        # catches a Ctrl-C landing in the window before they are installed.
        worker.start(collection=collection)
    except KeyboardInterrupt:
        worker.stop()


def main() -> None:
    """Runner entrypoint."""
    app()


if __name__ == "__main__":
    main()
