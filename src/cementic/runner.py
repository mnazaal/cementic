"""Internal background runner for source and pipeline workers."""

from __future__ import annotations

from typing import List

import typer
from rich.console import Console

from cementic.bootstrap import Bootstrapper
from cementic.config import get_config
from cementic.pipeline_worker import PipelineWorker
from cementic.source_watcher import SourceWatcher

app = typer.Typer(help="Internal cementic runner")
console = Console()


@app.command("source-watcher")
def run_source_watcher(
    directories: List[str] = typer.Argument(..., help="Directories to watch for PDFs"),
    collection: str = typer.Option(
        "default",
        "-c",
        "--collection",
        help="Collection name for indexed documents",
    ),
) -> None:
    """Run source watcher in foreground."""
    config = get_config()
    watcher = SourceWatcher(config)
    bootstrapper = Bootstrapper(config)

    try:
        bootstrapper.ensure_for_convert()
        watcher.start(directories, collection=collection)
    except RuntimeError as e:
        console.print(f"[red]Bootstrap failed: {e}[/red]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        watcher.stop()


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
    config = get_config()
    worker = PipelineWorker(config)
    bootstrapper = Bootstrapper(config)

    try:
        bootstrapper.ensure_for_index()
        worker.start(collection=collection)
    except RuntimeError as e:
        console.print(f"[red]Bootstrap failed: {e}[/red]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        worker.stop()


def main() -> None:
    """Runner entrypoint."""
    app()


if __name__ == "__main__":
    main()
