"""Internal background runner for converter and indexer daemons."""

from __future__ import annotations

from typing import List

import typer
from rich.console import Console

from seman.bootstrap import Bootstrapper
from seman.config import get_config
from seman.converter import ConverterDaemon
from seman.embedder import EmbedderDaemon

app = typer.Typer(help="Internal seman runner")
console = Console()


@app.command("converter")
def run_converter(
    directories: List[str] = typer.Argument(..., help="Directories to watch for PDFs"),
    collection: str = typer.Option(
        "default",
        "-c",
        "--collection",
        help="Collection name for indexed documents",
    ),
) -> None:
    """Run converter daemon in foreground."""
    config = get_config()
    daemon = ConverterDaemon(config)
    bootstrapper = Bootstrapper(config)

    try:
        bootstrapper.ensure_for_convert()
        daemon.start(directories, collection=collection)
    except RuntimeError as e:
        console.print(f"[red]Bootstrap failed: {e}[/red]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        daemon.stop()


@app.command("indexer")
def run_indexer() -> None:
    """Run indexer daemon in foreground."""
    config = get_config()
    daemon = EmbedderDaemon(config)
    bootstrapper = Bootstrapper(config)

    try:
        bootstrapper.ensure_for_index()
        daemon.start()
    except RuntimeError as e:
        console.print(f"[red]Bootstrap failed: {e}[/red]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        daemon.stop()


def main() -> None:
    """Runner entrypoint."""
    app()


if __name__ == "__main__":
    main()
