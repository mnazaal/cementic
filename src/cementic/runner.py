"""Internal background runner for source and pipeline workers."""

from __future__ import annotations

import typer
from pydantic import ValidationError
from pydantic_settings import SettingsError
from rich.console import Console

from cementic.bootstrap import Bootstrapper
from cementic.config import (
    Config,
    ConfigError,
    get_config,
    render_config_error,
)
from cementic.pipeline_worker import PipelineWorker
from cementic.source_watcher import SourceWatcher
from cementic.validation import validate_collection_name

app = typer.Typer(help="Internal cementic runner")
# soft_wrap: this output lands in the background log file, where rich's
# off-TTY 80-column fallback hard-wrapped paths mid-word.
console = Console(soft_wrap=True)
# Failures go to stderr, like the CLI's: the runner's stdout and stderr are
# merged into one background log file, so this costs nothing there -- but a
# runner invoked directly (debugging, a service unit that separates the two)
# used to report every failure on stdout, against the documented convention.
err_console = Console(stderr=True, soft_wrap=True)


def _load_config() -> Config:
    """Load config, reporting a broken one in one line instead of a traceback.

    The spawned workers write to a log file the CLI points the user at, so a
    pydantic traceback there is a wall of text in the one place that is supposed
    to explain why indexing never started.
    """
    try:
        return get_config()
    except (ConfigError, ValidationError, SettingsError) as error:
        err_console.print(f"[red]{render_config_error(error)}[/red]")
        raise typer.Exit(1)


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
        err_console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    config = _load_config()
    watcher = SourceWatcher(config)
    bootstrapper = Bootstrapper(config)

    try:
        bootstrapper.ensure_for_convert()
    except RuntimeError as e:
        err_console.print(f"[red]Bootstrap failed: {e}[/red]")
        raise typer.Exit(1)

    try:
        # start() installs SIGINT/SIGTERM handlers and returns on shutdown; this
        # catches a Ctrl-C landing in the window before they are installed.
        watcher.start(directories, collection=collection)
    except KeyboardInterrupt:
        # A clean Ctrl-C shutdown, not a startup failure: return rather than
        # falling through to the fatal-reason check below.
        watcher.stop()
        return
    except RuntimeError as e:
        # e.g. every watch directory vanished between `cementic start`'s check
        # and here. A one-line reason on stderr lands in the background log the
        # CLI points at; a traceback would not explain anything.
        err_console.print(f"[red]Source watcher failed: {e}[/red]")
        raise typer.Exit(1)
    # Outside the try: `typer.Exit` subclasses `RuntimeError`, so raising this
    # inside it would be caught by the handler above and reported as
    # "Source watcher failed: 1" -- swallowing the real reason on its way to
    # the background log the CLI points the user at.
    if watcher.fatal_reason:
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
        err_console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    config = _load_config()
    worker = PipelineWorker(config)
    bootstrapper = Bootstrapper(config)

    try:
        bootstrapper.ensure_for_index()
    except RuntimeError as e:
        err_console.print(f"[red]Bootstrap failed: {e}[/red]")
        raise typer.Exit(1)

    try:
        # start() installs SIGINT/SIGTERM handlers and returns on shutdown; this
        # catches a Ctrl-C landing in the window before they are installed.
        worker.start(collection=collection)
    except KeyboardInterrupt:
        worker.stop()
        return
    except RuntimeError as e:
        # Parity with the source watcher above: a startup RuntimeError (e.g. an
        # unindexable embedding dimension from _ensure_target_revision) reached
        # the background log as a traceback that explains nothing.
        err_console.print(f"[red]Pipeline worker failed: {e}[/red]")
        raise typer.Exit(1)
    # A fatal startup failure (already running, DB lock held, embedding runtime
    # unreachable) returns normally from start(), so without this the process
    # exits 0 and any systemd unit or CI check keying on exit status concludes
    # the worker is running. The source watcher already exits 1 for its own
    # fatal case.
    if worker.fatal_reason:
        raise typer.Exit(1)


def main() -> None:
    """Runner entrypoint."""
    app()


if __name__ == "__main__":
    main()
