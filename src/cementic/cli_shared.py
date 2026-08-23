"""Helpers shared by `cli.py` and `cli_collection.py`.

`cli.py` imports `collection_app` from `cli_collection.py`, so
`cli_collection.py` cannot import these by value from `cli.py` without
deadlocking on partial module initialisation (the names it needs are defined
in `cli.py` after the point where `cli.py` imports `cli_collection`). Both
modules import them from here instead, which has no dependency on either.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import typer
from pydantic import ValidationError
from pydantic_settings import SettingsError
from rich.console import Console
from sqlalchemy.exc import InterfaceError, OperationalError, ProgrammingError

from cementic import render
from cementic.collections import collection_exists
from cementic.config import (
    Config,
    ConfigError,
    get_config,
    render_config_error,
)
from cementic.db import get_engine, get_session_factory
from cementic.supervisor import (
    SupervisorState,
    is_managed_process_alive,
    load_supervisor_state,
    managed_process_pid,
    managed_process_start_token,
)
from cementic.validation import validate_collection_name

# soft_wrap: off a TTY rich falls back to an 80-column hard wrap, which split
# paths and aligned rows mid-word in piped or redirected output. Line breaking
# belongs to the terminal or the consuming program, not to us.
console = Console(soft_wrap=True)
# Errors and diagnostics go here so a failure never pollutes the data on
# stdout (which would otherwise be piped on as content, or break `--json | jq`).
err_console = Console(stderr=True, soft_wrap=True)

_config: Config | None = None


def _get_config() -> Config:
    """Lazy-load the config singleton, reporting config problems in one line.

    Every command routes through here, so this is where a broken config stops
    being a multi-screen pydantic traceback. A failed load is deliberately not
    cached: fixing the file and re-running must work.
    """
    global _config
    if _config is None:
        try:
            _config = get_config()
        except (ConfigError, ValidationError, SettingsError) as error:
            err_console.print(f"[red]{render_config_error(error)}[/red]")
            if isinstance(error, SettingsError):
                # pydantic-settings JSON-parses complex-typed fields from the
                # environment and raises SettingsError -- a ValueError, *not* a
                # ValidationError -- so a plausible spelling like
                # CEMENTIC_EXTRACT_BACKENDS=pdf=pymupdf4llm reached the user as
                # a multi-screen traceback from every command.
                err_console.print(
                    "hint: settings that take a list or table are read from the "
                    "environment as JSON, e.g. CEMENTIC_EXTRACT_BACKENDS='{\"pdf\": "
                    "\"pymupdf4llm\"}'"
                )
            raise typer.Exit(1)
    return _config


@contextmanager
def _db_session() -> Iterator[Any]:
    """One ORM session against the configured database.

    The engine/session-factory ritual around every database command appeared
    seven times; `collection remove` keeps its own copy because it needs the
    engine again after the session closes.
    """
    engine = get_engine(_get_config().database.url)
    session_factory = get_session_factory(engine)
    with session_factory() as session:
        yield session


def _validated_collection_name(collection: str) -> str:
    """validate_collection_name, rendered as a CLI error instead of a raise.

    The five-line try/except around it used to appear at every command that
    takes a collection name, verbatim.
    """
    try:
        return validate_collection_name(collection)
    except ValueError as e:
        err_console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


def _get_data_dir() -> Path:
    """Return cementic data directory path."""
    state_path = _get_config().source_watcher.state_path or _get_config().pipeline_worker.state_path
    if state_path is None:
        raise RuntimeError("State path is not configured")
    return state_path.parent


def _get_supervisor_state_path() -> Path:
    """Lazy supervisor state path."""
    return _get_data_dir() / "supervisor.json"


def _load_supervisor_state() -> SupervisorState:
    return load_supervisor_state(_get_supervisor_state_path())


def _supervisor_processes(state: SupervisorState) -> list[dict[str, object]]:
    processes = state.get("processes", [])
    if not isinstance(processes, list):
        return []
    return [proc for proc in processes if isinstance(proc, dict)]


def _is_managed_proc_alive(process: dict[str, object]) -> bool:
    return is_managed_process_alive(
        managed_process_pid(process), managed_process_start_token(process)
    )


def _is_database_unavailable(error: Exception) -> bool:
    """Return whether the error indicates an unreachable database."""
    return isinstance(error, (OperationalError, InterfaceError))


def _is_schema_missing(error: Exception) -> bool:
    """Return whether the error means cementic's tables don't exist yet.

    The schema is created by the pipeline worker on first `cementic start`, so a
    reachable-but-empty database is the normal pre-first-run state, not a fault.
    """
    return isinstance(error, ProgrammingError) and "does not exist" in str(error.orig)


_NO_SCHEMA_HINT = "nothing indexed yet — run `cementic start DIRECTORY -c COLLECTION` first"

_DB_HINT = (
    "hint: run `cementic init postgres ./cementic-postgres` once and follow its README, "
    "or set CEMENTIC_DB_URL to an existing Postgres with pgvector and pgvectorscale"
)


def _require_known_collection(session: Any, collection: str) -> None:
    """Exit 1 with a clear message when a named collection does not exist.

    "Does not exist" and "exists but has nothing to show" used to be
    indistinguishable -- a zero-filled status report, an empty revision list, or
    "no ready revision", each at exit 0 -- so a typo'd collection name read as a
    real but idle one, and no script could tell the difference.
    """
    if collection_exists(session, collection):
        return
    err_console.print(f"collection: {collection}")
    err_console.print("status: unknown collection (check `cementic collection list`)")
    raise typer.Exit(1)


def _report_db_error(error: Exception, action: str) -> None:
    """Print the right message for a failed database operation."""
    if _is_database_unavailable(error):
        render._print_database_unavailable(action, _DB_HINT)
    elif _is_schema_missing(error):
        err_console.print(_NO_SCHEMA_HINT)
    else:
        err_console.print(f"{action} failed: {error}")


@contextmanager
def _reporting_db_errors(action: str) -> Iterator[None]:
    """Report a failed database operation the same way at every call site.

    A context manager, not a decorator: several call sites do work before and
    after the guarded region (reading outcome fields inside the session, then
    rendering after it closes), and a decorator would pull that work inside the
    guarded scope and change which exceptions it sees.

    typer.Exit subclasses RuntimeError, so a bare `except Exception` here would
    swallow a deliberate exit and re-report it as "<action> failed: 1" -- this
    is the one place that has to get that re-raise right, instead of five.
    """
    try:
        yield
    except typer.Exit:
        raise
    except Exception as error:
        _report_db_error(error, action)
        raise typer.Exit(1)
