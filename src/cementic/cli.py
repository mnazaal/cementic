"""CLI interface for cementic using Typer."""

import json
import os
import shutil
import sys
import time
from importlib import resources
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import typer
from rich.markup import escape

from cementic import render
from cementic.bootstrap import Bootstrapper
from cementic.chunk import chunk_text
from cementic.cli_collection import collection_app
from cementic.cli_format import CementicTyper
from cementic.cli_shared import (
    _DB_HINT,
    _NO_SCHEMA_HINT,
    _db_session,
    _get_config,
    _get_data_dir,
    _get_supervisor_state_path,
    _is_database_unavailable,
    _is_managed_proc_alive,
    _is_schema_missing,
    _load_supervisor_state,
    _report_db_error,
    _reporting_db_errors,
    _require_known_collection,
    _supervisor_processes,
    _validated_collection_name,
    console,
    err_console,
)
from cementic.collections import collection_exists, list_collections
from cementic.config import config_path_error, default_config_path, resolve_config_path
from cementic.doctor import collect_doctor_report
from cementic.embedding_runtime import (
    create_provider,
    get_llama_cpp_runtime_client,
    llama_daemon_status,
    runtime_spec_from_config,
    stop_llama_cpp_runtime,
)
from cementic.embedding_text import describe_text_policy
from cementic.extract import extract_document
from cementic.filelock import LockUnavailableError, file_lock
from cementic.hybrid import scores_explain_order
from cementic.search import MAX_SEARCH_RESULTS, Searcher
from cementic.state import DaemonState, StateManager
from cementic.status_service import (
    build_supervisor_status,
    check_health,
    load_pipeline_status,
    load_pipeline_status_bulk,
    load_worker_statuses,
)
from cementic.supervisor import (
    ManagedProcess,
    force_kill,
    is_managed_process_alive,
    is_pid_running,
    managed_process_pid,
    process_start_token,
    save_supervisor_state,
    spawn_detached,
    wait_for_exit,
)

app = CementicTyper(help="Index and semantically search document collections")
embedding_app = CementicTyper(help="Manage embedding runtime service")
config_app = CementicTyper(help="View and manage the config file")
init_app = CementicTyper(help="Initialize local setup files")
app.add_typer(collection_app, name="collection")
app.add_typer(embedding_app, name="embedding")
app.add_typer(config_app, name="config")
app.add_typer(init_app, name="init")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(_get_cli_version())
        raise typer.Exit()


# A leading "\b" marks a paragraph as preformatted so Click does not re-wrap it.
_ROOT_EPILOG = """\
\b
EXAMPLES:
  cementic start ~/research-papers -c research
  cementic search "vector databases" -c research
  cementic status

\b
Run 'cementic COMMAND --help' for details on a command.
Docs and issues: https://github.com/mnazaal/cementic
"""

_START_EPILOG = """\
\b
EXAMPLES:
  cementic start ~/research-papers
  cementic start ~/papers ~/notes --collection research
"""

_SEARCH_EPILOG = """\
\b
EXAMPLES:
  cementic search "transformer inference"
  cementic search "graph theory" -n 5 -c math papers
  cementic search "graph theory" --json | jq .source_path
"""


@app.callback(epilog=_ROOT_EPILOG)
def _root(
    version: bool = typer.Option(
        None,
        "--version",
        "-V",
        help="Show the cementic version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Index and semantically search document collections.

    cementic watches directories of documents (PDF, Markdown, plain text) and
    builds a versioned extract -> chunk -> embed pipeline in Postgres (pgvector),
    then serves fast semantic search over the indexed chunks.
    """


def _default_config_toml() -> str:
    """The annotated default config, read from its packaged template file.

    Same `importlib.resources` pattern `init_postgres` below uses for its
    template directory, applied to a single file instead of a tree.
    """
    return (resources.files("cementic") / "templates" / "config.toml").read_text(encoding="utf-8")


@config_app.command("path", short_help="Print the active (or default) config path")
def config_path() -> None:
    """Print the active config file path, or the default location if none exists."""
    # The same guard every other command hits via get_config(): without it this
    # command printed the fallback path for an unusable CEMENTIC_CONFIG -- the
    # one symptom it exists to diagnose.
    problem = config_path_error()
    if problem is not None:
        err_console.print(f"config error: {problem}")
        raise typer.Exit(1)
    active = resolve_config_path()
    typer.echo(str(active if active is not None else default_config_path()))


@config_app.command("init", short_help="Write an annotated default config file")
def config_init(
    force: bool = typer.Option(False, "--force", help="Overwrite an existing config file"),
) -> None:
    """Write an annotated default config to the user config directory."""
    path = default_config_path()
    if path.exists() and not force:
        err_console.print(f"config already exists at {path} (use --force to overwrite)")
        raise typer.Exit(1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_default_config_toml(), encoding="utf-8")
    console.print(f"wrote {path}")


@config_app.command("show", short_help="Print the effective merged config as JSON")
def config_show() -> None:
    """Print the effective configuration (defaults + file + env) as JSON to stdout."""
    typer.echo(json.dumps(_get_config().model_dump(mode="json"), indent=2, sort_keys=True))


@init_app.command("postgres", short_help="Write a local Postgres setup directory")
def init_postgres(
    directory: Path = typer.Argument(..., help="Directory to write setup files into"),
    force: bool = typer.Option(
        False, "--force", help="Overwrite setup files in an existing non-empty directory"
    ),
) -> None:
    """Copy static Docker/Podman Postgres setup files for cementic."""
    if directory.exists() and not directory.is_dir():
        err_console.print(f"{directory} exists and is not a directory")
        raise typer.Exit(1)
    try:
        non_empty = directory.is_dir() and any(directory.iterdir())
    except OSError as error:
        err_console.print(f"cannot read {directory}: {error}")
        raise typer.Exit(1)
    if non_empty and not force:
        err_console.print(f"{directory} already exists and is not empty (use --force to overwrite)")
        raise typer.Exit(1)

    template_root = resources.files("cementic") / "templates" / "postgres"
    with resources.as_file(template_root) as source:
        if not source.is_dir():
            err_console.print("Postgres setup templates are missing from this installation")
            raise typer.Exit(1)
        # Overwrite the template files in place rather than clearing the
        # directory first: `--force` used to `rmtree` whatever it was pointed
        # at, so `cementic init postgres ~ --force` deleted the user's home
        # directory before writing five files into it.
        try:
            shutil.copytree(source, directory, dirs_exist_ok=True)
        except OSError as error:
            err_console.print(f"failed to write setup files to {directory}: {error}")
            raise typer.Exit(1)

    console.print(f"Wrote Postgres setup to {directory}")
    console.print("")
    console.print("Start the persistent local database once:")
    console.print(f"  cd {directory}")
    console.print("  docker compose up -d")
    console.print("  # or: podman compose up -d")
    console.print("")
    console.print("Then check readiness:")
    console.print("  cementic doctor")


def _save_supervisor_state(state: dict[str, object]) -> None:
    save_supervisor_state(_get_supervisor_state_path(), state)


def _known_collection_names(candidates: list[str]) -> set[str]:
    """Which of these collections cementic knows about at all."""
    try:
        with _db_session() as session:
            return {name for name in candidates if collection_exists(session, name)}
    except Exception:
        # Only ever refines an error message; never the reason a search fails.
        return set()


def _unsearchable_message(unknown: list[str], unindexed: set[str]) -> str:
    """One line naming what is wrong with each unsearchable collection."""
    missing = [name for name in unknown if name not in unindexed]
    parts = []
    if missing:
        parts.append(f"unknown collection {', '.join(missing)}")
    if unindexed:
        parts.append(
            f"no indexed revision for {', '.join(sorted(unindexed))}"
        )
    return f"search failed: {'; '.join(parts)} (check `cementic collection list`)"


def _stdin_is_a_terminal() -> bool:
    """Whether stdin is a terminal (a seam: test runners replace sys.stdin)."""
    return sys.stdin.isatty()


_STARTUP_GRACE_SECONDS = 2.0

#: How many files `status --verbose` lists before it stops. A screenful of the
#: rows that sort first, not one line per document: at corpus scale the
#: unlimited listing loaded tens of megabytes of ORM objects to flood a
#: terminal. `--limit 0` restores the full listing.
DEFAULT_STATUS_FILE_LIMIT = 20


def _pipeline_worker_activity() -> str | None:
    """What the pipeline worker is busy with, if it published anything.

    Read from its state file rather than inferred, and tolerant of every way
    that can fail -- this only ever adds an explanation to a stop message, so it
    must never be the reason a stop fails.
    """
    try:
        state = StateManager(_get_config().pipeline_worker.state_path).load()
    except Exception:
        return None
    return state.current_activity or None


def _worker_processes_from_state_files() -> list[dict[str, object]]:
    """Recover live worker records when the supervisor file is missing.

    ``supervisor.json`` is the normal source of PIDs, but it is a single file
    that every stop deletes and every start overwrites. Losing it left
    `cementic stop` unable to stop anything while reporting that nothing was
    running -- the user's only recourse being `ps` and `kill`. Each worker
    already persists its own pid and start-token, so the information was never
    actually lost; this just looks where it still is.
    """
    config = _get_config()
    records: list[dict[str, object]] = []
    for name, state_path in (
        ("source-watcher", config.source_watcher.state_path),
        ("pipeline-worker", config.pipeline_worker.state_path),
    ):
        if state_path is None:
            continue
        worker_state = StateManager(state_path).load()
        pid = worker_state.pid
        if pid and is_managed_process_alive(pid, worker_state.start_token):
            records.append(
                {"name": name, "pid": pid, "start_token": worker_state.start_token}
            )
    return records


def _terminate_managed(processes: list[ManagedProcess]) -> None:
    """Stop spawned workers, escalating to SIGKILL for anything that lingers."""
    pids = [
        proc.pid
        for proc in processes
        if is_managed_process_alive(proc.pid, proc.start_token)
    ]
    for pid in pids:
        try:
            os.kill(pid, 15)
        except OSError:
            continue
    remaining = wait_for_exit(pids, timeout_seconds=5.0)
    if remaining:
        force_kill(remaining)


def _clear_worker_state_files() -> None:
    """Mark both workers stopped after they were killed.

    A worker writes `stopped` from its own shutdown path, which never runs under
    SIGKILL. Without this, `status --verbose` reported "stopped, state=running,
    pid=<dead pid>" indefinitely -- a contradiction in the same line. The
    headline worker state was already correct (it checks liveness), so this
    aligns the detail with it.
    """
    config = _get_config()
    for state_path in (
        config.source_watcher.state_path,
        config.pipeline_worker.state_path,
    ):
        if state_path is None:
            continue
        try:
            StateManager(state_path).update(
                daemon_state=DaemonState.STOPPED, pid=None, start_token=None, current_file=None
            )
        except OSError:
            continue



def _wait_for_worker_startup(
    processes: list[ManagedProcess], grace_seconds: float = _STARTUP_GRACE_SECONDS
) -> list[ManagedProcess]:
    """Return the processes that died within the startup grace period.

    A worker that fails at startup (bad config, unreachable runtime, lock held)
    exits within a fraction of a second, so a short watch catches it; a healthy
    worker outlives the grace period and this returns empty.
    """
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        dead = [
            proc
            for proc in processes
            if not is_managed_process_alive(proc.pid, proc.start_token)
        ]
        if dead:
            return dead
        time.sleep(0.1)
    return []


def _get_cli_version() -> str:
    """Return installed cementic version, or unknown."""
    try:
        return version("cementic")
    except PackageNotFoundError:
        return "unknown"


def _get_start_lock_path() -> Path:
    """Lock file serialising `cementic start`'s check-then-spawn sequence."""
    return _get_data_dir() / "start.lock"


def _build_collection_filters(
    option_collections: list[str] | None,
    trailing_collections: list[str] | None,
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


def _llama_daemon_runtime_status() -> str:
    """Return llama.cpp daemon runtime status."""
    return llama_daemon_status(_get_config())


@app.command(
    "start",
    short_help="Start background indexing of one or more directories",
    epilog=_START_EPILOG,
    no_args_is_help=True,
)
def start_background(
    directories: list[str] = typer.Argument(..., help="Directories to watch for documents"),
    collection: str = typer.Option(
        "default",
        "-c",
        "--collection",
        help="Collection name for indexed documents",
    ),
) -> None:
    """Start source watcher and pipeline worker in the background."""
    collection = _validated_collection_name(collection)

    missing = [d for d in directories if not Path(d).is_dir()]
    if missing:
        for d in missing:
            # An existing file is not a missing directory; saying "does not
            # exist" about a path the user can see sends them the wrong way.
            reason = "is not a directory" if Path(d).exists() else "does not exist"
            err_console.print(f"[red]Error: {d} {reason}[/red]")
        raise typer.Exit(1)

    # Absolute from here on: the detached workers and any later `cementic status`
    # can run from a different working directory, where a relative path would
    # name something else entirely.
    directories = [str(Path(d).resolve()) for d in directories]

    # Serialise the whole check-then-spawn-then-record sequence. Two concurrent
    # `cementic start` runs could both see nothing running, both spawn, and the
    # second's supervisor record replace the first's -- leaving the first pair
    # running and invisible to `cementic stop`.
    try:
        with file_lock(_get_start_lock_path(), timeout=0):
            _start_background_locked(directories, collection)
    except LockUnavailableError:
        err_console.print("[yellow]Another `cementic start` is already in progress[/yellow]")
        raise typer.Exit(1)


def _start_background_locked(directories: list[str], collection: str) -> None:
    """The body of `cementic start`, run while holding the start lock."""
    state = _load_supervisor_state()
    running = [proc for proc in _supervisor_processes(state) if _is_managed_proc_alive(proc)]

    if running:
        err_console.print("[yellow]Background cementic processes already running:[/yellow]")
        for proc in running:
            err_console.print(f"- {proc.get('name')}: PID {proc.get('pid')}")
        raise typer.Exit(1)

    try:
        bootstrapper = Bootstrapper(_get_config())
        bootstrapper.ensure_for_convert()
        bootstrapper.ensure_for_index()
    except RuntimeError as error:
        err_console.print(f"[red]Bootstrap failed before background start: {error}[/red]")
        raise typer.Exit(1)

    base_cmd = [sys.executable, "-m", "cementic.runner"]
    data_dir = _get_data_dir()
    source_watcher_log = data_dir / "source-watcher-background.log"
    pipeline_log = data_dir / "pipeline-background.log"

    spawned: list[ManagedProcess] = []
    try:
        # `--` ends option parsing so a directory whose name begins with "-" is
        # not read as a flag by the runner.
        source_watcher_pid = spawn_detached(
            [*base_cmd, "source-watcher", "--collection", collection, "--", *directories],
            source_watcher_log,
        )
        spawned.append(
            ManagedProcess(
                "source-watcher",
                source_watcher_pid,
                str(source_watcher_log),
                process_start_token(source_watcher_pid),
            )
        )
        pipeline_pid = spawn_detached(
            [*base_cmd, "pipeline-worker", "--collection", collection],
            pipeline_log,
        )
        spawned.append(
            ManagedProcess(
                "pipeline-worker",
                pipeline_pid,
                str(pipeline_log),
                process_start_token(pipeline_pid),
            )
        )
    except OSError as error:
        # The first spawn can succeed and the second fail (ENOMEM, EMFILE,
        # unwritable log dir). Without this the survivor keeps running with no
        # supervisor record, so `cementic stop` could never find it again.
        _terminate_managed(spawned)
        err_console.print(f"[red]cementic failed to start: {error}[/red]")
        raise typer.Exit(1)

    _save_supervisor_state(
        {
            "collection": collection,
            "directories": directories,
            "processes": [proc.__dict__ for proc in spawned],
        }
    )

    # A worker can exit immediately (bootstrap failure, embedding runtime down,
    # lock held) writing only to its log file. Reporting success without looking
    # would leave the user believing indexing started.
    dead = _wait_for_worker_startup(spawned)
    if dead:
        err_console.print("[red]cementic failed to start:[/red]")
        for managed in dead:
            err_console.print(f"- {managed.name} exited immediately; see {managed.log_file}")
        # Stop whatever did come up and clear the record. Leaving a survivor
        # running would hold the collection's advisory lock and make the next
        # `cementic start` refuse with "already running" -- contradicting the
        # failure just reported, with no hint that `cementic stop` is the way out.
        dead_pids = {managed.pid for managed in dead}
        survivors = [managed for managed in spawned if managed.pid not in dead_pids]
        if survivors:
            _terminate_managed(survivors)
            for managed in survivors:
                err_console.print(f"- stopped {managed.name} (PID {managed.pid})")
        _get_supervisor_state_path().unlink(missing_ok=True)
        raise typer.Exit(1)

    # "Started" is as far as this can honestly claim: the grace period above
    # only rules out an immediate crash. A worker whose startup fails past it
    # (e.g. the embedding daemon still loading its model, up to
    # `daemon_start_timeout_seconds`) reports the reason to `cementic status`,
    # not here -- claiming success this early used to leave the user believing
    # indexing had started while it kept failing for another two minutes.
    console.print("[green]Source watcher and pipeline worker started[/green]")
    console.print(f"- source watcher PID: {source_watcher_pid}")
    console.print(f"- pipeline worker PID: {pipeline_pid}")
    console.print(f"- collection: {collection}")
    if collection == "default":
        console.print(
            "- note: no collection was specified, so documents will be indexed into 'default'"
        )
    console.print(
        "Run `cementic status` to confirm -- a startup failure in the next "
        "couple of minutes will show up there. `cementic stop` stops both."
    )


@app.command(
    "status",
    short_help="Show worker status and indexing progress",
)
def status(
    collection: str | None = typer.Option(
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
    limit: int = typer.Option(
        DEFAULT_STATUS_FILE_LIMIT,
        "--limit",
        min=0,
        help="Max files listed by --verbose; 0 for all",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output status as JSON",
    ),
) -> None:
    """Show background worker status and collection progress."""
    if collection is not None:
        collection = _validated_collection_name(collection)
    # 0 is the escape hatch, and None is how the query layer spells "no LIMIT".
    file_limit = limit or None

    state = _load_supervisor_state()
    source_watcher_status, pipeline_worker_status = load_worker_statuses(_get_config())
    supervisor_status = build_supervisor_status(state)
    directories = supervisor_status.directories or source_watcher_status.watched_directories

    try:
        health = check_health(_get_config())
        health_error = None
    except Exception as error:
        # The whole health section used to vanish without a word, making a
        # crashed health probe indistinguishable from health never being asked.
        health = None
        health_error = str(error)

    if json_output:
        document = render.build_status_document(
            _db_session,
            _is_schema_missing,
            _NO_SCHEMA_HINT,
            _get_config(),
            supervisor_status,
            source_watcher_status,
            pipeline_worker_status,
            directories,
            health,
            health_error,
            collection,
            verbose,
            file_limit,
        )
        failed = render.print_status_document(document)
        if failed:
            raise typer.Exit(1)
        return

    render._print_status_summary(
        supervisor_status.collection,
        directories,
        source_watcher_status,
        pipeline_worker_status,
        health,
        verbose,
        _get_config(),
    )
    if health is None and health_error is not None:
        err_console.print(f"health: unavailable ({health_error})")

    if health is None:
        # A health probe that raised leaves the database state unknown, which is
        # not the same as reachable: exiting 0 let `cementic status && deploy`
        # proceed on a report that was missing the very rows it would have
        # failed on.
        raise typer.Exit(1)
    if not health.db_reachable:
        err_console.print(_DB_HINT)
        # Non-zero so `cementic status && ...` cannot succeed against a database
        # cementic could not reach; every other database-backed command exits 1.
        raise typer.Exit(1)

    with _reporting_db_errors("status"):
        with _db_session() as session:
            if collection is None:
                rows = list_collections(session)
                console.print()
                console.print("collections")
                if not rows:
                    console.print("  (none)")
                    return
                status_by_collection = load_pipeline_status_bulk(
                    _get_config(), [row.name for row in rows]
                )
                items = [(row, status_by_collection[row.name]) for row in rows]
                name_w = max(len(row.name) for row, _ in items)
                doc_w = max(len(f"{ps.documents:,}") for _, ps in items)
                frac_w = max(
                    len(f"{ps.done_embeddings:,}/{ps.total_chunks:,}") for _, ps in items
                )
                for row, ps in items:
                    frac = f"{ps.done_embeddings:,}/{ps.total_chunks:,}"
                    # Naming the failures is what makes a finished collection
                    # readable as finished. Showing done/total alone, a corpus
                    # whose remainder can never succeed sits at "99.8%" forever
                    # and is indistinguishable from one still working.
                    failed_note = (
                        f", {ps.failed_embeddings:,} failed" if ps.failed_embeddings else ""
                    )
                    console.print(
                        f"  {row.name:<{name_w}}   {ps.documents:>{doc_w},} docs   "
                        f"{frac:>{frac_w}} embedded ({ps.embedding_pct}%{failed_note})"
                    )
                return
            # This session used to be opened and then discarded on the
            # named-collection path. It now earns its connection: a typo'd name
            # otherwise produced a full zero-filled report at exit 0, which
            # reads as a real collection that has not started yet.
            console.print()
            _require_known_collection(session, collection)

        pipeline_status = load_pipeline_status(_get_config(), collection)
        render._print_collection_detail(
            collection, pipeline_status, verbose, _get_config(), file_limit
        )


@app.command(
    "doctor",
    short_help="Run read-only runtime readiness diagnostics",
)
def doctor(
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output the report as JSON",
    ),
) -> None:
    """Run read-only runtime readiness diagnostics."""
    try:
        report = collect_doctor_report(_get_config())
    except typer.Exit:
        # _get_config already printed the precise config error to stderr.
        # A broken config is exactly what doctor exists to diagnose, so emit
        # the failing report it promises (--json consumers still get JSON)
        # instead of dying with less output than plain `status`.
        config_path = resolve_config_path()
        report = {
            "ok": False,
            "checks": {
                "config": {
                    "status": "fail",
                    "path": str(config_path) if config_path is not None else None,
                    "message": "config failed to load; the error is printed on stderr",
                }
            },
        }
    if json_output:
        typer.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        render._print_doctor_report(report)
    if not report["ok"]:
        raise typer.Exit(1)


@app.command(
    "stop",
    short_help="Stop cementic's background processes",
)
def stop_background(
    force: bool = typer.Option(
        False,
        "--force",
        help="Force kill processes that don't stop gracefully",
    ),
) -> None:
    """Stop cementic's background processes (source watcher, pipeline worker).

    The shared embedding daemon is intentionally left running so other
    collections and `search` stay warm; stop it separately with
    `cementic embedding stop`.

    Postgres is not managed by cementic; stop it with your container engine
    (e.g. `docker compose down` or `podman compose down`).
    """
    state = _load_supervisor_state()
    processes = _supervisor_processes(state)

    recovered = False
    if not processes:
        processes = _worker_processes_from_state_files()
        recovered = bool(processes)

    if not processes:
        console.print("no background cementic processes found")
        return

    if recovered:
        console.print(
            "supervisor record missing; found running workers from their own state files"
        )

    signaled_pids: list[int] = []
    unsignalable_pids: list[int] = []
    for proc in processes:
        pid = managed_process_pid(proc)
        # Only signal a process we can confirm is still ours; a recycled PID
        # (token mismatch) belongs to someone else and must not be killed.
        if not _is_managed_proc_alive(proc):
            continue
        try:
            os.kill(pid, 15)
            signaled_pids.append(pid)
        except ProcessLookupError:
            continue  # exited between the liveness check and the signal
        except OSError:
            # EPERM: alive, just not ours to signal (sudo, a service account, a
            # user namespace) -- the same reasoning as is_pid_running. Folding
            # this into "already gone" made stop print "cleared stale state"
            # and delete the state files of workers that kept indexing.
            unsignalable_pids.append(pid)

    # Grace period before --force is required. The worker answers SIGTERM only
    # between embedding requests, and a request is now claim-sized when nobody
    # is searching (embed_submit_size): 32 inputs measured 12.2 s against the
    # local llama-server, plus the write-back transaction. At the old 10 s this
    # timed out on an ordinary busy worker, and because `stop` then exits
    # non-zero, the documented `cementic stop && cementic start` restart
    # silently never started anything -- observed live on the papers import.
    timeout_seconds = 30.0
    remaining = wait_for_exit(signaled_pids, timeout_seconds=timeout_seconds)
    # SIGTERM never reached the EPERM'd pids, so they are certainly still
    # running: they rejoin the not-stopped set so no path below clears their
    # state, and --force reports them as unkillable instead of stale.
    remaining += unsignalable_pids

    if not remaining:
        _get_supervisor_state_path().unlink(missing_ok=True)
        _clear_worker_state_files()
        if signaled_pids:
            console.print(f"stopped {len(signaled_pids)} process(es)")
        else:
            # Recorded processes had already exited; "stopped 0 process(es)" read
            # as though a stop had happened and hid that they died on their own.
            console.print("no running processes found; cleared stale state")
        return

    if force:
        activity = _pipeline_worker_activity()
        if activity is not None:
            console.print(f"discarding in-progress work: {activity}")
        # force_kill returns the pids it could NOT kill; the old name `killed`
        # read as the opposite.
        unkilled = force_kill(remaining)
        time.sleep(0.5)
        still_alive = [pid for pid in unkilled if is_pid_running(pid)]
        if not still_alive:
            _get_supervisor_state_path().unlink(missing_ok=True)
            _clear_worker_state_files()
        if still_alive:
            # The supervisor record is deliberately kept: it is the only place
            # the collection and watched directories are written down, and
            # discarding it while the workers are still indexing left the next
            # `stop` to rediscover them from worker state files with no idea
            # what they were watching.
            err_console.print(
                f"force killed {len(remaining) - len(still_alive)} process(es); "
                f"could not kill: {', '.join(str(pid) for pid in still_alive)}"
            )
            # Exiting 0 here told `cementic stop && cementic start` that the
            # workers were gone; the start then refused with "already running".
            raise typer.Exit(1)
        console.print(f"force stopped {len(remaining)} process(es)")
        return

    still_running = [proc for proc in processes if managed_process_pid(proc) in remaining]
    stopped_pids = [pid for pid in signaled_pids if pid not in remaining]
    _save_supervisor_state(
        {
            "collection": state.get("collection"),
            "directories": state.get("directories", []),
            "processes": still_running,
        }
    )

    timed_out = [pid for pid in remaining if pid not in unsignalable_pids]
    if stopped_pids:
        # A mixed outcome used to report only the failure, so a stop that took
        # down one of two workers read as having done nothing.
        err_console.print(f"stopped {len(stopped_pids)} process(es)")
    if timed_out:
        err_console.print(
            f"stop timed out after {timeout_seconds}s; "
            f"still running PID(s): {', '.join(str(pid) for pid in timed_out)}"
        )
    if unsignalable_pids:
        # No timeout elapsed for these -- the signal itself was refused.
        err_console.print(
            f"could not signal PID(s) {', '.join(str(pid) for pid in unsignalable_pids)}: "
            "permission denied; still running, likely started by another user"
        )
    activity = _pipeline_worker_activity()
    if activity is not None:
        # Otherwise this reads as a hung worker. It is not: the worker cannot
        # answer SIGTERM from inside CREATE INDEX, and the statement is not
        # resumable, so forcing now throws the whole build away.
        err_console.print(f"the pipeline worker is {activity}, which does not stop on request")
        err_console.print("--force will discard that work; it restarts from scratch next run")
    elif timed_out:
        # Only a timed-out process can be helped by --force. Printing this when
        # every remaining PID was EPERM-unsignalable sent the user at a retry
        # that is guaranteed to fail the same way, for the same reason.
        err_console.print("use --force to kill stubborn processes")
    # Same reason as the force path above: nothing was stopped, so a caller
    # chaining on success must not proceed.
    raise typer.Exit(1)


@embedding_app.command("start", short_help="Start embedding runtime")
def start_embedding_runtime() -> None:
    """Start the configured embedding runtime service."""
    config = _get_config()
    if config.pipeline.embedding_provider != "llama-cpp":
        err_console.print(
            f"embedding start failed: unsupported provider "
            f"{config.pipeline.embedding_provider}"
        )
        raise typer.Exit(1)
    try:
        Bootstrapper(config).ensure_embedding_runtime()
    except RuntimeError as error:
        err_console.print(f"embedding start failed: {error}")
        raise typer.Exit(1)
    try:
        client = get_llama_cpp_runtime_client(config=config, autostart=True)
    except Exception as error:
        err_console.print(f"embedding start failed: {error}")
        raise typer.Exit(1)
    console.print("embedding: running")
    # Report the probed dimension, not the configured fallback: this command's
    # whole job is to confirm what is actually loaded.
    try:
        dimension: object = client.describe().embedding_dim
    except Exception as error:
        dimension = f"unknown ({error})"
    console.print(f"provider: llama-cpp, dim={dimension}")
    # Asymmetric models need task prefixes and the policy is chosen from the
    # model filename, so a renamed file silently disables it and quietly
    # degrades retrieval. Showing the choice makes that visible.
    console.print(f"text policy: {describe_text_policy(config.llama_cpp.model_path)}")


@embedding_app.command("stop", short_help="Stop embedding runtime")
def stop_embedding_runtime() -> None:
    """Stop the configured embedding runtime service."""
    config = _get_config()
    try:
        stopped = stop_llama_cpp_runtime(config)
    except RuntimeError as error:
        # The daemon is still up. Saying "stopped" here used to come with
        # discarding its pid file, so nothing could find it again.
        err_console.print(f"embedding stop failed: {error}")
        raise typer.Exit(1)
    console.print("embedding: stopped" if stopped else "embedding: already stopped")


@embedding_app.command("status", short_help="Show embedding runtime status")
def embedding_runtime_status() -> None:
    """Show configured embedding runtime service status."""
    console.print(f"embedding: {_llama_daemon_runtime_status()}")


@app.command(
    short_help="Semantic search over indexed documents",
    epilog=_SEARCH_EPILOG,
    no_args_is_help=True,
)
def search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(
        10, "-n", "--top-k", "--limit", min=1, max=MAX_SEARCH_RESULTS, help="Number of results"
    ),
    collections: list[str] | None = typer.Option(
        None,
        "-c",
        "--collection",
        help="Filter collections; supports '-c work personal' or repeated '-c'.",
    ),
    trailing_collections: list[str] | None = typer.Argument(
        None,
        help="Additional collections after --collection/-c",
    ),
    scores: bool = typer.Option(
        False,
        "--scores",
        help="Always show the relevance score, even when it does not explain the order",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output results as JSONL (one JSON object per line)",
    ),
) -> None:
    """Search indexed documents."""
    filters = _build_collection_filters(collections, trailing_collections)
    if filters is not None:
        filters = [_validated_collection_name(c) for c in filters]

    searcher = Searcher(_get_config())

    try:
        results = searcher.search(query, top_k=top_k, collections=filters)

        # Distinguish "no matches" from "that collection isn't indexed" — the two
        # are otherwise identical (empty output, exit 0), in both output modes.
        # Checked whenever collections were named, not only when nothing matched:
        # gating on an empty result set meant `-c work -c persnal` said nothing
        # about the typo as long as `work` returned a hit, so half the query was
        # dropped invisibly — and in --json mode the stream was well-formed and
        # the exit code 0, leaving a script no way to notice.
        unknown = searcher.unsearchable_collections(filters) if filters else []
        # "No indexed revision" was said for a name cementic has never heard of
        # as well as for a real collection still building, so a typo read as a
        # timing problem. Split them: the two need different actions.
        unindexed = _known_collection_names(unknown) if unknown else set()
        message = _unsearchable_message(unknown, unindexed)

        if json_output:
            for result in results:
                typer.echo(json.dumps(result))
            if unknown:
                # stdout is the JSONL stream; diagnostics go to stderr, and the
                # exit code has to distinguish this from a genuine no-match.
                err_console.print(message)
                raise typer.Exit(1)
            return

        if not results and not unknown:
            # Only when the query genuinely matched nothing: leading with "no
            # results" for a typo'd collection buried the actual answer under a
            # sentence that said the search worked.
            console.print("no results")
        elif results:
            rank_w = len(str(len(results)))
            # The score column is printed only while it still explains the
            # order. After fusion the list is ordered by summed reciprocal rank
            # while each result carries its own arm's score, so the numbers go
            # up and down the page and the listing reads as mis-sorted. Every
            # human-facing search tool surveyed (ripgrep, fzf, Recoll,
            # Spotlight, Google) shows no score at all; `--scores` is Recoll's
            # off-by-default escape hatch for when you want them anyway.
            show_scores = scores or scores_explain_order([r["score"] for r in results])
            for i, result in enumerate(results, 1):
                preview = " ".join(result["content"].split())
                score_column = f"{result['score']:.3f}  " if show_scores else ""
                console.print(
                    f"{i:>{rank_w}}. {score_column}{escape(result['source_path'])}"
                )
                # One line per hit. Truncation is for a terminal, where a long
                # preview would wrap and bury the ranking; off a TTY the width
                # is a fixed 80-column fallback that cut previews mid-sentence
                # and broke substring greps over redirected output.
                if console.is_terminal:
                    console.print(f"   {escape(preview)}", no_wrap=True, overflow="ellipsis")
                else:
                    console.print(f"   {escape(preview)}")

        if unknown:
            # Same condition, same exit code as the --json branch above: the two
            # modes used to disagree (0 here, 1 there) for identical input, so
            # whether a script could detect a typo'd collection depended on the
            # output format it happened to ask for.
            err_console.print(message)
            raise typer.Exit(1)

    except typer.Exit:
        # typer.Exit subclasses RuntimeError, so the broad handler below would
        # otherwise swallow a deliberate exit and report it as "search failed: 1".
        raise
    except Exception as e:
        if json_output:
            # Diagnostics must not land on stdout, which is the JSONL stream.
            if _is_database_unavailable(e):
                err_console.print(f"search failed: database unavailable: {e}")
            elif _is_schema_missing(e):
                err_console.print(f"search failed: {_NO_SCHEMA_HINT}")
            else:
                err_console.print(f"search failed: {e}")
        else:
            _report_db_error(e, "search")
        raise typer.Exit(1)


_FILTER_EPILOG = """\
\b
EXAMPLES:
  cementic extract paper.pdf
  cementic extract paper.pdf | cementic chunk | cementic embed
"""


@app.command(
    short_help="Extract a document to Markdown on stdout",
    epilog=_FILTER_EPILOG,
    no_args_is_help=True,
)
def extract(path: str = typer.Argument(..., help="Path to a document file")) -> None:
    """Extract one document to Markdown on stdout — no database, for piping/debugging."""
    cfg = _get_config()
    if path == "":
        # Path("") is Path("."), so the directory guard below reported an empty
        # subject: "extract failed:  is a directory, not a document file".
        err_console.print("extract failed: no path given")
        raise typer.Exit(1)
    if Path(path).is_dir():
        # Falling through said "no extractor for '(none)'" -- technically the
        # registry's answer for a suffixless path, but nonsense as a message.
        err_console.print(f"extract failed: {path} is a directory, not a document file")
        raise typer.Exit(1)
    try:
        markdown = extract_document(path, cfg)
    except (OSError, ValueError, RuntimeError) as error:
        # OSError rather than FileNotFoundError: a directory named `notes.md`
        # raises IsADirectoryError, an unreadable file PermissionError, and a
        # symlink loop ELOOP -- all ordinary inputs that produced a traceback.
        err_console.print(f"extract failed: {error}")
        raise typer.Exit(1)
    typer.echo(markdown)


@app.command(short_help="Chunk text from a file or stdin to JSONL on stdout")
def chunk(
    path: str | None = typer.Argument(None, help="Text file to chunk (default: stdin)"),
    chunk_size: int | None = typer.Option(
        None, "--chunk-size", min=1, help="Tokens per chunk (default: config value)"
    ),
    chunk_overlap: int | None = typer.Option(
        None,
        "--chunk-overlap",
        min=0,
        help="Overlap tokens between chunks (default: config value)",
    ),
) -> None:
    """Chunk text into JSONL on stdout (one object per line) — no database.

    Reads from PATH or stdin, so it pipes after `cementic extract`.
    """
    cfg = _get_config()
    try:
        # An empty PATH means "read stdin" when stdin is a pipe -- an unset
        # shell variable in `... | cementic chunk "$MAYBE_PATH"` is the ordinary
        # way to get one, and treating it as a path made that exit 1 with
        # `Is a directory: '.'`, naming a path the user never typed. Only when
        # stdin is a terminal is it a mistake, and then it is reported rather
        # than sat on, which is what looked hung.
        if path == "" and _stdin_is_a_terminal():
            err_console.print("chunk failed: no PATH given and stdin is a terminal")
            raise typer.Exit(1)
        read_stdin = path is None or path == ""
        text = sys.stdin.read() if read_stdin else Path(str(path)).read_text(encoding="utf-8")
    except (OSError, ValueError) as error:
        # ValueError covers UnicodeDecodeError: `chunk` takes text, and a binary
        # file should be a one-line error, not a traceback.
        err_console.print(f"chunk failed: {error}")
        raise typer.Exit(1)
    size = chunk_size if chunk_size is not None else cfg.pipeline.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else cfg.pipeline.chunk_overlap
    try:
        pieces = chunk_text(text, chunk_size=size, chunk_overlap=overlap)
    except ValueError as error:
        # Reachable from a plausible invocation: passing only --chunk-size leaves
        # the overlap at its (larger) configured default, so say where each value
        # came from rather than printing a traceback.
        err_console.print(f"chunk failed: {error} (chunk_size={size}, chunk_overlap={overlap})")
        if chunk_overlap is None:
            err_console.print(
                "hint: --chunk-overlap defaults to the config value; pass it explicitly"
            )
        raise typer.Exit(1)
    for piece in pieces:
        typer.echo(json.dumps({"index": piece.chunk_index, "content": piece.content}))


@app.command(short_help="Embed chunk JSONL from stdin to JSONL on stdout")
def embed() -> None:
    """Embed chunk JSONL from stdin, adding an "embedding" field to each line.

    Reads the JSONL produced by `cementic chunk` (objects with a "content" field);
    needs the embedding model/runtime.
    """
    cfg = _get_config()
    # Parsed with real stdin line numbers: blank lines used to shift every
    # reported number, so "line 2" could point at a perfectly good record.
    numbered: list[tuple[int, Any]] = []
    try:
        for stdin_line_number, line in enumerate(sys.stdin, 1):
            if not line.strip():
                continue
            try:
                numbered.append((stdin_line_number, json.loads(line)))
            except json.JSONDecodeError as error:
                err_console.print(
                    f"embed failed: invalid JSON on stdin line {stdin_line_number}: {error}"
                )
                raise typer.Exit(1)
    except UnicodeDecodeError as error:
        # Binary stdin surfaced as a raw traceback; `chunk` already handles it.
        err_console.print(f"embed failed: stdin is not text: {error}")
        raise typer.Exit(1)
    if not numbered:
        return
    records = [rec for _, rec in numbered]
    # Validate the whole input before embedding any of it, so a malformed line
    # is not reported only after some output has already been written.
    contents: list[str] = []
    for line_number, rec in numbered:
        if not isinstance(rec, dict):
            err_console.print(
                f"embed failed: line {line_number} is a JSON {type(rec).__name__}, "
                "not an object with a \"content\" field"
            )
            raise typer.Exit(1)
        content = rec.get("content")
        # str(rec.get("content", "")) used to turn a missing field into the
        # empty string and a null into the literal "None" -- both of which
        # embed happily into a plausible-looking vector for text that was never
        # there.
        if not isinstance(content, str):
            missing = "is missing" if "content" not in rec else f"is {json.dumps(content)}"
            err_console.print(
                f"embed failed: line {line_number} has no text to embed: \"content\" {missing}"
            )
            raise typer.Exit(1)
        if not content.strip():
            # The same hole as missing/null, one layer down: empty and
            # whitespace-only strings embed into a plausible-looking vector for
            # text that was never there. The pipeline worker never embeds such
            # chunks either.
            err_console.print(
                f"embed failed: line {line_number} has no text to embed: "
                "\"content\" is empty or whitespace"
            )
            raise typer.Exit(1)
        contents.append(content)

    batch_size = cfg.pipeline_worker.batch_size
    try:
        provider = create_provider(runtime_spec_from_config(cfg), cfg)
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            texts = [
                provider.format_document(content)
                for content in contents[start : start + batch_size]
            ]
            vectors = provider.embed_batch(texts)
            # strict: a provider returning the wrong count must be an error,
            # not records silently dropped from the output stream.
            for offset, (rec, vector) in enumerate(zip(batch, vectors, strict=True)):
                if vector is None:
                    # embed_batch reports per-item failure as None. Emitting
                    # "embedding": null would look like a successful record.
                    reason = getattr(provider, "over_budget_reason", lambda _text: None)(
                        texts[offset]
                    )
                    # The real stdin line, like every other error above it:
                    # `start + offset + 1` counted records, so blank lines in
                    # the input shifted it -- the exact mislocation the parse
                    # and validation errors were fixed to stop reporting.
                    err_console.print(
                        f"embed failed: line {numbered[start + offset][0]} "
                        "could not be embedded" + (f": {reason}" if reason else "")
                    )
                    raise typer.Exit(1)
                typer.echo(json.dumps({**rec, "embedding": vector}))
    except typer.Exit:
        raise
    except Exception as error:
        err_console.print(f"embed failed: {error}")
        raise typer.Exit(1)


def main() -> None:
    """Entry point — Typer handles help (-h/--help), --version, and dispatch."""
    app()


if __name__ == "__main__":
    main()
