"""Rendering `cementic status`, `cementic doctor`, and `--json` output.

Split out of cli.py. `_print_status_summary` and `_print_collection_detail`
used to reach for the config singleton themselves, and `_print_status_json`
opened its own database session and ran queries -- a command implementation
wearing a printer's name. Moving them here forces those dependencies into
parameters: `config` is passed in, and the status-document builder below takes
an `open_session` callable (and an `is_schema_missing` classifier) instead of
importing cli.py's `_db_session`/`_is_schema_missing` and risking an import
cycle back to the module that imports this one.
"""

import json
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import asdict
from typing import Any

import typer
from rich.console import Console
from rich.markup import escape

from cementic.collections import collection_exists, list_collections
from cementic.config import Config
from cementic.status_service import (
    load_file_progress,
    load_pipeline_status,
    load_pipeline_status_bulk,
)

# soft_wrap: off a TTY rich falls back to an 80-column hard wrap, which split
# paths and aligned rows mid-word in piped or redirected output. Line breaking
# belongs to the terminal or the consuming program, not to us.
console = Console(soft_wrap=True)
# Errors and diagnostics go here so a failure never pollutes the data on
# stdout (which would otherwise be piped on as content, or break `--json | jq`).
err_console = Console(stderr=True, soft_wrap=True)


def _state(ok: bool, ok_word: str, bad_word: str) -> str:
    """A status word, colored sparingly (rich drops color off-TTY / NO_COLOR)."""
    return f"[green]{ok_word}[/green]" if ok else f"[red]{bad_word}[/red]"


def _print_database_unavailable(action: str, hint: str) -> None:
    """Print a concise database-unavailable message with a recovery hint."""
    err_console.print(f"{action}: database not reachable")
    err_console.print(hint)


def _print_doctor_report(report: dict[str, Any]) -> None:
    """Print read-only doctor diagnostics in a compact human format."""
    doctor_status = "[green]ok[/green]" if report["ok"] else "[red]failed[/red]"
    console.print(f"cementic doctor: {doctor_status}")
    checks = report["checks"]
    for name, payload in checks.items():
        if name == "extensions":
            console.print("extensions:")
            for extension, extension_payload in payload.items():
                console.print(
                    f"  - {extension}: {extension_payload['status']} "
                    f"({extension_payload['message']})"
                )
            continue
        status_text = payload.get("status", "unknown")
        message = payload.get("message")
        console.print(f"{name}: {status_text}" + (f" — {message}" if message else ""))


def _print_status_summary(
    supervisor_collection: str,
    directories: list[str],
    source_watcher_status: Any,
    pipeline_worker_status: Any,
    health: Any,
    verbose: bool,
    config: Config,
) -> None:
    """Print the concise worker + health summary; full detail behind --verbose."""
    source_running = source_watcher_status.process == "running"
    pipeline_running = pipeline_worker_status.process == "running"
    running = int(source_running) + int(pipeline_running)
    if running == 2:
        workers = "[green]running[/green]"
    elif running == 0:
        workers = "[red]stopped[/red]"
    else:
        workers = "[yellow]partial[/yellow]"
    console.print(f"{'workers':<11} {workers}")
    if health is None:
        # Say the rows are missing rather than just omitting them: a summary two
        # rows short reads as a complete report of a healthy system, and the
        # explanation on stderr is lost to `2>/dev/null`.
        console.print(f"{'database':<11} [red]unknown (health check failed)[/red]")
        console.print(f"{'embedding':<11} [red]unknown (health check failed)[/red]")
    else:
        console.print(f"{'database':<11} {_state(health.db_reachable, 'reachable', 'unreachable')}")
        if health.embedding_healthy:
            embedding_text = "[green]healthy[/green]"
        elif config.llama_cpp.daemon_autostart:
            # Same state `cementic doctor` calls a warning: not running now, but
            # cementic starts it on demand. Not an error.
            embedding_text = "[yellow]stopped (autostarts when needed)[/yellow]"
        else:
            embedding_text = "[red]unhealthy[/red]"
        console.print(f"{'embedding':<11} {embedding_text}")

    # A worker looping on a permanent failure otherwise looks exactly like a
    # healthy idle one, so this is headline information rather than --verbose
    # detail: without it the only evidence is a log file the user must know about.
    for label, worker in (
        ("source watcher", source_watcher_status),
        ("pipeline worker", pipeline_worker_status),
    ):
        if worker.last_error:
            console.print(
                f"{'last error':<11} [red]{escape(f'{label}: {worker.last_error}')}[/red]"
            )

    # Headline, not --verbose detail: a skipped file never becomes a document,
    # so it is absent from every pipeline count. Without this a collection whose
    # watcher dropped a directory of symlinks still reported 100% complete and
    # promoted cleanly, with the only evidence a number behind --verbose and a
    # log file the user is never pointed at.
    if source_watcher_status.failed_count:
        console.print(
            f"{'skipped':<11} [yellow]{source_watcher_status.failed_count} file(s) not "
            "indexed[/yellow]" + ("" if verbose else " (run with --verbose for paths)")
        )

    if not verbose:
        return

    console.print()
    console.print(f"session collection: {supervisor_collection}")
    if directories:
        console.print("directories:")
        for directory in directories:
            console.print(f"- {directory}")
    console.print(
        f"source watcher: {source_watcher_status.process}, "
        f"state={source_watcher_status.state}, pid={source_watcher_status.pid}, "
        f"processed={source_watcher_status.processed_count}, "
        f"failed={source_watcher_status.failed_count}"
    )
    if source_watcher_status.current_file is not None:
        console.print(f"  current file: {source_watcher_status.current_file}")
    if source_watcher_status.skipped_files:
        console.print("  skipped files:")
        for entry in source_watcher_status.skipped_files:
            console.print(f"  - {escape(entry)}")
        recorded = len(source_watcher_status.skipped_files)
        if source_watcher_status.failed_count > recorded:
            console.print(
                f"  (showing the {recorded} most recent of "
                f"{source_watcher_status.failed_count})"
            )
    console.print(
        f"pipeline worker: {pipeline_worker_status.process}, "
        f"state={pipeline_worker_status.state}, pid={pipeline_worker_status.pid}"
    )
    if pipeline_worker_status.current_file is not None:
        console.print(f"  current file: {pipeline_worker_status.current_file}")
    if pipeline_worker_status.current_activity:
        console.print(f"  activity: {pipeline_worker_status.current_activity}")
        console.print("  (no other work happens until this finishes)")
    if health is not None and health.llama_daemon != "N/A":
        console.print(f"embedding daemon: {health.llama_daemon}")


def _in_flight_revision_text(ready_label: str | None, building_label: str | None) -> str:
    """Render the not-yet-active revision under the status it is actually in.

    (Both summary builders used to bucket ready and building together, so a
    finished revision was reported as `building=...` -- hiding the one fact the
    promote workflow turns on, that there is something ready to promote.)
    """
    # Labels are interpolated into a markup-enabled string, so escape them: a
    # label containing a closing tag would raise MarkupError mid-render, and one
    # containing an opening tag would be swallowed.
    if ready_label:
        return f"[green]ready={escape(ready_label)}[/green]"
    return f"building={escape(building_label) if building_label else '-'}"


def _print_collection_detail(
    collection: str,
    pipeline_status: Any,
    verbose: bool,
    config: Config,
) -> None:
    """Print a concise per-collection summary; per-file detail behind --verbose."""
    ps = pipeline_status
    console.print(f"{'collection':<11} {collection}")
    console.print(f"{'documents':<11} {ps.documents:,}")
    console.print(
        f"{'extracted':<11} {ps.extracted_done:,}/{ps.documents:,} ({ps.extraction_pct}%)"
    )
    console.print(
        f"{'chunked':<11} {ps.chunked_done:,}/{ps.extracted_done:,} ({ps.chunking_pct}%)"
    )
    # The denominator is chunks that exist *so far*, so mid-build this can read
    # 100% while most documents have not been chunked yet. Say so rather than
    # implying the collection is finished.
    #
    # total_chunks is final only once extraction has finished *and* chunking has
    # caught up with it -- the same two clauses as the worker's own completeness
    # check (pipeline_worker.revision_is_complete), so status and the worker
    # cannot disagree about whether a collection is done. Comparing chunking to
    # `documents` instead meant one document that failed to extract could never
    # be chunked, pinning the caveat on a collection that was in fact finished;
    # comparing to extracted_done alone would drop the caveat mid-extraction,
    # while more chunks were still on the way.
    extraction_complete = ps.extracted_done + ps.extracted_failed >= ps.documents
    chunking_complete = extraction_complete and (
        ps.chunked_done + ps.chunked_failed >= ps.extracted_done
    )
    embedded_suffix = "" if chunking_complete else " of chunks created so far"
    console.print(
        f"{'embedded':<11} {ps.done_embeddings:,}/{ps.total_chunks:,} "
        f"({ps.embedding_pct}%{embedded_suffix})"
    )
    console.print(
        f"{'revision':<11} active={ps.active_revision_label or '-'}  "
        f"{_in_flight_revision_text(ps.ready_revision_label, ps.building_revision_label)}"
    )
    if ps.ready_revision_label:
        console.print(
            f"{'':<11} run `cementic collection promote {collection}` to serve it"
        )
    failures = []
    if ps.extracted_failed:
        failures.append(f"extract={ps.extracted_failed}")
    if ps.chunked_failed:
        failures.append(f"chunk={ps.chunked_failed}")
    if ps.failed_embeddings:
        failures.append(f"embed={ps.failed_embeddings}")
    if failures:
        console.print(f"{'failures':<11} {', '.join(failures)}")

    if verbose and collection:
        try:
            files = load_file_progress(config, collection)
            if not files:
                console.print("files: none")
                return
            console.print("files:")
            for f in files:
                status_line = (
                    f"- {f.source_path} | extract={f.extraction_status}"
                    f" | chunk={f.chunking_status}"
                    f" | embeddings={f.embeddings_done}/{f.embeddings_total}"
                )
                if f.embeddings_failed:
                    status_line += f" (failed: {f.embeddings_failed})"
                if f.error_message:
                    status_line += f" | error: {f.error_message[:80]}"
                console.print(status_line)
        except Exception as error:
            # This used to be swallowed at exit 0 (and the database-unavailable
            # case printed nothing at all), so a truncated report read as the
            # complete answer.
            err_console.print(f"files: could not be listed: {error}")
            raise typer.Exit(1)


def build_status_document(
    open_session: Callable[[], AbstractContextManager[Any]],
    is_schema_missing: Callable[[Exception], bool],
    no_schema_hint: str,
    config: Config,
    supervisor_status: Any,
    source_watcher_status: Any,
    pipeline_worker_status: Any,
    directories: list[str],
    health: Any,
    health_error: str | None,
    collection: str | None,
    verbose: bool,
) -> dict[str, Any]:
    """Build the `status --json` document.

    Takes a session opener and a schema-missing classifier as parameters
    rather than importing cli.py's `_db_session`/`_is_schema_missing`
    directly, which would import back into the module that imports this one.
    Print with `print_status_document`.
    """
    output: dict[str, Any] = {
        "supervisor": {
            "state": supervisor_status.state,
            "collection": supervisor_status.collection,
            "directories": directories,
        },
        "source_watcher": {
            "process": source_watcher_status.process,
            "state": source_watcher_status.state,
            "pid": source_watcher_status.pid,
            "processed": source_watcher_status.processed_count,
            "failed": source_watcher_status.failed_count,
            "skipped_files": source_watcher_status.skipped_files,
            "current_file": source_watcher_status.current_file,
            "last_error": source_watcher_status.last_error,
            "last_error_at": source_watcher_status.last_error_at,
        },
        "pipeline_worker": {
            "process": pipeline_worker_status.process,
            "state": pipeline_worker_status.state,
            "pid": pipeline_worker_status.pid,
            "current_file": pipeline_worker_status.current_file,
            "last_error": pipeline_worker_status.last_error,
            "last_error_at": pipeline_worker_status.last_error_at,
            "current_activity": pipeline_worker_status.current_activity,
        },
    }

    if health is not None:
        output["health"] = {
            "db_reachable": health.db_reachable,
            "embedding_provider": health.embedding_provider,
            "embedding_healthy": health.embedding_healthy,
            "llama_daemon": health.llama_daemon,
        }
    elif health_error is not None:
        # Same keys, nulled, plus the reason: replacing the whole object made
        # `d["health"]["db_reachable"]` raise KeyError for a consumer that had
        # no way to know the shape could change.
        output["health"] = {
            "db_reachable": None,
            "embedding_provider": None,
            "embedding_healthy": None,
            "llama_daemon": None,
            "error": health_error,
        }

    try:
        with open_session() as session:
            if collection is None:
                rows = list_collections(session)
                status_by_collection = load_pipeline_status_bulk(
                    config, [row.name for row in rows]
                )
                # asdict: PipelineStatus *is* the JSON contract, so a hand-kept
                # key list here (which existed twice, identically) is a copy
                # waiting to drift, not one that had already drifted.
                output["collections"] = {
                    row.name: asdict(status_by_collection[row.name]) for row in rows
                }
            else:
                # Mirror _require_known_collection on the human path: a typo'd
                # name otherwise produced a full zero-filled pipeline block at
                # exit 0, indistinguishable from a real collection not started.
                if not collection_exists(session, collection):
                    raise ValueError(
                        f"unknown collection: {collection}"
                        " (check `cementic collection list`)"
                    )
                ps = load_pipeline_status(config, collection)
                output["pipeline"] = {"collection": collection, **asdict(ps)}
                if verbose:
                    files = load_file_progress(config, collection)
                    output["files"] = [asdict(f) for f in files]
    except Exception as error:
        output["error"] = no_schema_hint if is_schema_missing(error) else str(error)

    return output


def print_status_document(document: dict[str, Any]) -> bool:
    """Print a status document as JSON. Returns whether it recorded a failure."""
    typer.echo(json.dumps(document, indent=2, default=str))
    return "error" in document
