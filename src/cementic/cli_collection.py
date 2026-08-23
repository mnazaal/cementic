"""`cementic collection` subcommands: remove, list, promote, reindex, revisions."""

import typer

from cementic import render
from cementic.cli_format import CementicTyper
from cementic.cli_shared import (
    _db_session,
    _get_config,
    _is_managed_proc_alive,
    _load_supervisor_state,
    _reporting_db_errors,
    _require_known_collection,
    _supervisor_processes,
    _validated_collection_name,
    console,
    err_console,
)
from cementic.collections import (
    delete_collection_records,
    drop_orphan_vector_tables,
    list_collection_revisions,
    list_collections,
    promote_ready_revision,
    reindex_collection,
    remove_artifacts,
)
from cementic.db import get_engine, get_session_factory

collection_app = CementicTyper(help="Inspect and manage collections")


@collection_app.callback(invoke_without_command=True)
def collection_callback(ctx: typer.Context) -> None:
    """Show collection subcommand help when no subcommand is provided."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


@collection_app.command(
    "remove",
    short_help="Delete a collection and its artifacts",
    no_args_is_help=True,
)
def remove_collection(
    collection: str = typer.Argument(..., help="Collection name to delete"),
    force: bool = typer.Option(False, "--force", help="Skip confirmation prompt"),
) -> None:
    """Delete all documents and chunks belonging to a collection."""
    collection = _validated_collection_name(collection)

    if not force:
        confirm = typer.confirm(f"Delete collection '{collection}' and all associated chunks?")
        if not confirm:
            raise typer.Abort()

    with _reporting_db_errors(f"collection remove '{collection}'"):
        engine = get_engine(_get_config().database.url)
        session_factory = get_session_factory(engine)

        with session_factory() as session:
            result = delete_collection_records(session, collection)
            if result is None:
                console.print(f"collection: {collection}")
                console.print("status: not found")
                return

    # The rows are committed by here, so the collection *is* deleted. Leftover
    # artifacts/vector tables are reported as a warning rather than turning a
    # successful delete into a reported failure.
    console.print(f"collection: {collection}")
    console.print("status: deleted")
    console.print(f"documents: {result.deleted_docs}")
    console.print(f"chunks: {result.deleted_chunks}")
    # A running watcher re-registers the files it watches and resurrects the
    # collection; the pipeline worker notices the deleted revision and exits on
    # its next poll. Deleting is still allowed -- the rows cascade safely --
    # but silently racing the watcher is not.
    supervisor_state = _load_supervisor_state()
    if supervisor_state.get("collection") == collection and any(
        _is_managed_proc_alive(proc) for proc in _supervisor_processes(supervisor_state)
    ):
        console.print(
            "warning: background workers are still watching this collection; "
            "the watcher will re-register its files -- run `cementic stop` to stop them"
        )
    try:
        unremoved = remove_artifacts(result.artifact_paths, config=_get_config())
        drop_orphan_vector_tables(engine, result.vector_profile_ids)
    except Exception as e:
        # The delete is already committed, so this is a warning about leftovers
        # on disk, not a failed removal. Exiting non-zero here contradicted both
        # the comment above and the documented behaviour, and told scripts the
        # collection had not been removed when it had.
        console.print(f"warning: collection deleted but cleanup failed: {e}")
        return
    _print_unremoved_artifacts(unremoved)
    console.print(f"vector_tables_dropped: {len(result.vector_profile_ids)}")


@collection_app.command("list", short_help="List known collections")
def list_collection_command() -> None:
    """Show known collections."""
    with _reporting_db_errors("collection list"):
        with _db_session() as session:
            rows = list_collections(session)

        console.print("collections")
        if not rows:
            console.print("  (none)")
            return

        name_w = max(len(row.name) for row in rows)
        doc_w = max(len(f"{row.documents:,}") for row in rows)
        for row in rows:
            in_flight = render._in_flight_revision_text(
                row.ready_revision_label, row.building_revision_label
            )
            console.print(
                f"  {row.name:<{name_w}}   {row.documents:>{doc_w},} docs   "
                f"active={row.active_revision_label or '-'}  {in_flight}"
            )


@collection_app.command(
    "promote",
    short_help="Promote a collection's ready revision to active",
    no_args_is_help=True,
)
def promote_collection(
    collection: str = typer.Argument(..., help="Collection name to promote"),
    force: bool = typer.Option(
        False,
        "-f",
        "--force",
        help="Promote even if the ready revision built with failed documents or chunks",
    ),
) -> None:
    """Promote the ready pipeline revision for one collection."""
    collection = _validated_collection_name(collection)

    with _reporting_db_errors("collection promote"):
        with _db_session() as session:
            _require_known_collection(session, collection)
            outcome = promote_ready_revision(
                session, collection, config=_get_config(), force=force
            )
            # Read everything we need while the session is open. Not because
            # attributes expire on commit -- the session factory sets
            # expire_on_commit=False -- but because a lazy load after the
            # session closes has no connection to load through.
            status = outcome.status
            counts = outcome.counts
            unremoved = outcome.unremoved_artifacts
            cleanup_error = outcome.cleanup_error
            revision_label = (
                outcome.revision.label or outcome.revision.id
                if outcome.revision is not None
                else None
            )

    if status != "promoted":
        # A refusal is an error: exit 1 with nothing on stdout, per the stream
        # convention in the README. These lines used to go to stdout, so
        # `promote 2>errors.log || cat errors.log` printed nothing at all.
        err_console.print(f"collection: {collection}")
    else:
        console.print(f"collection: {collection}")
    if status == "no_ready":
        err_console.print("status: no ready revision")
        err_console.print("`cementic status -c` shows whether a build is still in progress")
        # Exit 1 like every other promote that promoted nothing: this was the
        # one no-op outcome that exited 0, so a script chaining
        # `promote && search` proceeded as if a revision had been published.
        raise typer.Exit(1)
    if status == "lost_race":
        err_console.print("status: another promote activated a revision first")
        err_console.print(
            "nothing was changed by this command; `cementic collection list` shows "
            "which revision is active now"
        )
        raise typer.Exit(1)
    if status == "empty":
        err_console.print("status: nothing to promote (revision has no documents)")
        err_console.print(
            "promoting would retire the active revision and leave nothing searchable"
        )
        raise typer.Exit(1)
    if status == "incomplete":
        pending = []
        if counts is not None:
            not_extracted = counts.documents - counts.extracted_done - counts.extracted_failed
            not_chunked = counts.extracted_done - counts.chunked_done - counts.chunked_failed
            not_embedded = counts.total_chunks - counts.done_embeddings - counts.failed_embeddings
            if not_extracted > 0:
                pending.append(f"extract={not_extracted}")
            if not_chunked > 0:
                pending.append(f"chunk={not_chunked}")
            if not_embedded > 0:
                pending.append(f"embed={not_embedded}")
        err_console.print(f"status: incomplete ({', '.join(pending)} pending)")
        err_console.print(
            "the revision took on new work after it was marked ready; "
            "wait for `cementic status` to show it finished, or --force to publish it as-is"
        )
        raise typer.Exit(1)
    if status == "blocked_by_failures":
        parts = []
        if counts is not None:
            if counts.extracted_failed:
                parts.append(f"extract={counts.extracted_failed}")
            if counts.chunked_failed:
                parts.append(f"chunk={counts.chunked_failed}")
            if counts.failed_embeddings:
                parts.append(f"embed={counts.failed_embeddings}")
        err_console.print(f"status: blocked ({', '.join(parts)} failed)")
        err_console.print("re-run with --force to promote anyway")
        raise typer.Exit(1)
    console.print("status: promoted")
    console.print(f"revision: {revision_label}")
    # Same contract as `collection remove`: the promote is committed, so
    # leftover files are a warning, not a failure -- but they used to be
    # discarded entirely here while remove reported them.
    if cleanup_error is not None:
        console.print(f"warning: promoted but cleanup failed: {cleanup_error}")
    _print_unremoved_artifacts(unremoved)


def _print_unremoved_artifacts(unremoved: list[str]) -> None:
    """Warn about leftover artifact files, truncated to the first five.

    Shared by `remove` and `promote`: both commands already committed the
    database change by the time this runs, so a leftover file on disk is a
    warning, not a reason to report either command as failed.
    """
    if not unremoved:
        return
    console.print(f"warning: {len(unremoved)} artifact file(s) could not be removed:")
    for path in unremoved[:5]:
        console.print(f"  {path}")
    if len(unremoved) > 5:
        console.print(f"  ... and {len(unremoved) - 5} more")


@collection_app.command(
    "reindex",
    short_help="Rebuild a collection's ANN index from current index config",
    no_args_is_help=True,
)
def reindex_collection_command(
    collection: str = typer.Argument(..., help="Collection name to reindex"),
    force: bool = typer.Option(
        False,
        "-f",
        "--force",
        help="Rebuild even if the index method is unchanged (picks up hnsw_m and "
        "ef_construction, which are fixed at build time)",
    ),
) -> None:
    """Reconcile the active revision's ANN index with the current `[index]` config.

    The index is built once, when a revision first completes, so editing
    `index.method` afterwards otherwise had no effect and no way to ask for one.
    """
    collection = _validated_collection_name(collection)

    with _reporting_db_errors("collection reindex"):
        with _db_session() as session:
            # Before the "this can take several minutes" line, so an unknown
            # collection does not first announce work that will never start.
            _require_known_collection(session, collection)
            console.print(f"collection: {collection}")
            console.print(
                "building the index — this can take several minutes on a large corpus"
            )
            outcome = reindex_collection(session, collection, config=_get_config(), force=force)

    if outcome.status == "no_active":
        err_console.print("status: no active revision — nothing has been promoted yet")
        raise typer.Exit(1)
    if outcome.status == "no_vectors":
        console.print("status: no vectors to index")
        return
    if outcome.previous_method is None:
        console.print(f"status: built ({outcome.method})")
    elif outcome.previous_method == outcome.method:
        console.print(
            f"status: rebuilt ({outcome.method})" if force else "status: unchanged"
        )
        if not force:
            console.print(
                f"the index is already {outcome.method}; --force rebuilds it anyway"
            )
    else:
        console.print(f"status: rebuilt ({outcome.previous_method} -> {outcome.method})")


@collection_app.command(
    "revisions",
    short_help="Show a collection's revision history",
    no_args_is_help=True,
)
def list_collection_revision_command(
    collection: str = typer.Argument(..., help="Collection name to inspect"),
) -> None:
    """Show revision history for one collection."""
    collection = _validated_collection_name(collection)

    with _reporting_db_errors("collection revisions"):
        with _db_session() as session:
            _require_known_collection(session, collection)
            rows = list_collection_revisions(session, collection)
            console.print(f"{'collection':<11} {collection}")
            console.print("revisions")
            if not rows:
                console.print("  (none)")
                return

            id_w = max(len(str(row.id)) for row in rows)
            status_w = max(len(row.status) for row in rows)
            label_w = max(len(row.label or "-") for row in rows)
            for row in rows:
                console.print(
                    f"  {row.id:>{id_w}}  {row.status:<{status_w}}  "
                    f"{(row.label or '-'):<{label_w}}  "
                    f"extract={row.extractor_profile.name} "
                    f"chunk={row.chunk_profile.fingerprint[:8]} "
                    f"embed={row.embedding_profile.provider}:{row.embedding_profile.fingerprint[:8]}"
                )
