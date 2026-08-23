"""`cementic collection` subcommands: remove, list, promote, reindex, revisions.

Split out of cli.py. These commands share `_db_session`, `_get_config`,
`_validated_collection_name`, `_reporting_db_errors`, `_require_known_collection`,
`_load_supervisor_state`, `_supervisor_processes`, `_is_managed_proc_alive`,
`console`, `err_console`, and (for `remove`/`list`/`promote`/`reindex`/`revisions`
specifically) `get_engine`, `get_session_factory`, `list_collections`,
`delete_collection_records`, `remove_artifacts`, `drop_orphan_vector_tables`,
`promote_ready_revision`, `reindex_collection` and `list_collection_revisions`
with the rest of the CLI.

Every one of those names is looked up as `cli.<name>` (an attribute read on the
`cementic.cli` module object) rather than imported by value, and cli.py keeps
its own top-level bindings for all of them -- including the ones its own code
no longer calls -- even though this is a module attribute cementic.cli's own
code doesn't read. Both are necessary for the same reason: the test suite
predates this split and still does `unittest.mock.patch("cementic.cli.<name>")`
against them, which only intercepts calls that resolve the name through the
`cementic.cli` module at call time, not calls against a copy this module
imported by value at load time. `from cementic.cli import <name>` would just
silently stop honouring those patches.

cli.py in turn imports `collection_app` and `collection_callback` from here;
that mutual dependency is why this imports the *module* (`from cementic import
cli`), never a name out of it, at import time -- a module reference needs
nothing from cli.py to exist yet, only `cli.<name>` attribute reads inside
function bodies do, and those run long after both modules have finished
loading.
"""

import typer

from cementic import cli, render
from cementic.cli_format import CementicTyper

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
    collection = cli._validated_collection_name(collection)

    if not force:
        confirm = typer.confirm(f"Delete collection '{collection}' and all associated chunks?")
        if not confirm:
            raise typer.Abort()

    with cli._reporting_db_errors(f"collection remove '{collection}'"):
        engine = cli.get_engine(cli._get_config().database.url)
        session_factory = cli.get_session_factory(engine)

        with session_factory() as session:
            result = cli.delete_collection_records(session, collection)
            if result is None:
                cli.console.print(f"collection: {collection}")
                cli.console.print("status: not found")
                return

    # The rows are committed by here, so the collection *is* deleted. Leftover
    # artifacts/vector tables are reported as a warning rather than turning a
    # successful delete into a reported failure.
    cli.console.print(f"collection: {collection}")
    cli.console.print("status: deleted")
    cli.console.print(f"documents: {result.deleted_docs}")
    cli.console.print(f"chunks: {result.deleted_chunks}")
    # A running watcher re-registers the files it watches and resurrects the
    # collection; the pipeline worker notices the deleted revision and exits on
    # its next poll. Deleting is still allowed -- the rows cascade safely --
    # but silently racing the watcher is not.
    supervisor_state = cli._load_supervisor_state()
    if supervisor_state.get("collection") == collection and any(
        cli._is_managed_proc_alive(proc) for proc in cli._supervisor_processes(supervisor_state)
    ):
        cli.console.print(
            "warning: background workers are still watching this collection; "
            "the watcher will re-register its files -- run `cementic stop` to stop them"
        )
    try:
        unremoved = cli.remove_artifacts(result.artifact_paths, config=cli._get_config())
        cli.drop_orphan_vector_tables(engine, result.vector_profile_ids)
    except Exception as e:
        # The delete is already committed, so this is a warning about leftovers
        # on disk, not a failed removal. Exiting non-zero here contradicted both
        # the comment above and the documented behaviour, and told scripts the
        # collection had not been removed when it had.
        cli.console.print(f"warning: collection deleted but cleanup failed: {e}")
        return
    if unremoved:
        # remove_artifacts has always returned the paths it could not remove;
        # both callers threw the list away, so files left behind were reported
        # only to a log file nobody is told about -- and the rows naming them
        # are gone, so nothing can find them again.
        cli.console.print(f"warning: {len(unremoved)} artifact file(s) could not be removed:")
        for path in unremoved[:5]:
            cli.console.print(f"  {path}")
        if len(unremoved) > 5:
            cli.console.print(f"  ... and {len(unremoved) - 5} more")
    cli.console.print(f"vector_tables_dropped: {len(result.vector_profile_ids)}")


@collection_app.command("list", short_help="List known collections")
def list_collection_command() -> None:
    """Show known collections."""
    with cli._reporting_db_errors("collection list"):
        with cli._db_session() as session:
            rows = cli.list_collections(session)

        cli.console.print("collections")
        if not rows:
            cli.console.print("  (none)")
            return

        name_w = max(len(row.name) for row in rows)
        doc_w = max(len(f"{row.documents:,}") for row in rows)
        for row in rows:
            in_flight = render._in_flight_revision_text(
                row.ready_revision_label, row.building_revision_label
            )
            cli.console.print(
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
    collection = cli._validated_collection_name(collection)

    with cli._reporting_db_errors("collection promote"):
        with cli._db_session() as session:
            cli._require_known_collection(session, collection)
            outcome = cli.promote_ready_revision(
                session, collection, config=cli._get_config(), force=force
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
        cli.err_console.print(f"collection: {collection}")
    else:
        cli.console.print(f"collection: {collection}")
    if status == "no_ready":
        cli.err_console.print("status: no ready revision")
        cli.err_console.print("`cementic status -c` shows whether a build is still in progress")
        # Exit 1 like every other promote that promoted nothing: this was the
        # one no-op outcome that exited 0, so a script chaining
        # `promote && search` proceeded as if a revision had been published.
        raise typer.Exit(1)
    if status == "lost_race":
        cli.err_console.print("status: another promote activated a revision first")
        cli.err_console.print(
            "nothing was changed by this command; `cementic collection list` shows "
            "which revision is active now"
        )
        raise typer.Exit(1)
    if status == "empty":
        cli.err_console.print("status: nothing to promote (revision has no documents)")
        cli.err_console.print(
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
        cli.err_console.print(f"status: incomplete ({', '.join(pending)} pending)")
        cli.err_console.print(
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
        cli.err_console.print(f"status: blocked ({', '.join(parts)} failed)")
        cli.err_console.print("re-run with --force to promote anyway")
        raise typer.Exit(1)
    cli.console.print("status: promoted")
    cli.console.print(f"revision: {revision_label}")
    # Same contract as `collection remove`: the promote is committed, so
    # leftover files are a warning, not a failure -- but they used to be
    # discarded entirely here while remove reported them.
    if cleanup_error is not None:
        cli.console.print(f"warning: promoted but cleanup failed: {cleanup_error}")
    if unremoved:
        cli.console.print(f"warning: {len(unremoved)} artifact file(s) could not be removed:")
        for path in unremoved[:5]:
            cli.console.print(f"  {path}")
        if len(unremoved) > 5:
            cli.console.print(f"  ... and {len(unremoved) - 5} more")


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
    collection = cli._validated_collection_name(collection)

    with cli._reporting_db_errors("collection reindex"):
        with cli._db_session() as session:
            # Before the "this can take several minutes" line, so an unknown
            # collection does not first announce work that will never start.
            cli._require_known_collection(session, collection)
            cli.console.print(f"collection: {collection}")
            cli.console.print(
                "building the index — this can take several minutes on a large corpus"
            )
            outcome = cli.reindex_collection(
                session, collection, config=cli._get_config(), force=force
            )

    if outcome.status == "no_active":
        cli.err_console.print("status: no active revision — nothing has been promoted yet")
        raise typer.Exit(1)
    if outcome.status == "no_vectors":
        cli.console.print("status: no vectors to index")
        return
    if outcome.previous_method is None:
        cli.console.print(f"status: built ({outcome.method})")
    elif outcome.previous_method == outcome.method:
        cli.console.print(
            f"status: rebuilt ({outcome.method})" if force else "status: unchanged"
        )
        if not force:
            cli.console.print(
                f"the index is already {outcome.method}; --force rebuilds it anyway"
            )
    else:
        cli.console.print(f"status: rebuilt ({outcome.previous_method} -> {outcome.method})")


@collection_app.command(
    "revisions",
    short_help="Show a collection's revision history",
    no_args_is_help=True,
)
def list_collection_revision_command(
    collection: str = typer.Argument(..., help="Collection name to inspect"),
) -> None:
    """Show revision history for one collection."""
    collection = cli._validated_collection_name(collection)

    with cli._reporting_db_errors("collection revisions"):
        with cli._db_session() as session:
            cli._require_known_collection(session, collection)
            rows = cli.list_collection_revisions(session, collection)
            cli.console.print(f"{'collection':<11} {collection}")
            cli.console.print("revisions")
            if not rows:
                cli.console.print("  (none)")
                return

            id_w = max(len(str(row.id)) for row in rows)
            status_w = max(len(row.status) for row in rows)
            label_w = max(len(row.label or "-") for row in rows)
            for row in rows:
                cli.console.print(
                    f"  {row.id:>{id_w}}  {row.status:<{status_w}}  "
                    f"{(row.label or '-'):<{label_w}}  "
                    f"extract={row.extractor_profile.name} "
                    f"chunk={row.chunk_profile.fingerprint[:8]} "
                    f"embed={row.embedding_profile.provider}:{row.embedding_profile.fingerprint[:8]}"
                )
