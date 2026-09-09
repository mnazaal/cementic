"""Document watcher that registers source files for downstream pipeline processing."""

from __future__ import annotations

import hashlib
import logging
import os
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from watchdog.events import DirMovedEvent, FileMovedEvent, FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from cementic.config import Config, get_config
from cementic.db import (
    Chunk,
    ChunkedDocument,
    ExtractedDocument,
    SourceDocument,
    create_tables,
    get_engine,
    get_session_factory,
)
from cementic.extract import supported_extensions
from cementic.state import DaemonState, StateManager
from cementic.supervisor import is_managed_process_alive, process_start_token
from cementic.worker_runtime import handle_shutdown_signal, report_fatal, setup_worker_logger

#: Minimum gap between "now working on X" state-file writes. Display only, so a
#: little staleness is fine; the alternative is one full read-modify-write per
#: file during a bulk scan.
_CURRENT_FILE_PUBLISH_INTERVAL_SECONDS = 0.5


def _is_missing(source_path: str) -> bool:
    """Whether a path is genuinely gone, as opposed to merely unreadable.

    ``Path.exists()`` cannot answer this: it suppresses only ENOENT-shaped
    errors and *raises* on a permission failure, so a chmod mishap or an
    NFS/automount hiccup took the whole watcher down mid-startup with a
    traceback. Anything other than "not found" means we do not know, and a
    document we cannot see is not evidence that the user deleted it -- marking
    it deleted drops it from search until some later run happens to rescan.
    """
    try:
        Path(source_path).stat()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return False


def _alternate_paths(file_path: str, aliases: list[tuple[Path, Path]]) -> list[str]:
    """The paths this file would have had under the roots as they were configured.

    A watched root given as a symlink is stored resolved, so every document
    under it is keyed on the real path. When the link is what moved -- the tree
    relocated and a link left behind -- the rows written before the move are
    keyed on the *old* real path, which is the configured path with the same
    tail. The watcher resolved that mapping itself, so it can name the old path
    outright rather than inferring it from the file's bytes, which is what a
    content hash does and what an edited file defeats. Pure; no filesystem.
    """
    path = Path(file_path)
    return [
        str(configured / path.relative_to(resolved))
        for configured, resolved in aliases
        if path.is_relative_to(resolved)
    ]


def _document_under_a_prior_root(
    session: Session, collection: str, alternate_paths: list[str]
) -> SourceDocument | None:
    """Find this file's document under a path a watched root used to have.

    A path lookup driven by the unique `(collection, source_path)` index -- one
    path per alias, so usually one -- rather than a scan, so it costs the same
    whether the collection holds ten documents or a million.

    This matches on names, and cannot do better: if a root that was a real
    directory is later repointed at an unrelated tree with the same relative
    filenames, the old row is repathed onto a different file, and no filesystem
    check can tell that from the tree having moved -- the old file is
    unreachable either way. It self-corrects rather than corrupts: the
    registration writes the new `file_hash`, which is what `_step_extract`
    claims on, so the wrong text is re-extracted and re-chunked. Ordered by id
    so the choice is at least stable when several aliases match.
    """
    if not alternate_paths:
        return None
    return (
        session.query(SourceDocument)
        .filter(
            SourceDocument.collection == collection,
            SourceDocument.source_path.in_(alternate_paths),
            SourceDocument.status != "deleted",
        )
        # Ordered because there can be more than one alternate: two configured
        # roots resolving under one another produce several, and an unordered
        # `first()` would then repath a different row on each run. Oldest wins,
        # which is the one carrying the extraction and chunks.
        .order_by(SourceDocument.id)
        .first()
    )


def _merge_prior_root_document(
    session: Session,
    collection: str,
    file_path: str,
    alternate_paths: list[str],
    occupant: SourceDocument,
) -> SourceDocument | None:
    """Fold a document registered under a prior root into the row at this path.

    The repath in `_register_document` only fires when nothing is registered at
    `file_path`. Once a scan has already inserted a twin there, every later
    scan short-circuits on that row and the pre-move row keeps its chunks under
    a path nothing writes to -- a duplicate that outlives any number of clean
    runs, which is what the corpus of 2026-09-09 was left holding.

    Resolved the way the manual repair resolved it: the twin holds nothing, so
    it is deleted, and the row holding the extraction, chunks and vectors takes
    over its path. Returns None unless the case is exactly that -- an empty
    occupant and a prior-root row with work -- because two rows that both hold
    chunks are a question this cannot answer without discarding one of them.

    The deleted twin's own extraction artifact is left on disk. Every path that
    drops a document here leaves its artifact (a document marked `deleted`
    keeps one too); `collection remove` and revision pruning are what sweep
    them. Bounded: it is one file per document that was registered twice.
    """
    if not alternate_paths or _documents_holding_chunks(session, [occupant.id]):
        return None
    prior = _document_under_a_prior_root(session, collection, alternate_paths)
    if prior is None or not _documents_holding_chunks(session, [prior.id]):
        return None
    session.delete(occupant)
    # Before the path is reassigned: the unique index on
    # (collection, source_path) would otherwise refuse it.
    session.flush()
    prior.source_path = file_path
    return prior


def _identity_is_stale(source_path: str) -> bool:
    """Whether a stored path is no longer the real path of the file it names.

    Every stored ``source_path`` is resolved at registration, so this can only
    become true after the fact: the tree was moved and a symlink left behind,
    or a parent directory became a link. The file is still there, but it is
    reachable under a different name -- and that name is a *different document*
    to a pipeline keyed on ``(collection, source_path)``. Both then extract,
    chunk and embed the same bytes.

    A path that cannot be resolved is not stale, it is missing, which
    ``_is_missing`` decides separately. Keeping the two apart matters: missing
    is a deletion, stale is a rename, and they call for opposite repairs.
    """
    try:
        resolved = Path(source_path).resolve(strict=True)
    except OSError:
        return False
    return str(resolved) != source_path


def _document_reached_by_another_path(
    session: Session, collection: str, file_path: str, file_hash: str
) -> SourceDocument | None:
    """Find the document that is this same file under a path it no longer has.

    Called when nothing is registered at ``file_path`` yet. A row whose stored
    path still resolves to exactly this file is not a second document, it is
    this one under its old name, so the registration repaths it instead of
    inserting a twin -- which keeps its extraction, chunks and vectors, and
    makes moving a corpus free rather than a full re-index.

    Candidates are drawn by content hash (indexed) rather than by scanning
    every live document: the whole-corpus case runs this once per file during
    the initial scan, so an O(documents) probe per file would be O(n^2) over
    the collection. The cost is one missed case -- a file whose *content* also
    changed while it was moved -- which registers as a new document and leaves
    the old one to ``_reconcile_deletions``. That is one file re-indexed, not a
    corpus duplicated.
    """
    candidates = (
        session.query(SourceDocument)
        .filter(
            SourceDocument.collection == collection,
            SourceDocument.file_hash == file_hash,
            SourceDocument.status != "deleted",
            SourceDocument.source_path != file_path,
        )
        .all()
    )
    for candidate in candidates:
        try:
            resolved = Path(candidate.source_path).resolve(strict=True)
        except OSError:
            continue
        if str(resolved) == file_path:
            return candidate
    return None


#: Documents per chunk statement. Bounds the IN list and the row count of any
#: single delete; the transaction around them is the caller's.
_DELETE_BATCH_SIZE = 500


def _documents_holding_chunks(session: Session, document_ids: list[int]) -> set[int]:
    """Which of these documents own at least one chunk."""
    holding: set[int] = set()
    for start in range(0, len(document_ids), _DELETE_BATCH_SIZE):
        batch = document_ids[start : start + _DELETE_BATCH_SIZE]
        holding.update(
            document_id
            for (document_id,) in session.query(Chunk.document_id)
            .filter(Chunk.document_id.in_(batch))
            .distinct()
            .all()
        )
    return holding


def _retirable_stale_documents(
    session: Session, candidates: list[SourceDocument], live: list[SourceDocument]
) -> list[SourceDocument]:
    """Stale identities that can be retired without deleting anything.

    Two conditions, deliberately strict. The row must hold no chunks, which
    makes retiring it a no-op on the chunk tables -- the reconcile cannot
    delete a chunk under any circumstances. And the collection must already
    hold a document at the row's real path, so a row nothing has replaced is
    left alone: a scan that skipped that file (the size cap, an unreadable
    subtree, a suffix no longer extracted) would otherwise have it retired with
    no successor.

    An earlier form also retired a stale row whose twin *held chunks*, reasoning
    that the twin covered the text. It does not. Holding chunks is a membership
    test, and a twin one chunk into a five-hundred-chunk document satisfies it
    while covering almost none of it -- and the stale row's chunks and vectors
    went with it. The asymmetry decides this: a duplicate pair left standing is
    visible in the document count and costs a re-index, while the deleted
    vectors are hours of embedding that nothing can bring back.
    """
    if not candidates:
        return []
    live_paths = {document.source_path for document in live}
    holding = _documents_holding_chunks(session, [candidate.id for candidate in candidates])
    return [
        candidate
        for candidate in candidates
        if candidate.id not in holding
        and str(Path(candidate.source_path).resolve()) in live_paths
    ]


def _purge_document_chunks(session: Session, document_ids: list[int]) -> None:
    """Delete the chunks of removed documents, and with them their vectors.

    Marking a document ``deleted`` used to be the whole story: its chunks,
    embeddings and vectors stayed. That was tolerable while search filtered
    ``sd.status <> 'deleted'`` at query time, but that filter lived on a joined
    table and was one of the reasons the planner could never use the ANN index.
    Search now filters on the vector row alone, so a deleted document's vectors
    have to actually go -- otherwise they would be returned.

    Deleting the chunks is enough to clear the vectors: ``chunk_embeddings``
    and every per-profile ``embedding_vectors_p*`` table carry
    ``ON DELETE CASCADE`` from ``chunks_v2``, so one statement clears all of
    them for every profile at once, including profiles this collection no
    longer uses.

    It is not enough for the *chunkings*, which must be invalidated too. A
    watcher sees delete-then-create for an unchanged file constantly -- a sync
    client or an editor writing a temp file and renaming it over the original
    -- and the file that comes back extracts to the same text it had before.
    ``_step_chunk`` re-claims a ``done`` chunking only when its
    ``source_content_hash`` differs from the extraction's, so a chunking left
    with a current hash and no chunks is never revisited: the document is
    unsearchable for good, while ``chunked_scope`` still counts it done and
    ``cementic status`` reads 100%. Clearing the hash re-opens the work, and
    keeps the row out of ``chunked_done`` until it is genuinely re-chunked --
    the same repair ``_purge_all_chunks`` makes for the re-extraction path.
    Measured on the live corpus 2026-09-09: 18,101 of 46,139 papers stranded
    this way, 1.8M chunks' worth, none of them reported by anything.
    """
    if not document_ids:
        return
    # Batched, because the whole-corpus case is real: removing a collection
    # passes every document at once. One statement with a 22k-element IN list
    # deletes ~2.3M chunk rows and their cascaded vectors in a single
    # transaction, which holds its snapshot open for the duration and keeps
    # autovacuum off every table it touches until it commits.
    #
    # The batches share the caller's transaction, so this is still all-or-
    # nothing; what it bounds is the size of each statement, not the atomicity.
    for start in range(0, len(document_ids), _DELETE_BATCH_SIZE):
        batch = document_ids[start : start + _DELETE_BATCH_SIZE]
        session.query(Chunk).filter(Chunk.document_id.in_(batch)).delete(
            synchronize_session=False
        )
        chunked_ids = (
            select(ChunkedDocument.id)
            .join(
                ExtractedDocument,
                ChunkedDocument.extracted_document_id == ExtractedDocument.id,
            )
            .where(ExtractedDocument.document_id.in_(batch))
        )
        session.query(ChunkedDocument).filter(ChunkedDocument.id.in_(chunked_ids)).update(
            {ChunkedDocument.source_content_hash: None}, synchronize_session=False
        )


class DocumentEventHandler(FileSystemEventHandler):
    """Handles file system events for any supported document type."""

    def __init__(
        self,
        callback: Callable[[str], None],
        delete_callback: Callable[[str], None] | None = None,
        ignore_directories: Iterable[str] = (),
        watched_roots: Iterable[Path] = (),
        delete_directory_callback: Callable[[str], None] | None = None,
        extensions: Iterable[str] | None = None,
    ) -> None:
        self.callback = callback
        self.delete_callback = delete_callback
        self.delete_directory_callback = delete_directory_callback
        self._ignored_directories = set(ignore_directories)
        # Plain data, like ignore_directories above, rather than a Config: the
        # set is config-dependent now (the `command` extractor's file types come
        # from `[extraction.commands]`), but resolving it is the caller's job.
        # None means the built-in extractors, which is what a handler built
        # without a config can honestly claim to handle.
        self._extensions = (
            frozenset(extensions) if extensions is not None else supported_extensions()
        )
        self._watched_roots = [Path(root) for root in watched_roots]
        self._timers: dict[str, Any] = {}
        self._debounce_seconds = 2.0

    def _is_ignored(self, file_path: str) -> bool:
        """Whether a directory *below the watched root* is one the watcher skips.

        Live events need this as well as the scan: a file written into
        node_modules while cementic is running arrives by inotify, never
        through the (pruned) directory walk.

        Only components below the root count. Testing the whole absolute path
        also tested the root's own ancestors, so watching a directory that
        happens to live under one named `build` or `node_modules` indexed
        everything on the initial scan -- which walks down from the root and
        never looks up -- and then silently dropped every create, modify and
        delete event for the life of the process.
        """
        if not self._ignored_directories:
            return False
        path = Path(file_path)
        for root in self._watched_roots:
            try:
                relative = path.relative_to(root)
            except ValueError:
                continue
            return any(part in self._ignored_directories for part in relative.parts[:-1])
        # No configured root contains it (or none was supplied): fall back to
        # testing the whole path rather than accepting it unchecked.
        return any(part in self._ignored_directories for part in path.parts[:-1])

    def _should_process(self, file_path: str) -> bool:
        if self._is_ignored(file_path):
            return False
        return Path(file_path).suffix.lower() in self._extensions

    def _debounced_process(self, file_path: str) -> None:
        existing_timer = self._timers.pop(file_path, None)
        if existing_timer:
            existing_timer.cancel()

        timer = threading.Timer(self._debounce_seconds, self._run_callback, args=(file_path,))
        timer.daemon = True
        self._timers[file_path] = timer
        timer.start()

    def _run_callback(self, file_path: str) -> None:
        self._timers.pop(file_path, None)
        self.callback(file_path)

    def cancel_all(self) -> None:
        """Cancel all pending debounce timers, preventing fire after shutdown."""
        for file_path, timer in list(self._timers.items()):
            timer.cancel()
        self._timers.clear()

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        src_path = event.src_path.decode() if isinstance(event.src_path, bytes) else event.src_path
        if self._should_process(src_path):
            self._debounced_process(src_path)

    # A creation and a modification are handled identically: both just need
    # the (debounced) processing check re-run against the changed path.
    on_created = on_modified

    def on_deleted(self, event: FileSystemEvent) -> None:
        src_path = event.src_path.decode() if isinstance(event.src_path, bytes) else event.src_path
        if event.is_directory:
            # A directory moved *out* of the watched tree (or rm -r'd) arrives
            # as one DirDeletedEvent with no per-file deletions -- measured
            # against watchdog's inotify backend, 2026-08-18. Returning here
            # left every document under it "present" until the next restart's
            # missing-file reconciliation, so searches kept matching paths that
            # no longer existed.
            if self.delete_directory_callback and not self._is_ignored(
                str(Path(src_path) / "x")
            ):
                self.delete_directory_callback(src_path)
            return
        timer = self._timers.pop(src_path, None)
        if timer:
            timer.cancel()
        if self.delete_callback and self._should_process(src_path):
            self.delete_callback(src_path)

    def on_moved(self, event: DirMovedEvent | FileMovedEvent) -> None:
        if event.is_directory:
            return
        src_path = event.src_path.decode() if isinstance(event.src_path, bytes) else event.src_path
        dest_path = (
            event.dest_path.decode() if isinstance(event.dest_path, bytes) else event.dest_path
        )
        if self.delete_callback and self._should_process(src_path):
            self.delete_callback(src_path)
        if self._should_process(dest_path):
            self._debounced_process(dest_path)


class SourceWatcher:
    """Background watcher that registers source documents for pipeline processing."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or get_config()
        self.state_manager = StateManager(self.config.source_watcher.state_path)
        self._shutdown_event = threading.Event()
        self._shutdown_signal: int | None = None
        #: Whether this process has published a last_error that is still
        #: standing. Gates the retraction so a clean registration does not write
        #: the state file on every single file it processes.
        self._reported_error = False
        self.watcher: Any = None
        self._event_handler: DocumentEventHandler | None = None
        self._logger = self._setup_logging()
        self.Session: Any = None
        self.collection = "default"
        self._watched_roots: list[Path] = []
        #: (configured, resolved) for each watched root that is not its own real
        #: path. What a document registered before the root became a link is
        #: keyed on, and the only record of that mapping.
        self._root_aliases: list[tuple[Path, Path]] = []
        self._last_current_file_publish = 0.0
        #: Why startup aborted, or None. The "already running" path returns
        #: normally, so without this the runner exits 0 on a worker that never
        #: started and anything keying on exit status concludes success.
        self.fatal_reason: str | None = None

    def _setup_logging(self) -> logging.Logger:
        return setup_worker_logger(
            "cementic.source_watcher", self.config.source_watcher.log_file, "Source watcher"
        )

    def start(self, directories: list[str], collection: str = "default") -> None:
        self.collection = collection
        # A previous run's reason must not make this start look failed.
        self.fatal_reason = None
        state = self.state_manager.load()
        # PID + start-token: a recycled PID after `stop --force` must not block
        # a fresh start.
        if (
            state.daemon_state == DaemonState.RUNNING
            and state.pid
            and is_managed_process_alive(state.pid, state.start_token)
        ):
            self._fatal(
                "Source watcher already running with PID %s", state.pid, publish=False
            )
            return

        engine = get_engine(self.config.database.url)
        create_tables(engine)
        self.Session = get_session_factory(engine)

        # Counters are per-session: reset so `status` reports this run, not an
        # ever-growing total across restarts. skipped_files with them -- it is
        # the paths behind failed_count, and resetting one but not the other
        # showed a fresh run still "skipping" last run's files.
        self.state_manager.update(
            daemon_state=DaemonState.RUNNING,
            watched_directories=directories,
            pid=os.getpid(),
            start_token=process_start_token(os.getpid()),
            processed_count=0,
            failed_count=0,
            current_file=None,
            skipped_files=[],
            # Clear any fatal reason a previous failed start published to
            # `last_error` (via `report_fatal`). Without this a startup failure
            # that has since been fixed would be reported by `cementic status`
            # forever, mirroring the pipeline worker's clear at the same point.
            last_error=None,
            last_error_at=None,
        )

        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

        try:
            self._start_watcher(directories)
            self._logger.info(
                "Source watcher started watching: %s (collection=%s)", directories, collection
            )
            # The signal handlers set the shutdown event, so a plain wait loop
            # suffices past this point.
            while not self._shutdown_event.is_set():
                self._shutdown_event.wait(1)
        finally:
            # However this run ends, the state file must not be left claiming
            # the watcher is RUNNING.
            self.stop()

    def _fatal(self, message: str, *args: Any, publish: bool = True) -> None:
        self.fatal_reason = report_fatal(
            self._logger, self.state_manager, message, *args, publish=publish
        )

    def _configure_watched_roots(self, directories: list[str]) -> list[str]:
        """Resolve the watched roots, remembering what each resolved from.

        Returns the directories that are not usable. The roots are resolved
        because document identity is: a file is stored under its real path.
        Keeping the configured form beside it is what lets a registration
        recognise a document written before a root became a symlink.
        """
        self._watched_roots = []
        self._root_aliases = []
        missing: list[str] = []
        for directory in directories:
            resolved = Path(directory).resolve()
            if not resolved.is_dir():
                missing.append(directory)
                continue
            self._watched_roots.append(resolved)
            configured = Path(os.path.abspath(directory))
            if configured != resolved:
                self._root_aliases.append((configured, resolved))
        return missing

    def _start_watcher(self, directories: list[str]) -> None:
        observer = Observer()
        # Resolve the roots before building the handler: it filters ignored
        # directory names relative to a root, so handing it an empty root list
        # would make it fall back to matching against whole absolute paths.
        missing = self._configure_watched_roots(directories)
        for directory in missing:
            # A warning, not _fatal: with other directories surviving the
            # watcher still has work, but the poisoned fatal_reason made
            # the runner exit 1 for a "startup failure" when that run
            # finally shut down cleanly hours later. The all-missing case
            # raises below and stays fatal.
            self._logger.error("Watch directory does not exist: %s", directory)
            print(
                f"Watch directory does not exist: {directory}",
                file=sys.stderr,
                flush=True,
            )

        event_handler = DocumentEventHandler(
            self._on_file_detected,
            self._on_file_deleted,
            ignore_directories=self.config.source_watcher.ignore_directories,
            watched_roots=self._watched_roots,
            delete_directory_callback=self._on_directory_deleted,
            extensions=supported_extensions(self.config),
        )
        self._event_handler = event_handler
        for path in self._watched_roots:
            observer.schedule(event_handler, str(path), recursive=True)
        if not self._watched_roots:
            # Skipping every directory used to be log-only: the watcher still
            # published RUNNING, survived the startup check, and `cementic
            # status` showed healthy workers indexing nothing forever. There is
            # no useful work to do, so fail where the user can see it.
            raise RuntimeError(
                "No watchable directories: " + ", ".join(missing) + ". "
                "Nothing would be indexed."
            )
        # Start observing *before* the initial scan: the scan hashes every
        # existing file and can take minutes, and events are only delivered
        # after start(). Double-seeing a file during the overlap is handled by
        # _register_document's retry, not by idempotence: the insert races the
        # unique index on (collection, source_path).
        observer.start()
        self.watcher = observer
        for root in self._watched_roots:
            if self._shutdown_event.is_set():
                return
            self._scan_existing(root)
        if not self._shutdown_event.is_set():
            self._reconcile_deletions()

    def _reconcile_deletions(self) -> None:
        """Retire documents that no longer name a file of their own.

        Two ways that happens while cementic is not running. The file vanished,
        which is otherwise only noticed through a live filesystem event, so a
        file removed between runs kept ``status="pending"`` forever and kept
        matching searches with a path that no longer exists. Or the file moved
        and the stored path now resolves elsewhere, which makes the row a second
        identity for a document the collection already holds under its real
        path. Both are scoped to the currently-watched roots, so documents
        indexed from other directories (or other collections) are never touched.
        """
        if not self._watched_roots:
            return
        with self.Session() as session:
            documents = (
                session.query(SourceDocument)
                .filter(
                    SourceDocument.collection == self.collection,
                    SourceDocument.status != "deleted",
                )
                .all()
            )
            missing = [
                document
                for document in documents
                if self._is_under_watched_roots(document.source_path)
                and _is_missing(document.source_path)
            ]
            # Rows the scan could not repath. The scan runs first and claims
            # every one it can identify, so anything still holding a stale path
            # here is a second identity for a document the collection already
            # has under its real path. Unlike the missing ones these are
            # invisible to the literal containment test -- a path under the
            # pre-move location is under no watched root -- which is why they
            # survived every restart.
            # By id, not by `in missing`: that is a list scan per document with
            # ORM identity comparison, and this loop runs over every live
            # document in the collection at every startup.
            missing_ids = {document.id for document in missing}
            stale = _retirable_stale_documents(
                session,
                [
                    document
                    for document in documents
                    if document.id not in missing_ids
                    and _identity_is_stale(document.source_path)
                    and self._resolves_under_watched_roots(document.source_path)
                ],
                documents,
            )
            retired = missing + stale
            # Read the paths before committing: ORM attributes expire on commit
            # and these instances are detached once the session closes.
            missing_paths = [document.source_path for document in missing]
            stale_paths = [document.source_path for document in stale]
            for document in retired:
                document.status = "deleted"
                document.file_hash = None
            if retired:
                _purge_document_chunks(session, [document.id for document in retired])
                session.commit()
        for source_path in missing_paths:
            self._logger.info(
                "Marked document deleted while stopped: %s (collection=%s)",
                source_path,
                self.collection,
            )
        for source_path in stale_paths:
            self._logger.info(
                "Marked document deleted, its path is no longer its own: %s (collection=%s)",
                source_path,
                self.collection,
            )

    def _scan_existing(self, directory: Path) -> None:
        extensions = supported_extensions(self.config)
        ignored = set(self.config.source_watcher.ignore_directories)
        # os.walk rather than rglob so ignored directories can be pruned from
        # the traversal itself: rglob would still descend into node_modules and
        # .git to discover files it then discards.
        def _on_walk_error(error: OSError) -> None:
            # os.walk swallows errors by default, so an unreadable subtree was
            # skipped in complete silence: no log line, no counter, and a
            # collection quietly missing however many documents it held. The
            # path goes through record_skipped so `status --verbose` can name
            # the subtree, not just count it.
            self._logger.error("Could not read directory during scan: %s", error)
            failed_path = str(error.filename) if error.filename else str(directory)
            self.state_manager.record_skipped(failed_path, f"unreadable during scan: {error}")

        for dirpath, dirnames, filenames in os.walk(directory, onerror=_on_walk_error):
            # The scan can walk a large tree for minutes; without this a
            # `cementic stop` during startup waits out its whole grace period
            # and then reports a timeout, while the watcher keeps indexing.
            if self._shutdown_event.is_set():
                return
            dirnames[:] = [name for name in dirnames if name not in ignored]
            root = Path(dirpath)
            for filename in filenames:
                if self._shutdown_event.is_set():
                    return
                file_path = root / filename
                if file_path.suffix.lower() not in extensions:
                    continue
                # Symlinks pass through so they get *recorded*: filtering them
                # here dropped them with no counter and no skip entry, while a
                # live inotify event for the same file reached
                # _register_document's record_skipped. is_symlink() is checked
                # separately because a broken symlink fails is_file() and would
                # otherwise vanish the same way.
                if file_path.is_file() or file_path.is_symlink():
                    self._on_file_detected(str(file_path))

    def _on_file_detected(self, file_path: str) -> None:
        try:
            self._register_document(file_path)
            # Mirrors the pipeline worker clearing last_error after a clean
            # loop. Without it the watcher published errors and never retracted
            # them, so one transient failure was reported by `cementic status`
            # forever -- observed live 2026-08-23 against a healthy watcher.
            self._clear_error()
        except Exception as error:
            self._logger.error("Failed to register %s: %s", file_path, error)
            # record_skipped, not a bare failed increment: the file never
            # became a document, so without the path in skipped_files the only
            # evidence was a log line the user has to know to look for.
            self.state_manager.record_skipped(
                file_path, f"registration failed: {error}", current_file=None
            )
            self._record_error(error)

    def _on_file_deleted(self, file_path: str) -> None:
        try:
            self._mark_document_deleted(file_path)
        except Exception as error:
            self._logger.error("Failed to mark deleted %s: %s", file_path, error)
            # Same reasoning as _on_file_detected above: without record_skipped
            # the deleted file kept "existing" in search with nothing visible
            # in `status` beyond a log line the user has to know to look for.
            self.state_manager.record_skipped(file_path, f"deletion failed: {error}")
            self._record_error(error)

    def _on_directory_deleted(self, dir_path: str) -> None:
        try:
            self._mark_documents_deleted_under(dir_path)
        except Exception as error:
            self._logger.error("Failed to mark directory deleted %s: %s", dir_path, error)
            self.state_manager.record_skipped(dir_path, f"directory deletion failed: {error}")
            self._record_error(error)

    def _clear_error(self) -> None:
        """Retract a previously published error after a clean registration."""
        if not self._reported_error:
            return
        try:
            self.state_manager.update(last_error=None, last_error_at=None)
        except Exception:  # pragma: no cover - never mask a successful register
            self._logger.exception("Could not clear the source watcher error state")
        else:
            self._reported_error = False

    def _record_error(self, error: Exception) -> None:
        """Publish an event-handler failure to the worker state file.

        Mirrors ``PipelineWorker._record_loop_error``: without this the
        watcher never wrote ``last_error`` at all, so the "last error" row
        `cli.py` renders for it was permanently dead -- an operational
        failure here was only visible in a log file the user has to know
        about.
        """
        message = f"{type(error).__name__}: {error}"
        try:
            self.state_manager.update(
                last_error=message[:500],
                last_error_at=datetime.now(timezone.utc).isoformat(),
            )
            self._reported_error = True
        except Exception:  # pragma: no cover - state file must never mask the real error
            self._logger.exception("Could not record source watcher error to the state file")

    def _publish_current_file(self, file_path: str) -> None:
        """Publish "now working on X", at most once per interval.

        Purely for display in `cementic status --verbose`, but each write is a
        JSON read, a full write and a rename. During the initial scan of a large
        tree that is one such round-trip per file on top of the SHA-256 read, so
        it is rate-limited. The processed/failed counters are *not* throttled --
        those are exact.
        """
        now = time.monotonic()
        if now - self._last_current_file_publish < _CURRENT_FILE_PUBLISH_INTERVAL_SECONDS:
            return
        self._last_current_file_publish = now
        self.state_manager.update(current_file=file_path)

    def _is_under_watched_roots(self, file_path: str) -> bool:
        """Whether a stored path lies under a root this run is watching.

        Deliberately does not resolve: the path is already stored resolved, and
        the file may no longer exist. That is also why the configured form of
        each root counts as watched. A row written before a root became a
        symlink is stored under the path that root used to resolve to, and if
        its file is then deleted there is nothing left to resolve -- so a test
        that only knows today's real roots cannot see it, and the document
        stays in search under a path that no longer exists. Found live on
        2026-09-09: 38 documents in exactly that state, holding chunks.
        """
        path = Path(file_path)
        roots = self._watched_roots + [configured for configured, _ in self._root_aliases]
        return any(path.is_relative_to(root) for root in roots)

    def _resolves_under_watched_roots(self, file_path: str) -> bool:
        """Whether a stored path reaches a file under a root this run watches.

        The literal test above is the right one for a path that is still its
        own real path, which is every path at the moment it is stored. This one
        is for the path that has stopped being that: the row belongs to this
        run's corpus, it just names it by a route that no longer is the file's
        own name.
        """
        try:
            resolved = Path(file_path).resolve(strict=True)
        except OSError:
            return False
        return any(resolved.is_relative_to(root) for root in self._watched_roots)

    def _normalize_watched_path(self, file_path: str, *, must_exist: bool) -> str | None:
        path = Path(file_path)
        try:
            resolved_path = path.resolve(strict=must_exist)
        except OSError:
            self._logger.error("Cannot resolve file: %s", file_path)
            return None
        if self._watched_roots and not any(
            resolved_path.is_relative_to(root) for root in self._watched_roots
        ):
            self._logger.error("File is outside watched roots: %s", file_path)
            return None
        return str(resolved_path)

    def _register_document(self, file_path: str) -> None:
        # Each rejection below counts. They are logged at ERROR level but used to
        # touch neither counter, so `status --verbose` read "processed=N,
        # failed=0" while an arbitrary number of documents had been dropped --
        # the only evidence in a log file the user has to know to look for.
        path = Path(file_path)
        if path.is_symlink():
            self._logger.error("Refusing symlinked file: %s", file_path)
            self.state_manager.record_skipped(file_path, "symlink")
            return
        normalized_path = self._normalize_watched_path(file_path, must_exist=True)
        if normalized_path is None:
            self.state_manager.record_skipped(file_path, "unreadable or outside watched roots")
            return
        file_path = normalized_path
        # Guard against exceedingly large files
        max_size_bytes = 512 * 1024 * 1024  # 512 MiB
        try:
            file_size = path.stat().st_size
        except OSError:
            self._logger.error("Cannot stat file: %s", file_path)
            self.state_manager.record_skipped(file_path, "cannot stat")
            return
        if file_size > max_size_bytes:
            self._logger.error(
                "File too large (%d bytes, max %d): %s", file_size, max_size_bytes, file_path
            )
            self.state_manager.record_skipped(
                file_path, f"too large ({file_size} bytes, max {max_size_bytes})"
            )
            return

        sha256 = hashlib.sha256()
        with open(file_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                sha256.update(chunk)
        file_hash = sha256.hexdigest()

        self._publish_current_file(file_path)
        # Read-then-insert is not atomic, and the initial scan runs on the main
        # thread while debounce timers fire on their own. The same file can be
        # registered twice concurrently: both see no row, both insert, and the
        # loser hits the unique index on (collection, source_path). That is a
        # benign race -- the winner registered the file -- but it surfaced as
        # "Failed to register ..." and a permanently inflated failure count.
        for attempt in range(2):
            try:
                with self.Session() as session:
                    document = (
                        session.query(SourceDocument)
                        .filter_by(source_path=file_path, collection=self.collection)
                        .first()
                    )
                    alternates = _alternate_paths(file_path, self._root_aliases)
                    if document is not None:
                        merged = _merge_prior_root_document(
                            session, self.collection, file_path, alternates, document
                        )
                        if merged is not None:
                            self._logger.info(
                                "Document folded back into its move: %s -> %s "
                                "(collection=%s)",
                                merged.source_path,
                                file_path,
                                self.collection,
                            )
                            document = merged
                    if document is None:
                        # The alias first: it is an indexed lookup and it holds
                        # whether or not the file was edited on its way, which
                        # the hash probe below cannot say.
                        document = _document_under_a_prior_root(
                            session, self.collection, alternates
                        ) or _document_reached_by_another_path(
                            session, self.collection, file_path, file_hash
                        )
                        if document is not None:
                            self._logger.info(
                                "Document moved: %s -> %s (collection=%s)",
                                document.source_path,
                                file_path,
                                self.collection,
                            )
                            document.source_path = file_path
                    if document is None:
                        document = SourceDocument(
                            source_path=file_path, collection=self.collection
                        )
                        session.add(document)

                    document.file_hash = file_hash
                    document.status = "pending"
                    session.commit()
                break
            except IntegrityError:
                if attempt == 1:
                    raise
                # The concurrent insert has committed; the retry now updates it.
                self._logger.debug("Concurrent registration of %s; retrying", file_path)

        self.state_manager.increment(processed=1, current_file=None)
        self._logger.info("Registered document: %s (collection=%s)", file_path, self.collection)

    def _mark_document_deleted(self, file_path: str) -> None:
        normalized_path = self._normalize_watched_path(file_path, must_exist=False)
        if normalized_path is None:
            return
        with self.Session() as session:
            document = (
                session.query(SourceDocument)
                .filter_by(source_path=normalized_path, collection=self.collection)
                .first()
            )
            if document is None:
                return
            document.status = "deleted"
            document.file_hash = None
            _purge_document_chunks(session, [document.id])
            session.commit()
        self._logger.info(
            "Marked document deleted: %s (collection=%s)", normalized_path, self.collection
        )

    def _mark_documents_deleted_under(self, dir_path: str) -> None:
        """Mark every document under a vanished directory deleted.

        The directory is gone, so its path cannot be resolved; the prefix is
        normalized textually instead, and matched with a trailing separator so
        ``/a/docs`` never claims ``/a/docs-archive``'s documents.
        """
        prefix = str(Path(dir_path).absolute())
        if self._watched_roots and not any(
            Path(prefix).is_relative_to(root) or Path(root).is_relative_to(prefix)
            for root in self._watched_roots
        ):
            return
        with self.Session() as session:
            documents = (
                session.query(SourceDocument)
                .filter(
                    SourceDocument.collection == self.collection,
                    SourceDocument.status != "deleted",
                    # autoescape=True because `startswith` compiles to LIKE and
                    # defaults to leaving the prefix raw: a directory named
                    # `2024_papers` then also matches `2024-papers`, and one
                    # containing `%` matches an arbitrary suffix. This query
                    # feeds a hard delete of the matched documents' chunks and
                    # vectors, so an over-match silently destroys a sibling
                    # directory's index.
                    SourceDocument.source_path.startswith(prefix + os.sep, autoescape=True),
                )
                .all()
            )
            if not documents:
                return
            deleted_paths = [document.source_path for document in documents]
            for document in documents:
                document.status = "deleted"
                document.file_hash = None
            _purge_document_chunks(session, [document.id for document in documents])
            session.commit()
        for source_path in deleted_paths:
            self._logger.info(
                "Marked document deleted (directory removed): %s (collection=%s)",
                source_path,
                self.collection,
            )

    def _handle_shutdown(self, signum: int, frame: object) -> None:
        handle_shutdown_signal(self, signum, frame)

    def stop(self) -> None:
        self._shutdown_event.set()
        if self._shutdown_signal is not None:
            self._logger.info("Received signal %s, shutting down...", self._shutdown_signal)
            self._shutdown_signal = None
        # Observer first, timers second: an event delivered after cancel_all
        # but before the observer stopped re-armed a debounce timer, which then
        # fired into a watcher whose state already said STOPPED.
        if self.watcher:
            self.watcher.stop()
            self.watcher.join()
        if self._event_handler:
            self._event_handler.cancel_all()
        self.state_manager.update(daemon_state=DaemonState.STOPPED, pid=None)
        self._logger.info("Source watcher stopped")
