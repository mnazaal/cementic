"""Collection and revision persistence helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

from sqlalchemy import func
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from cementic.config import Config
from cementic.db import Chunk, ExtractedDocument, PipelineRevision, SourceDocument
from cementic.pipeline_worker import (
    PipelineCounts,
    compute_revision_counts,
    revision_failure_total,
    revision_is_complete,
)
from cementic.revisions import (
    drain_pending_artifact_removals,
    drain_pending_vector_table_drops,
    ensure_revision_ann_index,
    get_active_revision,
    promote_revision,
)
from cementic.storage import safe_remove_artifact
from cementic.vector_store import (
    drop_vector_table,
    index_access_method,
    vector_index_name,
    vector_table_exists,
)

logger = logging.getLogger("cementic.collections")


@dataclass(frozen=True)
class DeleteCollectionResult:
    """Summary of collection deletion work."""

    deleted_docs: int
    deleted_chunks: int
    artifact_paths: list[str]
    vector_profile_ids: list[int]


@dataclass(frozen=True)
class CollectionSummary:
    """Summary of one collection."""

    name: str
    documents: int
    active_revision_label: str | None
    ready_revision_label: str | None
    building_revision_label: str | None


def list_collections(session: Session) -> list[CollectionSummary]:
    """Return collection summaries ordered by collection name.

    Uses a fixed number of grouped queries regardless of collection count,
    instead of 3 queries per collection.
    """
    collection_names = [
        str(name)
        for (name,) in session.query(SourceDocument.collection)
        .group_by(SourceDocument.collection)
        .order_by(SourceDocument.collection)
        .all()
    ]

    documents_by_collection: dict[str, int] = {
        str(name): int(count)
        for name, count in session.query(SourceDocument.collection, func.count(SourceDocument.id))
        .filter(SourceDocument.status != "deleted")
        .group_by(SourceDocument.collection)
        .all()
    }

    active_by_collection: dict[str, PipelineRevision] = {}
    ready_by_collection: dict[str, PipelineRevision] = {}
    building_by_collection: dict[str, PipelineRevision] = {}
    # Ready and building need separate slots, not one "not active" slot chosen
    # by highest id: a ready revision sitting behind a newer building one is the
    # normal state after any profile-affecting config change, and collapsing
    # them hid the promotable revision that `collection promote` targets.
    for revision in (
        session.query(PipelineRevision)
        .filter(PipelineRevision.status.in_(["active", "building", "ready"]))
        .order_by(PipelineRevision.collection, PipelineRevision.id.desc())
        .all()
    ):
        if revision.status == "active":
            active_by_collection.setdefault(revision.collection, revision)
        elif revision.status == "ready":
            ready_by_collection.setdefault(revision.collection, revision)
        else:
            building_by_collection.setdefault(revision.collection, revision)

    return [
        CollectionSummary(
            name=name,
            documents=documents_by_collection.get(name, 0),
            active_revision_label=getattr(active_by_collection.get(name), "label", None),
            ready_revision_label=getattr(ready_by_collection.get(name), "label", None),
            building_revision_label=getattr(building_by_collection.get(name), "label", None),
        )
        for name in collection_names
    ]


def delete_collection_records(session: Session, collection: str) -> DeleteCollectionResult | None:
    """Delete a collection from the database and return removed artifact paths.

    Returns None only when the collection is genuinely unknown. A collection can
    own revisions without owning any documents -- `cementic start` on a directory
    with no supported files creates exactly that -- and returning early on the
    document check left those rows undeletable: `collection list` (which reads
    document rows) never showed them, and `collection remove` said "not found".
    """
    docs = session.query(SourceDocument).filter_by(collection=collection).all()
    revision_count = session.query(PipelineRevision).filter_by(collection=collection).count()
    if not docs and not revision_count:
        return None

    doc_ids = [doc.id for doc in docs]
    artifact_paths = [
        path
        for (path,) in session.query(ExtractedDocument.artifact_path)
        .filter(ExtractedDocument.document_id.in_(doc_ids))
        .all()
        if path
    ]
    candidate_profile_ids = [
        int(profile_id)
        for (profile_id,) in session.query(PipelineRevision.embedding_profile_id)
        .filter_by(collection=collection)
        .all()
        if profile_id is not None
    ]
    deleted_chunks = (
        session.query(Chunk)
        .filter(Chunk.document_id.in_(doc_ids))
        .delete(synchronize_session=False)
    )
    session.query(PipelineRevision).filter_by(collection=collection).delete(
        synchronize_session=False
    )
    deleted_docs = (
        session.query(SourceDocument)
        .filter(SourceDocument.id.in_(doc_ids))
        .delete(synchronize_session=False)
    )
    vector_profile_ids = []
    unique_profile_ids = sorted(set(candidate_profile_ids))
    if unique_profile_ids:
        remaining_by_profile: dict[int, int] = {
            int(profile_id): int(count)
            for profile_id, count in session.query(
                PipelineRevision.embedding_profile_id, func.count(PipelineRevision.id)
            )
            .filter(PipelineRevision.embedding_profile_id.in_(unique_profile_ids))
            .group_by(PipelineRevision.embedding_profile_id)
            .all()
        }
        vector_profile_ids = [
            profile_id
            for profile_id in unique_profile_ids
            if remaining_by_profile.get(profile_id, 0) == 0
        ]
    session.commit()
    return DeleteCollectionResult(
        deleted_docs=deleted_docs,
        deleted_chunks=deleted_chunks,
        artifact_paths=artifact_paths,
        vector_profile_ids=vector_profile_ids,
    )


def remove_artifacts(paths: list[str], *, config: Config) -> list[str]:
    """Best-effort removal of extracted document artifacts.

    Returns the paths that could not be removed. ``safe_remove_artifact`` raises
    for a path outside the artifacts root or a symlink -- a deliberate safety
    stop, but one that previously aborted the whole loop, so a single rejected
    path left every later artifact on disk and could propagate out of a
    promotion. Each path now fails on its own.
    """
    failures: list[str] = []
    for artifact_path in paths:
        try:
            safe_remove_artifact(config, artifact_path)
        except Exception:
            logger.warning("Could not remove artifact %s", artifact_path, exc_info=True)
            failures.append(artifact_path)
    return failures


def drop_orphan_vector_tables(engine: Any, profile_ids: list[int]) -> None:
    """Best-effort drop of vector tables no longer referenced by revisions."""
    for profile_id in profile_ids:
        drop_vector_table(engine, profile_id)


def find_ready_revision(session: Session, collection: str) -> PipelineRevision | None:
    """Return the newest ready (promotable) revision for one collection."""
    return (
        session.query(PipelineRevision)
        .filter_by(collection=collection, status="ready")
        .order_by(PipelineRevision.id.desc())
        .first()
    )


@dataclass(frozen=True)
class PromotionOutcome:
    """Result of attempting to promote a collection's ready revision.

    ``status`` is one of ``"promoted"``, ``"no_ready"`` (nothing to promote),
    ``"empty"`` (the ready revision has no live documents), ``"incomplete"`` (it
    has unfinished work and ``force`` was not set), or ``"blocked_by_failures"``
    (it built with failures and ``force`` was not set). Every non-promoted
    outcome leaves the revision untouched.
    """

    status: str
    revision: PipelineRevision | None = None
    counts: PipelineCounts | None = None


def promote_ready_revision(
    session: Session, collection: str, *, config: Config, force: bool = False
) -> PromotionOutcome:
    """Promote the newest ready revision for one collection.

    ``ready`` records that a revision was complete once, not that it still is:
    nothing flips it back to ``building`` when the watcher registers new
    documents, and ``requeue_interrupted_artifacts`` resets failed rows to
    ``pending`` on every ``cementic start``. So completeness is re-checked here
    against current counts rather than trusted from the status.

    A revision can also reach ``ready`` with failed documents/chunks (failures
    are terminal so the build can finish). Promoting one silently would publish
    a partial index, so unless ``force`` is set this refuses and reports the
    counts instead.
    """
    revision = find_ready_revision(session, collection)
    if revision is None:
        return PromotionOutcome("no_ready")

    counts = compute_revision_counts(session, collection, revision)
    if counts.documents == 0:
        # Promotion retires whatever is currently active, so publishing an
        # empty revision removes search coverage rather than merely adding
        # none. There is no reading of --force under which that is wanted.
        return PromotionOutcome("empty", revision=revision, counts=counts)
    if not force and not revision_is_complete(counts):
        return PromotionOutcome("incomplete", revision=revision, counts=counts)
    if not force and revision_failure_total(counts) > 0:
        return PromotionOutcome("blocked_by_failures", revision=revision, counts=counts)

    promote_revision(session, collection, revision, config=config)
    session.commit()
    # Only now that the promotion is durable are the superseded revisions'
    # artifact files safe to unlink, and their vector tables safe to drop --
    # the latter also because DROP TABLE takes its own transaction, which would
    # have blocked on the locks the session held until this commit.
    remove_artifacts(drain_pending_artifact_removals(session), config=config)
    drop_orphan_vector_tables(session.get_bind(), drain_pending_vector_table_drops(session))
    return PromotionOutcome("promoted", revision=revision)


@dataclass(frozen=True)
class ReindexOutcome:
    """Result of reconciling a collection's ANN index with current config.

    ``status`` is one of ``"reindexed"``, ``"no_active"`` (nothing promoted yet)
    or ``"no_vectors"`` (the active revision embedded nothing, so there is no
    table to index).
    """

    status: str
    method: str | None = None
    previous_method: str | None = None


def reindex_collection(
    session: Session, collection: str, *, config: Config, force: bool = False
) -> ReindexOutcome:
    """Rebuild the active revision's ANN index against current ``index`` config.

    Nothing else triggers this. The index is built once, when a revision first
    completes, so editing ``index.method`` or the build-time knobs afterwards had
    no effect on an already-built collection and no way to ask for one -- config
    said one thing and the database did another, indefinitely.

    Plain runs only rebuild when the *method* changed, which is what
    ``ensure_embedding_ann_index`` already reconciles. ``force`` drops the index
    first, so a changed ``hnsw_m``/``ef_construction`` is picked up too: those
    are baked in at build time and ``CREATE INDEX IF NOT EXISTS`` would silently
    keep the old graph.
    """
    revision = get_active_revision(session, collection)
    if revision is None:
        return ReindexOutcome("no_active")

    profile_id = revision.embedding_profile_id
    engine = cast(Engine, session.get_bind())
    with engine.connect() as conn:
        if not vector_table_exists(conn, profile_id):
            return ReindexOutcome("no_vectors")
        previous_method = index_access_method(conn, vector_index_name(profile_id))

    # The drop rides along inside ensure_revision_ann_index's own connection and
    # commit rather than happening here first. Committing it separately meant a
    # rebuild that failed -- a >2000-dim profile HNSW rejects, a statement
    # timeout, Ctrl-C, disk full -- left the collection with no ANN index at
    # all, permanently and silently: search still succeeds by sequential scan.
    ensure_revision_ann_index(session, revision, config, force_rebuild=force)
    session.commit()
    return ReindexOutcome(
        "reindexed", method=config.index.method, previous_method=previous_method
    )


def list_collection_revisions(session: Session, collection: str) -> list[PipelineRevision]:
    """Return revision history for one collection."""
    return (
        session.query(PipelineRevision)
        .filter_by(collection=collection)
        .order_by(PipelineRevision.id.desc())
        .all()
    )
