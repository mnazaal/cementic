"""Pipeline revision lifecycle helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import cast

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from cementic.config import Config
from cementic.db import (
    Chunk,
    ChunkedDocument,
    ChunkEmbedding,
    ExtractedDocument,
    PipelineRevision,
    SourceDocument,
    ensure_embedding_ann_index,
)
from cementic.embedding_provider import EmbeddingProvider
from cementic.index_strategies import IndexParams
from cementic.profiles import (
    get_or_create_chunk_profile,
    get_or_create_embedding_profile,
    get_or_create_extractor_profile,
)
from cementic.storage import safe_remove_artifact

#: Revision statuses that represent an in-flight (not yet promoted) build.
BUILDING_STATUSES = ("building", "ready")


@dataclass(frozen=True)
class RevisionPrunePlan:
    """Pure pruning decision for old pipeline revisions."""

    removable_revision_ids: set[int]
    keep_extractor_profile_ids: set[int]
    keep_chunk_profile_ids: set[int]
    keep_embedding_profile_ids: set[int]


def _revision_prune_plan(revisions: Sequence[PipelineRevision]) -> RevisionPrunePlan:
    """Plan which revision history can be removed without DB or file I/O."""
    retired = [revision for revision in revisions if revision.status == "retired"]
    keep_retired_id = retired[0].id if retired else None

    removable_revision_ids = {
        revision.id
        for revision in revisions
        if revision.status == "superseded"
        or (revision.status == "retired" and revision.id != keep_retired_id)
    }
    kept = [revision for revision in revisions if revision.id not in removable_revision_ids]

    return RevisionPrunePlan(
        removable_revision_ids=removable_revision_ids,
        keep_extractor_profile_ids={revision.extractor_profile_id for revision in kept},
        keep_chunk_profile_ids={revision.chunk_profile_id for revision in kept},
        keep_embedding_profile_ids={revision.embedding_profile_id for revision in kept},
    )


def _default_revision_label(revision: PipelineRevision) -> str:
    return (
        f"{revision.extractor_profile.name}-"
        f"{revision.chunk_profile.fingerprint[:8]}-"
        f"{revision.embedding_profile.provider}-{revision.embedding_profile.fingerprint[:8]}"
    )


def get_active_revision(session: Session, collection: str) -> PipelineRevision | None:
    """Return the active revision for one collection."""
    return (
        session.query(PipelineRevision)
        .filter_by(collection=collection, status="active")
        .order_by(PipelineRevision.id.desc())
        .first()
    )


def get_target_revision(
    session: Session,
    collection: str,
    config: Config,
    provider: EmbeddingProvider | None = None,
) -> PipelineRevision:
    """Return the revision matching current config, creating it if needed."""
    extractor_profile = get_or_create_extractor_profile(session, config)
    chunk_profile = get_or_create_chunk_profile(session, config)
    embedding_profile = get_or_create_embedding_profile(session, config, provider)

    current = (
        session.query(PipelineRevision)
        .filter_by(
            collection=collection,
            extractor_profile_id=extractor_profile.id,
            chunk_profile_id=chunk_profile.id,
            embedding_profile_id=embedding_profile.id,
        )
        .order_by(PipelineRevision.id.desc())
        .first()
    )
    if current is not None:
        if current.status in ("retired", "superseded"):
            # Config reverted to a previously built revision (e.g. rolling back a
            # model change). Resurrect it as the building target — its artifacts
            # are reused, so it completes and becomes promotable again. Whatever
            # else was building for this collection is no longer the target.
            (
                session.query(PipelineRevision)
                .filter(
                    PipelineRevision.collection == collection,
                    PipelineRevision.status.in_(BUILDING_STATUSES),
                )
                .update({"status": "superseded"}, synchronize_session=False)
            )
            current.status = "building"
            session.flush()
        return current

    (
        session.query(PipelineRevision)
        .filter(
            PipelineRevision.collection == collection,
            PipelineRevision.status.in_(BUILDING_STATUSES),
        )
        .update({"status": "superseded"}, synchronize_session=False)
    )

    revision = PipelineRevision(
        collection=collection,
        extractor_profile_id=extractor_profile.id,
        chunk_profile_id=chunk_profile.id,
        embedding_profile_id=embedding_profile.id,
        status="building",
    )
    session.add(revision)
    session.flush()
    revision.label = _default_revision_label(revision)
    session.flush()
    return revision


def mark_revision_ready(session: Session, revision: PipelineRevision) -> None:
    """Mark a revision ready for promotion."""
    if revision.status == "building":
        revision.status = "ready"
        session.flush()


def requeue_interrupted_artifacts(
    session: Session, collection: str, revision: PipelineRevision
) -> None:
    """Reset failed/interrupted artifacts of a revision to ``pending``.

    Failures are terminal *within* a single worker run (so a build can still
    finish), but a fresh ``cementic start`` re-queues them — along with any rows a
    crashed worker left in ``processing`` — for another attempt. Deleted source
    documents are skipped so they leave no dangling ``pending`` rows.
    """
    interrupted = ("failed", "processing")
    doc_ids = select(SourceDocument.id).where(
        SourceDocument.collection == collection,
        SourceDocument.status != "deleted",
    )

    session.query(ExtractedDocument).filter(
        ExtractedDocument.document_id.in_(doc_ids),
        ExtractedDocument.extractor_profile_id == revision.extractor_profile_id,
        ExtractedDocument.status.in_(interrupted),
    ).update({"status": "pending"}, synchronize_session=False)

    extracted_ids = select(ExtractedDocument.id).where(
        ExtractedDocument.document_id.in_(doc_ids),
        ExtractedDocument.extractor_profile_id == revision.extractor_profile_id,
    )
    session.query(ChunkedDocument).filter(
        ChunkedDocument.extracted_document_id.in_(extracted_ids),
        ChunkedDocument.chunk_profile_id == revision.chunk_profile_id,
        ChunkedDocument.status.in_(interrupted),
    ).update({"status": "pending"}, synchronize_session=False)

    # Scope to the revision's chunk chain: embeddings of chunks under another
    # chunk profile are never embed candidates for this revision, so resetting
    # them would only create permanently-orphaned pending rows.
    chunked_ids = select(ChunkedDocument.id).where(
        ChunkedDocument.extracted_document_id.in_(extracted_ids),
        ChunkedDocument.chunk_profile_id == revision.chunk_profile_id,
    )
    chunk_ids = select(Chunk.id).where(Chunk.chunked_document_id.in_(chunked_ids))
    session.query(ChunkEmbedding).filter(
        ChunkEmbedding.chunk_id.in_(chunk_ids),
        ChunkEmbedding.embedding_profile_id == revision.embedding_profile_id,
        ChunkEmbedding.status.in_(interrupted),
    ).update({"status": "pending"}, synchronize_session=False)

    session.flush()


def promote_revision(
    session: Session, collection: str, revision: PipelineRevision, *, config: Config
) -> PipelineRevision:
    """Promote a ready revision to active for one collection."""
    if revision.collection != collection:
        raise ValueError("Revision does not belong to the requested collection")
    if revision.status != "ready":
        raise ValueError("Only ready revisions can be promoted")

    (
        session.query(PipelineRevision)
        .filter_by(collection=collection, status="active")
        .update({"status": "retired"}, synchronize_session=False)
    )
    revision.status = "active"
    revision.promoted_at = datetime.now(timezone.utc)
    session.flush()
    prune_collection_history(session, collection, config=config)
    return revision


def ensure_revision_ann_index(
    session: Session, revision: PipelineRevision, config: Config
) -> None:
    """Ensure the configured ANN index exists for the revision's vector table."""
    bind = session.get_bind()
    if bind is None:
        raise RuntimeError("Session is not bound to an engine")
    params = IndexParams(
        hnsw_m=config.index.hnsw_m,
        hnsw_ef_construction=config.index.hnsw_ef_construction,
        diskann_num_neighbors=config.index.diskann_num_neighbors,
        diskann_search_list_size=config.index.diskann_search_list_size,
    )
    ensure_embedding_ann_index(
        cast(Engine, bind),
        profile_id=revision.embedding_profile_id,
        method=config.index.method,
        params=params,
        distance_metric=revision.embedding_profile.distance_metric,
    )


def prune_collection_history(session: Session, collection: str, *, config: Config) -> None:
    """Keep only active and most recent retired history for one collection."""
    session.flush()
    session.expire_all()
    revisions = (
        session.query(PipelineRevision)
        .filter_by(collection=collection)
        .order_by(PipelineRevision.id.desc())
        .all()
    )
    plan = _revision_prune_plan(revisions)
    if not plan.removable_revision_ids:
        return

    extracted_to_remove = (
        session.query(ExtractedDocument.id, ExtractedDocument.artifact_path)
        .join(SourceDocument, ExtractedDocument.document_id == SourceDocument.id)
        .filter(
            SourceDocument.collection == collection,
            ExtractedDocument.extractor_profile_id.not_in(plan.keep_extractor_profile_ids),
        )
        .all()
    )
    extracted_ids = [row.id for row in extracted_to_remove]
    artifact_paths = [row.artifact_path for row in extracted_to_remove if row.artifact_path]

    chunked_ids = (
        select(ChunkedDocument.id)
        .join(ExtractedDocument, ChunkedDocument.extracted_document_id == ExtractedDocument.id)
        .join(SourceDocument, ExtractedDocument.document_id == SourceDocument.id)
        .filter(
            SourceDocument.collection == collection,
            ChunkedDocument.chunk_profile_id.not_in(plan.keep_chunk_profile_ids),
        )
    )
    chunk_ids = select(Chunk.id).where(Chunk.chunked_document_id.in_(chunked_ids))

    session.query(ChunkEmbedding).filter(
        ChunkEmbedding.chunk_id.in_(chunk_ids),
        ChunkEmbedding.embedding_profile_id.not_in(plan.keep_embedding_profile_ids),
    ).delete(synchronize_session=False)
    session.query(Chunk).filter(Chunk.chunked_document_id.in_(chunked_ids)).delete(
        synchronize_session=False
    )
    session.query(ChunkedDocument).filter(ChunkedDocument.id.in_(chunked_ids)).delete(
        synchronize_session=False
    )

    if extracted_ids:
        session.query(ExtractedDocument).filter(ExtractedDocument.id.in_(extracted_ids)).delete(
            synchronize_session=False
        )

    session.query(PipelineRevision).filter(PipelineRevision.id.in_(plan.removable_revision_ids)).delete(
        synchronize_session=False
    )
    session.flush()

    for artifact_path in artifact_paths:
        safe_remove_artifact(config, artifact_path)
