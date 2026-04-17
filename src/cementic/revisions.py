"""Pipeline revision lifecycle helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
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
from cementic.profiles import (
    get_or_create_chunk_profile,
    get_or_create_embedding_profile,
    get_or_create_extractor_profile,
)

BUILDING_STATUSES = {"building", "ready"}


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


def get_target_revision(session: Session, collection: str, config: Config) -> PipelineRevision:
    """Return the revision matching current config, creating it if needed."""
    extractor_profile = get_or_create_extractor_profile(session, config)
    chunk_profile = get_or_create_chunk_profile(session, config)
    embedding_profile = get_or_create_embedding_profile(session, config)

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
        return current

    (
        session.query(PipelineRevision)
        .filter(
            PipelineRevision.collection == collection,
            PipelineRevision.status.in_(["building", "ready"]),
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


def promote_revision(
    session: Session, collection: str, revision: PipelineRevision
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
    prune_collection_history(session, collection)
    return revision


def ensure_revision_ann_index(session: Session, revision: PipelineRevision) -> None:
    """Ensure the ANN index exists for the revision's embedding profile."""
    bind = session.get_bind()
    if bind is None:
        raise RuntimeError("Session is not bound to an engine")
    ensure_embedding_ann_index(
        cast(Engine, bind),
        profile_id=revision.embedding_profile_id,
        embedding_dim=revision.embedding_profile.embedding_dim,
        distance_metric=revision.embedding_profile.distance_metric,
    )


def prune_collection_history(session: Session, collection: str) -> None:
    """Keep only active and most recent retired history for one collection."""
    session.flush()
    session.expire_all()
    revisions = (
        session.query(PipelineRevision)
        .filter_by(collection=collection)
        .order_by(PipelineRevision.id.desc())
        .all()
    )
    retired = [revision for revision in revisions if revision.status == "retired"]
    keep_retired_id = retired[0].id if retired else None

    removable = [
        revision
        for revision in revisions
        if revision.status == "superseded"
        or (revision.status == "retired" and revision.id != keep_retired_id)
    ]
    if not removable:
        return

    removable_revision_ids = {revision.id for revision in removable}
    kept = [revision for revision in revisions if revision.id not in removable_revision_ids]
    keep_extractor_profile_ids = {revision.extractor_profile_id for revision in kept}
    keep_chunk_profile_ids = {revision.chunk_profile_id for revision in kept}
    keep_embedding_profile_ids = {revision.embedding_profile_id for revision in kept}

    extracted_to_remove = (
        session.query(ExtractedDocument.id, ExtractedDocument.artifact_path)
        .join(SourceDocument, ExtractedDocument.document_id == SourceDocument.id)
        .filter(
            SourceDocument.collection == collection,
            ExtractedDocument.extractor_profile_id.not_in(keep_extractor_profile_ids),
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
            ChunkedDocument.chunk_profile_id.not_in(keep_chunk_profile_ids),
        )
    )
    chunk_ids = select(Chunk.id).where(Chunk.chunked_document_id.in_(chunked_ids))

    session.query(ChunkEmbedding).filter(
        ChunkEmbedding.chunk_id.in_(chunk_ids),
        ChunkEmbedding.embedding_profile_id.not_in(keep_embedding_profile_ids),
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

    session.query(PipelineRevision).filter(PipelineRevision.id.in_(removable_revision_ids)).delete(
        synchronize_session=False
    )
    session.flush()

    for artifact_path in artifact_paths:
        try:
            Path(artifact_path).unlink(missing_ok=True)
        except OSError:
            pass
