"""Collection and revision persistence helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from cementic.config import Config
from cementic.db import Chunk, ExtractedDocument, PipelineRevision, SourceDocument
from cementic.pipeline_worker import (
    PipelineCounts,
    compute_revision_counts,
    revision_failure_total,
)
from cementic.revisions import promote_revision
from cementic.storage import safe_remove_artifact
from cementic.vector_store import drop_vector_table


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
    building_revision_label: str | None


def list_collections(session: Session) -> list[CollectionSummary]:
    """Return collection summaries ordered by collection name."""
    collection_names = [
        str(name)
        for (name,) in session.query(SourceDocument.collection)
        .group_by(SourceDocument.collection)
        .order_by(SourceDocument.collection)
        .all()
    ]

    summaries: list[CollectionSummary] = []
    for name in collection_names:
        active_revision = (
            session.query(PipelineRevision)
            .filter_by(collection=name, status="active")
            .order_by(PipelineRevision.id.desc())
            .first()
        )
        building_revision = (
            session.query(PipelineRevision)
            .filter(
                PipelineRevision.collection == name,
                PipelineRevision.status.in_(["building", "ready"]),
            )
            .order_by(PipelineRevision.id.desc())
            .first()
        )
        documents = (
            session.query(func.count(SourceDocument.id))
            .filter(SourceDocument.collection == name, SourceDocument.status != "deleted")
            .scalar()
        ) or 0
        summaries.append(
            CollectionSummary(
                name=name,
                documents=int(documents),
                active_revision_label=getattr(active_revision, "label", None),
                building_revision_label=getattr(building_revision, "label", None),
            )
        )

    return summaries


def delete_collection_records(session: Session, collection: str) -> DeleteCollectionResult | None:
    """Delete a collection from the database and return removed artifact paths."""
    docs = session.query(SourceDocument).filter_by(collection=collection).all()
    if not docs:
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
    for profile_id in sorted(set(candidate_profile_ids)):
        remaining = (
            session.query(func.count(PipelineRevision.id))
            .filter_by(embedding_profile_id=profile_id)
            .scalar()
        ) or 0
        if int(remaining) == 0:
            vector_profile_ids.append(profile_id)
    session.commit()
    return DeleteCollectionResult(
        deleted_docs=deleted_docs,
        deleted_chunks=deleted_chunks,
        artifact_paths=artifact_paths,
        vector_profile_ids=vector_profile_ids,
    )


def remove_artifacts(paths: list[str], *, config: Config) -> None:
    """Best-effort removal of extracted document artifacts."""
    for artifact_path in paths:
        safe_remove_artifact(config, artifact_path)


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

    ``status`` is one of ``"promoted"``, ``"no_ready"`` (nothing to promote), or
    ``"blocked_by_failures"`` (a ready revision exists but built with failures and
    ``force`` was not set, so it was left untouched).
    """

    status: str
    revision: PipelineRevision | None = None
    failures: PipelineCounts | None = None


def promote_ready_revision(
    session: Session, collection: str, *, config: Config, force: bool = False
) -> PromotionOutcome:
    """Promote the newest ready revision for one collection.

    A revision can reach ``ready`` with failed documents/chunks (failures are
    terminal so the build can finish). Promoting one silently would publish a
    partial index, so unless ``force`` is set this refuses and reports the failure
    counts instead.
    """
    revision = find_ready_revision(session, collection)
    if revision is None:
        return PromotionOutcome("no_ready")

    counts = compute_revision_counts(session, collection, revision)
    if not force and revision_failure_total(counts) > 0:
        return PromotionOutcome("blocked_by_failures", revision=revision, failures=counts)

    promote_revision(session, collection, revision, config=config)
    session.commit()
    return PromotionOutcome("promoted", revision=revision)


def list_collection_revisions(session: Session, collection: str) -> list[PipelineRevision]:
    """Return revision history for one collection."""
    return (
        session.query(PipelineRevision)
        .filter_by(collection=collection)
        .order_by(PipelineRevision.id.desc())
        .all()
    )
