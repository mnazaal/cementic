"""Collection and revision persistence helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import func
from sqlalchemy.orm import Session

from cementic.db import Chunk, ExtractedDocument, PipelineRevision, SourceDocument
from cementic.revisions import promote_revision


@dataclass(frozen=True)
class DeleteCollectionResult:
    """Summary of collection deletion work."""

    deleted_docs: int
    deleted_chunks: int
    artifact_paths: list[str]


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
            session.query(func.count(SourceDocument.id)).filter_by(collection=name).scalar()
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
    session.commit()
    return DeleteCollectionResult(
        deleted_docs=deleted_docs,
        deleted_chunks=deleted_chunks,
        artifact_paths=artifact_paths,
    )


def remove_artifacts(paths: list[str]) -> None:
    """Best-effort removal of extracted document artifacts."""
    for artifact_path in paths:
        try:
            Path(artifact_path).unlink(missing_ok=True)
        except OSError:
            pass


def promote_ready_revision(session: Session, collection: str) -> PipelineRevision | None:
    """Promote the newest ready revision for one collection."""
    revision = (
        session.query(PipelineRevision)
        .filter_by(collection=collection, status="ready")
        .order_by(PipelineRevision.id.desc())
        .first()
    )
    if revision is None:
        return None

    promote_revision(session, collection, revision)
    session.commit()
    return revision


def list_collection_revisions(session: Session, collection: str) -> list[PipelineRevision]:
    """Return revision history for one collection."""
    return (
        session.query(PipelineRevision)
        .filter_by(collection=collection)
        .order_by(PipelineRevision.id.desc())
        .all()
    )
