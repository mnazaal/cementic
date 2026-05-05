"""Search functionality over active pipeline revisions."""

from __future__ import annotations

import json

# mypy: disable-error-code="import-untyped"
from typing import Any, Optional, TypedDict

from pgvector.sqlalchemy import Vector
from sqlalchemy import bindparam, cast

from cementic.config import Config, get_config
from cementic.db import (
    Chunk,
    ChunkedDocument,
    ChunkEmbedding,
    PipelineRevision,
    SourceDocument,
    get_engine,
    get_session_factory,
)
from cementic.embedding_providers import get_embedding_provider
from cementic.embedding_runtime import get_llama_cpp_runtime_client
from cementic.embedding_text import format_query_text_for_model


class SearchResult(TypedDict):
    """Type for search results."""

    collection: str
    source_path: str
    content: str
    score: float
    page_start: int
    page_end: int


class Searcher:
    """Searcher for semantic search over active revisions."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or get_config()
        engine = get_engine(self.config.database.url)
        self.Session = get_session_factory(engine)

    def search(
        self,
        query: str,
        top_k: int = 10,
        collections: Optional[list[str]] = None,
    ) -> list[SearchResult]:
        with self.Session() as session:
            revisions = self._load_searchable_revisions(session, collections)
            if not revisions:
                return []

            embedding_profile_ids = {revision.embedding_profile_id for revision in revisions}
            if len(embedding_profile_ids) != 1:
                raise RuntimeError(
                    "Selected collections use different active embedding models; "
                    "search them separately"
                )

            embedding_profile = revisions[0].embedding_profile
            embedding_client = _create_embedding_provider(
                embedding_profile.config_json, self.config
            )
            if not embedding_client.health_check():
                raise RuntimeError("Active embedding provider is not healthy")
            query_embedding = embedding_client.embed(
                format_query_text_for_model(query, embedding_profile.model_identifier)
            )

            combined: list[SearchResult] = []
            for revision in revisions:
                query_param: Any = bindparam("query_embedding", value=query_embedding)
                vector_type = Vector(revision.embedding_profile.embedding_dim)
                distance_expression = cast(
                    ChunkEmbedding.embedding,
                    vector_type,
                ).cosine_distance(
                    cast(
                        query_param,
                        vector_type,
                    )
                )
                rows = (
                    session.query(
                        Chunk,
                        distance_expression.label("distance"),
                    )
                    .join(ChunkEmbedding, ChunkEmbedding.chunk_id == Chunk.id)
                    .join(ChunkedDocument, Chunk.chunked_document_id == ChunkedDocument.id)
                    .join(SourceDocument, Chunk.document_id == SourceDocument.id)
                    .filter(
                        SourceDocument.collection == revision.collection,
                        ChunkedDocument.chunk_profile_id == revision.chunk_profile_id,
                        ChunkEmbedding.embedding_profile_id == revision.embedding_profile_id,
                        ChunkEmbedding.status == "done",
                        ChunkEmbedding.embedding.isnot(None),
                    )
                    .order_by(distance_expression)
                    .limit(top_k)
                    .all()
                )
                for chunk, distance in rows:
                    combined.append(
                        SearchResult(
                            collection=chunk.document.collection,
                            source_path=chunk.document.source_path,
                            content=chunk.content,
                            score=1.0 - distance,
                            page_start=chunk.page_start or 0,
                            page_end=chunk.page_end or 0,
                        )
                    )

        combined.sort(key=lambda result: result["score"], reverse=True)
        return combined[:top_k]

    def _searchable_revisions(self, revisions: list[PipelineRevision]) -> list[PipelineRevision]:
        searchable_by_collection: dict[str, PipelineRevision] = {}
        for revision in revisions:
            current = searchable_by_collection.get(revision.collection)
            if current is None:
                searchable_by_collection[revision.collection] = revision
                continue
            if current.status != "active" and revision.status == "active":
                searchable_by_collection[revision.collection] = revision
        return list(searchable_by_collection.values())

    def _load_searchable_revisions(
        self,
        session: Any,
        collections: Optional[list[str]],
    ) -> list[PipelineRevision]:
        if collections is not None:
            wanted = list(dict.fromkeys(collections))
            revisions: list[PipelineRevision] = []
            for collection in wanted:
                active_revision = (
                    session.query(PipelineRevision)
                    .filter_by(collection=collection, status="active")
                    .order_by(PipelineRevision.id.desc())
                    .first()
                )
                if active_revision is not None:
                    revisions.append(active_revision)
                    continue

                fallback_revision = (
                    session.query(PipelineRevision)
                    .filter(
                        PipelineRevision.collection == collection,
                        PipelineRevision.status.in_(["building", "ready"]),
                    )
                    .order_by(PipelineRevision.id.desc())
                    .first()
                )
                if fallback_revision is not None:
                    revisions.append(fallback_revision)
            return revisions

        active_revisions = (
            session.query(PipelineRevision)
            .filter_by(status="active")
            .order_by(PipelineRevision.collection, PipelineRevision.id.desc())
            .all()
        )
        if active_revisions:
            return self._searchable_revisions(active_revisions)

        building_revisions = (
            session.query(PipelineRevision)
            .filter(PipelineRevision.status.in_(["building", "ready"]))
            .order_by(PipelineRevision.collection, PipelineRevision.id.desc())
            .all()
        )
        return self._searchable_revisions(building_revisions)


def _create_embedding_provider(config_json: str, config: Config | None = None) -> Any:
    payload = json.loads(config_json)
    provider = payload["provider"]
    if provider == "ollama":
        return get_embedding_provider(
            "ollama",
            host=payload["host"],
            model=payload["model_identifier"],
            embedding_dim=payload["embedding_dim"],
        )
    if config is not None:
        return get_llama_cpp_runtime_client(config)
    return get_embedding_provider(
        "llama-cpp",
        model_path=payload["model_identifier"],
        n_ctx=payload["n_ctx"],
        n_gpu_layers=payload["n_gpu_layers"],
        embedding_dim=payload["embedding_dim"],
        verbose=payload["verbose"],
    )
