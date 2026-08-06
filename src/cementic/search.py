"""Search functionality over active pipeline revisions."""

from __future__ import annotations

# mypy: disable-error-code="import-untyped"
from typing import Any, TypedDict

from sqlalchemy import text

from cementic.config import Config, get_config
from cementic.db import PipelineRevision, get_engine, get_session_factory
from cementic.embedding_runtime import (
    client_is_healthy_or_busy,
    create_provider,
    runtime_spec_from_profile_json,
)
from cementic.revisions import BUILDING_STATUSES, get_active_revision
from cementic.vector_store import (
    index_access_method,
    query_tuning_sql,
    to_vector_literal,
    vector_index_name,
    vector_table_exists,
    vector_table_name,
)

MAX_SEARCH_RESULTS = 50
MAX_QUERY_CHARS = 8_000


class SearchResult(TypedDict):
    """Type for search results."""

    collection: str
    source_path: str
    content: str
    score: float
    distance: float
    score_kind: str


_DISTANCE_OPERATORS = {
    "cosine": "<=>",
    "l2": "<->",
    "ip": "<#>",
}


def _distance_operator(metric: str) -> str:
    """Return the pgvector distance operator for a configured metric."""
    try:
        return _DISTANCE_OPERATORS[metric]
    except KeyError:
        raise ValueError(f"Unsupported distance metric: {metric}") from None


def _score_from_distance(metric: str, distance: float) -> tuple[float, str]:
    """Convert pgvector distance to a higher-is-better score with explicit semantics."""
    if metric == "cosine":
        return 1.0 - distance, "cosine_similarity"
    if metric == "l2":
        return -distance, "negative_l2_distance"
    if metric == "ip":
        return -distance, "inner_product"
    raise ValueError(f"Unsupported distance metric: {metric}")


def _searchable_revisions(revisions: list[PipelineRevision]) -> list[PipelineRevision]:
    """Select one searchable revision per collection, preferring active."""
    status_rank = {"active": 3, "ready": 2, "building": 1}
    searchable_by_collection: dict[str, PipelineRevision] = {}
    for revision in revisions:
        current = searchable_by_collection.get(revision.collection)
        if current is None:
            searchable_by_collection[revision.collection] = revision
            continue
        current_rank = status_rank.get(current.status, 0)
        revision_rank = status_rank.get(revision.status, 0)
        if revision_rank > current_rank:
            searchable_by_collection[revision.collection] = revision
    return list(searchable_by_collection.values())


class Searcher:
    """Searcher for semantic search over active revisions."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or get_config()
        engine = get_engine(self.config.database.url)
        self.Session = get_session_factory(engine)

    def search(
        self,
        query: str,
        top_k: int = 10,
        collections: list[str] | None = None,
    ) -> list[SearchResult]:
        if top_k < 1 or top_k > MAX_SEARCH_RESULTS:
            raise ValueError(f"top_k must be between 1 and {MAX_SEARCH_RESULTS}")
        if len(query) > MAX_QUERY_CHARS:
            raise ValueError(f"query too long: {len(query)} characters (max {MAX_QUERY_CHARS})")

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
            if not client_is_healthy_or_busy(embedding_client, self.config):
                raise RuntimeError("Active embedding provider is not healthy")
            query_embedding = embedding_client.embed(embedding_client.format_query(query))
            query_literal = to_vector_literal(query_embedding)
            distance_metric = embedding_profile.distance_metric
            distance_operator = _distance_operator(distance_metric)

            combined: list[SearchResult] = []
            for revision in revisions:
                table = vector_table_name(revision.embedding_profile_id)
                if not vector_table_exists(session.connection(), revision.embedding_profile_id):
                    continue
                # Tune for the index that actually exists rather than the
                # configured method, which can drift until the index is rebuilt.
                actual_method = (
                    index_access_method(
                        session.connection(),
                        vector_index_name(revision.embedding_profile_id),
                    )
                    or self.config.index.method
                )
                tuning = query_tuning_sql(
                    actual_method,
                    hnsw_ef_search=self.config.index.hnsw_ef_search,
                    diskann_query_rescore=self.config.index.diskann_query_rescore,
                )
                if tuning is not None:
                    session.execute(text(tuning))
                statement = text(
                    "SELECT sd.collection AS collection, sd.source_path AS source_path, "
                    "c.content AS content, "
                    f"ev.embedding {distance_operator} (:query)::vector AS distance "
                    f"FROM {table} ev "
                    "JOIN chunks_v2 c ON c.id = ev.chunk_id "
                    "JOIN chunked_documents cd ON c.chunked_document_id = cd.id "
                    "JOIN source_documents sd ON c.document_id = sd.id "
                    "WHERE sd.collection = :collection "
                    "AND sd.status <> 'deleted' "
                    "AND cd.chunk_profile_id = :chunk_profile_id "
                    f"ORDER BY ev.embedding {distance_operator} (:query)::vector "
                    "LIMIT :k"
                )
                rows = session.execute(
                    statement,
                    {
                        "query": query_literal,
                        "collection": revision.collection,
                        "chunk_profile_id": revision.chunk_profile_id,
                        "k": top_k,
                    },
                )
                for row in rows:
                    distance = float(row.distance)
                    score, score_kind = _score_from_distance(distance_metric, distance)
                    combined.append(
                        SearchResult(
                            collection=row.collection,
                            source_path=row.source_path,
                            content=row.content,
                            score=score,
                            distance=distance,
                            score_kind=score_kind,
                        )
                    )

        combined.sort(key=lambda result: result["score"], reverse=True)
        return combined[:top_k]

    def unsearchable_collections(self, collections: list[str]) -> list[str]:
        """Return requested collections that have no searchable revision.

        A typo'd or not-yet-indexed collection otherwise looks exactly like a
        query with no matches, so callers can tell the two apart.
        """
        with self.Session() as session:
            found = {
                revision.collection
                for revision in self._load_searchable_revisions(session, collections)
            }
        return [collection for collection in collections if collection not in found]

    def _load_searchable_revisions(
        self,
        session: Any,
        collections: list[str] | None,
    ) -> list[PipelineRevision]:
        if collections is not None:
            wanted = list(dict.fromkeys(collections))
            revisions: list[PipelineRevision] = []
            for collection in wanted:
                active_revision = get_active_revision(session, collection)
                if active_revision is not None:
                    revisions.append(active_revision)
                    continue

                fallback_revision = (
                    session.query(PipelineRevision)
                    .filter(
                        PipelineRevision.collection == collection,
                        PipelineRevision.status.in_(BUILDING_STATUSES),
                    )
                    .order_by(PipelineRevision.id.desc())
                    .first()
                )
                if fallback_revision is not None:
                    revisions.append(fallback_revision)
            return revisions

        revisions = (
            session.query(PipelineRevision)
            .filter(PipelineRevision.status.in_(["active", "ready", "building"]))
            .order_by(PipelineRevision.collection, PipelineRevision.id.desc())
            .all()
        )
        return _searchable_revisions(revisions)


def _create_embedding_provider(config_json: str, config: Config) -> Any:
    spec = runtime_spec_from_profile_json(config_json)
    return create_provider(spec, config)
