"""Search functionality over active pipeline revisions."""

from __future__ import annotations

# mypy: disable-error-code="import-untyped"
from typing import Any, TypedDict

import tiktoken
from sqlalchemy import text

from cementic.chunk import TOKENIZER
from cementic.config import Config, get_config
from cementic.db import PipelineRevision, get_engine, get_session_factory
from cementic.embedding_runtime import (
    create_provider,
    runtime_spec_from_profile_json,
)
from cementic.revisions import BUILDING_STATUSES
from cementic.vector_store import (
    ensure_vector_table_schema,
    index_access_method,
    knn_sql,
    pgvector_version,
    query_tuning_statements,
    supports_hnsw_iterative_scan,
    to_vector_literal,
    vector_index_name,
    vector_table_exists,
)

MAX_SEARCH_RESULTS = 50
MAX_QUERY_CHARS = 8_000


#: Fraction of the context window a query may occupy. The counting tokenizer
#: (cl100k_base, shared with chunking) is not the embedding model's own, so the
#: margin absorbs the disagreement plus any task prefix the provider prepends.
_QUERY_CONTEXT_MARGIN = 0.9


def _reject_query_over_context(query: str, n_ctx: int) -> None:
    """Raise if the query cannot fit the embedding model's context window.

    The server truncates over-long input silently, so the tail of a long query
    simply stopped affecting the results: two queries sharing a long prefix and
    differing only in their final words returned bit-identical scores. Refusing
    is the honest answer -- a silently truncated query looks like a working one.

    This is the cheap, local pre-check only. It is deliberately generous,
    because ``TOKENIZER`` is not the model's own tokenizer and the ratio between
    them runs to 1.33 on ordinary English: the exact check happens against the
    served model in ``RemoteEmbeddingClient.embed``, which is the only place the
    real count is available.
    """
    budget = max(1, int(n_ctx * _QUERY_CONTEXT_MARGIN))
    tokens = len(tiktoken.get_encoding(TOKENIZER).encode(query))
    if tokens > budget:
        raise ValueError(
            f"query too long: about {tokens} tokens, but the embedding model's "
            f"context window is {n_ctx} (usable {budget}). Shorten the query -- "
            "the model would silently ignore everything past the limit."
        )


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


def _mixed_model_message(revisions: list[PipelineRevision]) -> str:
    """Name which collection uses which model, and how to proceed (pure).

    The old message -- "different active embedding models; search them
    separately" -- named no collection, so working out which one to drop meant
    reading `collection list` and comparing revision labels by hand. It also
    said "active" about revisions that may be `building` or `ready`, which is
    how a half-built collection ended up blamed for a config change nobody had
    made.

    Refusing rather than dropping the odd collection is deliberate: distances
    from different models are not comparable, so a merged ranking would be
    silently meaningless -- worse than an error, which is at least visible.
    """
    by_model: dict[str, list[str]] = {}
    for revision in sorted(revisions, key=lambda item: item.collection):
        model = revision.embedding_profile.model_identifier
        by_model.setdefault(model, []).append(f"{revision.collection} ({revision.status})")

    groups = "; ".join(
        f"{model}: {', '.join(collections)}" for model, collections in sorted(by_model.items())
    )
    largest = max(by_model.values(), key=len)
    example = " ".join(entry.split(" ")[0] for entry in largest)
    return (
        f"cannot search these collections together -- they are indexed by different "
        f"embedding models, and scores from different models are not comparable. "
        f"{groups}. Search one model's collections at a time, e.g. -c {example}"
    )


#: Statuses a revision may be searched in, in no particular order; the *rank*
#: between them lives in _searchable_revisions.
SEARCHABLE_STATUSES = ("active", *BUILDING_STATUSES)


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
        self._supports_iterative_scan: bool | None = None

    def _iterative_scan_mode(self, session: Any) -> str | None:
        """The configured iterative-scan mode, if this server understands it.

        Cached per Searcher and gated on the pgvector version rather than
        attempted-and-caught: an unknown qualified setting is accepted as a
        placeholder until pgvector's module loads on that connection and
        rejected afterwards, so an ungated SET fails only on pooled connections
        that had already run a vector query.
        """
        if self.config.index.hnsw_iterative_scan == "off":
            return None
        if self._supports_iterative_scan is None:
            self._supports_iterative_scan = supports_hnsw_iterative_scan(
                pgvector_version(session.connection())
            )
        if not self._supports_iterative_scan:
            return None
        return self.config.index.hnsw_iterative_scan

    def search(
        self,
        query: str,
        top_k: int = 10,
        collections: list[str] | None = None,
    ) -> list[SearchResult]:
        if top_k < 1 or top_k > MAX_SEARCH_RESULTS:
            raise ValueError(f"top_k must be between 1 and {MAX_SEARCH_RESULTS}")
        if not query.strip():
            # An empty query embeds to a real vector, so this used to return a
            # confidently ranked top-k -- the nearest neighbours of nothing --
            # with no indication the query was empty.
            raise ValueError("query cannot be empty")
        if len(query) > MAX_QUERY_CHARS:
            raise ValueError(f"query too long: {len(query)} characters (max {MAX_QUERY_CHARS})")

        with self.Session() as session:
            revisions = self._load_searchable_revisions(session, collections)
            if not revisions:
                return []

            embedding_profile_ids = {revision.embedding_profile_id for revision in revisions}
            if len(embedding_profile_ids) != 1:
                raise RuntimeError(_mixed_model_message(revisions))

            embedding_profile = revisions[0].embedding_profile
            # Bound the query against the window of the profile the daemon will
            # actually load, not current config. They diverge whenever config
            # changed after indexing, and the wrong one is wrong both ways: it
            # rejects queries that would have fit, and admits queries that get
            # silently truncated -- the failure this check exists to prevent.
            _reject_query_over_context(
                query,
                runtime_spec_from_profile_json(embedding_profile.config_json).n_ctx
                or self.config.llama_cpp.n_ctx,
            )
            # No separate health probe here: _create_embedding_provider only
            # returns once the served fingerprint is confirmed, and raises with a
            # better message otherwise. Re-probing re-asked a question already
            # answered, and if the worker started a batch in between, that second
            # probe could itself wait out the batch -- adding minutes to a search.
            embedding_client = _create_embedding_provider(
                embedding_profile.config_json, self.config
            )
            query_embedding = embedding_client.embed(embedding_client.format_query(query))
            query_literal = to_vector_literal(query_embedding)
            distance_metric = embedding_profile.distance_metric
            distance_operator = _distance_operator(distance_metric)

            combined: list[SearchResult] = []
            for revision in revisions:
                if not vector_table_exists(session.connection(), revision.embedding_profile_id):
                    continue
                # Migrate here too, not only in the embed step. The query below
                # names the filter columns, so on a database written by an older
                # cementic search would fail with UndefinedColumn until someone
                # happened to run the worker -- an upgrade that breaks reading
                # until you write. Idempotent, and a catalog lookup once done.
                ensure_vector_table_schema(
                    session.connection(),
                    revision.embedding_profile_id,
                    embedding_profile.embedding_dim,
                )
                # ...and commit it. Without this the session's context manager
                # closes and rolls the DDL back, so "once done" never arrived:
                # on a pre-migration database every search re-ran the full-table
                # backfill, held ACCESS EXCLUSIVE on the vector table for the
                # whole query, and threw the work away. Committing here also
                # ends the transaction the tuning SETs below would otherwise
                # join, which is why they are re-applied per revision anyway.
                session.commit()
                # Tune for the index that actually exists rather than the
                # configured method, which can drift until the index is rebuilt.
                actual_method = (
                    index_access_method(
                        session.connection(),
                        vector_index_name(revision.embedding_profile_id),
                    )
                    or self.config.index.method
                )
                for tuning in query_tuning_statements(
                    actual_method,
                    hnsw_ef_search=self.config.index.hnsw_ef_search,
                    diskann_query_rescore=self.config.index.diskann_query_rescore,
                    top_k=top_k,
                    hnsw_iterative_scan=self._iterative_scan_mode(session),
                ):
                    session.execute(text(tuning))
                # Every filter is on the vector row itself. The vector table is
                # per *embedding* profile, so it also holds vectors from other
                # collections and from other revisions sharing that model --
                # restricting to the chunk profile alone is not enough, since a
                # revision whose extractor changed reuses the same chunk and
                # embedding profiles and pruning keeps the newest retired
                # revision. What is *not* here is the freshness join: stale rows
                # are deleted when content is superseded or a document removed,
                # rather than filtered out at query time, because a filter on a
                # joined table stops the planner using the ANN index at all.
                statement = text(
                    knn_sql(
                        revision.embedding_profile_id, distance_operator=distance_operator
                    )
                )
                rows = session.execute(
                    statement,
                    {
                        "query": query_literal,
                        "collection": revision.collection,
                        "chunk_profile_id": revision.chunk_profile_id,
                        "extractor_profile_id": revision.extractor_profile_id,
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
        # One revision-choice rule for both paths, via _searchable_revisions.
        # The -c path used to take the newest in-flight revision by id, so a
        # `building` one out-ranked a `ready` one -- while searching the same
        # collection without -c preferred ready. Same query, same data,
        # different answer depending on how the collection was named.
        query = session.query(PipelineRevision).filter(
            PipelineRevision.status.in_(SEARCHABLE_STATUSES)
        )
        if collections is not None:
            query = query.filter(
                PipelineRevision.collection.in_(list(dict.fromkeys(collections)))
            )
        revisions = query.order_by(
            PipelineRevision.collection, PipelineRevision.id.desc()
        ).all()
        return _searchable_revisions(revisions)


def _create_embedding_provider(config_json: str, config: Config) -> Any:
    spec = runtime_spec_from_profile_json(config_json)
    return create_provider(spec, config)
