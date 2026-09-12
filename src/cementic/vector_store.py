"""Per-profile vector tables: pure SQL builders + thin execution shells.

Each embedding profile gets its own fixed-dimension ``embedding_vectors_p{id}``
table that holds only completed vectors, so a plain whole-table ANN index (HNSW
or DiskANN) can be built on it — no partial/expression index needed. The pure
builders are unit-testable without a database; the shells just execute them.
"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

_MAX_EMBEDDING_DIM = 8192


def _validate_profile_id(profile_id: int) -> int:
    # Profile ids flow into table/index identifiers, so guard them strictly.
    if not isinstance(profile_id, int) or isinstance(profile_id, bool) or profile_id < 0:
        raise ValueError(f"profile_id must be a non-negative integer, got {profile_id!r}")
    return profile_id


def vector_table_name(profile_id: int) -> str:
    """Name of the per-profile vector table."""
    return f"embedding_vectors_p{_validate_profile_id(profile_id)}"


def vector_index_name(profile_id: int) -> str:
    """Name of the ANN index on the per-profile vector table."""
    return f"ix_embedding_vectors_p{_validate_profile_id(profile_id)}_ann"


def to_vector_literal(vector: Sequence[float]) -> str:
    """Render a vector as a pgvector text literal: ``[1.0,2.0,3.0]``."""
    return "[" + ",".join(repr(float(value)) for value in vector) + "]"


#: Columns carried on the vector row purely so search can filter without
#: joining. Each is immutable for a given chunk: a chunk belongs to exactly one
#: chunked document (hence one chunk profile), which belongs to one extracted
#: document (one extractor profile), under one source document (one collection).
#:
#: They exist because the planner cannot drive an ANN index scan from a filter
#: that lives on a joined table. With the filters here it can: measured at 100k
#: rows and 768 dimensions, the joined form took 407ms and never touched the
#: index, while this form takes about 1ms and does.
FILTER_COLUMNS = ("collection", "extractor_profile_id", "chunk_profile_id")


def create_table_sql(profile_id: int, dim: int) -> str:
    """DDL to create one profile's fixed-dimension vector table."""
    if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 or dim > _MAX_EMBEDDING_DIM:
        raise ValueError(f"dim must be a positive integer <= {_MAX_EMBEDDING_DIM}, got {dim!r}")
    table = vector_table_name(profile_id)
    return (
        f"CREATE TABLE IF NOT EXISTS {table} ("
        "chunk_id INTEGER PRIMARY KEY REFERENCES chunks_v2(id) ON DELETE CASCADE, "
        f"embedding vector({dim}) NOT NULL, "
        "collection TEXT NOT NULL, "
        "extractor_profile_id INTEGER NOT NULL, "
        "chunk_profile_id INTEGER NOT NULL)"
    )


def backfill_filter_columns_sql(profile_id: int) -> str:
    """Populate the filter columns from the joins they replace.

    Derived from the existing rows, so an upgrade costs one UPDATE rather than
    re-embedding the corpus -- which at a few hundred thousand chunks is the
    difference between a minute and a day.
    """
    table = vector_table_name(profile_id)
    return (
        f"UPDATE {table} ev SET collection = sd.collection, "
        "extractor_profile_id = ed.extractor_profile_id, "
        "chunk_profile_id = cd.chunk_profile_id "
        "FROM chunks_v2 c "
        "JOIN chunked_documents cd ON c.chunked_document_id = cd.id "
        "JOIN extracted_documents ed ON cd.extracted_document_id = ed.id "
        "JOIN source_documents sd ON c.document_id = sd.id "
        "WHERE ev.chunk_id = c.id AND ev.collection IS NULL"
    )


def enforce_filter_columns_sql(profile_id: int) -> list[str]:
    """Drop any row the backfill could not resolve, then require the columns.

    A leftover NULL row is unreachable by search anyway; leaving it nullable
    would let it sit there looking indexed. There should be none -- the vector
    table's foreign key cascades from ``chunks_v2`` -- so this is a guard, not a
    routine step.
    """
    table = vector_table_name(profile_id)
    return [
        f"DELETE FROM {table} WHERE collection IS NULL",
        f"ALTER TABLE {table} ALTER COLUMN collection SET NOT NULL",
        f"ALTER TABLE {table} ALTER COLUMN extractor_profile_id SET NOT NULL",
        f"ALTER TABLE {table} ALTER COLUMN chunk_profile_id SET NOT NULL",
    ]


def drop_table_sql(profile_id: int) -> str:
    """DDL to drop one profile's vector table."""
    return f"DROP TABLE IF EXISTS {vector_table_name(profile_id)}"


def delete_vectors_for_collection_sql(profile_id: int) -> str:
    """Parameterised delete of one collection's vectors from a profile's table.

    For pruning a retired model whose profile another collection still uses: the
    table has to stay, but this collection's vectors in it are dead weight. The
    ``ON DELETE CASCADE`` from ``chunks_v2`` does not cover it, because a model
    swap leaves the chunks themselves untouched.
    """
    table = vector_table_name(profile_id)
    return (
        f"DELETE FROM {table} WHERE chunk_id IN ("
        "SELECT c.id FROM chunks_v2 c "
        "JOIN source_documents sd ON c.document_id = sd.id "
        "WHERE sd.collection = :collection)"
    )


def upsert_sql(profile_id: int) -> str:
    """Parameterised upsert of one vector row.

    Binds ``:chunk_id``, ``:embedding`` and the three filter columns. They are
    updated on conflict as well as inserted: re-embedding an existing chunk
    under a corrected profile should not leave the old routing behind.
    """
    table = vector_table_name(profile_id)
    return (
        f"INSERT INTO {table} "
        "(chunk_id, embedding, collection, extractor_profile_id, chunk_profile_id) "
        "VALUES (:chunk_id, (:embedding)::vector, :collection, "
        ":extractor_profile_id, :chunk_profile_id) "
        "ON CONFLICT (chunk_id) DO UPDATE SET embedding = EXCLUDED.embedding, "
        "collection = EXCLUDED.collection, "
        "extractor_profile_id = EXCLUDED.extractor_profile_id, "
        "chunk_profile_id = EXCLUDED.chunk_profile_id"
    )


def knn_sql(profile_id: int, *, distance_operator: str) -> str:
    """The filtered KNN query, with every filter on the vector table itself.

    The joins that remain are projections -- the chunk's text and the document's
    path -- reached by primary key, which a nested loop can satisfy while
    preserving the index scan's ordering. It is the *filters* that had to move:
    while they sat on joined tables the planner drove from ``chunked_documents``
    and probed this table by primary key, reading every row.
    """
    table = vector_table_name(profile_id)
    return (
        "SELECT sd.collection AS collection, sd.source_path AS source_path, "
        "c.content AS content, "
        f"ev.embedding {distance_operator} (:query)::vector AS distance "
        f"FROM {table} ev "
        "JOIN chunks_v2 c ON c.id = ev.chunk_id "
        "JOIN source_documents sd ON c.document_id = sd.id "
        "WHERE ev.collection = :collection "
        "AND ev.chunk_profile_id = :chunk_profile_id "
        "AND ev.extractor_profile_id = :extractor_profile_id "
        f"ORDER BY ev.embedding {distance_operator} (:query)::vector "
        "LIMIT :k"
    )


#: Chunks pulled from the full-text index before ranking and scope filtering.
#: Two reasons it exists. ``ts_rank`` recomputes a tsvector per candidate row,
#: so ranking every match of a common term scores hundreds of thousands of
#: chunks -- measured as a hang of minutes. And the scope filter is applied
#: *after* this cap, so the pool needs headroom over ``top_k``: mid-rebuild a
#: collection has two revisions in ``chunks_v2`` and up to half the candidates
#: can be out of scope.
LEXICAL_CANDIDATE_POOL = 2000


def lexical_sql(profile_id: int, *, text_config: str) -> str:
    """Full-text search over the same rows ``knn_sql`` searches.

    Scope parity with the vector arm is structural, not coincidental: this joins
    the *same* per-profile vector table and filters on the *same* three
    denormalised columns. ``chunks_v2`` is a single global table holding chunks
    from every collection and every revision, including superseded ones, so a
    lexical query written against it directly would happily return documents the
    vector arm cannot reach -- and the two halves of one search would disagree
    about what the collection contains.

    The text predicate drives an inner LIMIT rather than sitting beside the scope
    filters. Measured 2026-09-06: with the filters and the text match in one
    WHERE clause the planner drives from the joined table and applies the text
    match as a filter, never touching the GIN index -- a term matching nothing
    then costs a full scan of every row to prove absence. Retrieving from the
    index first and filtering the candidates afterwards keeps the bitmap index
    scan.
    """
    table = vector_table_name(profile_id)
    rank = (
        f"ts_rank(to_tsvector('{text_config}', c.content), "
        f"plainto_tsquery('{text_config}', :query))"
    )
    return (
        "SELECT sd.collection AS collection, sd.source_path AS source_path, "
        f"c.content AS content, {rank} AS rank "
        "FROM ("
        "  SELECT id, document_id, content FROM chunks_v2 "
        f"  WHERE to_tsvector('{text_config}', content) "
        f"        @@ plainto_tsquery('{text_config}', :query) "
        "  LIMIT :pool"
        ") c "
        f"JOIN {table} ev ON ev.chunk_id = c.id "
        "JOIN source_documents sd ON sd.id = c.document_id "
        "WHERE ev.collection = :collection "
        "AND ev.chunk_profile_id = :chunk_profile_id "
        "AND ev.extractor_profile_id = :extractor_profile_id "
        f"ORDER BY {rank} DESC "
        "LIMIT :k"
    )


#: pgvector release that introduced ``hnsw.iterative_scan``.
_ITERATIVE_SCAN_SINCE = (0, 8, 0)

#: Accepted values for ``hnsw.iterative_scan``. Anything else is refused before
#: it reaches the server, where an invalid value aborts the whole query.
HNSW_ITERATIVE_SCAN_MODES = ("off", "relaxed_order", "strict_order")


def parse_extension_version(raw: str | None) -> tuple[int, ...] | None:
    """Parse ``pg_extension.extversion`` into comparable integers (pure).

    Returns None for anything non-numeric rather than guessing, so an unusual
    build is treated as "cannot confirm support" instead of assumed capable.
    """
    if not raw:
        return None
    numbers: list[int] = []
    for part in raw.split("."):
        if not part.isdigit():
            break
        numbers.append(int(part))
    return tuple(numbers) if numbers else None


def supports_hnsw_iterative_scan(version: tuple[int, ...] | None) -> bool:
    """Whether this pgvector can be told to scan iteratively (pure)."""
    if version is None:
        return False
    return version >= _ITERATIVE_SCAN_SINCE


def query_tuning_statements(
    method: str,
    *,
    hnsw_ef_search: int,
    diskann_query_rescore: int,
    top_k: int = 1,
    hnsw_iterative_scan: str | None = None,
) -> list[str]:
    """``SET LOCAL`` statements for the method's query-time knobs (pure).

    An HNSW scan yields at most ``ef_search`` candidates, so a configured value
    below the requested ``top_k`` caps the result count with no indication. The
    default is 100, above the documented maximum of 50 results, so only a
    lowered setting can hit this -- it was reachable with the previous default
    of 40, which is why the guard exists.

    Iterative scanning matters because the filters are applied *during* the
    index scan: without it the scan stops after ``ef_search`` candidates
    regardless of how many passed the filter, which for a collection holding a
    small share of a shared vector table measured 0 results out of 10 requested.

    ``hnsw_iterative_scan`` is passed only when the server is known to support
    it. That gate is not optional: PostgreSQL accepts an unknown qualified
    setting as a placeholder *until* the defining module loads on that
    connection, then rejects it with InvalidName. Behind a connection pool,
    whether pgvector had already loaded decides whether the statement succeeds,
    so an ungated SET breaks search intermittently rather than honestly.
    """
    if method == "hnsw":
        statements = [f"SET LOCAL hnsw.ef_search = {max(int(hnsw_ef_search), int(top_k))}"]
        if hnsw_iterative_scan in HNSW_ITERATIVE_SCAN_MODES:
            statements.append(f"SET LOCAL hnsw.iterative_scan = {hnsw_iterative_scan}")
        return statements
    if method == "diskann":
        return [f"SET LOCAL diskann.query_rescore = {int(diskann_query_rescore)}"]
    return []


# --- imperative shells -------------------------------------------------------


def vector_table_exists(conn: Connection, profile_id: int) -> bool:
    """Whether a profile's vector table has been created yet.

    The table is created on demand, so a revision that has not embedded anything
    yet has none. Callers that build indexes or query vectors must check first
    rather than assume.
    """
    result = conn.execute(
        text("SELECT to_regclass(:name)"), {"name": vector_table_name(profile_id)}
    ).scalar()
    return result is not None


def vector_table_has_rows(conn: Connection, profile_id: int) -> bool:
    """Whether a profile's vector table holds any vectors at all.

    Callers must check ``vector_table_exists`` first; probing a missing table
    raises. An EXISTS probe rather than COUNT so the answer costs one row.
    """
    result = conn.execute(
        text(f"SELECT EXISTS (SELECT 1 FROM {vector_table_name(profile_id)})")
    ).scalar()
    return bool(result)


def pgvector_version(conn: Connection) -> tuple[int, ...] | None:
    """Installed pgvector version, or None when it cannot be determined."""
    raw = conn.execute(
        text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    ).scalar()
    return parse_extension_version(str(raw) if raw is not None else None)


def index_access_method(conn: Connection, index_name: str) -> str | None:
    """Return the access method (``hnsw``/``diskann``) of an index, or None.

    Used so callers can reconcile against the index that actually exists rather
    than assuming the currently-configured method.
    """
    row = conn.execute(
        text(
            # Restricted to the connection's own search_path: an index of the
            # same name in another schema would otherwise be reported here,
            # suppressing a genuine method-change rebuild and misdirecting the
            # query-time tuning in search.
            "SELECT am.amname FROM pg_class c "
            "JOIN pg_am am ON am.oid = c.relam "
            "WHERE c.relname = :name AND c.relkind = 'i' "
            "AND pg_catalog.pg_table_is_visible(c.oid)"
        ),
        {"name": index_name},
    ).scalar()
    return str(row) if row is not None else None


def drop_vector_table(engine: Engine, profile_id: int) -> None:
    """Drop the profile's vector table if it exists."""
    with engine.begin() as conn:
        conn.execute(text(drop_table_sql(profile_id)))


def upsert_vectors(
    conn: Connection,
    profile_id: int,
    rows: Sequence[tuple[int, Sequence[float]]],
    *,
    collection: str,
    extractor_profile_id: int,
    chunk_profile_id: int,
) -> None:
    """Insert/replace completed vectors for one profile.

    The routing columns are keyword-only and required: a vector written without
    them is one search can never return, and defaulting them would make that
    failure silent.
    """
    if not rows:
        return
    params = [
        {
            "chunk_id": chunk_id,
            "embedding": to_vector_literal(vector),
            "collection": collection,
            "extractor_profile_id": extractor_profile_id,
            "chunk_profile_id": chunk_profile_id,
        }
        for chunk_id, vector in rows
    ]
    conn.execute(text(upsert_sql(profile_id)), params)


def ensure_vector_table_schema(conn: Connection, profile_id: int, dim: int) -> None:
    """Create the vector table if it does not exist yet.

    Until 2026-08 this also migrated pre-denormalisation tables (adding,
    backfilling and enforcing the filter columns). Every database in existence
    has run that migration, so the shim was deleted rather than carried;
    ``create_table_sql`` creates a fresh table in its final shape.
    """
    if not vector_table_exists(conn, profile_id):
        conn.execute(text(create_table_sql(profile_id, dim)))
