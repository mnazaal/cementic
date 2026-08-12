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


def create_table_sql(profile_id: int, dim: int) -> str:
    """DDL to create one profile's fixed-dimension vector table."""
    if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 or dim > _MAX_EMBEDDING_DIM:
        raise ValueError(f"dim must be a positive integer <= {_MAX_EMBEDDING_DIM}, got {dim!r}")
    table = vector_table_name(profile_id)
    return (
        f"CREATE TABLE IF NOT EXISTS {table} ("
        "chunk_id INTEGER PRIMARY KEY REFERENCES chunks_v2(id) ON DELETE CASCADE, "
        f"embedding vector({dim}) NOT NULL)"
    )


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
    """Parameterised upsert of one vector row (``:chunk_id``, ``:embedding``)."""
    table = vector_table_name(profile_id)
    return (
        f"INSERT INTO {table} (chunk_id, embedding) "
        "VALUES (:chunk_id, (:embedding)::vector) "
        "ON CONFLICT (chunk_id) DO UPDATE SET embedding = EXCLUDED.embedding"
    )


def query_tuning_sql(
    method: str, *, hnsw_ef_search: int, diskann_query_rescore: int, top_k: int = 1
) -> str | None:
    """``SET LOCAL`` statement for the method's query-time knob, or None.

    An HNSW scan yields at most ``ef_search`` candidates, so a configured value
    below the requested ``top_k`` caps the result count with no indication: the
    default 40 is under the documented maximum of 50 results.
    """
    if method == "hnsw":
        return f"SET LOCAL hnsw.ef_search = {max(int(hnsw_ef_search), int(top_k))}"
    if method == "diskann":
        return f"SET LOCAL diskann.query_rescore = {int(diskann_query_rescore)}"
    return None


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
    conn: Connection, profile_id: int, rows: Sequence[tuple[int, Sequence[float]]]
) -> None:
    """Insert/replace completed vectors for one profile."""
    if not rows:
        return
    params = [
        {"chunk_id": chunk_id, "embedding": to_vector_literal(vector)}
        for chunk_id, vector in rows
    ]
    conn.execute(text(upsert_sql(profile_id)), params)
