"""Pure builders for ANN index DDL, selected by method name (data-driven).

Adding an index method is one entry in ``_INDEX_STRATEGIES`` plus a pure builder
function — no branching at call sites. This is the functional core; executing the
DDL is the caller's (imperative) job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

_DISTANCE_TO_OPCLASS = {
    "cosine": "vector_cosine_ops",
    "l2": "vector_l2_ops",
    "ip": "vector_ip_ops",
}


@dataclass(frozen=True)
class IndexParams:
    """Build-time index parameters. Query-time knobs are applied per query."""

    hnsw_m: int = 16
    hnsw_ef_construction: int = 64
    diskann_num_neighbors: int = 50
    diskann_search_list_size: int = 100


def _hnsw_ddl(index_name: str, table: str, column: str, opclass: str, params: IndexParams) -> str:
    return (
        f"CREATE INDEX IF NOT EXISTS {index_name} ON {table} "
        f"USING hnsw ({column} {opclass}) "
        f"WITH (m = {params.hnsw_m}, ef_construction = {params.hnsw_ef_construction})"
    )


def _diskann_ddl(
    index_name: str, table: str, column: str, opclass: str, params: IndexParams
) -> str:
    return (
        f"CREATE INDEX IF NOT EXISTS {index_name} ON {table} "
        f"USING diskann ({column} {opclass}) "
        f"WITH (num_neighbors = {params.diskann_num_neighbors}, "
        f"search_list_size = {params.diskann_search_list_size})"
    )


_INDEX_STRATEGIES: dict[str, Callable[[str, str, str, str, IndexParams], str]] = {
    "hnsw": _hnsw_ddl,
    "diskann": _diskann_ddl,
}


#: Largest vector dimension each method can actually index. pgvector stores up
#: to 16000 dimensions in a `vector` column but its HNSW index stops at 2000;
#: pgvectorscale's DiskANN documents no limit of its own. Storage limit and
#: index limit are different facts, so they live apart -- `vector_store` guards
#: the column, this guards the index.
_MAX_INDEXABLE_DIM: dict[str, int | None] = {
    "hnsw": 2000,
    "diskann": None,
}


def supported_index_methods() -> tuple[str, ...]:
    """Return the registered index method names."""
    return tuple(_INDEX_STRATEGIES)


def max_indexable_dim(method: str) -> int | None:
    """Largest dimension ``method`` can index, or None when unbounded (pure)."""
    return _MAX_INDEXABLE_DIM.get(method)


def index_dimension_error(method: str, embedding_dim: int) -> str | None:
    """Why ``embedding_dim`` cannot be indexed by ``method``, or None (pure).

    Checked before embedding starts. The index is only built once a revision
    first completes, so an oversized model previously embedded the entire corpus
    -- potentially hours -- and only then failed at CREATE INDEX, leaving the
    revision retrying that failure forever.
    """
    limit = max_indexable_dim(method)
    if limit is None or embedding_dim <= limit:
        return None
    return (
        f"the {method} index supports at most {limit} dimensions, but the "
        f"embedding model produces {embedding_dim}. Choose a smaller model, or "
        f"set index.method to a method without that limit "
        f"({', '.join(m for m in supported_index_methods() if max_indexable_dim(m) is None)})."
    )


def build_index_ddl(
    *,
    method: str,
    index_name: str,
    table: str,
    column: str,
    metric: str,
    params: IndexParams,
) -> str:
    """Return the ``CREATE INDEX`` statement for one method. Pure; no DB."""
    try:
        opclass = _DISTANCE_TO_OPCLASS[metric]
    except KeyError:
        raise ValueError(f"Unsupported distance metric: {metric}") from None
    try:
        builder = _INDEX_STRATEGIES[method]
    except KeyError:
        raise ValueError(
            f"Unknown index method: {method}. Available: {sorted(_INDEX_STRATEGIES)}"
        ) from None
    return builder(index_name, table, column, opclass, params)
