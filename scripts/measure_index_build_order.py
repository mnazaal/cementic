#!/usr/bin/env python
"""Compare the two ANN index build orders, on a real PostgreSQL.

  insert then build  -- what cementic does today: embed everything, then one
                        bulk CREATE INDEX at the building->ready transition,
                        during which the pipeline worker does nothing else.
  build then insert  -- create the index while the table is empty (HNSW has no
                        training step) and let each insert maintain the graph.

Run against the *test* database, never a real one:

    PYTHONPATH=. python scripts/measure_index_build_order.py

Two traps, both of which produced wrong numbers before they were understood
(see PLAN.md, "Still open — a decision, with the measurements taken"):

1. Vectors must be clustered. Uniformly random unit vectors in 768 dimensions
   sit at near-identical distances from any query, so the "true" nearest
   neighbours are arbitrary and any recall figure measures noise.
2. Every timed query must have its plan asserted. Without ANALYZE the planner
   declines the index it has just built, and a sequential scan then gets
   reported as ANN latency -- which is how a spurious 100x difference appeared.

If you extend this to score recall, note the third trap: forcing an exact
baseline with `enable_indexscan = off` leaks onto pooled connections, so the
baseline needs its own engine and its plan asserted to be a sequential scan.
"""

from __future__ import annotations

import random
import time

from sqlalchemy import create_engine, text

from tests.integration.conftest import _pg_url

DIM = 768
ROWS = 100_000
BATCH = 2000
TOP_K = 10
CLUSTERS = 200
BUILD_MEMORY = "2GB"

engine = create_engine(_pg_url().render_as_string(hide_password=False))
rng = random.Random(101)
CENTROIDS = [[rng.gauss(0, 1) for _ in range(DIM)] for _ in range(CLUSTERS)]


def _vector() -> str:
    centre = CENTROIDS[rng.randrange(CLUSTERS)]
    values = [value + rng.gauss(0, 0.35) for value in centre]
    norm = sum(value * value for value in values) ** 0.5
    return "[" + ",".join(f"{value / norm:.6f}" for value in values) + "]"


def _reset(table: str) -> None:
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
        conn.execute(
            text(f"CREATE TABLE {table} (id serial PRIMARY KEY, embedding vector({DIM}))")
        )


def _insert(table: str, corpus: list[str]) -> float:
    start = time.perf_counter()
    for offset in range(0, len(corpus), BATCH):
        with engine.begin() as conn:
            conn.execute(
                text(f"INSERT INTO {table} (embedding) VALUES ((:e)::vector)"),
                [{"e": vector} for vector in corpus[offset : offset + BATCH]],
            )
    return time.perf_counter() - start


def _build(table: str, index: str) -> float:
    start = time.perf_counter()
    with engine.connect() as conn:
        conn.execute(text(f"SET maintenance_work_mem = '{BUILD_MEMORY}'"))
        conn.execute(
            text(f"CREATE INDEX {index} ON {table} USING hnsw (embedding vector_cosine_ops)")
        )
        conn.commit()
    return time.perf_counter() - start


def _analyze(table: str) -> None:
    with engine.connect() as conn:
        conn.execute(text(f"ANALYZE {table}"))
        conn.commit()


def _assert_index_used(table: str, index: str, query: str) -> None:
    with engine.connect() as conn:
        conn.execute(text("SET LOCAL hnsw.ef_search = 40"))
        plan = "\n".join(
            row[0]
            for row in conn.execute(
                text(
                    f"EXPLAIN (COSTS OFF) SELECT id FROM {table} "
                    f"ORDER BY embedding <=> (:q)::vector LIMIT {TOP_K}"
                ),
                {"q": query},
            ).all()
        )
    if index not in plan:
        raise SystemExit(f"{table}: planner did not use {index}; timings are meaningless\n{plan}")


def main() -> None:
    print(f"generating {ROWS} clustered vectors at {DIM} dims...", flush=True)
    corpus = [_vector() for _ in range(ROWS)]
    probe = _vector()

    print("\n=== insert then build (today) ===", flush=True)
    _reset("ibo_after")
    insert_after = _insert("ibo_after", corpus)
    build_after = _build("ibo_after", "ix_ibo_after")
    _analyze("ibo_after")
    _assert_index_used("ibo_after", "ix_ibo_after", probe)
    print(f"  insert {insert_after:7.1f}s   build {build_after:7.1f}s (worker blocked)"
          f"   total {insert_after + build_after:7.1f}s", flush=True)

    print("\n=== build then insert ===", flush=True)
    _reset("ibo_first")
    build_first = _build("ibo_first", "ix_ibo_first")
    insert_first = _insert("ibo_first", corpus)
    _analyze("ibo_first")
    _assert_index_used("ibo_first", "ix_ibo_first", probe)
    print(f"  build  {build_first:7.1f}s (empty table)   insert {insert_first:7.1f}s"
          f"   total {insert_first + build_first:7.1f}s", flush=True)

    print("\n=== summary ===", flush=True)
    print(f"  total wall time          after={insert_after + build_after:7.1f}s"
          f"   first={insert_first + build_first:7.1f}s", flush=True)
    print(f"  longest unresumable step after={build_after:7.1f}s   first=    0.0s", flush=True)

    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS ibo_after"))
        conn.execute(text("DROP TABLE IF EXISTS ibo_first"))


if __name__ == "__main__":
    main()
