#!/usr/bin/env python3
"""Does the embedding claim cost stay flat as a collection fills up?

The embed step used to find work with an anti-join, so its cost tracked corpus
size rather than remaining work: measured at 178,681 buffers to claim one batch,
and 64,536 buffers to return nothing against a finished collection. This script
is the check that the replacement is actually flat, at a size worth trusting --
every bug found in this pipeline so far was invisible at 153 documents.

Builds a synthetic collection, drives it to 10/50/90% embedded, and reports what
each claim costs. Run against the *test* database, never a real one:

    PYTHONPATH=. .venv/bin/python scripts/measure_embedding_claim_scale.py [n_chunks]

It removes its own collection on exit, including after a failure.
"""

from __future__ import annotations

import sys

from sqlalchemy import text

from cementic.config import Config
from cementic.db import get_engine

COLLECTION = "__claim_scale_probe__"
CHUNKS_PER_DOC = 50
BATCH = 32


def _seed(conn, n_chunks: int) -> int:
    """Create profiles, a revision, and n_chunks chunks. Returns the profile id."""
    ext = conn.execute(text(
        "INSERT INTO extractor_profiles (name, config_json, fingerprint, created_at) "
        "VALUES ('probe','{}',:f, now()) RETURNING id"), {"f": f"ext-{COLLECTION}"}).scalar()
    chunkp = conn.execute(text(
        "INSERT INTO chunk_profiles (config_json, fingerprint, created_at) "
        "VALUES ('{}',:f, now()) RETURNING id"), {"f": f"chk-{COLLECTION}"}).scalar()
    embp = conn.execute(text(
        "INSERT INTO embedding_profiles (provider, model_identifier, embedding_dim, "
        "distance_metric, config_json, fingerprint, created_at) "
        "VALUES ('probe','probe-model',768,'cosine','{}',:f, now()) RETURNING id"),
        {"f": f"emb-{COLLECTION}"}).scalar()
    conn.execute(text(
        "INSERT INTO pipeline_revisions (collection,label,extractor_profile_id,"
        "chunk_profile_id,embedding_profile_id,status,created_at) "
        "VALUES (:c,'probe',:e,:k,:m,'building', now())"),
        {"c": COLLECTION, "e": ext, "k": chunkp, "m": embp})

    n_docs = max(1, n_chunks // CHUNKS_PER_DOC)
    conn.execute(text(
        "INSERT INTO source_documents (collection, source_path, file_hash, status, "
        "created_at, updated_at) SELECT :c, '/probe/' || g, md5(g::text), 'done', now(), now() "
        "FROM generate_series(1,:n) g"), {"c": COLLECTION, "n": n_docs})
    conn.execute(text(
        "INSERT INTO extracted_documents (document_id, extractor_profile_id, artifact_path, "
        "content_hash, source_file_hash, status, created_at, updated_at) "
        "SELECT sd.id, :e, '/a', md5(sd.id::text), sd.file_hash, 'done', now(), now() "
        "FROM source_documents sd WHERE sd.collection = :c"), {"e": ext, "c": COLLECTION})
    conn.execute(text(
        "INSERT INTO chunked_documents (extracted_document_id, chunk_profile_id, "
        "source_content_hash, status, total_chunks, created_at, updated_at) "
        "SELECT ed.id, :k, ed.content_hash, 'done', :cpd, now(), now() "
        "FROM extracted_documents ed JOIN source_documents sd ON sd.id = ed.document_id "
        "WHERE sd.collection = :c"), {"k": chunkp, "cpd": CHUNKS_PER_DOC, "c": COLLECTION})
    conn.execute(text(
        "INSERT INTO chunks_v2 (document_id, chunked_document_id, chunk_index, content, "
        "created_at, updated_at) SELECT sd.id, cd.id, g, repeat('x', 800), now(), now() "
        "FROM source_documents sd "
        "JOIN extracted_documents ed ON ed.document_id = sd.id "
        "JOIN chunked_documents cd ON cd.extracted_document_id = ed.id "
        "CROSS JOIN generate_series(1,:cpd) g WHERE sd.collection = :c"),
        {"cpd": CHUNKS_PER_DOC, "c": COLLECTION})
    conn.execute(text(
        "INSERT INTO chunk_embeddings (chunk_id, embedding_profile_id, status, created_at, "
        "updated_at, collection, extractor_profile_id, chunk_profile_id) "
        "SELECT ch.id, :m, 'pending', now(), now(), :c, :e, :k FROM chunks_v2 ch "
        "JOIN source_documents sd ON sd.id = ch.document_id WHERE sd.collection = :c"),
        {"m": embp, "c": COLLECTION, "e": ext, "k": chunkp})
    conn.execute(text("ANALYZE chunk_embeddings"))
    conn.execute(text("ANALYZE chunks_v2"))
    return embp, ext, chunkp


CLAIM = """
EXPLAIN (ANALYZE, BUFFERS, TIMING OFF) SELECT ce.chunk_id, ce.id
FROM chunk_embeddings ce
WHERE ce.embedding_profile_id = :m AND ce.status IN ('pending','processing')
  AND ce.collection = :c AND ce.extractor_profile_id = :e AND ce.chunk_profile_id = :k
LIMIT :b
"""


def _measure(conn, embp: int, ext: int, chunkp: int) -> tuple[int, float, int]:
    """(buffers, ms, rows read by the driving index scan) for one claim."""
    args = {"m": embp, "c": COLLECTION, "e": ext, "k": chunkp, "b": BATCH}
    plan = [r[0] for r in conn.execute(text(CLAIM), args)]
    buffers = max(
        (int(line.split("shared hit=")[1].split()[0])
         for line in plan if "Buffers: shared hit=" in line), default=0)
    ms = next((float(line.split(":")[1].strip().split()[0])
               for line in plan if "Execution Time" in line), 0.0)
    driving = next((line for line in plan if "chunk_embeddings ce" in line), "")
    read = -1
    if "actual rows=" in driving:
        read = int(float(driving.split("actual rows=")[1].split()[0]))
    return buffers, ms, read


def main() -> int:
    n_chunks = int(sys.argv[1]) if len(sys.argv) > 1 else 300_000
    engine = get_engine(Config().database.url)
    with engine.begin() as conn:
        exists = conn.execute(text("SELECT 1 FROM source_documents WHERE collection = :c LIMIT 1"),
                              {"c": COLLECTION}).scalar()
        if exists:
            print(f"{COLLECTION} already present; remove it first")
            return 1
    try:
        with engine.begin() as conn:
            print(f"seeding {n_chunks:,} chunks ...", flush=True)
            embp, ext, chunkp = _seed(conn, n_chunks)
            total = conn.execute(text(
                "SELECT count(*) FROM chunk_embeddings WHERE embedding_profile_id = :m"),
                {"m": embp}).scalar()
            print(f"seeded {total:,} chunks / {total // CHUNKS_PER_DOC:,} documents\n")
            print(f"{'% embedded':>12} {'pending':>10} {'buffers':>10} {'ms':>8} {'rows read':>10}")
            for pct in (10, 50, 90):
                conn.execute(text(
                    "UPDATE chunk_embeddings SET status = CASE WHEN rn <= :k THEN 'done' "
                    "ELSE 'pending' END FROM (SELECT id, row_number() OVER (ORDER BY id) rn "
                    "FROM chunk_embeddings WHERE embedding_profile_id = :m) s "
                    "WHERE chunk_embeddings.id = s.id"),
                    {"k": total * pct // 100, "m": embp})
                conn.execute(text("ANALYZE chunk_embeddings"))
                pending = conn.execute(text(
                    "SELECT count(*) FROM chunk_embeddings WHERE embedding_profile_id = :m "
                    "AND status = 'pending'"), {"m": embp}).scalar()
                buffers, ms, read = _measure(conn, embp, ext, chunkp)
                print(f"{pct:>11}% {pending:>10,} {buffers:>10,} {ms:>8.2f} {read:>10}")
            print("\nFlat buffers/rows across the three rows is the property under test:")
            print("cost must track the batch size, not the corpus or the completed fraction.")
    finally:
        with engine.begin() as conn:
            args = {"c": COLLECTION}
            conn.execute(text("DELETE FROM source_documents WHERE collection = :c"), args)
            conn.execute(text("DELETE FROM pipeline_revisions WHERE collection = :c"), args)
            for tbl, fp in (("extractor_profiles", f"ext-{COLLECTION}"),
                            ("chunk_profiles", f"chk-{COLLECTION}"),
                            ("embedding_profiles", f"emb-{COLLECTION}")):
                conn.execute(text(f"DELETE FROM {tbl} WHERE fingerprint = :f"), {"f": fp})
        print(f"\nremoved {COLLECTION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
