"""Optional Postgres-backed smoke test for the full revision pipeline."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import SecretStr
from sqlalchemy import text

from cementic.config import Config
from cementic.db import (
    Chunk,
    ChunkEmbedding,
    PipelineRevision,
    SourceDocument,
    create_tables,
    get_engine,
    get_session_factory,
)
from cementic.embedding_provider import EmbeddingProvider
from cementic.embedding_text import format_document_text_for_model, format_query_text_for_model
from cementic.pipeline_worker import PipelineWorker
from cementic.revisions import promote_revision
from cementic.search import Searcher
from cementic.source_watcher import SourceWatcher
from tests.integration.conftest import _pg_url

#: Needs a real Postgres, so it belongs to the ``pg`` job. Without this the
#: module is excluded from ``-m pg`` and self-skips under ``-m "not pg"``,
#: leaving the only end-to-end build/promote/search path with no CI coverage.
pytestmark = pytest.mark.pg

#: Generated on demand by tests/fixtures/generate_pdfs.py (see conftest's
#: autouse fixture). Its text is built around FakeEmbeddingClient.TERMS below,
#: so the search assertion is meaningful.
PDF_FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "test_doc_a.pdf"


class FakeEmbeddingClient(EmbeddingProvider):
    """Tiny deterministic embedding client for smoke tests."""

    TERMS = ["computer", "symbiosis", "man", "time"]

    def health_check(self) -> bool:
        return True

    def embed(self, text: str) -> list[float]:
        lowered = text.lower().replace("search_document: ", "").replace("search_query: ", "")
        return [float(lowered.count(term)) for term in self.TERMS]

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        return [self.embed(text) for text in texts]

    @property
    def embedding_dim(self) -> int:
        return len(self.TERMS)


def _config_for(temp_dir: Path) -> Config:
    config = Config()
    # This test opens its own engine rather than using the pg_engine fixture, so
    # it must be pointed at the dedicated test database explicitly -- otherwise
    # it creates tables in whatever database the user's real config names.
    config.database.url_override = SecretStr(
        _pg_url().render_as_string(hide_password=False)
    )
    config.storage.artifacts_path = temp_dir / "artifacts"
    config.source_watcher.log_file = temp_dir / "source_watcher.log"
    config.pipeline_worker.log_file = temp_dir / "pipeline_worker.log"
    config.llama_cpp.model_path = str(temp_dir / "fake-model.gguf")
    config.llama_cpp.embedding_dim = 4
    config.pipeline.embedding_provider = "llama-cpp"
    config.pipeline_worker.batch_size = 8
    config.pipeline_worker.poll_interval = 0.01
    return config


def _require_postgres(config: Config) -> None:
    engine = get_engine(config.database.url)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as error:  # pragma: no cover - environment dependent
        pytest.skip(f"Postgres smoke test skipped: {error}")


def _run_pipeline_once(daemon: PipelineWorker, revision_id: int) -> bool:
    if daemon._step_extract(revision_id):
        return True
    if daemon._step_chunk(revision_id):
        return True
    if daemon._step_embed(revision_id):
        return True
    daemon._mark_revision_ready_if_complete(revision_id)
    return False


def test_postgres_smoke_build_search_and_promote(
    temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config_for(temp_dir)
    _require_postgres(config)

    collection = f"smoke-{uuid4().hex[:8]}"
    engine = get_engine(config.database.url)
    create_tables(engine)
    session_factory = get_session_factory(engine)

    source_watcher = SourceWatcher(config)
    source_watcher.Session = session_factory
    source_watcher.collection = collection
    monkeypatch.setattr(source_watcher.state_manager, "update", lambda **kwargs: None)

    pipeline = PipelineWorker(config)
    pipeline.Session = session_factory
    pipeline.collection = collection
    pipeline.embedding_client = FakeEmbeddingClient()
    monkeypatch.setattr(pipeline.state_manager, "update", lambda **kwargs: None)
    # Signature must match search._create_embedding_provider(config_json, config):
    # a mismatched patch raises TypeError inside search and silently voids the
    # search assertion below.
    monkeypatch.setattr(
        "cementic.search._create_embedding_provider",
        lambda config_json, config: FakeEmbeddingClient(),
    )

    try:
        source_watcher._register_document(str(PDF_FIXTURE))
        revision_id = pipeline._ensure_target_revision()

        for _ in range(500):
            if not _run_pipeline_once(pipeline, revision_id):
                break
        else:  # pragma: no cover - guard rail
            raise AssertionError("Pipeline did not settle within the expected iteration budget")

        with session_factory() as session:
            revision = session.get(PipelineRevision, revision_id)
            assert revision is not None
            assert revision.status == "ready"
            promote_revision(session, collection, revision, config=config)
            session.commit()

        with session_factory() as session:
            revision = session.get(PipelineRevision, revision_id)
            assert revision is not None
            sample_doc_vector = FakeEmbeddingClient().embed(
                format_document_text_for_model(
                    "computer symbiosis", revision.embedding_profile.model_identifier
                )
            )
            sample_query_vector = FakeEmbeddingClient().embed(
                format_query_text_for_model(
                    "computer symbiosis", revision.embedding_profile.model_identifier
                )
            )
            assert sample_doc_vector == sample_query_vector

            done_embeddings = session.query(ChunkEmbedding).filter_by(status="done").count()
            all_embeddings = session.query(ChunkEmbedding).count()
            chunks = session.query(Chunk).count()
            active_revisions = session.query(PipelineRevision).filter_by(status="active").count()

        assert all_embeddings > 0
        assert done_embeddings > 0
        assert chunks > 0
        assert active_revisions > 0

        searcher = Searcher(config)
        results = searcher.search("computer symbiosis", top_k=5, collections=[collection])

        assert results, "search returned no results for an indexed collection"
        assert any(PDF_FIXTURE.name in result["source_path"] for result in results)
    finally:
        with session_factory() as session:
            doc_ids = [
                doc_id
                for (doc_id,) in session.query(SourceDocument.id)
                .filter_by(collection=collection)
                .all()
            ]
            session.query(PipelineRevision).filter_by(collection=collection).delete(
                synchronize_session=False
            )
            if doc_ids:
                session.query(SourceDocument).filter(SourceDocument.id.in_(doc_ids)).delete(
                    synchronize_session=False
                )
            session.commit()
        # This test creates its own schema rather than using the pg_engine
        # fixture, so it also owns the per-profile vector tables it produced.
        # They carry a foreign key to chunks_v2 and would otherwise block the
        # session-scoped teardown's drop_all.
        with engine.connect() as conn:
            for (table_name,) in conn.execute(
                text("SELECT tablename FROM pg_tables WHERE tablename LIKE 'embedding_vectors_p%'")
            ).all():
                conn.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
            conn.commit()
