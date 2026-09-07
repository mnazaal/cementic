"""Tests for pipeline revision lifecycle helpers."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cementic.config import Config
from cementic.db import (
    Base,
    Chunk,
    ChunkedDocument,
    ChunkEmbedding,
    ChunkProfile,
    EmbeddingProfile,
    ExtractedDocument,
    ExtractorProfile,
    PipelineRevision,
    SourceDocument,
)
from cementic.revisions import (
    _default_revision_label,
    _revision_prune_plan,
    drain_pending_vector_table_drops,
    get_active_revision,
    get_target_revision,
    mark_revision_ready,
    promote_revision,
    prune_collection_history,
    unreferenced_profile_ids,
)


class TestDefaultRevisionLabel:
    """Tests for _default_revision_label."""

    def test_label_format(self) -> None:
        revision = MagicMock(spec=PipelineRevision)
        revision.extractor_profile.name = "pdfplumber"
        revision.chunk_profile.fingerprint = "abcdef1234567890"
        revision.embedding_profile.provider = "llama-cpp"
        revision.embedding_profile.fingerprint = "fedcba9876543210"

        label = _default_revision_label(revision)
        assert label.startswith("pdfplumber-")
        assert "abcdef12" in label  # first 8 chars of chunk fingerprint
        assert "llama-cpp-fedcba98" in label  # provider + first 8 chars of embedding fingerprint


class TestGetActiveRevision:
    """Tests for get_active_revision."""

    def test_returns_active(self) -> None:
        session = MagicMock()
        expected = MagicMock(spec=PipelineRevision)
        session.query().filter_by().order_by().first.return_value = expected

        result = get_active_revision(session, "col")
        assert result is expected

    def test_no_active_returns_none(self) -> None:
        session = MagicMock()
        session.query().filter_by().order_by().first.return_value = None

        result = get_active_revision(session, "col")
        assert result is None


def _sqlite_session():
    """A real session against an empty in-memory schema."""
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _sqlite_session_with_profiles():
    """A real session plus one extractor/chunk profile and two embedding profiles."""
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    extractor = ExtractorProfile(fingerprint="ext", name="default", config_json="{}")
    chunk = ChunkProfile(fingerprint="chk", config_json="{}")
    embeddings = [
        EmbeddingProfile(
            fingerprint=f"emb-{name}",
            provider="llama-cpp",
            model_identifier=name,
            embedding_dim=4,
            distance_metric="cosine",
            config_json="{}",
        )
        for name in ("a", "b")
    ]
    session.add_all([extractor, chunk, *embeddings])
    session.flush()
    return session, (extractor, chunk, *embeddings)


class TestGetTargetRevision:
    """Tests for get_target_revision."""

    @patch("cementic.revisions.get_or_create_extractor_profile")
    @patch("cementic.revisions.get_or_create_chunk_profile")
    @patch("cementic.revisions.get_or_create_embedding_profile")
    def test_returns_existing(self, mock_emb, mock_chk, mock_ext) -> None:
        session = MagicMock()
        cfg = MagicMock()

        existing = MagicMock(spec=PipelineRevision)
        session.query().filter_by().order_by().first.return_value = existing

        result = get_target_revision(session, "col", cfg)
        assert result is existing

    @pytest.mark.parametrize("target_status", ["active", "retired", "superseded"])
    def test_other_in_flight_revision_is_superseded(self, target_status: str) -> None:
        """Only one revision per collection may stay in flight.

        Regression: the supersede step ran only when the target was `retired` or
        `superseded`. Reverting config back to the *active* revision therefore
        left the abandoned build stuck in `building` forever -- nothing ever
        worked on it, `cementic status` displayed it as building indefinitely,
        and prune_collection_history (which collects only `superseded` and old
        `retired` revisions) pinned its artifacts on disk permanently.
        """
        session, profiles = _sqlite_session_with_profiles()
        extractor, chunk, embedding_a, embedding_b = profiles

        target = PipelineRevision(
            collection="c",
            extractor_profile_id=extractor.id,
            chunk_profile_id=chunk.id,
            embedding_profile_id=embedding_a.id,
            status=target_status,
            label="target",
        )
        abandoned = PipelineRevision(
            collection="c",
            extractor_profile_id=extractor.id,
            chunk_profile_id=chunk.id,
            embedding_profile_id=embedding_b.id,
            status="building",
            label="abandoned",
        )
        session.add_all([target, abandoned])
        session.commit()

        with (
            patch("cementic.revisions.get_or_create_extractor_profile", return_value=extractor),
            patch("cementic.revisions.get_or_create_chunk_profile", return_value=chunk),
            patch("cementic.revisions.get_or_create_embedding_profile", return_value=embedding_a),
        ):
            result = get_target_revision(session, "c", Config())
        session.commit()

        assert result.id == target.id
        assert abandoned.status == "superseded"
        # A reverted-to revision is resurrected; an active one stays active.
        expected = "active" if target_status == "active" else "building"
        assert target.status == expected

    @patch("cementic.revisions._default_revision_label", return_value="test-label")
    @patch("cementic.revisions.get_or_create_extractor_profile")
    @patch("cementic.revisions.get_or_create_chunk_profile")
    @patch("cementic.revisions.get_or_create_embedding_profile")
    def test_creates_new_when_none_exists(self, mock_emb, mock_chk, mock_ext, mock_label) -> None:
        session = MagicMock()
        cfg = MagicMock()

        session.query().filter_by().order_by().first.return_value = None
        session.query().filter().update.return_value = None

        result = get_target_revision(session, "col", cfg)
        assert result.status == "building"
        assert session.add.called
        assert result.label == "test-label"


class TestMarkRevisionReady:
    """Tests for mark_revision_ready."""

    def test_marks_building_to_ready(self) -> None:
        session = MagicMock()
        revision = MagicMock(spec=PipelineRevision)
        revision.status = "building"

        mark_revision_ready(session, revision)
        assert revision.status == "ready"
        assert session.flush.called

    def test_does_not_change_other_status(self) -> None:
        session = MagicMock()
        revision = MagicMock(spec=PipelineRevision)
        revision.status = "ready"

        mark_revision_ready(session, revision)
        assert revision.status == "ready"


class TestPromoteRevision:
    """Tests for promote_revision."""

    @pytest.fixture(autouse=True)
    def _config(self) -> Config:
        return Config()

    def test_raises_when_wrong_collection(self, _config: Config) -> None:
        session = MagicMock()
        revision = MagicMock(spec=PipelineRevision)
        revision.collection = "other"

        with pytest.raises(ValueError, match="does not belong"):
            promote_revision(session, "col", revision)

    @staticmethod
    def _mock_db_status(session: MagicMock, status: str | None) -> None:
        """Answer promote_revision's FOR UPDATE status re-read."""
        chain = session.query.return_value.filter_by.return_value.with_for_update
        chain.return_value.scalar.return_value = status

    def test_raises_when_not_ready(self, _config: Config) -> None:
        session = MagicMock()
        self._mock_db_status(session, "building")
        revision = MagicMock(spec=PipelineRevision)
        revision.collection = "col"
        revision.status = "building"

        with pytest.raises(ValueError, match="Only ready"):
            promote_revision(session, "col", revision)

    @patch("cementic.revisions.prune_collection_history")
    def test_promotes_ready_revision(self, mock_prune: MagicMock, _config: Config) -> None:
        session = MagicMock()
        self._mock_db_status(session, "ready")
        revision = MagicMock(spec=PipelineRevision)
        revision.collection = "col"
        revision.status = "ready"

        result = promote_revision(session, "col", revision)
        assert result.status == "active"
        assert result.promoted_at is not None
        mock_prune.assert_called_once_with(session, "col")

    def test_stale_orm_status_is_not_trusted(self, _config: Config) -> None:
        """Regression: the readiness check read revision.status off the ORM
        object, so two concurrent promotes could both see a stale "ready" and
        both pass -- the second retiring the first's newly-active revision
        while activating its own. The status is re-read from the database
        under FOR UPDATE."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        session = sessionmaker(bind=engine, expire_on_commit=False)()
        revision = PipelineRevision(
            collection="col",
            extractor_profile_id=1,
            chunk_profile_id=1,
            embedding_profile_id=1,
            status="ready",
            label="rev",
        )
        session.add(revision)
        session.commit()

        # Another actor promoted (and later retired) it; this ORM object is stale.
        session.execute(
            PipelineRevision.__table__.update()
            .where(PipelineRevision.id == revision.id)
            .values(status="retired")
        )
        assert revision.status == "ready"  # the stale attribute the bug trusted

        with pytest.raises(ValueError, match="Only ready"):
            promote_revision(session, "col", revision)


class TestOneActiveRevisionPerCollection:
    """The database itself must refuse a second active revision (regression:
    nothing did, so the promote race left two actives for search and status to
    disagree about)."""

    def test_second_active_is_refused(self) -> None:
        from sqlalchemy.exc import IntegrityError

        from cementic.db import create_tables

        engine = create_engine("sqlite:///:memory:")
        create_tables(engine)
        session = sessionmaker(bind=engine, expire_on_commit=False)()

        def revision(status: str, chunk_profile_id: int) -> PipelineRevision:
            return PipelineRevision(
                collection="col",
                extractor_profile_id=1,
                chunk_profile_id=chunk_profile_id,
                embedding_profile_id=1,
                status=status,
                label=f"rev-{chunk_profile_id}",
            )

        session.add(revision("active", 1))
        session.commit()
        # A second *retired* revision is fine; the index is partial.
        session.add(revision("retired", 2))
        session.commit()

        session.add(revision("active", 3))
        with pytest.raises(IntegrityError):
            session.commit()


class TestEnsureRevisionAnnIndex:
    """Tests for ensure_revision_ann_index."""

    def test_raises_when_session_not_bound(self) -> None:
        from cementic.config import Config
        from cementic.revisions import ensure_revision_ann_index

        session = MagicMock()
        session.get_bind.return_value = None
        revision = MagicMock(spec=PipelineRevision)

        with pytest.raises(RuntimeError, match="not bound"):
            ensure_revision_ann_index(session, revision, Config())

    @patch("cementic.revisions.ensure_embedding_ann_index")
    def test_calls_ensure_ann_index(self, mock_ensure) -> None:
        from cementic.config import Config
        from cementic.revisions import ensure_revision_ann_index

        session = MagicMock()
        revision = MagicMock(spec=PipelineRevision)
        revision.embedding_profile_id = 5
        revision.embedding_profile.distance_metric = "cosine"

        config = Config()
        config.index.method = "diskann"
        ensure_revision_ann_index(session, revision, config)

        mock_ensure.assert_called_once()
        _, kwargs = mock_ensure.call_args
        assert kwargs["profile_id"] == 5
        assert kwargs["method"] == "diskann"


class TestRevisionPrunePlan:
    """Tests for pure revision history pruning decisions."""

    def test_keeps_active_and_newest_retired(self) -> None:
        revisions = [
            PipelineRevision(
                id=4,
                status="retired",
                extractor_profile_id=40,
                chunk_profile_id=400,
                embedding_profile_id=4000,
            ),
            PipelineRevision(
                id=3,
                status="retired",
                extractor_profile_id=30,
                chunk_profile_id=300,
                embedding_profile_id=3000,
            ),
            PipelineRevision(
                id=2,
                status="superseded",
                extractor_profile_id=20,
                chunk_profile_id=200,
                embedding_profile_id=2000,
            ),
            PipelineRevision(
                id=1,
                status="active",
                extractor_profile_id=10,
                chunk_profile_id=100,
                embedding_profile_id=1000,
            ),
        ]

        plan = _revision_prune_plan(revisions)

        assert plan.removable_revision_ids == {2, 3}
        assert plan.keep_extractor_profile_ids == {10, 40}
        assert plan.keep_chunk_profile_ids == {100, 400}
        assert plan.keep_embedding_profile_ids == {1000, 4000}

    def test_keeps_all_when_no_removable_history(self) -> None:
        revisions = [
            PipelineRevision(
                id=2,
                status="ready",
                extractor_profile_id=20,
                chunk_profile_id=200,
                embedding_profile_id=2000,
            ),
            PipelineRevision(
                id=1,
                status="active",
                extractor_profile_id=10,
                chunk_profile_id=100,
                embedding_profile_id=1000,
            ),
        ]

        plan = _revision_prune_plan(revisions)

        assert plan.removable_revision_ids == set()
        assert plan.keep_extractor_profile_ids == {10, 20}


class TestPruneCollectionHistory:
    """Tests for prune_collection_history side effects."""

    @staticmethod
    def _make_query_mock(rows: list | None = None) -> MagicMock:
        """Return a query-chain mock that is fluent and optionally returns *rows* via .all()."""
        qm = MagicMock()
        qm.filter_by.return_value = qm
        qm.filter.return_value = qm
        qm.join.return_value = qm
        qm.order_by.return_value = qm
        qm.delete.return_value = None
        if rows is not None:
            qm.all.return_value = rows
        return qm

    @patch("cementic.storage.Path.unlink")
    @patch("cementic.revisions.select")
    @patch("cementic.revisions._revision_prune_plan")
    def test_pruning_defers_artifact_removal_until_after_commit(
        self, mock_prune_plan, mock_select, mock_unlink, tmp_path: Path
    ) -> None:
        """Pruning must not unlink while the caller's transaction is open.

        Regression: files were removed inline, so a failed or rolled-back commit
        left rows pointing at artifacts that no longer existed and every later
        chunk step failed on them. The paths are returned instead, for the
        caller to remove once the delete is durable.
        """
        from cementic.revisions import RevisionPrunePlan

        plan = RevisionPrunePlan(
            removable_revision_ids={99},
            keep_extractor_profile_ids={1},
            keep_chunk_profile_ids={10},
            keep_embedding_profile_ids={100},
        )
        mock_prune_plan.return_value = plan

        extracted_row = MagicMock()
        extracted_row.id = 200
        extracted_row.artifact_path = str(tmp_path / "artifact.pdf")
        mock_select.return_value = MagicMock()

        session = MagicMock()
        session.query.side_effect = [
            self._make_query_mock([]),                          # 1) revisions
            self._make_query_mock([extracted_row]),             # 2) extracted_to_remove
            self._make_query_mock(),                            # 3) ChunkEmbedding delete
            self._make_query_mock(),                            # 4) Chunk delete
            self._make_query_mock(),                            # 5) ChunkedDocument delete
            self._make_query_mock(),                            # 6) ExtractedDocument delete
            self._make_query_mock(),                            # 7) PipelineRevision delete
        ]

        config = Config()
        config.storage.artifacts_path = tmp_path
        pending = prune_collection_history(session, "test-collection")

        assert pending == [str(tmp_path / "artifact.pdf")]
        mock_unlink.assert_not_called()


class TestUnreferencedProfileIds:
    """Only a globally unreferenced embedding profile may lose its table."""

    def test_profile_still_used_elsewhere_is_kept(self) -> None:
        assert unreferenced_profile_ids([1, 2, 3], {1: 2, 3: 0}) == [2, 3]

    def test_no_candidates_is_empty(self) -> None:
        assert unreferenced_profile_ids([], {1: 0}) == []


def _seed_model_swap_history(session, collection: str = "c1"):
    """One document chunked once, embedded by three successive models.

    The scenario the pruning leak lived in: extractor and chunking never change,
    so every revision shares those profiles and only the embedding profile moves.
    """
    extractor = ExtractorProfile(name="x", fingerprint="ext", config_json="{}")
    chunk_profile = ChunkProfile(fingerprint="chunk", config_json="{}")
    embeddings = [
        EmbeddingProfile(
            fingerprint=f"embed-{suffix}",
            provider="test",
            model_identifier=f"model-{suffix}",
            embedding_dim=2,
            distance_metric="cosine",
            config_json="{}",
        )
        for suffix in ("a", "b", "c")
    ]
    session.add_all([extractor, chunk_profile, *embeddings])
    session.flush()

    document = SourceDocument(
        collection=collection, source_path="/doc.pdf", file_hash="h", status="indexed"
    )
    session.add(document)
    session.flush()
    extracted = ExtractedDocument(
        document_id=document.id,
        extractor_profile_id=extractor.id,
        source_file_hash="h",
        content_hash="ch",
        status="done",
    )
    session.add(extracted)
    session.flush()
    chunked = ChunkedDocument(
        extracted_document_id=extracted.id,
        chunk_profile_id=chunk_profile.id,
        source_content_hash="ch",
        status="done",
        total_chunks=1,
    )
    session.add(chunked)
    session.flush()
    chunk = Chunk(
        document_id=document.id, chunked_document_id=chunked.id, chunk_index=0, content="text"
    )
    session.add(chunk)
    session.flush()

    # Older retired, newest retired, active -- so only the first is removable.
    for embedding, status in zip(embeddings, ("retired", "retired", "active")):
        session.add(
            PipelineRevision(
                collection=collection,
                status=status,
                extractor_profile_id=extractor.id,
                chunk_profile_id=chunk_profile.id,
                embedding_profile_id=embedding.id,
            )
        )
        session.add(
            ChunkEmbedding(
                chunk_id=chunk.id,
                embedding_profile_id=embedding.id,
                status="done",
                collection=collection,
                extractor_profile_id=extractor.id,
                chunk_profile_id=chunk_profile.id,
            )
        )
    session.commit()
    return embeddings, chunk


class TestPruningAfterAModelSwap:
    """A retired model must not keep a full copy of the corpus's vectors.

    Regression: the embedding delete was scoped through the *chunk* profiles
    being removed, so a plain model swap -- same extractor, same chunking, new
    model -- matched nothing. Every `chunk_embeddings` row for the retired model
    survived, as did its whole `embedding_vectors_p{id}` table: one full copy of
    the corpus per swap, forever.
    """

    def test_dropped_model_loses_its_rows_and_its_table(self) -> None:
        session = _sqlite_session()
        embeddings, _chunk = _seed_model_swap_history(session)

        prune_collection_history(session, "c1")
        session.commit()

        surviving = {
            row.embedding_profile_id for row in session.query(ChunkEmbedding).all()
        }
        assert surviving == {embeddings[1].id, embeddings[2].id}
        assert drain_pending_vector_table_drops(session) == [embeddings[0].id]

    def test_the_chunks_themselves_are_untouched(self) -> None:
        """Only the model changed, so the extraction and chunking still stand."""
        session = _sqlite_session()
        _seed_model_swap_history(session)

        prune_collection_history(session, "c1")
        session.commit()

        assert session.query(Chunk).count() == 1
        assert session.query(ChunkedDocument).count() == 1
        assert session.query(ExtractedDocument).count() == 1

    def test_a_profile_another_collection_still_uses_keeps_its_table(self) -> None:
        """The vector table is global, so "this collection is done with it" is
        not the question -- dropping it would blank the other collection."""
        session = _sqlite_session()
        embeddings, _chunk = _seed_model_swap_history(session)
        session.add(
            PipelineRevision(
                collection="other",
                status="active",
                extractor_profile_id=1,
                chunk_profile_id=1,
                embedding_profile_id=embeddings[0].id,
            )
        )
        session.commit()

        prune_collection_history(session, "c1")
        session.commit()

        assert drain_pending_vector_table_drops(session) == []
        # This collection's rows for it still go: the table stays for the other
        # collection's sake, not this one's.
        assert (
            session.query(ChunkEmbedding)
            .filter_by(embedding_profile_id=embeddings[0].id)
            .count()
            == 0
        )


class TestRemoveArtifacts:
    """Cleanup must not stop at the first rejected path."""

    def test_one_rejected_path_does_not_abort_the_rest(self, tmp_path: Path) -> None:
        """Regression: safe_remove_artifact raises for a path outside the
        artifacts root, which aborted the whole loop -- leaving every later
        artifact on disk and propagating out of a promotion."""
        from cementic.collections import remove_artifacts

        config = Config()
        config.storage.artifacts_path = tmp_path
        good = tmp_path / "keep.md.gz"
        good.write_bytes(b"x")
        outside = "/etc/passwd"  # rejected: outside the artifacts root

        failures = remove_artifacts([outside, str(good)], config=config)

        assert failures == [outside]
        assert not good.exists(), "the valid artifact must still be removed"
