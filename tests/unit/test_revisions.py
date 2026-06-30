"""Tests for pipeline revision lifecycle helpers."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cementic.config import Config
from cementic.db import PipelineRevision
from cementic.revisions import (
    _default_revision_label,
    _revision_prune_plan,
    get_active_revision,
    get_target_revision,
    mark_revision_ready,
    promote_revision,
    prune_collection_history,
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
            promote_revision(session, "col", revision, config=_config)

    def test_raises_when_not_ready(self, _config: Config) -> None:
        session = MagicMock()
        revision = MagicMock(spec=PipelineRevision)
        revision.collection = "col"
        revision.status = "building"

        with pytest.raises(ValueError, match="Only ready"):
            promote_revision(session, "col", revision, config=_config)

    @patch("cementic.revisions.prune_collection_history")
    def test_promotes_ready_revision(self, mock_prune: MagicMock, _config: Config) -> None:
        session = MagicMock()
        revision = MagicMock(spec=PipelineRevision)
        revision.collection = "col"
        revision.status = "ready"

        result = promote_revision(session, "col", revision, config=_config)
        assert result.status == "active"
        assert result.promoted_at is not None
        mock_prune.assert_called_once_with(session, "col", config=_config)


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

    @patch("cementic.storage.Path.unlink", side_effect=OSError("permission denied"))
    @patch("cementic.revisions.select")
    @patch("cementic.revisions._revision_prune_plan")
    def test_oserror_during_artifact_unlink_is_suppressed(
        self, mock_prune_plan, mock_select, mock_unlink, tmp_path: Path
    ) -> None:
        """Artifact unlink OSError is caught and suppressed during pruning."""
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

        # Must not raise
        config = Config()
        config.storage.artifacts_path = tmp_path
        prune_collection_history(session, "test-collection", config=config)

        mock_unlink.assert_called_once_with(missing_ok=True)
