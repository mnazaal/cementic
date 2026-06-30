"""Tests for collection and revision persistence helpers."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cementic.collections import (
    DeleteCollectionResult,
    delete_collection_records,
    drop_orphan_vector_tables,
    list_collection_revisions,
    list_collections,
    promote_ready_revision,
    remove_artifacts,
)
from cementic.config import Config
from cementic.db import PipelineRevision, SourceDocument
from cementic.pipeline_worker import PipelineCounts


def _counts(*, extracted_failed: int = 0, chunked_failed: int = 0, failed_embeddings: int = 0):
    return PipelineCounts(
        documents=1,
        extracted_done=1,
        chunked_done=1,
        total_chunks=1,
        done_embeddings=1,
        extracted_failed=extracted_failed,
        chunked_failed=chunked_failed,
        failed_embeddings=failed_embeddings,
    )


class TestListCollections:
    """Tests for list_collections."""

    def test_empty(self) -> None:
        session = MagicMock()
        session.query().group_by().order_by().all.return_value = []

        result = list_collections(session)
        assert result == []

    def test_single_collection(self) -> None:
        session = MagicMock()
        session.query().group_by().order_by().all.return_value = [("default",)]
        session.query().filter_by().order_by().first.return_value = None
        session.query().filter().order_by().first.return_value = None
        session.query().filter.return_value.scalar.return_value = 3

        result = list_collections(session)
        assert len(result) == 1
        assert result[0].name == "default"
        assert result[0].documents == 3
        assert result[0].active_revision_label is None
        assert result[0].building_revision_label is None

    def test_with_active_and_building_revisions(self) -> None:
        session = MagicMock()
        session.query().group_by().order_by().all.return_value = [("c1",)]

        active_rev = MagicMock(spec=PipelineRevision)
        active_rev.label = "v1"
        building_rev = MagicMock(spec=PipelineRevision)
        building_rev.label = "v2"

        # First call: active revision query
        # Second call: building revision query
        # Third call: count query
        session.query().filter_by().order_by().first.side_effect = [active_rev, building_rev]
        session.query().filter().order_by().first.return_value = building_rev
        session.query().filter.return_value.scalar.return_value = 5

        result = list_collections(session)
        assert len(result) == 1
        assert result[0].name == "c1"
        assert result[0].documents == 5
        assert result[0].active_revision_label == "v1"
        assert result[0].building_revision_label == "v2"


class TestDeleteCollectionRecords:
    """Tests for delete_collection_records."""

    def test_nonexistent_collection_returns_none(self) -> None:
        session = MagicMock()
        session.query().filter_by().all.return_value = []

        result = delete_collection_records(session, "missing")
        assert result is None

    def test_deletes_existing_collection(self) -> None:
        session = MagicMock()
        doc = MagicMock(spec=SourceDocument)
        doc.id = 1
        session.query().filter_by().all.side_effect = [[doc], [(7,)]]

        session.query().filter().all.return_value = [("/path/to/artifact.md.gz",)]
        session.query().filter().delete.return_value = 5  # chunks
        session.query().filter_by().delete.return_value = 1  # revisions
        session.query().filter().delete.side_effect = [5, 1]  # chunks, then source docs
        session.query().filter_by().scalar.return_value = 0

        result = delete_collection_records(session, "c1")
        assert isinstance(result, DeleteCollectionResult)
        assert result.deleted_docs == 1
        assert result.artifact_paths == ["/path/to/artifact.md.gz"]
        assert result.vector_profile_ids == [7]
        assert session.commit.called


class TestDropOrphanVectorTables:
    """Tests for vector table cleanup shell."""

    @patch("cementic.collections.drop_vector_table")
    def test_drops_each_orphan_profile_table(self, mock_drop: MagicMock) -> None:
        engine = MagicMock()

        drop_orphan_vector_tables(engine, [3, 7])

        assert mock_drop.call_args_list == [((engine, 3),), ((engine, 7),)]


class TestRemoveArtifacts:
    """Tests for remove_artifacts."""

    @pytest.fixture(autouse=True)
    def _config(self, temp_dir: Path | None) -> Config:
        config = Config()
        if temp_dir is not None:
            config.storage.artifacts_path = temp_dir
        return config

    def test_removes_existing_files(self, temp_dir: Path, _config: Config) -> None:
        path = temp_dir / "artifact.md.gz"
        path.write_text("temp")
        remove_artifacts([str(path)], config=_config)
        assert not path.exists()

    def test_no_error_if_missing(self, temp_dir: Path, _config: Config) -> None:
        path = temp_dir / "nonexistent.md.gz"
        # Should not raise
        remove_artifacts([str(path)], config=_config)

    def test_empty_list(self, _config: Config) -> None:
        # Should not raise
        remove_artifacts([], config=_config)

    def test_swallows_oserror(self, temp_dir: Path, _config: Config) -> None:
        """remove_artifacts should silently pass on OSError (e.g. PermissionError)."""
        # Place file under artifacts root so path validation passes.
        f = temp_dir / "fails.md.gz"
        f.write_text("content")
        with patch.object(Path, "unlink", side_effect=OSError):
            # Should not raise
            remove_artifacts([str(f)], config=_config)
        f.unlink(missing_ok=True)


class TestListCollectionRevisions:
    """Tests for list_collection_revisions."""

    def test_returns_revisions(self) -> None:
        session = MagicMock()
        expected = [MagicMock(spec=PipelineRevision), MagicMock(spec=PipelineRevision)]
        session.query().filter_by().order_by().all.return_value = expected

        result = list_collection_revisions(session, "c1")
        assert result == expected


class TestPromoteReadyRevision:
    """Tests for promote_ready_revision."""

    @pytest.fixture(autouse=True)
    def _config(self) -> Config:
        return Config()

    def test_no_ready_revision_returns_no_ready(self, _config: Config) -> None:
        session = MagicMock()
        session.query().filter_by().order_by().first.return_value = None

        outcome = promote_ready_revision(session, "c1", config=_config)
        assert outcome.status == "no_ready"
        assert outcome.revision is None

    @patch("cementic.collections.compute_revision_counts", return_value=_counts())
    @patch("cementic.collections.promote_revision")
    def test_promotes_ready_revision(
        self, mock_promote: MagicMock, _mock_counts: MagicMock, _config: Config
    ) -> None:
        session = MagicMock()
        revision = MagicMock(spec=PipelineRevision)
        session.query().filter_by().order_by().first.return_value = revision

        outcome = promote_ready_revision(session, "c1", config=_config)
        assert outcome.status == "promoted"
        assert outcome.revision is revision
        mock_promote.assert_called_once_with(session, "c1", revision, config=_config)
        assert session.commit.called

    @patch("cementic.collections.compute_revision_counts", return_value=_counts(extracted_failed=2))
    @patch("cementic.collections.promote_revision")
    def test_blocks_when_revision_has_failures(
        self, mock_promote: MagicMock, _mock_counts: MagicMock, _config: Config
    ) -> None:
        session = MagicMock()
        revision = MagicMock(spec=PipelineRevision)
        session.query().filter_by().order_by().first.return_value = revision

        outcome = promote_ready_revision(session, "c1", config=_config)
        assert outcome.status == "blocked_by_failures"
        assert outcome.failures is not None and outcome.failures.extracted_failed == 2
        mock_promote.assert_not_called()
        assert not session.commit.called

    @patch(
        "cementic.collections.compute_revision_counts",
        return_value=_counts(failed_embeddings=1),
    )
    @patch("cementic.collections.promote_revision")
    def test_force_promotes_despite_failures(
        self, mock_promote: MagicMock, _mock_counts: MagicMock, _config: Config
    ) -> None:
        session = MagicMock()
        revision = MagicMock(spec=PipelineRevision)
        session.query().filter_by().order_by().first.return_value = revision

        outcome = promote_ready_revision(session, "c1", config=_config, force=True)
        assert outcome.status == "promoted"
        mock_promote.assert_called_once_with(session, "c1", revision, config=_config)
        assert session.commit.called
