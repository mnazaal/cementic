"""Tests for collection and revision persistence helpers."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

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
from cementic.db import (
    Base,
    Chunk,
    ChunkedDocument,
    ChunkProfile,
    EmbeddingProfile,
    ExtractedDocument,
    ExtractorProfile,
    PipelineRevision,
    SourceDocument,
)
from cementic.pipeline_worker import PipelineCounts


def _counts(*, extracted_failed: int = 0, chunked_failed: int = 0, failed_embeddings: int = 0):
    """Counts for a revision that is *complete*, with the given failures.

    Failures are folded into each stage's total so the stage equalities still
    balance. Left unbalanced, the revision reads as merely unfinished and
    promotion is refused for that reason instead of the one under test.
    """
    extracted_done = 1 + chunked_failed
    return PipelineCounts(
        documents=extracted_done + extracted_failed,
        extracted_done=extracted_done,
        chunked_done=1,
        total_chunks=1 + failed_embeddings,
        done_embeddings=1,
        extracted_failed=extracted_failed,
        chunked_failed=chunked_failed,
        failed_embeddings=failed_embeddings,
    )


def _new_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine, sessionmaker(bind=engine, expire_on_commit=False)()


def _count_select_queries(engine):
    class _Counter:
        value = 0

    counter = _Counter()

    def _on_execute(conn, cursor, statement, *args, **kwargs):
        if statement.lstrip().upper().startswith("SELECT"):
            counter.value += 1

    event.listen(engine, "before_cursor_execute", _on_execute)
    return counter


def _seed_profiles(session, suffix: str = "1"):
    extractor = ExtractorProfile(name="x", fingerprint=f"ext-{suffix}", config_json="{}")
    chunk_profile = ChunkProfile(fingerprint=f"chunk-{suffix}", config_json="{}")
    embedding_profile = EmbeddingProfile(
        fingerprint=f"embed-{suffix}",
        provider="test",
        model_identifier="test",
        embedding_dim=2,
        distance_metric="cosine",
        config_json="{}",
    )
    session.add_all([extractor, chunk_profile, embedding_profile])
    session.flush()
    return extractor, chunk_profile, embedding_profile


class TestListCollections:
    """Tests for list_collections."""

    def test_empty(self) -> None:
        _, session = _new_session()
        assert list_collections(session) == []

    def test_single_collection(self) -> None:
        _, session = _new_session()
        for i in range(3):
            session.add(
                SourceDocument(collection="default", source_path=f"/{i}.pdf", file_hash=f"h{i}")
            )
        session.commit()

        result = list_collections(session)
        assert len(result) == 1
        assert result[0].name == "default"
        assert result[0].documents == 3
        assert result[0].active_revision_label is None
        assert result[0].building_revision_label is None

    def test_with_active_and_building_revisions(self) -> None:
        _, session = _new_session()
        for i in range(5):
            session.add(SourceDocument(collection="c1", source_path=f"/{i}.pdf", file_hash=f"h{i}"))
        extractor1, chunk_profile1, embedding_profile1 = _seed_profiles(session, suffix="a")
        extractor2, chunk_profile2, embedding_profile2 = _seed_profiles(session, suffix="b")
        session.add(
            PipelineRevision(
                collection="c1",
                label="v1",
                status="active",
                extractor_profile_id=extractor1.id,
                chunk_profile_id=chunk_profile1.id,
                embedding_profile_id=embedding_profile1.id,
            )
        )
        session.add(
            PipelineRevision(
                collection="c1",
                label="v2",
                status="building",
                extractor_profile_id=extractor2.id,
                chunk_profile_id=chunk_profile2.id,
                embedding_profile_id=embedding_profile2.id,
            )
        )
        session.commit()

        result = list_collections(session)
        assert len(result) == 1
        assert result[0].name == "c1"
        assert result[0].documents == 5
        assert result[0].active_revision_label == "v1"
        assert result[0].building_revision_label == "v2"
        assert result[0].ready_revision_label is None

    def test_ready_revision_is_not_reported_as_building(self) -> None:
        """A finished revision awaiting promotion must say so.

        Both summary builders bucket every non-active revision together, which
        target selection needs but reporting must not: shown as `building`, the
        one fact the promote workflow turns on -- that something is ready --
        was never surfaced by any command.
        """
        engine, session = _new_session()
        extractor, chunk_profile, embedding_profile = _seed_profiles(session, suffix="r")
        session.add(SourceDocument(collection="c1", source_path="/a.pdf", file_hash="h"))
        session.add(
            PipelineRevision(
                collection="c1",
                label="v9",
                status="ready",
                extractor_profile_id=extractor.id,
                chunk_profile_id=chunk_profile.id,
                embedding_profile_id=embedding_profile.id,
            )
        )
        session.commit()

        result = list_collections(session)

        assert result[0].ready_revision_label == "v9"
        assert result[0].building_revision_label is None

    def test_ready_revision_is_still_reported_behind_a_newer_building_one(self) -> None:
        """The normal state after a config change: a finished revision awaiting
        promotion, plus a newer build. Reporting the highest-id non-active
        revision alone showed only the build, hiding the promotable one that
        `collection promote` actually targets."""
        engine, session = _new_session()
        extractor, chunk_profile, embedding_profile = _seed_profiles(session, suffix="rb")
        extractor2, chunk_profile2, embedding_profile2 = _seed_profiles(session, suffix="rb2")
        session.add(SourceDocument(collection="c1", source_path="/a.pdf", file_hash="h"))
        session.add(
            PipelineRevision(
                collection="c1",
                label="ready-rev",
                status="ready",
                extractor_profile_id=extractor.id,
                chunk_profile_id=chunk_profile.id,
                embedding_profile_id=embedding_profile.id,
            )
        )
        session.flush()
        session.add(
            PipelineRevision(
                collection="c1",
                label="building-rev",
                status="building",
                extractor_profile_id=extractor2.id,
                chunk_profile_id=chunk_profile2.id,
                embedding_profile_id=embedding_profile2.id,
            )
        )
        session.commit()

        result = list_collections(session)

        assert result[0].ready_revision_label == "ready-rev"
        assert result[0].building_revision_label == "building-rev"

    def test_query_count_does_not_scale_with_collection_count(self) -> None:
        def run(n_collections: int) -> int:
            engine, session = _new_session()
            for i in range(n_collections):
                collection = f"col-{i}"
                session.add(
                    SourceDocument(
                        collection=collection, source_path=f"/{i}.pdf", file_hash=f"h{i}"
                    )
                )
                extractor, chunk_profile, embedding_profile = _seed_profiles(session, suffix=str(i))
                session.add(
                    PipelineRevision(
                        collection=collection,
                        label=f"v{i}",
                        status="active",
                        extractor_profile_id=extractor.id,
                        chunk_profile_id=chunk_profile.id,
                        embedding_profile_id=embedding_profile.id,
                    )
                )
            session.commit()

            counter = _count_select_queries(engine)
            result = list_collections(session)
            assert len(result) == n_collections
            return counter.value

        small = run(3)
        large = run(30)
        assert small == large


class TestDeleteCollectionRecordsWithoutDocuments:
    """A collection can own revisions but no documents.

    Regression: `cementic start` on a directory with no supported files creates
    exactly that, and the early return on the document check made those rows
    invisible to `collection list` and undeletable via `collection remove`,
    which reported "not found".
    """

    def test_revision_only_collection_is_deleted(self) -> None:
        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        session = sessionmaker(bind=engine)()
        extractor = ExtractorProfile(fingerprint="e", name="default", config_json="{}")
        chunk_profile = ChunkProfile(fingerprint="c", config_json="{}")
        embedding_profile = EmbeddingProfile(
            fingerprint="m",
            provider="llama-cpp",
            model_identifier="x",
            embedding_dim=4,
            distance_metric="cosine",
            config_json="{}",
        )
        session.add_all([extractor, chunk_profile, embedding_profile])
        session.flush()
        session.add(
            PipelineRevision(
                collection="ghost",
                extractor_profile_id=extractor.id,
                chunk_profile_id=chunk_profile.id,
                embedding_profile_id=embedding_profile.id,
                status="building",
            )
        )
        session.commit()

        result = delete_collection_records(session, "ghost")

        assert result is not None
        assert result.deleted_docs == 0
        assert session.query(PipelineRevision).filter_by(collection="ghost").count() == 0

    def test_genuinely_unknown_collection_still_returns_none(self) -> None:
        engine = create_engine("sqlite://")
        Base.metadata.create_all(engine)
        session = sessionmaker(bind=engine)()

        assert delete_collection_records(session, "never-existed") is None


class TestDeleteCollectionRecords:
    """Tests for delete_collection_records."""

    def test_nonexistent_collection_returns_none(self) -> None:
        _, session = _new_session()
        result = delete_collection_records(session, "missing")
        assert result is None

    def test_deletes_existing_collection(self) -> None:
        _, session = _new_session()
        extractor, chunk_profile, embedding_profile = _seed_profiles(session)
        session.add(
            PipelineRevision(
                collection="c1",
                label="v1",
                status="active",
                extractor_profile_id=extractor.id,
                chunk_profile_id=chunk_profile.id,
                embedding_profile_id=embedding_profile.id,
            )
        )
        doc = SourceDocument(collection="c1", source_path="/a.pdf", file_hash="h1")
        session.add(doc)
        session.flush()
        extracted = ExtractedDocument(
            document_id=doc.id,
            extractor_profile_id=extractor.id,
            status="done",
            artifact_path="/path/to/artifact.md.gz",
        )
        session.add(extracted)
        session.flush()
        chunked = ChunkedDocument(
            extracted_document_id=extracted.id, chunk_profile_id=chunk_profile.id, status="done"
        )
        session.add(chunked)
        session.flush()
        session.add(
            Chunk(document_id=doc.id, chunked_document_id=chunked.id, chunk_index=0, content="c")
        )
        session.commit()

        result = delete_collection_records(session, "c1")
        assert isinstance(result, DeleteCollectionResult)
        assert result.deleted_docs == 1
        assert result.deleted_chunks == 1
        assert result.artifact_paths == ["/path/to/artifact.md.gz"]
        assert result.vector_profile_ids == [embedding_profile.id]
        assert session.query(SourceDocument).count() == 0
        assert session.query(PipelineRevision).count() == 0

    def test_vector_profile_still_referenced_elsewhere_is_kept(self) -> None:
        """A profile still used by another collection's revision must not be dropped."""
        _, session = _new_session()
        extractor, chunk_profile, embedding_profile = _seed_profiles(session)
        session.add(
            PipelineRevision(
                collection="c1",
                label="v1",
                status="active",
                extractor_profile_id=extractor.id,
                chunk_profile_id=chunk_profile.id,
                embedding_profile_id=embedding_profile.id,
            )
        )
        session.add(
            PipelineRevision(
                collection="c2",
                label="v1",
                status="active",
                extractor_profile_id=extractor.id,
                chunk_profile_id=chunk_profile.id,
                embedding_profile_id=embedding_profile.id,
            )
        )
        doc = SourceDocument(collection="c1", source_path="/a.pdf", file_hash="h1")
        session.add(doc)
        session.commit()

        result = delete_collection_records(session, "c1")
        assert result is not None
        assert result.vector_profile_ids == []

    def test_query_count_bounded_by_profile_count(self) -> None:
        """Regression: remaining-reference check groups by profile, not one query each."""

        def run(n_old_profiles: int) -> int:
            engine, session = _new_session()
            extractor, chunk_profile, _ = _seed_profiles(session, suffix=str(n_old_profiles))
            doc = SourceDocument(collection="c1", source_path="/a.pdf", file_hash="h1")
            session.add(doc)
            for i in range(n_old_profiles):
                embedding_profile = EmbeddingProfile(
                    fingerprint=f"embed-{n_old_profiles}-{i}",
                    provider="test",
                    model_identifier="test",
                    embedding_dim=2,
                    distance_metric="cosine",
                    config_json="{}",
                )
                session.add(embedding_profile)
                session.flush()
                session.add(
                    PipelineRevision(
                        collection="c1",
                        label=f"v{i}",
                        status="retired",
                        extractor_profile_id=extractor.id,
                        chunk_profile_id=chunk_profile.id,
                        embedding_profile_id=embedding_profile.id,
                    )
                )
            session.commit()

            counter = _count_select_queries(engine)
            delete_collection_records(session, "c1")
            return counter.value

        small = run(2)
        large = run(30)
        assert small == large


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
        assert outcome.counts is not None and outcome.counts.extracted_failed == 2
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

    @patch(
        "cementic.collections.compute_revision_counts",
        return_value=PipelineCounts(
            documents=5, extracted_done=1, chunked_done=1, total_chunks=1, done_embeddings=1
        ),
    )
    @patch("cementic.collections.promote_revision")
    def test_refuses_a_ready_revision_carrying_unfinished_work(
        self, mock_promote: MagicMock, _mock_counts: MagicMock, _config: Config
    ) -> None:
        """`ready` says a revision was complete once, not that it still is.

        Nothing flips it back to `building` when the watcher registers new
        documents, and every `cementic start` resets failed rows to pending, so
        the failure counts alone can read zero while work is outstanding.
        """
        session = MagicMock()
        session.query().filter_by().order_by().first.return_value = MagicMock(
            spec=PipelineRevision
        )

        outcome = promote_ready_revision(session, "c1", config=_config)
        assert outcome.status == "incomplete"
        assert outcome.counts is not None and outcome.counts.documents == 5
        mock_promote.assert_not_called()
        assert not session.commit.called

    @patch(
        "cementic.collections.compute_revision_counts",
        return_value=PipelineCounts(
            documents=5, extracted_done=1, chunked_done=1, total_chunks=1, done_embeddings=1
        ),
    )
    @patch("cementic.collections.promote_revision")
    def test_force_publishes_an_incomplete_revision(
        self, mock_promote: MagicMock, _mock_counts: MagicMock, _config: Config
    ) -> None:
        session = MagicMock()
        revision = MagicMock(spec=PipelineRevision)
        session.query().filter_by().order_by().first.return_value = revision

        outcome = promote_ready_revision(session, "c1", config=_config, force=True)
        assert outcome.status == "promoted"
        mock_promote.assert_called_once_with(session, "c1", revision, config=_config)

    @patch(
        "cementic.collections.compute_revision_counts",
        return_value=PipelineCounts(
            documents=0, extracted_done=0, chunked_done=0, total_chunks=0, done_embeddings=0
        ),
    )
    @patch("cementic.collections.promote_revision")
    def test_refuses_an_empty_revision_even_with_force(
        self, mock_promote: MagicMock, _mock_counts: MagicMock, _config: Config
    ) -> None:
        """Promotion retires the active revision, so publishing an empty one
        removes search coverage rather than merely adding none."""
        session = MagicMock()
        session.query().filter_by().order_by().first.return_value = MagicMock(
            spec=PipelineRevision
        )

        outcome = promote_ready_revision(session, "c1", config=_config, force=True)
        assert outcome.status == "empty"
        mock_promote.assert_not_called()
        assert not session.commit.called
