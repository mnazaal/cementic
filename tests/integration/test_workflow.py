"""Integration tests for revision lifecycle behavior."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cementic.config import Config
from cementic.db import Base, PipelineRevision
from cementic.profiles import (
    build_chunk_profile_payload,
    build_embedding_profile_payload,
    build_extractor_profile_payload,
)
from cementic.revisions import get_active_revision, get_target_revision, promote_revision


def _config_for(temp_dir: Path) -> Config:
    config = Config()
    config.storage.artifacts_path = temp_dir / "artifacts"
    config.llama_cpp.model_path = str(temp_dir / "model.gguf")
    return config


def test_model_change_creates_new_revision_without_reusing_old_embedding_profile(
    temp_dir: Path,
) -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    config = _config_for(temp_dir)

    with session_factory() as session:
        first = get_target_revision(session, "research", config)
        session.commit()
        first_embedding_profile_id = first.embedding_profile_id

        config.pipeline.embedding_provider = "ollama"
        second = get_target_revision(session, "research", config)
        session.commit()

        assert second.id != first.id
        assert second.embedding_profile_id != first_embedding_profile_id
        assert second.status == "building"

    with session_factory() as session:
        refreshed_first = session.query(PipelineRevision).filter_by(id=first.id).first()
        assert refreshed_first is not None
        assert refreshed_first.status == "superseded"


def test_ready_revision_can_be_promoted_per_collection(temp_dir: Path) -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    config = _config_for(temp_dir)

    with session_factory() as session:
        first = get_target_revision(session, "research", config)
        first.status = "ready"
        promote_revision(session, "research", first)
        session.commit()

        config.pipeline.embedding_provider = "ollama"
        second = get_target_revision(session, "research", config)
        second.status = "ready"
        promote_revision(session, "research", second)
        session.commit()

    with session_factory() as session:
        active = get_active_revision(session, "research")
        retired = session.query(PipelineRevision).filter_by(id=first.id).first()

        assert active is not None
        assert active.id == second.id
        assert retired is not None
        assert retired.status == "retired"


def test_profile_payloads_change_only_when_relevant_config_changes(temp_dir: Path) -> None:
    config = _config_for(temp_dir)
    extractor_payload = build_extractor_profile_payload(config)
    chunk_payload = build_chunk_profile_payload(config)
    embedding_payload = build_embedding_profile_payload(config)

    config.pipeline_worker.batch_size = 99
    assert build_extractor_profile_payload(config) == extractor_payload
    assert build_chunk_profile_payload(config) == chunk_payload
    assert build_embedding_profile_payload(config) == embedding_payload

    config.extraction.use_ocr = not config.extraction.use_ocr
    assert build_extractor_profile_payload(config) != extractor_payload


def test_promotion_prunes_older_retired_revisions(temp_dir: Path) -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    config = _config_for(temp_dir)

    with session_factory() as session:
        first = get_target_revision(session, "research", config)
        first.status = "ready"
        promote_revision(session, "research", first)
        session.commit()

        config.pipeline.embedding_provider = "ollama"
        second = get_target_revision(session, "research", config)
        second.status = "ready"
        promote_revision(session, "research", second)
        session.commit()

        config.pipeline.embedding_provider = "llama-cpp"
        config.llama_cpp.model_path = str(temp_dir / "model-v2.gguf")
        third = get_target_revision(session, "research", config)
        third.status = "ready"
        promote_revision(session, "research", third)
        session.commit()

    with session_factory() as session:
        revisions = (
            session.query(PipelineRevision)
            .filter_by(collection="research")
            .order_by(PipelineRevision.id)
            .all()
        )

        assert [revision.status for revision in revisions] == ["retired", "active"]
        assert revisions[0].id == second.id
        assert revisions[1].id == third.id
