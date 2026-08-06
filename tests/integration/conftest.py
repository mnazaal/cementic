"""Fixtures for integration tests — SQLite + PostgreSQL.

The PostgreSQL tests bring the compose stack (pgvector + vectorscale) up and down
themselves, using whichever container engine is available (docker or podman).
"""

import shutil
import subprocess
import time
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from cementic.config import Config
from cementic.db import Base, create_tables

_COMPOSE_FILE = Path(__file__).resolve().parents[2] / "compose.yml"
_STARTUP_TIMEOUT = 600  # first run builds the pgvector + vectorscale image


def _pg_url() -> object:
    """Target Postgres URL, honoring the real config (CEMENTIC_DB_* / CEMENTIC_DB_URL).

    Defaults to the local ``cementic`` credentials when nothing is set, which is
    what the bundled ``compose.yml`` also defaults to — so the probe matches the
    database the suite would otherwise bring up itself.
    """
    return Config().database.url


def _pg_reachable() -> bool:
    """Check if PostgreSQL is reachable."""
    engine = None
    try:
        engine = create_engine(_pg_url(), connect_args={"connect_timeout": 2})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
    finally:
        if engine is not None:
            engine.dispose()


def _detect_compose_engine() -> list[str] | None:
    """Return a working ``<engine> compose`` base command, or None.

    Checks docker then podman, and verifies the ``compose`` subcommand actually
    resolves a provider (a bare `podman` without a compose provider is skipped).
    """
    for engine in ("docker", "podman"):
        if not shutil.which(engine):
            continue
        try:
            result = subprocess.run(
                [engine, "compose", "version"],
                capture_output=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if result.returncode == 0:
            return [engine, "compose"]
    return None


def _wait_pg_reachable(timeout: int) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _pg_reachable():
            return True
        time.sleep(2)
    return False


@pytest.fixture(scope="session")
def _compose_postgres():
    """Bring the compose Postgres up for the session (down after) if not already up."""
    if _pg_reachable():
        # Already running (CI service, or a dev started it) — leave it untouched.
        yield
        return

    engine = _detect_compose_engine()
    if engine is None:
        pytest.skip("docker or podman is required for PostgreSQL integration tests")

    compose = [*engine, "-f", str(_COMPOSE_FILE)]
    subprocess.run([*compose, "up", "-d"], check=True)
    try:
        if not _wait_pg_reachable(_STARTUP_TIMEOUT):
            raise RuntimeError("compose Postgres did not become reachable in time")
        yield
    finally:
        subprocess.run([*compose, "down"], check=False)


def _drop_vector_tables(engine) -> None:
    """Drop the dynamically-created per-profile vector tables.

    They are not part of ``Base.metadata`` but carry a foreign key to
    ``chunks_v2``, so leaving one behind makes ``drop_all`` fail with
    DependentObjectsStillExist and takes the whole session down with it.
    """
    with engine.connect() as conn:
        names = conn.execute(
            text("SELECT tablename FROM pg_tables WHERE tablename LIKE 'embedding_vectors_p%'")
        ).all()
        for (table_name,) in names:
            conn.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
        conn.commit()


@pytest.fixture(scope="session")
def pg_engine(_compose_postgres):
    """Session-scoped PostgreSQL engine backed by the compose-managed service."""
    if not _pg_reachable():
        pytest.skip("PostgreSQL not reachable")
    engine = create_engine(_pg_url(), connect_args={"connect_timeout": 2})
    _drop_vector_tables(engine)
    Base.metadata.drop_all(engine)
    create_tables(engine)
    yield engine
    _drop_vector_tables(engine)
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def pg_session(pg_engine):
    """Function-scoped PostgreSQL session with committed cleanup.

    Tests often open separate connections through production code, so fixture data must be
    committed rather than hidden inside an outer rollback transaction.
    """
    session_factory = sessionmaker(bind=pg_engine)
    session = session_factory()

    try:
        yield session
    finally:
        session.rollback()
        session.execute(
            text(
                "TRUNCATE chunk_embeddings, chunks_v2, chunked_documents, "
                "extracted_documents, source_documents, pipeline_revisions, "
                "embedding_profiles, extractor_profiles, chunk_profiles RESTART IDENTITY CASCADE"
            )
        )
        vector_tables = session.execute(
            text("SELECT tablename FROM pg_tables WHERE tablename LIKE 'embedding_vectors_p%'")
        ).all()
        for (table_name,) in vector_tables:
            session.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
        session.commit()
        session.close()


@pytest.fixture
def pg_config(pg_engine) -> Config:
    """Config pointing at the same PostgreSQL the engine fixture uses.

    Honors the real environment (CEMENTIC_DB_* / CEMENTIC_DB_URL) rather than
    forcing the default password, so the suite works against any reachable DB.
    """
    return Config()


@pytest.fixture(scope="session")
def pdf_fixtures_dir() -> Path:
    """Directory containing test PDF fixtures."""
    return Path(__file__).resolve().parent.parent / "fixtures"
