"""Fixtures for integration tests — SQLite + PostgreSQL.

The PostgreSQL tests bring the compose stack (pgvector + vectorscale) up and down
themselves, using whichever container engine is available (docker or podman).
"""

import shutil
import subprocess
import time
from pathlib import Path

import pytest
from pydantic import SecretStr
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from cementic.config import Config
from cementic.db import Base, create_tables

_COMPOSE_FILE = Path(__file__).resolve().parents[2] / "compose.yml"
#: Covers container startup only -- `compose up -d` builds the pgvector +
#: vectorscale image before it returns, so the (long) first-run build is not on
#: this clock. Kept short enough that a container which fails to start says so
#: in minutes instead of stalling the CI job for ten.
_STARTUP_TIMEOUT = 180


#: These fixtures call Base.metadata.drop_all(), so they must never target a
#: database anyone keeps real data in. The configured database name gets this
#: suffix appended, and the suite refuses to run without it.
_TEST_DB_SUFFIX = "_test"


def _configured_url():
    """The URL the user's own cementic uses (CEMENTIC_DB_* / CEMENTIC_DB_URL)."""
    return Config().database.url


def _pg_url():
    """Target Postgres URL for tests: always a dedicated ``*_test`` database.

    Host, port and credentials come from the real config so the suite runs
    against whatever Postgres is available -- but the database *name* never
    does. These fixtures drop every cementic table on setup and again on
    teardown; pointing that at the configured database destroys the user's
    index with no warning and no opt-in.
    """
    configured = _configured_url()
    name = configured.database or "cementic"
    if name.endswith(_TEST_DB_SUFFIX):
        return configured
    return configured.set(database=f"{name}{_TEST_DB_SUFFIX}")


def _require_isolated_test_database() -> None:
    """Refuse to run destructive fixtures against a non-test database."""
    target = _pg_url().database or ""
    if not target.endswith(_TEST_DB_SUFFIX):
        pytest.exit(
            f"refusing to run destructive integration tests against {target!r}: "
            f"the target database name must end in {_TEST_DB_SUFFIX!r}",
            returncode=1,
        )


def _url_reachable(url) -> bool:
    engine = None
    try:
        engine = create_engine(url, connect_args={"connect_timeout": 2})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
    finally:
        if engine is not None:
            engine.dispose()


def _server_reachable() -> bool:
    """Whether the Postgres server answers, independent of the test database."""
    return _url_reachable(_pg_url().set(database="postgres"))


def _ensure_test_database() -> None:
    """Create the dedicated test database if it does not exist yet."""
    target = _pg_url()
    admin = create_engine(
        target.set(database="postgres"),
        isolation_level="AUTOCOMMIT",
        connect_args={"connect_timeout": 2},
    )
    try:
        with admin.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": target.database},
            ).scalar()
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{target.database}"'))
    finally:
        admin.dispose()


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


def _wait_server_reachable(timeout: int) -> bool:
    """Wait for the *server*, not for the test database.

    Waiting on ``_pg_url()`` here cannot ever succeed on a cold machine: that
    database is created by ``_ensure_test_database()``, which runs in
    ``pg_engine`` -- i.e. only after this fixture has yielded. Compose came up
    fine and the suite still sat out the whole timeout and failed, which is how
    CI's pg job burned 19 minutes per run.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _server_reachable():
            return True
        time.sleep(2)
    return False


def _compose_diagnostics(compose: list[str]) -> str:
    """Container status and recent logs, for a startup failure to explain itself."""
    parts = []
    for label, args in (("ps", ["ps", "-a"]), ("logs", ["logs", "--tail", "50"])):
        try:
            result = subprocess.run(
                [*compose, *args], capture_output=True, text=True, timeout=30
            )
        except (OSError, subprocess.SubprocessError) as error:
            parts.append(f"--- compose {label} unavailable: {error}")
            continue
        parts.append(f"--- compose {label} ---\n{result.stdout}{result.stderr}")
    return "\n".join(parts)


@pytest.fixture(scope="session")
def _compose_postgres():
    """Bring the compose Postgres up for the session (down after) if not already up."""
    _require_isolated_test_database()
    if _server_reachable():
        # Already running (CI service, or a dev started it) — leave it untouched.
        yield
        return

    engine = _detect_compose_engine()
    if engine is None:
        pytest.skip("docker or podman is required for PostgreSQL integration tests")

    compose = [*engine, "-f", str(_COMPOSE_FILE)]
    subprocess.run([*compose, "up", "-d"], check=True)
    try:
        if not _wait_server_reachable(_STARTUP_TIMEOUT):
            raise RuntimeError(
                f"compose Postgres did not become reachable within "
                f"{_STARTUP_TIMEOUT}s at {_pg_url().set(database='postgres')}\n"
                f"{_compose_diagnostics(compose)}"
            )
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
    """Session-scoped engine bound to the dedicated ``*_test`` database."""
    _require_isolated_test_database()
    if not _server_reachable():
        pytest.skip("PostgreSQL not reachable")
    _ensure_test_database()
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
    """Config pointing at the same database the engine fixture uses.

    Server and credentials come from the real environment (CEMENTIC_DB_* /
    CEMENTIC_DB_URL), but the URL is pinned to the dedicated test database.
    Returning a bare Config() here made anything that opens its own connection
    from this config -- Searcher, most obviously -- read the *user's* database
    while the fixtures seeded the test one.
    """
    config = Config()
    config.database.url_override = SecretStr(
        _pg_url().render_as_string(hide_password=False)
    )
    return config


@pytest.fixture(scope="session")
def pdf_fixtures_dir() -> Path:
    """Directory containing test PDF fixtures."""
    return Path(__file__).resolve().parent.parent / "fixtures"


#: Fixtures that require a real PostgreSQL server. Anything depending on one of
#: these (directly or transitively) belongs to the ``pg`` job.
_PG_FIXTURES = frozenset({"pg_engine", "pg_session", "pg_config", "pg_setup"})


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark every Postgres-dependent test ``pg``, rather than trusting authors to.

    Hand-marking drifted: five tests took the Postgres fixtures without the
    marker, so ``-m pg`` deselected them and ``-m "not pg"`` selected them only
    for their own fixture to skip. They ran in neither CI job -- including the
    full build-to-ready pipeline test -- which is exactly the coverage gap that
    lets a promote regression reach main.
    """
    for item in items:
        if _PG_FIXTURES & set(getattr(item, "fixturenames", ())):
            item.add_marker(pytest.mark.pg)
