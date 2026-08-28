"""Test fixtures and configuration."""

import os
import sqlite3
import tempfile
import warnings
import weakref
from pathlib import Path
from typing import Generator

import pytest
from sqlalchemy import event
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.pool import Pool

from tests.fixtures.generate_pdfs import ensure_fixture_pdfs

# Ensure DB password env var is set for all tests; matches compose.yml's default.
os.environ.setdefault("CEMENTIC_DB_PASSWORD", "cementic")


@pytest.fixture(scope="session", autouse=True)
def _ensure_pdf_fixtures() -> None:
    """Regenerate any missing fixture PDF so a fresh clone can run the suite.

    ``*.pdf`` is gitignored; see tests/fixtures/generate_pdfs.py.
    """
    ensure_fixture_pdfs()


@pytest.fixture(autouse=True)
def _isolate_user_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Keep tests hermetic: never pick up a developer's real config file.

    Clears CEMENTIC_CONFIG and points the user-config dir at an empty temp dir,
    so resolve_config_path() returns None unless a test opts in. Tests that
    exercise the config file set CEMENTIC_CONFIG themselves.
    """
    monkeypatch.delenv("CEMENTIC_CONFIG", raising=False)
    empty = tmp_path_factory.mktemp("cementic-no-user-config")
    monkeypatch.setattr("cementic.config.user_config_dir", lambda *a, **k: str(empty))


@pytest.fixture(autouse=True)
def _isolate_user_data_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Keep tests hermetic: never write into the developer's real data directory.

    The config-file isolation above covers where settings are *read* from; this
    covers where state, artifacts and the model cache are *written*. Without it
    a test that constructs a real worker gets a StateManager pointed at
    ``~/.local/share/cementic/``: a `SourceWatcher()` built in a unit test wrote
    `last_error: "RuntimeError: db down"` into the live state file, where
    `cementic status` then reported it as a real failure of a healthy watcher.
    Found by running the CLI against the live corpus, not by the suite.

    Defaults are derived from this directory, so it must be patched before any
    Config is constructed -- hence autouse and session-independent.
    """
    data_dir = tmp_path_factory.mktemp("cementic-data")
    monkeypatch.setattr("cementic.config.user_data_dir", lambda *a, **k: str(data_dir))


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(autouse=True)
def _close_sqlite_connections_after_each_test() -> Generator[None, None, None]:
    """Close every SQLite connection a test opened, however it was created.

    Tests across this suite build ~30 ad hoc ``create_engine(...)`` instances
    (sqlite, mostly in-memory) with no ``dispose()``, and hand many of them to a
    ``sessionmaker(...)()`` that is never closed. Harmless on 3.12; from 3.13 on
    ``sqlite3.Connection`` emits ``ResourceWarning: unclosed database`` when it
    is finalized, which -- with this project's ``filterwarnings = ["error"]``
    and pytest 9's unraisable-exception hook -- fails whichever *unrelated* test
    happens to be running when the collector gets to it. Locally that was
    ``test_search.py``'s sessions failing ``test_status_service.py``.

    Editing each call site would mean touching ~30 places across 10 files and
    would silently miss the next one added, so this closes what the test opened,
    at two levels:

    - Engines that opened a connection are disposed, via the ``engine_connect``
      event registered globally on the ``Engine`` class -- documented, public
      SQLAlchemy API, unlike two rejected alternatives: wrapping
      ``Engine.__init__`` broke ``create_engine()``'s own kwarg-routing
      introspection for the psycopg2 dialect (a real engine with
      ``echo=``/``pool_pre_ping=`` started raising ``TypeError: Invalid
      argument(s) 'echo'``), and a ``gc.get_objects()`` sweep at every teardown
      was correct but slowed the full unit suite roughly 3x (16s -> 55s).
      An engine that's built but never used never opens a DBAPI connection, so
      it has nothing to leak and needs no entry here.
    - The DBAPI connections themselves, via the pool-level ``connect`` event.
      ``dispose()`` only closes what is *in* the pool: a connection checked out
      by a session the test never closed stays open, which is the leak the
      engine sweep alone left behind. Disposal runs first so that every
      connection a pool still owns is closed through its pool, leaving this
      sweep to catch only what no pool tracks any more -- nothing here can then
      close a connection out from under a pool a later test would draw from.

    Only SQLite is touched. psycopg2 connections raise no such warning, and the
    session-scoped ``pg_engine`` must survive the test that first used it.
    Every connection this sees was therefore opened by a test: ``src/`` never
    opens a SQLite connection at all, and ``db.get_engine`` could not hand it
    one -- that function passes ``connect_timeout``, which ``sqlite3.connect``
    rejects.
    """
    engines: "weakref.WeakSet[Engine]" = weakref.WeakSet()
    connections: list[sqlite3.Connection] = []

    def _on_engine_connect(connection: Connection) -> None:
        engines.add(connection.engine)

    def _on_pool_connect(dbapi_connection: object, _record: object) -> None:
        if isinstance(dbapi_connection, sqlite3.Connection):
            connections.append(dbapi_connection)

    event.listen(Engine, "engine_connect", _on_engine_connect)
    event.listen(Pool, "connect", _on_pool_connect)
    try:
        yield
    finally:
        event.remove(Engine, "engine_connect", _on_engine_connect)
        event.remove(Pool, "connect", _on_pool_connect)
        for engine in engines:
            if engine.dialect.name == "sqlite":
                engine.dispose()
        for connection in connections:
            try:
                connection.close()
            except sqlite3.ProgrammingError as error:
                # The one connection this cannot free: an in-memory database
                # opened on a worker thread keeps check_same_thread=True, so
                # only that thread may close it, and SQLAlchemy's own dispose()
                # hits the same wall. Closing an already-closed connection is a
                # no-op, so this is the only case that reaches here -- and no
                # test produces it today (measured: none of the 94 connections
                # the two suites open). Naming it here beats letting it return
                # as a ResourceWarning against some unrelated test, which is
                # the failure this fixture exists to end.
                warnings.warn(
                    f"SQLite connection opened on another thread, left open: {error}",
                    ResourceWarning,
                    stacklevel=1,
                )
