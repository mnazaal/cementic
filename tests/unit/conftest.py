"""Fixtures for tests/unit."""

from __future__ import annotations

import weakref
from collections.abc import Iterator

import pytest
from sqlalchemy import event
from sqlalchemy.engine import Connection, Engine


@pytest.fixture(autouse=True)
def _dispose_sqlalchemy_engines_after_each_test() -> Iterator[None]:
    """Dispose every ``Engine`` a unit test connects, however it was created.

    Unit tests across this directory build ~23 ad hoc ``create_engine(...)``
    instances (sqlite, mostly in-memory) with no ``dispose()``. Harmless on
    3.12; on 3.14 the resulting ``ResourceWarning: unclosed database``, combined
    with this project's ``filterwarnings = ["error"]`` and pytest 9's
    unraisable-exception hook, turns into a GC-timing-dependent failing test.

    Editing each call site would mean touching ~23 places across 7 files and
    would silently miss the next one added, so this tracks every ``Engine``
    that actually opens a connection during the test and disposes it at
    teardown, via the ``engine_connect`` event registered globally on the
    ``Engine`` class -- documented, public SQLAlchemy API, unlike two rejected
    alternatives: wrapping ``Engine.__init__`` broke ``create_engine()``'s own
    kwarg-routing introspection for the psycopg2 dialect (a real engine with
    ``echo=``/``pool_pre_ping=`` started raising ``TypeError: Invalid
    argument(s) 'echo'``), and a ``gc.get_objects()`` sweep at every teardown
    was correct but slowed the full unit suite roughly 3x (16s -> 55s).
    An engine that's built but never used never opens a DBAPI connection, so
    it has nothing to leak and needs no entry here.
    """
    engines: "weakref.WeakSet[Engine]" = weakref.WeakSet()

    def _on_engine_connect(conn: Connection) -> None:
        engines.add(conn.engine)

    event.listen(Engine, "engine_connect", _on_engine_connect)
    try:
        yield
    finally:
        event.remove(Engine, "engine_connect", _on_engine_connect)
        for engine in engines:
            engine.dispose()
