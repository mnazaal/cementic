"""The integration suite must never touch the database the user actually uses.

These fixtures call ``Base.metadata.drop_all()`` on setup and teardown. Pointing
that at the configured database silently destroys a real index -- which is
exactly what happened during development, repeatedly, before this guard existed.
"""

from __future__ import annotations

import pytest

from cementic.config import Config
from tests.integration.conftest import _TEST_DB_SUFFIX, _configured_url, _pg_url


def test_target_database_is_never_the_configured_one() -> None:
    assert _pg_url().database != _configured_url().database


def test_target_database_name_carries_the_test_suffix() -> None:
    assert (_pg_url().database or "").endswith(_TEST_DB_SUFFIX)


def test_server_and_credentials_still_come_from_the_real_config() -> None:
    """Only the database *name* is overridden, so any reachable server works."""
    configured, target = _configured_url(), _pg_url()
    assert (target.host, target.port, target.username) == (
        configured.host,
        configured.port,
        configured.username,
    )


@pytest.mark.pg
def test_pg_config_fixture_points_at_the_test_database(pg_config: Config) -> None:
    """Anything opening its own connection from this config (e.g. Searcher)
    must land in the test database, not the user's."""
    assert pg_config.database.url.database == _pg_url().database
