"""Test fixtures and configuration."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest

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


