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


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


