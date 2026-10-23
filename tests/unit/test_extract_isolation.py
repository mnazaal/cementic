"""PDF extraction runs in a child process with a time and memory limit."""

import os
from pathlib import Path

import pytest

from cementic import extract as extract_mod
from cementic.config import Config
from cementic.extract import ExtractorSpec, extract_document, extractor_registry_payload
from cementic.profiles import build_extractor_profile_payload

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures"


def use_extractor(monkeypatch: pytest.MonkeyPatch, fn_name: str) -> Config:
    """Register `tests.unit.isolation_extractors.<fn_name>` as the isolated .pdf extractor."""
    from tests.unit import isolation_extractors

    spec = ExtractorSpec(name="pymupdf4llm", version=1, extensions=(".pdf",), isolated=True)
    monkeypatch.setitem(
        extract_mod._EXTRACTORS, "pymupdf4llm", (spec, getattr(isolation_extractors, fn_name))
    )
    monkeypatch.setenv("PYTHONPATH", str(REPO_ROOT))
    config = Config()
    config.extraction.timeout_seconds = 20
    config.extraction.memory_limit_mb = 512
    return config


@pytest.mark.parametrize(
    ("fn_name", "reason"),
    [
        ("hard_exit", "exited 7"),
        ("segfault", r"crashed \(SIGSEGV\)"),
        ("hog", "out of memory"),
        ("raises", "bad pdf"),
    ],
)
def test_a_dying_child_becomes_an_exception_with_its_reason(monkeypatch, fn_name, reason):
    config = use_extractor(monkeypatch, fn_name)

    with pytest.raises(RuntimeError, match=reason):
        extract_document("/x/a.pdf", config)


def test_a_hung_child_is_killed_at_the_timeout(monkeypatch):
    config = use_extractor(monkeypatch, "hang")
    config.extraction.timeout_seconds = 1

    with pytest.raises(RuntimeError, match="timed out after 1s"):
        extract_document("/x/a.pdf", config)


def test_text_survives_the_pipe_byte_for_byte(monkeypatch):
    """Stdout noise from the parser is kept out of the text, and a lone surrogate
    round-trips, so isolation cannot move the extracted bytes."""
    config = use_extractor(monkeypatch, "chatty")

    assert extract_document("/x/a.pdf", config) == "text \ud800 with a lone surrogate"


def test_a_real_pdf_extracts_identically_in_a_child():
    """pymupdf4llm is left to the 200-PDF measurement: its in-process output
    depends on whether `pymupdf.layout` was already imported by an earlier test."""
    config = Config()
    config.extraction.backends = {"pdf": "pymupdf-raw"}
    path = str(FIXTURES / "test_doc_a.pdf")
    _name, fn = extract_mod.extractor_for(path, config)

    assert extract_document(path, config) == extract_mod.strip_unstorable(fn(path, config))


def test_isolation_does_not_move_the_extractor_fingerprint():
    config = Config()
    payload = build_extractor_profile_payload(config)

    assert all(set(entry) == {"name", "version"} for entry in extractor_registry_payload(config))
    assert "timeout_seconds" not in str(payload) and "memory_limit_mb" not in str(payload)
    config.extraction.timeout_seconds = 5
    config.extraction.memory_limit_mb = 0
    assert build_extractor_profile_payload(config) == payload


def pid_of_child(monkeypatch, config) -> int:
    use_extractor(monkeypatch, "child_pid")
    return int(extract_document("/x/a.pdf", config))


def test_documents_reuse_one_child(monkeypatch):
    config = use_extractor(monkeypatch, "child_pid")

    first = pid_of_child(monkeypatch, config)
    assert pid_of_child(monkeypatch, config) == first
    assert first != os.getpid()


@pytest.mark.parametrize("fn_name", ["hard_exit", "segfault"])
def test_a_death_restarts_the_child_for_the_next_document(monkeypatch, fn_name):
    config = use_extractor(monkeypatch, "child_pid")
    first = pid_of_child(monkeypatch, config)

    use_extractor(monkeypatch, fn_name)
    with pytest.raises(RuntimeError):
        extract_document("/x/a.pdf", config)

    second = pid_of_child(monkeypatch, config)
    assert second != first
    assert pid_of_child(monkeypatch, config) == second


def test_a_timeout_restarts_the_child_for_the_next_document(monkeypatch):
    config = use_extractor(monkeypatch, "child_pid")
    config.extraction.timeout_seconds = 5
    first = pid_of_child(monkeypatch, config)

    use_extractor(monkeypatch, "hang")
    config.extraction.timeout_seconds = 5
    with pytest.raises(RuntimeError, match="timed out"):
        extract_document("/x/a.pdf", config)

    assert pid_of_child(monkeypatch, config) != first


def test_a_parser_exception_keeps_the_child(monkeypatch):
    config = use_extractor(monkeypatch, "child_pid")
    first = pid_of_child(monkeypatch, config)

    use_extractor(monkeypatch, "raises")
    with pytest.raises(RuntimeError, match="bad pdf"):
        extract_document("/x/a.pdf", config)

    assert pid_of_child(monkeypatch, config) == first


def test_the_child_exits_when_its_parent_closes_it(monkeypatch):
    config = use_extractor(monkeypatch, "child_pid")
    pid = pid_of_child(monkeypatch, config)

    extract_mod._close_children()

    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
