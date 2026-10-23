"""PDF extraction runs in a child process with a time and memory limit."""

import contextlib
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


def test_a_stop_signal_during_a_parse_does_not_lose_the_document(monkeypatch):
    """Systemd's control-group SIGTERM reaches the worker and the child together."""
    import subprocess
    import sys
    import textwrap

    use_extractor(monkeypatch, "slow")
    script = textwrap.dedent(
        """
        import os, signal, threading, time
        from cementic import extract
        from cementic.config import Config
        from tests.unit import isolation_extractors as ie
        from cementic.extract import ExtractorSpec
        spec = ExtractorSpec(name="pymupdf4llm", version=1, extensions=(".pdf",), isolated=True)
        extract._EXTRACTORS["pymupdf4llm"] = (spec, ie.slow)
        signal.signal(signal.SIGTERM, lambda s, f: None)
        def stop():
            time.sleep(1)
            for pid in [os.getpid()] + [c.pid for c in extract._children.values()]:
                os.kill(pid, signal.SIGTERM)
        threading.Thread(target=stop, daemon=True).start()
        config = Config()
        config.extraction.backends = {}
        print(extract.extract_document("/x/a.pdf", config))
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60,
        cwd=REPO_ROOT, env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )

    assert done.stdout.strip() == "full text", done.stderr


def test_default_timeouts_by_kind():
    timeout = extract_mod.extraction_timeout
    assert Config().extraction.timeout_seconds is None
    assert timeout(None, isolated=True) == 3600.0
    assert timeout(None, isolated=False) == 900.0
    assert timeout(5.0, isolated=True) == timeout(5.0, isolated=False) == 5.0


def test_a_large_reply_is_read_whole(monkeypatch):
    config = use_extractor(monkeypatch, "big")
    config.extraction.memory_limit_mb = 0

    assert len(extract_document("/x/a.pdf", config)) == 20_000_000


def test_a_child_dies_with_a_killed_parent():
    import signal
    import subprocess
    import sys
    import time

    parent_code = (
        "import subprocess, sys, os;"
        "c = subprocess.Popen([sys.executable, '-m', 'cementic.extract_child', '{}',"
        " str(os.getpid())], stdin=sys.stdin);"
        "print(c.pid, flush=True); c.wait()"
    )
    read_end, write_end = os.pipe()
    parent = subprocess.Popen(
        [sys.executable, "-c", parent_code], stdin=read_end, stdout=subprocess.PIPE,
        cwd=REPO_ROOT, env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    os.close(read_end)
    assert parent.stdout is not None
    child_pid = int(parent.stdout.readline())
    parent.stdout.close()
    # The child is parsing (stdin stays open, so EOF cannot be what ends it).
    os.write(write_end, b'{"target": "tests.unit.isolation_extractors:hang", "path": "x"}\n')
    time.sleep(1)
    parent.send_signal(signal.SIGKILL)
    parent.wait()
    try:
        for _ in range(50):
            try:
                alive = "Z" not in Path(f"/proc/{child_pid}/stat").read_text().split(")")[-1][:3]
            except OSError:
                alive = False
            if not alive:
                break
            time.sleep(0.1)
        assert not alive
    finally:
        os.close(write_end)
        with contextlib.suppress(OSError):
            os.kill(child_pid, signal.SIGKILL)
