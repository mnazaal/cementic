"""docs/operations.md documents the JSON shapes; these tests keep it honest."""

import json
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cementic.config import Config
from cementic.doctor import collect_doctor_report
from cementic.embedding_runtime import (
    AmbiguousDaemonPidsError,
    llama_daemon_state,
    llama_daemon_status,
)
from cementic.render import build_status_document, collection_record, revision_record
from cementic.status_service import FileProgress, PipelineStatus, StatusData, WorkerStatus

OPERATIONS = Path(__file__).resolve().parents[2] / "docs" / "operations.md"


def _documented(name: str) -> dict:
    match = re.search(
        rf"<!-- json-shape: {name} -->\s*```json\n(.*?)```", OPERATIONS.read_text(), re.S
    )
    assert match, f"no json-shape block named {name!r} in {OPERATIONS}"
    return json.loads(match.group(1))


def _keys(value: object) -> object:
    """The key structure of a JSON value, with leaves erased."""
    if isinstance(value, dict):
        return {key: _keys(item) for key, item in value.items()}
    return None


def _worker() -> WorkerStatus:
    return WorkerStatus(
        state="running",
        pid="1",
        process="running",
        current_file=None,
        watched_directories=[],
        processed_count=0,
        failed_count=0,
    )


def _build(data: StatusData | None, error: str | None = None, **kwargs) -> dict:
    return build_status_document(
        SimpleNamespace(state="running", collection="NAME"),
        _worker(),
        _worker(),
        [],
        SimpleNamespace(
            db_reachable=True,
            embedding_provider="llama-cpp",
            embedding_healthy=True,
            llama_daemon="running",
        ),
        None,
        kwargs.get("collection"),
        data,
        error,
        kwargs.get("file_limit"),
    )


def _pipeline() -> PipelineStatus:
    return PipelineStatus(**{name: 0 for name in PipelineStatus.__dataclass_fields__})


def _file(path: str) -> FileProgress:
    return FileProgress(path, "done", "done", 1, 0, 1)


def test_status_json_matches_the_documented_shape() -> None:
    document = _build(StatusData(collections={"NAME": _pipeline()}))

    assert _keys(document) == _keys(_documented("status"))


def test_status_document_for_one_collection_lists_files_and_flags_truncation() -> None:
    data = StatusData(pipeline=_pipeline(), files=[_file("a"), _file("b")])

    capped = _build(data, collection="NAME", file_limit=2)
    uncapped = _build(data, collection="NAME", file_limit=None)

    assert capped["pipeline"]["collection"] == "NAME"
    assert [f["source_path"] for f in capped["files"]] == ["a", "b"]
    assert capped["files_truncated"] is True
    assert uncapped["files_truncated"] is False
    assert "collections" not in capped


def test_status_document_without_verbose_has_no_files() -> None:
    document = _build(StatusData(pipeline=_pipeline()), collection="NAME")

    assert "files" not in document
    assert "files_truncated" not in document


def test_status_document_records_a_load_failure_as_the_error() -> None:
    document = _build(None, "unknown collection: X (check `cementic collection list`)")

    assert document["error"] == "unknown collection: X (check `cementic collection list`)"
    assert "collections" not in document
    assert "pipeline" not in document


@patch("cementic.doctor._daemon_state", return_value=(True, True, "reachable"))
@patch("cementic.doctor.get_engine", side_effect=Exception("database down"))
def test_doctor_json_names_the_documented_checks(_engine, _daemon) -> None:
    report = collect_doctor_report(Config())
    documented = _documented("doctor")

    assert set(report) == set(documented)
    assert list(report["checks"]) == list(documented["checks"])


def test_collection_list_record_matches_the_documented_shape() -> None:
    row = SimpleNamespace(
        name="papers",
        documents=3,
        active_revision_label="r1",
        ready_revision_label=None,
        building_revision_label="r2",
    )

    record = collection_record(row)

    assert _keys(record) == _keys(_documented("collection_list"))
    assert record["ready_revision_label"] is None
    assert record["building_revision_label"] == "r2"


def test_collection_revisions_record_matches_the_documented_shape() -> None:
    row = SimpleNamespace(
        id=3,
        status="active",
        label=None,
        extractor_profile=SimpleNamespace(name="pymupdf4llm", fingerprint="e" * 64),
        chunk_profile=SimpleNamespace(fingerprint="c" * 64),
        embedding_profile=SimpleNamespace(provider="llama-cpp", fingerprint="m" * 64),
        created_at=datetime(2026, 1, 2, 3, 4, 5),
        promoted_at=None,
    )

    record = revision_record(row)

    assert _keys(record) == _keys(_documented("collection_revisions"))
    assert record["chunk_profile"]["fingerprint"] == "c" * 64
    assert record["created_at"] == "2026-01-02T03:04:05"
    assert record["promoted_at"] is None
    json.dumps(record)


def test_embedding_status_state_matches_the_documented_shape_and_the_string() -> None:
    cases = [
        ((None, False), "stopped", "stopped"),
        ((7, False), "running", "running, pid=7"),
        ((7, True), "running", "running, pid=7 (recovered: pid file was missing or stale)"),
    ]
    for source, state, text in cases:
        with patch("cementic.embedding_runtime._live_daemon_pid_with_source", return_value=source):
            daemon = llama_daemon_state(Config())
            assert llama_daemon_status(Config()) == text
        assert _keys(daemon) == _keys(_documented("embedding_status"))
        assert daemon["state"] == state

    with patch(
        "cementic.embedding_runtime._live_daemon_pid_with_source",
        side_effect=AmbiguousDaemonPidsError([2, 1]),
    ):
        assert llama_daemon_state(Config())["state"] == "ambiguous"
        assert llama_daemon_status(Config()).startswith("ambiguous: found multiple processes")
