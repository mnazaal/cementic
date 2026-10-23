"""docs/operations.md documents the JSON shapes; these tests keep it honest."""

import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cementic.config import Config
from cementic.doctor import collect_doctor_report
from cementic.render import build_status_document
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
