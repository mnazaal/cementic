"""docs/operations.md documents the JSON shapes; these tests keep it honest."""

import json
import re
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cementic.config import Config
from cementic.doctor import collect_doctor_report
from cementic.render import build_status_document
from cementic.status_service import PipelineStatus, WorkerStatus

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


def test_status_json_matches_the_documented_shape() -> None:
    session = MagicMock()

    @contextmanager
    def open_session():
        yield session

    pipeline = PipelineStatus(**{name: 0 for name in PipelineStatus.__dataclass_fields__})
    with (
        patch("cementic.render.list_collections", return_value=[SimpleNamespace(name="NAME")]),
        patch("cementic.render.load_pipeline_status_bulk", return_value={"NAME": pipeline}),
    ):
        document = build_status_document(
            open_session,
            lambda _error: False,
            "",
            Config(),
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
            None,
            False,
        )

    assert _keys(document) == _keys(_documented("status"))


@patch("cementic.doctor._daemon_state", return_value=(True, True, "reachable"))
@patch("cementic.doctor.get_engine", side_effect=Exception("database down"))
def test_doctor_json_names_the_documented_checks(_engine, _daemon) -> None:
    report = collect_doctor_report(Config())
    documented = _documented("doctor")

    assert set(report) == set(documented)
    assert list(report["checks"]) == list(documented["checks"])
