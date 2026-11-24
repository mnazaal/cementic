"""Credential masking for database URLs and driver errors (PLAN.md C4).

Only synthetic sentinels appear here. The query-string classes are the ones
SQLAlchemy's ``hide_password=True`` leaves in place: ``?password=`` and
``?sslpassword=``.
"""

import io
import json
import logging
from unittest.mock import patch

import pytest
from sqlalchemy.engine import make_url
from typer.testing import CliRunner

from cementic.bootstrap import Bootstrapper
from cementic.cli import app as cli_app
from cementic.config import Config, DatabaseConfig
from cementic.config_diagnostics import format_config_error
from cementic.doctor import collect_doctor_report
from cementic.redact import RedactingFormatter, redact_text, redact_url

QS_PW = "SENTINEL_QS_PW"
QS_SSLPW = "SENTINEL_QS_SSLPW"
AUTH_PW = "SENTINEL_AUTH_PW"
SENTINELS = (QS_PW, QS_SSLPW, AUTH_PW)
DSN = (
    f"postgresql+psycopg://u:{AUTH_PW}@h/d?password={QS_PW}&sslpassword={QS_SSLPW}&sslmode=require"
)


def _config() -> Config:
    return Config(database=DatabaseConfig(CEMENTIC_DB_URL=DSN))


def _clean(text: str) -> None:
    for sentinel in SENTINELS:
        assert sentinel not in text


class TestRedactUrl:
    def test_masks_authority_and_credential_query_values(self) -> None:
        rendered = redact_url(make_url(DSN))
        _clean(rendered)
        assert "sslmode=require" in rendered
        assert "u:***@h/d" in rendered

    def test_accepts_a_string(self) -> None:
        _clean(redact_url(DSN))

    def test_url_without_query_is_unchanged_apart_from_the_password(self) -> None:
        assert (
            redact_url("postgresql+psycopg://u:pw@h:5432/d")
            == "postgresql+psycopg://u:***@h:5432/d"
        )

    def test_repeated_query_key_masks_every_value(self) -> None:
        _clean(redact_url(f"postgresql://h/d?password={QS_PW}&password={QS_SSLPW}"))


class TestRedactText:
    def test_masks_known_values_even_when_bare(self) -> None:
        text = f"auth failed with {QS_PW} and {AUTH_PW} and {QS_SSLPW}"
        _clean(redact_text(text, DSN))

    def test_masks_dsn_shapes_without_a_url(self) -> None:
        text = (
            f"bad: postgresql://u:{AUTH_PW}@h/d?password={QS_PW}&sslpassword={QS_SSLPW}; "
            f"host=h password={QS_PW} sslpassword='{QS_SSLPW}'"
        )
        _clean(redact_text(text))

    def test_leaves_unrelated_text_alone(self) -> None:
        assert redact_text("connection refused on port 5432") == "connection refused on port 5432"

    def test_a_value_shorter_than_three_characters_is_not_replaced_globally(self) -> None:
        # Replacing every "a" in the text would wreck the message.
        assert redact_text("a banana", "postgresql://u:a@h/d") == "a banana"


class TestRedactTextShapes:
    @pytest.mark.parametrize(
        "text",
        [
            "password = SENT_X1",
            "host=h password =SENT_X1 user=u",
            "{'password': 'SENT_X1', 'user': 'u'}",
            '{"password": "SENT_X1"}',
            '{"sslpassword":"SENT_X1"}',
            "host=h password=ab&SENT_X1 user=u",
            "password='ab\\'SENT_X1'",
            'password="ab\\"SENT_X1"',
            "postgresql://u:ab@SENT_X1@h/d",
            "postgresql://u:ab/SENT_X1@h/d",
            "failed (postgresql://u:SENT_X1@h/d)",
            "x?password=SENT_X1)",
            "{'api_key': 'SENT_X1'}",
            "client_secret=SENT_X1",
            "access_token=SENT_X1",
        ],
    )
    def test_shape_is_masked(self, text: str) -> None:
        assert "SENT_X1" not in redact_text(text)

    @pytest.mark.parametrize(
        "text",
        [
            # a value directly followed by a closing quote
            '{"dsn": "host=h password=SENT_X1"}',
            "{'url': 'postgresql://h/d?password=SENT_X1'}",
            "connection failed: 'host=h password=SENT_X1'",
            "input_value='postgresql://h/d?password=SENT_X1', input_type=str",
            "['host=h password=SENT_X1']",
            'OperationalError("host=h password=SENT_X1")',
            # key preceded by - . /
            "cmd: psql --password=SENT_X1 -h h",
            "datasource.password=SENT_X1",
            "db-password=SENT_X1",
            "/etc/app/password=SENT_X1",
            # quote inside an unquoted value
            "password=ab'SENT_X1",
            'password=ab"SENT_X1',
            "api-key=SENT_X1",
        ],
    )
    def test_reviewer_probe_shape_is_masked(self, text: str) -> None:
        assert "SENT_X1" not in redact_text(text)

    def test_closing_quote_is_kept(self) -> None:
        assert redact_text("'host=h password=SENT_X1'") == "'host=h password=***'"

    @pytest.mark.parametrize(
        "text",
        [
            "Enter password:\nnext_value here",
            "password:\nnext_value",
            "password=\nnext_value",
            "fe_sendauth: no password supplied",
            "llama-server --port 8080 --api-key-file /x",
        ],
    )
    def test_separator_does_not_cross_a_line(self, text: str) -> None:
        assert redact_text(text) == text

    def test_prose_value_masks_one_token_on_the_same_line(self) -> None:
        assert redact_text("secret: not set\nsecond line") == "secret: *** set\nsecond line"

    def test_port_then_path_is_masked_like_a_password_with_a_slash(self) -> None:
        # Structurally identical to ``u:ab/c@h``; masking is the safe side.
        assert "8080" not in redact_text("see http://host:8080/a@b for details")

    def test_ampersand_in_value_keeps_following_param(self) -> None:
        out = redact_text("?password=a&b&sslmode=require")
        assert "sslmode=require" in out
        assert "a&b" not in out

    def test_url_with_space_in_password_masks_every_encoding(self) -> None:
        url = "postgresql://u:p@h/d?password=hunter%20SENT_X1"
        for rendered in ("hunter SENT_X1", "hunter+SENT_X1", "hunter%20SENT_X1"):
            assert "SENT_X1" not in redact_text(f"bad {rendered} here", url)

    @pytest.mark.parametrize(
        "text",
        [
            "max_token=128",
            "num_token=5",
            "PWD=/home/u",
            "cwd /x/token=1/y",
            "token=5 tokens_per_s=3",
            "passfile=/home/u/.pgpass sslkey=/k",
        ],
    )
    def test_non_credential_keys_are_left_alone(self, text: str) -> None:
        assert redact_text(text) == text

    def test_non_credential_query_keys_survive_redact_url(self) -> None:
        out = redact_url("postgresql://h/d?max_token=128&pwd=/x&password=SENT_X1")
        assert "max_token=128" in out
        assert "SENT_X1" not in out


def test_formatter_masks_message_and_traceback() -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(RedactingFormatter("%(message)s"))
    log = logging.getLogger("test_redact_formatter")
    log.propagate = False
    log.addHandler(handler)
    try:
        try:
            raise RuntimeError(f"cannot connect: {DSN}")
        except RuntimeError:
            log.error("failed with password=%s", QS_PW, exc_info=True)
    finally:
        log.removeHandler(handler)
    _clean(stream.getvalue())
    assert "cannot connect" in stream.getvalue()


class TestSites:
    def test_bootstrap_message(self) -> None:
        bootstrapper = Bootstrapper(_config())
        cause = RuntimeError(f"connection failed for {DSN}")
        with patch.object(bootstrapper, "_database_error", return_value=cause):
            with pytest.raises(RuntimeError) as excinfo:
                bootstrapper._ensure_postgres_ready()
        _clean(str(excinfo.value))

    def test_doctor_json(self) -> None:
        err = RuntimeError(f"could not connect: {DSN} password={QS_PW}")
        with patch("cementic.doctor.get_engine", side_effect=err):
            report = collect_doctor_report(_config())
        _clean(json.dumps(report, default=str))

    def test_runner_bootstrap_failure(self) -> None:
        from cementic.runner import app as runner_app

        with (
            patch("cementic.runner.get_config", return_value=_config()),
            patch("cementic.runner.Bootstrapper") as boot,
            patch("cementic.runner.SourceWatcher"),
        ):
            boot.return_value.ensure_for_convert.side_effect = RuntimeError(f"x {DSN}")
            result = CliRunner().invoke(runner_app, ["source-watcher", "/tmp/x"])
        assert result.exit_code == 1
        _clean(result.stderr + result.stdout)

    def test_report_db_error(self) -> None:
        from cementic.cli_shared import report_db_error

        err = io.StringIO()
        with patch("cementic.cli_shared.err_console") as console:
            console.print.side_effect = lambda *a, **k: err.write(" ".join(map(str, a)))
            report_db_error(RuntimeError(f"boom {DSN}"), "search")
        _clean(err.getvalue())

    def test_invalid_url_error_does_not_echo_url_fragments(self) -> None:
        # SQLAlchemy's int() failure on the port quotes the offending text.
        bad = f"postgresql://u:{AUTH_PW}@h:{QS_PW}/d?password={QS_SSLPW}"
        with pytest.raises(Exception) as excinfo:
            DatabaseConfig(CEMENTIC_DB_URL=bad)
        _clean(str(excinfo.value.__cause__) + format_config_error(excinfo.value, None))

    def test_worker_log_file_is_clean(self, tmp_path) -> None:
        from cementic.worker_runtime import setup_worker_logger

        log_file = tmp_path / "w.log"
        log = setup_worker_logger("test_redact_worker", log_file, "worker")
        try:
            raise RuntimeError(f"dsn {DSN}")
        except RuntimeError:
            log.exception("Pipeline step failed")
        for handler in log.handlers:
            handler.flush()
        _clean(log_file.read_text())
        assert "Pipeline step failed" in log_file.read_text()

    def test_status_json_error(self) -> None:
        with (
            patch("cementic.cli.cli_config", return_value=_config()),
            patch("cementic.cli.db_session", side_effect=RuntimeError(f"boom {DSN}")),
        ):
            result = CliRunner().invoke(cli_app, ["status", "--json"])
        _clean(result.stdout + result.stderr)


def test_runner_entry_redacts_the_last_resort_log_handler() -> None:
    from cementic import runner

    logging.lastResort.setFormatter(logging.Formatter())
    with patch.object(runner, "app"):
        runner.main()
    assert isinstance(logging.lastResort.formatter, RedactingFormatter)


class _State:
    def __init__(self) -> None:
        self.reasons: list[str] = []

    def record_skipped(self, path: str, reason: str, **kwargs: object) -> None:
        self.reasons.append(reason)


class TestSiblingErrorSites:
    def test_source_watcher_state_errors_are_redacted(self) -> None:
        from cementic.source_watcher import SourceWatcher

        w = SourceWatcher.__new__(SourceWatcher)
        w._logger = logging.getLogger("test_sw")
        state = _State()
        w.state_manager = state  # type: ignore[assignment]
        w._record_error = lambda e: None  # type: ignore[method-assign]

        def boom(*args: object) -> None:
            raise RuntimeError(f"db {DSN}")

        w._mark_document_deleted = boom  # type: ignore[method-assign]
        w._mark_document_moved = boom  # type: ignore[method-assign]
        w._mark_documents_deleted_under = boom  # type: ignore[method-assign]
        w._on_file_deleted("/f")
        w._on_file_moved("/a", "/b")
        w._on_directory_deleted("/d")
        assert len(state.reasons) == 3
        for reason in state.reasons:
            _clean(reason)

    def test_runner_watcher_failure_line_is_redacted(self) -> None:
        from cementic.runner import app as runner_app

        with (
            patch("cementic.runner.get_config", return_value=_config()),
            patch("cementic.runner.Bootstrapper"),
            patch("cementic.runner.SourceWatcher") as sw,
        ):
            sw.return_value.start.side_effect = RuntimeError(f"x {DSN}")
            result = CliRunner().invoke(runner_app, ["source-watcher", "/tmp/x"])
        assert result.exit_code == 1
        _clean(result.stderr + result.stdout)

    def test_runner_pipeline_failure_line_is_redacted(self) -> None:
        from cementic.runner import app as runner_app

        with (
            patch("cementic.runner.get_config", return_value=_config()),
            patch("cementic.runner.Bootstrapper"),
            patch("cementic.runner.PipelineWorker") as pw,
        ):
            pw.return_value.start.side_effect = RuntimeError(f"x {DSN}")
            result = CliRunner().invoke(runner_app, ["pipeline-worker"])
        assert result.exit_code == 1
        _clean(result.stderr + result.stdout)


def test_cli_redacts_the_last_resort_log_handler() -> None:
    CliRunner().invoke(cli_app, ["doctor", "--help"])
    assert isinstance(logging.lastResort.formatter, RedactingFormatter)
