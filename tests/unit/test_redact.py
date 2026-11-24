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

    def test_malformed_toml_message_is_clean(self, tmp_path, monkeypatch) -> None:
        from cementic.config_diagnostics import config_file_error

        path = tmp_path / "config.toml"
        path.write_text(f'[database]\nCEMENTIC_DB_URL = "{DSN}\n')
        monkeypatch.setenv("CEMENTIC_CONFIG", str(path))
        message = config_file_error() or ""
        _clean(message)

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


def test_cli_redacts_the_last_resort_log_handler() -> None:
    CliRunner().invoke(cli_app, ["doctor", "--help"])
    assert isinstance(logging.lastResort.formatter, RedactingFormatter)
