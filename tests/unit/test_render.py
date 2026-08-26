"""Tests for the status/doctor rendering helpers in render.py."""

from types import SimpleNamespace

from cementic import render
from cementic.config import Config
from cementic.embedding_runtime import DaemonHealth


def _health(
    *,
    healthy: bool,
    llama_daemon: str,
    llama_daemon_health: DaemonHealth | None = DaemonHealth.DOWN,
) -> SimpleNamespace:
    return SimpleNamespace(
        db_reachable=True,
        embedding_provider="llama-cpp",
        embedding_healthy=healthy,
        llama_daemon=llama_daemon,
        llama_daemon_health=llama_daemon_health,
    )


def _worker() -> SimpleNamespace:
    return SimpleNamespace(
        process="stopped",
        state="stopped",
        pid="N/A",
        current_file=None,
        current_activity=None,
        processed_count=0,
        failed_count=0,
        skipped_files=[],
        last_error=None,
        last_error_at=None,
        watched_directories=[],
    )


def _summary(health: SimpleNamespace, *, autostart: bool = True) -> str:
    config = Config()
    config.llama_cpp.daemon_autostart = autostart
    with render.console.capture() as captured:
        render._print_status_summary("test", [], _worker(), _worker(), health, False, config)
    return captured.get()


class TestEmbeddingRowDistinguishesRunningFromStopped:
    """A daemon that is running but unusable must not be reported as stopped.

    check_health already works out *why* an unhealthy daemon is unhealthy --
    "serving a different model", or wedged with the command that fixes it -- and
    the summary used to discard that and print "stopped (autostarts when
    needed)" for every unhealthy state. That said "stopped" about a live process
    and promised autostart would resolve something autostart cannot.
    """

    def test_a_wrong_model_daemon_is_not_called_stopped(self) -> None:
        output = _summary(
            _health(
                healthy=False,
                llama_daemon="serving a different model than this config expects",
                llama_daemon_health=DaemonHealth.WRONG_MODEL,
            )
        )
        assert "serving a different model" in output
        assert "stopped (autostarts when needed)" not in output

    def test_a_wedged_daemon_keeps_its_remediation(self) -> None:
        wedged = (
            "running but not answering embeddings; restart it with "
            "`cementic embedding stop && cementic embedding start`"
        )
        output = _summary(
            _health(healthy=False, llama_daemon=wedged, llama_daemon_health=DaemonHealth.WEDGED)
        )
        assert "not answering embeddings" in output
        assert "cementic embedding stop" in output

    def test_a_genuinely_stopped_daemon_still_says_autostart(self) -> None:
        """The existing, deliberate message for the not-running case survives."""
        output = _summary(_health(healthy=False, llama_daemon="stopped"))
        assert "stopped (autostarts when needed)" in output

    def test_a_stopped_daemon_without_autostart_is_unhealthy(self) -> None:
        output = _summary(_health(healthy=False, llama_daemon="stopped"), autostart=False)
        assert "unhealthy" in output

    def test_a_healthy_daemon_is_unchanged(self) -> None:
        output = _summary(
            _health(
                healthy=True,
                llama_daemon="running, pid=1234",
                llama_daemon_health=DaemonHealth.HEALTHY,
            )
        )
        assert "healthy" in output
        assert "serving a different model" not in output

    def test_a_failed_probe_is_not_promised_an_autostart(self) -> None:
        """llama_daemon_health None means the probe itself failed (e.g. an
        ambiguous-PID refusal). Autostart hits the same refusal, so the yellow
        autostart line would promise a repair that cannot happen."""
        output = _summary(
            _health(
                healthy=False,
                llama_daemon="unknown (found multiple processes ...)",
                llama_daemon_health=None,
            )
        )
        assert "unknown (found multiple processes" in output
        assert "stopped (autostarts when needed)" not in output
