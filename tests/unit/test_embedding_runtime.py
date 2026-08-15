"""Tests for persistent llama.cpp runtime helpers."""

import json
import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from cementic.config import Config, resolve_llama_model_path
from cementic.embedding_provider import EmbeddingProvider
from cementic.embedding_runtime import (
    DaemonHealth,
    EmbeddingRuntimeSpec,
    RemoteEmbeddingClient,
    _read_daemon_pid_file,
    _start_llama_cpp_daemon,
    _stop_mismatched_llama_cpp_daemon,
    _wait_for_daemon_ready,
    create_provider,
    get_llama_cpp_runtime_client,
    llama_cpp_runtime_fingerprint,
    llama_daemon_status,
    probe_daemon,
    runtime_spec_from_config,
    runtime_spec_from_profile_json,
    stop_llama_cpp_runtime,
)


class TestRemoteEmbeddingClient:
    """Tests for RemoteEmbeddingClient."""

    def test_embedding_dim_property(self) -> None:
        client = RemoteEmbeddingClient(
            host="localhost", port=8081, embedding_dim=384, expected_fingerprint="abc"
        )
        assert client.embedding_dim == 384

    def test_base_url(self) -> None:
        client = RemoteEmbeddingClient(
            host="localhost", port=8081, embedding_dim=384, expected_fingerprint="abc"
        )
        assert client.base_url == "http://localhost:8081"

    @patch("cementic.embedding_runtime.requests.get")
    def test_health_check_matches_served_model(self, mock_get) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": [{"id": "abc"}]}
        mock_get.return_value = mock_resp

        client = RemoteEmbeddingClient(
            host="localhost", port=8081, embedding_dim=384, expected_fingerprint="abc"
        )
        assert client.health_check() is True

    @patch("cementic.embedding_runtime.requests.get")
    def test_health_check_mismatch(self, mock_get) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": [{"id": "wrong"}]}
        mock_get.return_value = mock_resp

        client = RemoteEmbeddingClient(
            host="localhost", port=8081, embedding_dim=384, expected_fingerprint="abc"
        )
        assert client.health_check() is False

    @patch("cementic.embedding_runtime.requests.get", side_effect=requests.ConnectionError)
    def test_health_check_connection_error(self, mock_get) -> None:
        client = RemoteEmbeddingClient(
            host="localhost", port=8081, embedding_dim=384, expected_fingerprint="abc"
        )
        assert client.health_check() is False

    @patch("cementic.embedding_runtime.requests.post")
    def test_embed_single(self, mock_post) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]}
        mock_post.return_value = mock_resp

        client = RemoteEmbeddingClient(
            host="localhost", port=8081, embedding_dim=3, expected_fingerprint="abc"
        )
        result = client.embed("hello")
        assert result == [0.1, 0.2, 0.3]

    @patch("cementic.embedding_runtime.requests.post")
    def test_embed_batch_preserves_input_order(self, mock_post) -> None:
        # Server may return rows out of order; they must be sorted by index.
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                {"index": 1, "embedding": [3.0, 4.0]},
                {"index": 0, "embedding": [1.0, 2.0]},
            ]
        }
        mock_post.return_value = mock_resp

        client = RemoteEmbeddingClient(
            host="localhost", port=8081, embedding_dim=2, expected_fingerprint="abc"
        )
        result = client.embed_batch(["a", "b"])
        assert result == [[1.0, 2.0], [3.0, 4.0]]

    @patch("cementic.embedding_runtime.requests.post")
    def test_embed_batch_count_mismatch_raises(self, mock_post) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": [{"index": 0, "embedding": [1.0, 2.0]}]}
        mock_post.return_value = mock_resp

        client = RemoteEmbeddingClient(
            host="localhost", port=8081, embedding_dim=2, expected_fingerprint="abc"
        )
        with pytest.raises(ValueError, match="does not match input count"):
            client.embed_batch(["a", "b"])

    def test_embed_batch_empty(self) -> None:
        client = RemoteEmbeddingClient(
            host="localhost", port=8081, embedding_dim=2, expected_fingerprint="abc"
        )
        assert client.embed_batch([]) == []

    @patch("cementic.embedding_runtime.requests.post")
    def test_describe_probes_live_dim_once(self, mock_post) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": [{"index": 0, "embedding": [0.0, 0.0, 0.0]}]}
        mock_post.return_value = mock_resp

        client = RemoteEmbeddingClient(
            host="localhost", port=8081, embedding_dim=768, expected_fingerprint="abc"
        )
        facts = client.describe()
        assert facts.embedding_dim == 3  # probed, not the configured 768
        assert facts.name == "llama-cpp"
        # Cached: a second describe() does not probe again.
        client.describe()
        assert mock_post.call_count == 1

    @patch("cementic.embedding_runtime._PROBE_RETRY_DELAY_SECONDS", 0)
    @patch("cementic.embedding_runtime.requests.post", side_effect=requests.ConnectionError)
    def test_describe_raises_rather_than_guessing_the_dim(self, mock_post) -> None:
        """A failed probe must not mint a profile from the configured fallback.

        The dimension is written into an immutable embedding profile, so a
        guess forks the profile and revision and silently re-embeds the whole
        collection under a new vector table.
        """
        client = RemoteEmbeddingClient(
            host="localhost", port=8081, embedding_dim=768, expected_fingerprint="abc"
        )

        with pytest.raises(RuntimeError, match="Could not determine the embedding dimension"):
            client.describe()
        assert mock_post.call_count == 3  # retried before giving up

    @patch("cementic.embedding_runtime._PROBE_RETRY_DELAY_SECONDS", 0)
    @patch("cementic.embedding_runtime.requests.post")
    def test_describe_retries_a_transient_probe_failure(self, mock_post) -> None:
        ok = MagicMock()
        ok.json.return_value = {"data": [{"index": 0, "embedding": [0.0] * 5}]}
        mock_post.side_effect = [requests.ConnectionError(), ok]

        client = RemoteEmbeddingClient(
            host="localhost", port=8081, embedding_dim=768, expected_fingerprint="abc"
        )

        assert client.describe().embedding_dim == 5


class TestTokenBudgetGuard:
    """The window is counted with the model's own tokenizer, not chunk.TOKENIZER.

    The server truncates over-long input silently, so an unguarded over-budget
    chunk embeds "successfully" into a vector representing only its head.
    """

    @staticmethod
    def _client(n_ctx: int = 512) -> RemoteEmbeddingClient:
        return RemoteEmbeddingClient(
            host="localhost",
            port=8081,
            embedding_dim=768,
            expected_fingerprint="abc",
            n_ctx=n_ctx,
        )

    def test_short_text_never_pays_for_a_round_trip(self) -> None:
        """The cheap local pre-filter must settle the common case on its own."""
        with patch("cementic.embedding_runtime.requests.post") as mock_post:
            assert self._client().over_budget_tokens("a short chunk of text") is None
            mock_post.assert_not_called()

    @patch("cementic.embedding_runtime.requests.post")
    def test_long_text_is_measured_with_the_models_own_tokenizer(self, mock_post) -> None:
        counted = MagicMock()
        counted.status_code = 200
        counted.json.return_value = {"count": 640}
        mock_post.return_value = counted

        over = self._client().over_budget_tokens("word " * 2000)

        assert over == 640
        assert mock_post.call_args.args[0].endswith("/extras/tokenize/count")

    @patch("cementic.embedding_runtime.requests.post")
    def test_long_text_that_actually_fits_is_allowed(self, mock_post) -> None:
        """The pre-filter over-estimates on purpose; the exact count overrules it."""
        counted = MagicMock()
        counted.status_code = 200
        counted.json.return_value = {"count": 500}
        mock_post.return_value = counted

        assert self._client().over_budget_tokens("word " * 2000) is None

    @patch("cementic.embedding_runtime.requests.post")
    def test_embed_batch_reports_per_item_failure_not_a_dead_batch(self, mock_post) -> None:
        """One over-long chunk must not cost the whole batch its embeddings."""

        def responses(url, **kwargs):
            response = MagicMock()
            response.status_code = 200
            if url.endswith("/extras/tokenize/count"):
                response.json.return_value = {"count": 900}
                return response
            response.json.return_value = {"data": [{"index": 0, "embedding": [0.5, 0.5]}]}
            return response

        mock_post.side_effect = responses
        client = self._client()

        vectors = client.embed_batch(["short one", "word " * 2000])

        assert vectors[0] == [0.5, 0.5]
        assert vectors[1] is None
        assert "over its 512-token context window" in (
            client.over_budget_reason("word " * 2000) or ""
        )

    @patch("cementic.embedding_runtime.requests.post")
    def test_embed_refuses_rather_than_returning_a_truncated_vector(self, mock_post) -> None:
        counted = MagicMock()
        counted.status_code = 200
        counted.json.return_value = {"count": 900}
        mock_post.return_value = counted

        with pytest.raises(ValueError, match="over its 512-token context window"):
            self._client().embed("word " * 2000)

    @patch("cementic.embedding_runtime.requests.post")
    def test_a_server_without_the_endpoint_falls_back_to_the_estimate(self, mock_post) -> None:
        """A missing tokenizer endpoint must not silently re-admit truncation."""
        missing = MagicMock()
        missing.status_code = 404
        mock_post.return_value = missing

        assert self._client().over_budget_tokens("word " * 2000) is not None

    def test_the_guard_is_off_when_the_window_is_unknown(self) -> None:
        with patch("cementic.embedding_runtime.requests.post") as mock_post:
            assert self._client(n_ctx=0).over_budget_tokens("word " * 2000) is None
            mock_post.assert_not_called()


class TestDaemonLifecycle:
    """Tests for daemon lifecycle helpers."""

    def test_restart_no_pid_file(self, temp_dir) -> None:
        config = Config()
        config.llama_cpp.daemon_pid_file = temp_dir / "nonexistent.pid"
        # Should not raise
        _stop_mismatched_llama_cpp_daemon(config)

    def test_restart_invalid_pid_content(self, temp_dir) -> None:
        pid_file = temp_dir / "daemon.pid"
        pid_file.write_text("not-a-pid")
        config = Config()
        config.llama_cpp.daemon_pid_file = pid_file
        _stop_mismatched_llama_cpp_daemon(config)
        # Should unlink the invalid pid file
        assert not pid_file.exists()

    @patch("cementic.embedding_runtime.is_managed_process_alive", return_value=False)
    def test_restart_stale_pid(self, mock_running, temp_dir) -> None:
        pid_file = temp_dir / "daemon.pid"
        pid_file.write_text("99999")
        config = Config()
        config.llama_cpp.daemon_pid_file = pid_file
        _stop_mismatched_llama_cpp_daemon(config)
        assert not pid_file.exists()

    @patch("cementic.embedding_runtime.wait_for_exit", return_value=[])
    @patch("cementic.embedding_runtime.os.kill")
    @patch("cementic.embedding_runtime.is_managed_process_alive", return_value=True)
    def test_restart_live_pid_sigterm_clean_exit(
        self, mock_running, mock_kill, mock_wait, temp_dir
    ) -> None:
        """Live PID → SIGTERM → all exited → pid file unlinked."""
        pid_file = temp_dir / "daemon.pid"
        pid_file.write_text("12345")
        config = Config()
        config.llama_cpp.daemon_pid_file = pid_file
        _stop_mismatched_llama_cpp_daemon(config)
        mock_kill.assert_any_call(12345, 15)  # signal.SIGTERM
        assert not pid_file.exists()

    # Survives SIGTERM, then exits under SIGKILL: the wait reports it alive the
    # first time and gone the second.
    @patch("cementic.embedding_runtime.wait_for_exit", side_effect=[[12345], []])
    @patch("cementic.embedding_runtime.os.kill")
    @patch("cementic.embedding_runtime.is_managed_process_alive", return_value=True)
    def test_restart_live_pid_sigterm_then_sigkill(
        self, mock_running, mock_kill, mock_wait, temp_dir
    ) -> None:
        """Live PID → SIGTERM → still there → SIGKILL → gone → pid file unlinked."""
        pid_file = temp_dir / "daemon.pid"
        pid_file.write_text("12345")
        config = Config()
        config.llama_cpp.daemon_pid_file = pid_file
        _stop_mismatched_llama_cpp_daemon(config)
        mock_kill.assert_any_call(12345, 15)  # signal.SIGTERM
        mock_kill.assert_any_call(12345, 9)   # signal.SIGKILL
        assert not pid_file.exists()

    @patch("cementic.embedding_runtime.wait_for_exit", return_value=[12345])
    @patch("cementic.embedding_runtime.os.kill")
    @patch("cementic.embedding_runtime.is_managed_process_alive", return_value=True)
    def test_a_daemon_that_survives_sigkill_is_reported_not_stopped(
        self, mock_running, mock_kill, mock_wait, temp_dir
    ) -> None:
        """Reporting success here dropped the pid file too, stranding a live
        daemon on the port with nothing able to find it again."""
        pid_file = temp_dir / "daemon.pid"
        pid_file.write_text("12345")
        config = Config()
        config.llama_cpp.daemon_pid_file = pid_file

        with pytest.raises(RuntimeError, match="did not exit"):
            stop_llama_cpp_runtime(config)

        assert pid_file.exists()

    @patch("cementic.embedding_runtime.os.kill", side_effect=PermissionError)
    @patch("cementic.embedding_runtime.is_managed_process_alive", return_value=True)
    def test_a_daemon_owned_by_another_user_is_not_treated_as_gone(
        self, mock_running, mock_kill, temp_dir
    ) -> None:
        """"Cannot signal" was conflated with "already exited", so the pid file
        was deleted and the stop reported as a no-op."""
        pid_file = temp_dir / "daemon.pid"
        pid_file.write_text("12345")
        config = Config()
        config.llama_cpp.daemon_pid_file = pid_file

        with pytest.raises(RuntimeError, match="cannot be signalled"):
            stop_llama_cpp_runtime(config)

        assert pid_file.exists()

    def test_start_daemon_missing_log_file_raises(self) -> None:
        config = Config()
        config.llama_cpp.daemon_log_file = None
        with pytest.raises(RuntimeError, match="not configured"):
            _start_llama_cpp_daemon(config)

    def test_start_daemon_missing_pid_file_raises(self) -> None:
        config = Config()
        config.llama_cpp.daemon_log_file = "/tmp/test.log"
        config.llama_cpp.daemon_pid_file = None
        with pytest.raises(RuntimeError, match="not configured"):
            _start_llama_cpp_daemon(config)

    @patch("cementic.embedding_runtime.spawn_detached", return_value=4321)
    def test_start_daemon_runs_llama_cpp_server(self, mock_spawn, temp_dir) -> None:
        config = Config()
        config.llama_cpp.daemon_log_file = temp_dir / "daemon.log"
        config.llama_cpp.daemon_pid_file = temp_dir / "daemon.pid"
        config.llama_cpp.verbose = False

        _start_llama_cpp_daemon(config)

        command = mock_spawn.call_args[0][0]
        assert "llama_cpp.server" in command
        assert command[command.index("--embedding") + 1] == "true"
        assert command[command.index("--verbose") + 1] == "false"
        # --model is resolved to the physical path the daemon will open, while
        # the model_alias (fingerprint) still derives from the logical identifier.
        assert command[command.index("--model") + 1] == str(
            resolve_llama_model_path(config.llama_cpp.model_path)
        )
        # llama_cpp.server does not write its own PID file; we record it as a
        # {pid, start_token} JSON record for recycled-PID-safe teardown.
        record = json.loads((temp_dir / "daemon.pid").read_text())
        assert record["pid"] == 4321
        assert "start_token" in record

    @patch("cementic.embedding_runtime.spawn_detached", return_value=4321)
    def test_start_daemon_verbose_true(self, mock_spawn, temp_dir) -> None:
        config = Config()
        config.llama_cpp.daemon_log_file = temp_dir / "daemon.log"
        config.llama_cpp.daemon_pid_file = temp_dir / "daemon.pid"
        config.llama_cpp.verbose = True

        _start_llama_cpp_daemon(config)

        command = mock_spawn.call_args[0][0]
        assert command[command.index("--verbose") + 1] == "true"


class TestAutostartChecksTheModelFirst:
    """Autostart must verify the model before spawning a server around it.

    Regression: `search` and `embed` reach the autostart path without ever
    running the bootstrapper, so a missing or mistyped model path spawned
    `llama_cpp.server --model /does/not/exist`, the child died instantly, and
    the user got a generic startup failure instead of "model not found at ...".
    """

    def test_missing_model_reported_before_spawning(self, temp_dir) -> None:
        config = Config()
        config.llama_cpp.model_path = str(temp_dir / "absent.gguf")
        config.bootstrap.auto_download_llama_model = False
        config.llama_cpp.daemon_pid_file = temp_dir / "d.pid"
        config.llama_cpp.daemon_log_file = temp_dir / "d.log"

        with (
            patch("cementic.embedding_runtime.RemoteEmbeddingClient._list_models",
                  return_value=None),
            patch("cementic.embedding_runtime._daemon_pid_alive", return_value=False),
            patch("cementic.embedding_runtime._start_llama_cpp_daemon") as spawn,
        ):
            with pytest.raises(RuntimeError, match="model not found"):
                get_llama_cpp_runtime_client(config=config, autostart=True)

        spawn.assert_not_called()


class TestWaitForDaemon:
    """Tests for _wait_for_daemon_ready."""

    def test_daemon_immediately_ready(self) -> None:
        client = MagicMock(spec=RemoteEmbeddingClient)
        client.matches_expected_runtime.return_value = True
        # Should not raise
        _wait_for_daemon_ready(client, timeout_seconds=5, config=Config())

    def test_daemon_timeout_raises(self) -> None:
        client = MagicMock(spec=RemoteEmbeddingClient)
        client.matches_expected_runtime.return_value = False
        with pytest.raises(RuntimeError, match="did not become ready"):
            _wait_for_daemon_ready(client, timeout_seconds=0.01, config=Config())

    def test_dead_daemon_fails_immediately_instead_of_waiting(self, temp_dir) -> None:
        """A child that exits must be noticed, not waited out.

        Regression: corrupt model, missing file, port in use and OOM all produced
        the same "did not become ready in time" sentence after the full timeout,
        with no hint that a log existed.
        """
        config = Config()
        config.llama_cpp.daemon_log_file = temp_dir / "daemon.log"
        client = MagicMock(spec=RemoteEmbeddingClient)
        client.matches_expected_runtime.return_value = False

        started = time.monotonic()
        with patch("cementic.embedding_runtime.is_managed_process_alive", return_value=False):
            with pytest.raises(RuntimeError, match="exited during startup"):
                _wait_for_daemon_ready(
                    client, timeout_seconds=30, config=config, pid=4242, start_token="t"
                )
        assert time.monotonic() - started < 5  # did not wait out the 30s budget

    def test_failure_message_carries_the_log_path_and_tail(self, temp_dir) -> None:
        config = Config()
        log_file = temp_dir / "daemon.log"
        log_file.write_text(
            "loading model...\nerror loading model: unable to open GGUF\n", encoding="utf-8"
        )
        config.llama_cpp.daemon_log_file = log_file
        client = MagicMock(spec=RemoteEmbeddingClient)
        client.matches_expected_runtime.return_value = False

        with patch("cementic.embedding_runtime.is_managed_process_alive", return_value=False):
            with pytest.raises(RuntimeError) as excinfo:
                _wait_for_daemon_ready(
                    client, timeout_seconds=5, config=config, pid=4242, start_token="t"
                )

        message = str(excinfo.value)
        assert str(log_file) in message
        assert "unable to open GGUF" in message

    def test_missing_log_file_is_not_fatal(self, temp_dir) -> None:
        config = Config()
        config.llama_cpp.daemon_log_file = temp_dir / "absent.log"
        client = MagicMock(spec=RemoteEmbeddingClient)
        client.matches_expected_runtime.return_value = False

        with pytest.raises(RuntimeError, match="did not become ready"):
            _wait_for_daemon_ready(client, timeout_seconds=0.01, config=config)


class TestEmbeddingRuntime:
    """Test llama.cpp daemon client helpers."""

    def test_runtime_fingerprint_is_stable(self) -> None:
        first = llama_cpp_runtime_fingerprint(
            model_path="model.gguf",
            n_ctx=512,
            n_batch=512,
            n_gpu_layers=0,
            verbose=False,
        )
        second = llama_cpp_runtime_fingerprint(
            model_path="model.gguf",
            n_ctx=512,
            n_batch=512,
            n_gpu_layers=0,
            verbose=False,
        )
        third = llama_cpp_runtime_fingerprint(
            model_path="other.gguf",
            n_ctx=512,
            n_batch=512,
            n_gpu_layers=0,
            verbose=False,
        )

        assert first == second
        assert first != third

    def test_get_runtime_client_reuses_matching_daemon(self) -> None:
        config = Config()

        def fake_list_models(self: RemoteEmbeddingClient) -> list[dict]:
            return [{"id": self.expected_fingerprint}]

        with (
            patch(
                "cementic.embedding_runtime.RemoteEmbeddingClient._list_models",
                fake_list_models,
            ),
            patch("cementic.embedding_runtime._stop_mismatched_llama_cpp_daemon") as restart,
            patch("cementic.embedding_runtime._start_llama_cpp_daemon") as start,
            patch("cementic.embedding_runtime._wait_for_daemon_ready") as wait,
        ):
            client = get_llama_cpp_runtime_client(config)

        assert client.embedding_dim == config.llama_cpp.embedding_dim
        restart.assert_not_called()
        start.assert_not_called()
        wait.assert_not_called()

    def test_get_runtime_client_refuses_autostart_when_disabled(self) -> None:
        config = Config()
        config.llama_cpp.daemon_autostart = False

        with (
            patch(
                "cementic.embedding_runtime.RemoteEmbeddingClient._list_models",
                return_value=None,
            ),
            patch("cementic.embedding_runtime._daemon_pid_alive", return_value=False),
            patch("cementic.embedding_runtime._stop_mismatched_llama_cpp_daemon") as restart,
            patch("cementic.embedding_runtime._start_llama_cpp_daemon") as start,
            patch("cementic.embedding_runtime._wait_for_daemon_ready") as wait,
            pytest.raises(RuntimeError, match="cementic embedding start"),
        ):
            get_llama_cpp_runtime_client(config)

        restart.assert_not_called()
        start.assert_not_called()
        wait.assert_not_called()

    def test_get_runtime_client_starts_daemon_when_autostart_enabled(self) -> None:
        config = Config()
        config.llama_cpp.daemon_autostart = True

        with (
            patch(
                "cementic.embedding_runtime.RemoteEmbeddingClient._list_models",
                return_value=None,
            ),
            patch("cementic.embedding_runtime._daemon_pid_alive", return_value=False),
            patch("cementic.embedding_runtime._stop_mismatched_llama_cpp_daemon") as restart,
            patch("cementic.embedding_runtime._start_llama_cpp_daemon") as start,
            patch("cementic.embedding_runtime._wait_for_daemon_ready") as wait,
        ):
            client = get_llama_cpp_runtime_client(config)

        assert client.embedding_dim == config.llama_cpp.embedding_dim
        restart.assert_called_once_with(config)
        _, kwargs = start.call_args
        assert kwargs["spec"].model_identifier == config.llama_cpp.model_path
        wait.assert_called_once()

    def test_get_runtime_client_prints_latency_notice_to_stderr_on_autostart(
        self, capsys
    ) -> None:
        """A cold daemon start (30s+ model load) must not read as a silent hang."""
        config = Config()
        config.llama_cpp.daemon_autostart = True

        with (
            patch(
                "cementic.embedding_runtime.RemoteEmbeddingClient._list_models",
                return_value=None,
            ),
            patch("cementic.embedding_runtime._daemon_pid_alive", return_value=False),
            patch("cementic.embedding_runtime._stop_mismatched_llama_cpp_daemon"),
            patch("cementic.embedding_runtime._start_llama_cpp_daemon"),
            patch("cementic.embedding_runtime._wait_for_daemon_ready"),
        ):
            get_llama_cpp_runtime_client(config)

        captured = capsys.readouterr()
        assert "starting embedding daemon" in captured.err
        assert captured.out == ""

    def test_get_runtime_client_restarts_on_real_mismatch_even_if_daemon_alive(self) -> None:
        """A reachable daemon serving the wrong model is a real config change,
        not busyness -- it must restart even though the PID is alive."""
        config = Config()
        config.llama_cpp.daemon_autostart = True

        with (
            patch(
                "cementic.embedding_runtime.RemoteEmbeddingClient._list_models",
                return_value=[{"id": "some-other-model"}],
            ),
            patch("cementic.embedding_runtime._daemon_pid_alive", return_value=True),
            patch("cementic.embedding_runtime._stop_mismatched_llama_cpp_daemon") as restart,
            patch("cementic.embedding_runtime._start_llama_cpp_daemon") as start,
            patch("cementic.embedding_runtime._wait_for_daemon_ready") as wait,
        ):
            get_llama_cpp_runtime_client(config)

        restart.assert_called_once_with(config)
        start.assert_called_once()
        wait.assert_called_once()

    def test_get_runtime_client_treats_unreachable_alive_daemon_as_busy(self) -> None:
        """/v1/models failing to respond while the PID is confirmed alive means
        busy, not down -- must wait it out rather than kill the process."""
        config = Config()

        with (
            patch(
                "cementic.embedding_runtime.RemoteEmbeddingClient._list_models",
                return_value=None,
            ),
            patch("cementic.embedding_runtime._daemon_pid_alive", return_value=True),
            patch("cementic.embedding_runtime._poll_until_ready", return_value=True) as poll,
            patch("cementic.embedding_runtime._stop_mismatched_llama_cpp_daemon") as restart,
            patch("cementic.embedding_runtime._start_llama_cpp_daemon") as start,
        ):
            client = get_llama_cpp_runtime_client(config)

        assert client.embedding_dim == config.llama_cpp.embedding_dim
        poll.assert_called_once()
        restart.assert_not_called()
        start.assert_not_called()

    def test_get_runtime_client_raises_when_busy_daemon_never_frees(self) -> None:
        config = Config()

        with (
            patch(
                "cementic.embedding_runtime.RemoteEmbeddingClient._list_models",
                return_value=None,
            ),
            patch("cementic.embedding_runtime._daemon_pid_alive", return_value=True),
            patch("cementic.embedding_runtime._poll_until_ready", return_value=False),
            patch("cementic.embedding_runtime._stop_mismatched_llama_cpp_daemon") as restart,
            patch("cementic.embedding_runtime._start_llama_cpp_daemon") as start,
            pytest.raises(RuntimeError, match="busy"),
        ):
            get_llama_cpp_runtime_client(config)

        restart.assert_not_called()
        start.assert_not_called()

    def test_get_runtime_client_uses_profile_spec(self) -> None:
        config = Config()
        spec = EmbeddingRuntimeSpec(
            provider="llama-cpp",
            model_identifier="profile-model.gguf",
            embedding_dim=384,
            n_ctx=1024,
            n_gpu_layers=2,
            verbose=True,
        )

        with (
            patch(
                "cementic.embedding_runtime.RemoteEmbeddingClient._list_models",
                return_value=[{"id": "profile-fingerprint"}],
            ),
            patch("cementic.embedding_runtime.llama_cpp_runtime_fingerprint") as fingerprint,
        ):
            fingerprint.return_value = "profile-fingerprint"
            client = get_llama_cpp_runtime_client(config, spec=spec)

        assert client.embedding_dim == 384
        fingerprint.assert_called_once_with(
            model_path="profile-model.gguf",
            n_ctx=1024,
            n_batch=1024,
            n_gpu_layers=2,
            verbose=True,
        )

    def test_get_runtime_client_uses_embed_timeout_not_start_timeout(self) -> None:
        config = Config()
        config.llama_cpp.daemon_start_timeout_seconds = 30
        config.llama_cpp.llama_embed_timeout_seconds = 200

        def fake_list_models(self: RemoteEmbeddingClient) -> list[dict]:
            return [{"id": self.expected_fingerprint}]

        with patch(
            "cementic.embedding_runtime.RemoteEmbeddingClient._list_models",
            fake_list_models,
        ):
            client = get_llama_cpp_runtime_client(config)

        assert client.timeout == 200.0

    def test_runtime_spec_from_config_for_llama_cpp(self) -> None:
        config = Config()
        config.pipeline.embedding_provider = "llama-cpp"
        config.llama_cpp.model_path = "model.gguf"
        config.llama_cpp.embedding_dim = 384

        spec = runtime_spec_from_config(config)

        assert spec.provider == "llama-cpp"
        assert spec.model_identifier == "model.gguf"
        assert spec.embedding_dim == 384

    def test_runtime_spec_from_profile_json_generic(self) -> None:
        spec = runtime_spec_from_profile_json(
            '{"provider":"llama-cpp","model_identifier":"model.gguf",'
            '"embedding_dim":768,"n_ctx":512,"n_gpu_layers":0,"verbose":false}'
        )

        assert spec.provider == "llama-cpp"
        assert spec.model_identifier == "model.gguf"
        assert spec.embedding_dim == 768
        assert spec.n_ctx == 512


class TestCreateProvider:
    """Tests for the single create_provider resolver."""

    def test_llama_cpp_with_config_uses_runtime_client(self) -> None:
        config = Config()
        spec = runtime_spec_from_profile_json(
            '{"provider": "llama-cpp", "model_identifier": "profile-model.gguf", '
            '"n_ctx": 512, "n_gpu_layers": 0, "embedding_dim": 768, "verbose": false}'
        )
        with patch("cementic.embedding_runtime.get_llama_cpp_runtime_client") as mock_runtime:
            create_provider(spec, config)

        _, kwargs = mock_runtime.call_args
        assert kwargs["config"] is config
        assert kwargs["spec"].model_identifier == "profile-model.gguf"
        assert kwargs["spec"].embedding_dim == 768

    def test_unknown_provider_raises(self) -> None:
        spec = EmbeddingRuntimeSpec(provider="nope", model_identifier="x", embedding_dim=1)
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            create_provider(spec, Config())


class TestProbeDaemon:
    """One probe with an explicit budget, replacing three that disagreed.

    Collapsing "no answer" into False made a daemon serving the *wrong* model
    indistinguishable from one merely mid-batch, so a fallback meant to excuse
    busyness also excused a genuine mismatch.
    """

    def test_serving_our_runtime_is_healthy(self) -> None:
        client = MagicMock(spec=RemoteEmbeddingClient)
        client.probe_served_runtime.return_value = True
        assert probe_daemon(client, Config()) is DaemonHealth.HEALTHY

    def test_answering_with_another_model_is_wrong_model(self) -> None:
        client = MagicMock(spec=RemoteEmbeddingClient)
        client.probe_served_runtime.return_value = False
        assert probe_daemon(client, Config()) is DaemonHealth.WRONG_MODEL

    @patch("cementic.embedding_runtime._daemon_pid_alive", return_value=False)
    def test_no_answer_and_no_process_is_down(self, mock_alive) -> None:
        client = MagicMock(spec=RemoteEmbeddingClient)
        client.probe_served_runtime.return_value = None
        assert probe_daemon(client, Config()) is DaemonHealth.DOWN

    @patch("cementic.embedding_runtime._poll_until_ready")
    @patch("cementic.embedding_runtime._daemon_pid_alive", return_value=True)
    def test_no_answer_with_a_live_process_is_busy_and_does_not_wait(
        self, mock_alive, mock_poll
    ) -> None:
        """The default budget is zero: a status read must not wait out a batch."""
        client = MagicMock(spec=RemoteEmbeddingClient)
        client.probe_served_runtime.return_value = None

        assert probe_daemon(client, Config()) is DaemonHealth.BUSY
        mock_poll.assert_not_called()

    @patch("cementic.embedding_runtime._poll_until_ready", return_value=True)
    @patch("cementic.embedding_runtime._daemon_pid_alive", return_value=True)
    def test_a_caller_that_asks_to_wait_can_recover(self, mock_alive, mock_poll) -> None:
        client = MagicMock(spec=RemoteEmbeddingClient)
        client.probe_served_runtime.return_value = None

        assert probe_daemon(client, Config(), wait_seconds=30) is DaemonHealth.HEALTHY
        mock_poll.assert_called_once()

    @patch("cementic.embedding_runtime._poll_until_ready", return_value=False)
    @patch("cementic.embedding_runtime._daemon_pid_alive", return_value=True)
    def test_waiting_and_never_freeing_stays_busy(self, mock_alive, mock_poll) -> None:
        client = MagicMock(spec=RemoteEmbeddingClient)
        client.probe_served_runtime.return_value = None

        assert probe_daemon(client, Config(), wait_seconds=30) is DaemonHealth.BUSY

    def test_a_non_remote_provider_degrades_to_its_own_check(self) -> None:
        client = MagicMock(spec=EmbeddingProvider)
        client.health_check.return_value = False
        assert probe_daemon(client, Config()) is DaemonHealth.DOWN


class TestDaemonPidFile:
    """The daemon pid-file carries an identity token so a recycled PID is safe."""

    def test_read_legacy_bare_int(self, temp_dir) -> None:
        pid_file = temp_dir / "daemon.pid"
        pid_file.write_text("4321")
        assert _read_daemon_pid_file(pid_file) == (4321, None)

    def test_read_json_record(self, temp_dir) -> None:
        pid_file = temp_dir / "daemon.pid"
        pid_file.write_text(json.dumps({"pid": 4321, "start_token": "tok"}))
        assert _read_daemon_pid_file(pid_file) == (4321, "tok")

    def test_read_invalid_returns_none(self, temp_dir) -> None:
        pid_file = temp_dir / "daemon.pid"
        pid_file.write_text("not-a-pid")
        assert _read_daemon_pid_file(pid_file) is None

    @patch("cementic.embedding_runtime.os.kill")
    @patch("cementic.embedding_runtime.is_managed_process_alive", return_value=False)
    def test_stop_does_not_kill_recycled_pid(
        self, mock_alive, mock_kill, temp_dir
    ) -> None:
        """A token mismatch (recycled PID) must not be signalled."""
        pid_file = temp_dir / "daemon.pid"
        pid_file.write_text(json.dumps({"pid": 4321, "start_token": "stale"}))
        config = Config()
        config.llama_cpp.daemon_pid_file = pid_file

        assert stop_llama_cpp_runtime(config) is False
        mock_kill.assert_not_called()
        assert not pid_file.exists()

    def test_status_stopped_when_no_pid_file(self, temp_dir) -> None:
        config = Config()
        config.llama_cpp.daemon_pid_file = temp_dir / "missing.pid"
        assert llama_daemon_status(config) == "stopped"

    @patch("cementic.embedding_runtime.is_managed_process_alive", return_value=True)
    def test_status_running_reports_pid(self, mock_alive, temp_dir) -> None:
        pid_file = temp_dir / "daemon.pid"
        pid_file.write_text(json.dumps({"pid": 4321, "start_token": "tok"}))
        config = Config()
        config.llama_cpp.daemon_pid_file = pid_file
        status = llama_daemon_status(config)
        assert "running" in status
        assert "4321" in status


def test_runtime_fingerprint_tracks_the_batch_size() -> None:
    """n_batch caps how many tokens the server embeds per input, so it changes
    what the daemon does. It also has to be in the alias so a daemon left
    running by a version that never passed --n_batch is not silently reused
    with its old 512-token cap."""
    small = llama_cpp_runtime_fingerprint(
        model_path="m.gguf", n_ctx=2048, n_batch=512, n_gpu_layers=0, verbose=False
    )
    matched = llama_cpp_runtime_fingerprint(
        model_path="m.gguf", n_ctx=2048, n_batch=2048, n_gpu_layers=0, verbose=False
    )

    assert small != matched
