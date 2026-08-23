"""Tests for persistent llama.cpp runtime helpers."""

import json
import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from cementic.config import Config, resolve_llama_model_path
from cementic.embedding_provider import EmbeddingProvider
from cementic.embedding_runtime import (
    AmbiguousDaemonPidsError,
    DaemonHealth,
    EmbeddingRuntimeSpec,
    RemoteEmbeddingClient,
    _read_daemon_pid_file,
    _start_llama_cpp_daemon,
    _stop_mismatched_llama_cpp_daemon,
    _wait_for_daemon_ready,
    build_llama_cpp_client,
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

    @patch("cementic.embedding_runtime.os.kill")
    @patch("cementic.embedding_runtime._write_daemon_pid_file", side_effect=OSError("read-only"))
    @patch("cementic.embedding_runtime.spawn_detached", return_value=4321)
    def test_start_daemon_kills_the_child_when_the_pid_file_write_fails(
        self, mock_spawn, mock_write, mock_kill, temp_dir
    ) -> None:
        """Regression: an unguarded `_write_daemon_pid_file` after

        `spawn_detached` left the server running with a multi-GB model
        resident and nothing recording its PID -- `embedding status` then
        said "stopped" and `embedding stop` returned False forever.
        """
        config = Config()
        config.llama_cpp.daemon_log_file = temp_dir / "daemon.log"
        config.llama_cpp.daemon_pid_file = temp_dir / "daemon.pid"

        with pytest.raises(OSError, match="read-only"):
            _start_llama_cpp_daemon(config)

        mock_kill.assert_called_once()
        assert mock_kill.call_args.args[0] == 4321


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

    @staticmethod
    def _client(served: list[str] | None, expected: str = "ours") -> MagicMock:
        """A client whose /v1/models probe reports the given model ids (None =
        not answering)."""
        client = MagicMock(spec=RemoteEmbeddingClient)
        client._list_models.return_value = (
            None if served is None else [{"id": model_id} for model_id in served]
        )
        client.expected_fingerprint = expected
        return client

    def test_daemon_immediately_ready(self) -> None:
        # Should not raise
        _wait_for_daemon_ready(self._client(["ours"]), timeout_seconds=5, config=Config())

    def test_daemon_timeout_raises(self) -> None:
        with pytest.raises(RuntimeError, match="did not become ready"):
            _wait_for_daemon_ready(self._client(None), timeout_seconds=0.01, config=Config())

    def test_transient_empty_model_list_is_waited_out_not_failed(self) -> None:
        """An empty list mid-load is not a definitive wrong-model answer."""
        with pytest.raises(RuntimeError, match="did not become ready"):
            _wait_for_daemon_ready(self._client([]), timeout_seconds=0.01, config=Config())

    def test_wrong_model_fails_fast_instead_of_waiting_out_the_timeout(
        self, temp_dir
    ) -> None:
        """Regression: a daemon answering /v1/models with a different model is
        a definitive mismatch (one model per daemon, fixed at launch), yet the
        loop polled it for the full startup timeout -- up to 120s -- and then
        reported the generic "did not become ready"."""
        config = Config()
        config.llama_cpp.daemon_log_file = temp_dir / "daemon.log"

        started = time.monotonic()
        with pytest.raises(RuntimeError, match="serving a different"):
            _wait_for_daemon_ready(
                self._client(["someone-elses-model"]), timeout_seconds=30, config=config
            )
        assert time.monotonic() - started < 5  # did not wait out the 30s budget

    def test_dead_daemon_fails_immediately_instead_of_waiting(self, temp_dir) -> None:
        """A child that exits must be noticed, not waited out.

        Regression: corrupt model, missing file, port in use and OOM all produced
        the same "did not become ready in time" sentence after the full timeout,
        with no hint that a log existed.
        """
        config = Config()
        config.llama_cpp.daemon_log_file = temp_dir / "daemon.log"
        client = self._client(None)

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
        client = self._client(None)

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

        with pytest.raises(RuntimeError, match="did not become ready"):
            _wait_for_daemon_ready(self._client(None), timeout_seconds=0.01, config=config)


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

    def test_a_wedged_daemon_is_detected_by_the_embed_probe(self) -> None:
        """Regression (the 21-hour incident): /v1/models is served without the
        model lock, so a daemon whose embedding path was dead answered listings
        and read as healthy for as long as nobody restarted it."""
        client = MagicMock(spec=RemoteEmbeddingClient)
        client.probe_served_runtime.return_value = True
        client.probe_embedding.return_value = False

        with patch(
            "cementic.embedding_runtime._worker_load_explains_slow_embeddings",
            return_value=False,
        ):
            health = probe_daemon(client, Config(), embed_probe_seconds=5.0)

        assert health is DaemonHealth.WEDGED
        client.probe_embedding.assert_called_once_with(5.0)

    def test_a_worker_batch_explains_a_slow_embed_probe(self) -> None:
        """A live worker mid-batch legitimately holds the model lock for tens
        of seconds; misreporting that as a wedge would alarm on every index."""
        client = MagicMock(spec=RemoteEmbeddingClient)
        client.probe_served_runtime.return_value = True
        client.probe_embedding.return_value = False

        with patch(
            "cementic.embedding_runtime._worker_load_explains_slow_embeddings",
            return_value=True,
        ):
            health = probe_daemon(client, Config(), embed_probe_seconds=5.0)

        assert health is DaemonHealth.BUSY

    def test_an_answering_embed_probe_is_healthy(self) -> None:
        client = MagicMock(spec=RemoteEmbeddingClient)
        client.probe_served_runtime.return_value = True
        client.probe_embedding.return_value = True

        assert probe_daemon(client, Config(), embed_probe_seconds=5.0) is DaemonHealth.HEALTHY

    def test_without_the_embed_budget_the_probe_stays_one_stage(self) -> None:
        """Callers that only need liveness (search's autostart) pay nothing new."""
        client = MagicMock(spec=RemoteEmbeddingClient)
        client.probe_served_runtime.return_value = True

        assert probe_daemon(client, Config()) is DaemonHealth.HEALTHY
        client.probe_embedding.assert_not_called()

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

    @patch("cementic.embedding_runtime.find_pids_by_cmdline", return_value=[])
    @patch("cementic.embedding_runtime.os.kill")
    @patch("cementic.embedding_runtime.is_managed_process_alive", return_value=False)
    def test_stop_does_not_kill_recycled_pid(
        self, mock_alive, mock_kill, mock_scan, temp_dir
    ) -> None:
        """A token mismatch (recycled PID) must not be signalled.

        No recoverable process either (the /proc scan is mocked empty here,
        not the recovery path under test), so this is genuinely stopped.
        """
        pid_file = temp_dir / "daemon.pid"
        pid_file.write_text(json.dumps({"pid": 4321, "start_token": "stale"}))
        config = Config()
        config.llama_cpp.daemon_pid_file = pid_file

        assert stop_llama_cpp_runtime(config) is False
        mock_kill.assert_not_called()
        assert not pid_file.exists()

    @patch("cementic.embedding_runtime.find_pids_by_cmdline", return_value=[])
    def test_status_stopped_when_no_pid_file(self, mock_scan, temp_dir) -> None:
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


class TestDaemonRecoveryFromProc:
    """Batch A: an orphaned daemon -- pid file missing or stale -- can be
    recovered straight from the OS, by matching the exact command line
    `_start_llama_cpp_daemon` spawns it with, and the pid file is repaired.

    Fakes `/proc` under `tmp_path` (never spawns real processes) and always
    goes through the real entry points (`llama_daemon_status`,
    `stop_llama_cpp_runtime`): a test that called the recovery helper
    directly would pass against the pre-fix code too, which is the exact
    mistake Batch C made.
    """

    @staticmethod
    def _write_daemon_entry(proc_root, pid: int, *, port: int, model_alias: str) -> None:
        pid_dir = proc_root / str(pid)
        pid_dir.mkdir(parents=True)
        argv = [
            "/usr/bin/python3",
            "-m",
            "llama_cpp.server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--model",
            "/models/whatever.gguf",
            "--model_alias",
            model_alias,
        ]
        (pid_dir / "cmdline").write_bytes("\0".join(argv).encode("utf-8") + b"\0")

    @staticmethod
    def _config(temp_dir) -> Config:
        config = Config()
        config.llama_cpp.daemon_pid_file = temp_dir / "daemon.pid"
        return config

    def test_missing_pid_file_matching_process_is_recovered_and_pid_file_repaired(
        self, temp_dir, tmp_path
    ) -> None:
        config = self._config(temp_dir)
        fingerprint = build_llama_cpp_client(config).expected_fingerprint
        self._write_daemon_entry(
            tmp_path, 4242, port=config.llama_cpp.daemon_port, model_alias=fingerprint
        )

        assert not config.llama_cpp.daemon_pid_file.exists()
        with patch("cementic.supervisor._PROC_ROOT", tmp_path):
            status = llama_daemon_status(config)

        assert "running, pid=4242" in status
        assert "recovered" in status
        record = json.loads(config.llama_cpp.daemon_pid_file.read_text())
        assert record["pid"] == 4242

    def test_missing_pid_file_process_on_a_different_port_is_not_matched(
        self, temp_dir, tmp_path
    ) -> None:
        config = self._config(temp_dir)
        fingerprint = build_llama_cpp_client(config).expected_fingerprint
        self._write_daemon_entry(
            tmp_path, 4242, port=config.llama_cpp.daemon_port + 1, model_alias=fingerprint
        )

        with patch("cementic.supervisor._PROC_ROOT", tmp_path):
            status = llama_daemon_status(config)

        assert status == "stopped"
        assert not config.llama_cpp.daemon_pid_file.exists()

    def test_missing_pid_file_process_with_a_different_model_alias_is_not_matched(
        self, temp_dir, tmp_path
    ) -> None:
        config = self._config(temp_dir)
        self._write_daemon_entry(
            tmp_path,
            4242,
            port=config.llama_cpp.daemon_port,
            model_alias="some-other-runtimes-fingerprint",
        )

        with patch("cementic.supervisor._PROC_ROOT", tmp_path):
            status = llama_daemon_status(config)

        assert status == "stopped"
        assert not config.llama_cpp.daemon_pid_file.exists()

    def test_two_matching_processes_refuses_and_names_both(self, temp_dir, tmp_path) -> None:
        config = self._config(temp_dir)
        fingerprint = build_llama_cpp_client(config).expected_fingerprint
        self._write_daemon_entry(
            tmp_path, 4242, port=config.llama_cpp.daemon_port, model_alias=fingerprint
        )
        self._write_daemon_entry(
            tmp_path, 4343, port=config.llama_cpp.daemon_port, model_alias=fingerprint
        )

        with patch("cementic.supervisor._PROC_ROOT", tmp_path):
            status = llama_daemon_status(config)

        assert "ambiguous" in status
        assert "4242" in status
        assert "4343" in status
        # Refusing means refusing: never write a pid file naming a guess.
        assert not config.llama_cpp.daemon_pid_file.exists()

    def test_ambiguous_recovery_raises_for_callers_that_decide_whether_to_spawn(
        self, temp_dir, tmp_path
    ) -> None:
        """`_daemon_pid_alive` (used to decide whether to autostart a
        competing daemon) must not silently pick one -- it has to raise."""
        from cementic.embedding_runtime import _daemon_pid_alive

        config = self._config(temp_dir)
        fingerprint = build_llama_cpp_client(config).expected_fingerprint
        self._write_daemon_entry(
            tmp_path, 4242, port=config.llama_cpp.daemon_port, model_alias=fingerprint
        )
        self._write_daemon_entry(
            tmp_path, 4343, port=config.llama_cpp.daemon_port, model_alias=fingerprint
        )

        with (
            patch("cementic.supervisor._PROC_ROOT", tmp_path),
            pytest.raises(AmbiguousDaemonPidsError),
        ):
            _daemon_pid_alive(config)

    def test_no_proc_degrades_to_not_running_never_raises(self, temp_dir) -> None:
        config = self._config(temp_dir)
        with patch("cementic.supervisor._PROC_ROOT", temp_dir / "no-such-proc-here"):
            status = llama_daemon_status(config)
        assert status == "stopped"

    def test_process_owned_by_another_uid_is_not_matched(self, temp_dir, tmp_path) -> None:
        config = self._config(temp_dir)
        fingerprint = build_llama_cpp_client(config).expected_fingerprint
        self._write_daemon_entry(
            tmp_path, 4242, port=config.llama_cpp.daemon_port, model_alias=fingerprint
        )

        with (
            patch("cementic.supervisor._PROC_ROOT", tmp_path),
            patch("cementic.supervisor.os.getuid", return_value=999999),
        ):
            status = llama_daemon_status(config)

        assert status == "stopped"
        assert not config.llama_cpp.daemon_pid_file.exists()

    def test_stop_recovers_a_lost_pid_file_and_actually_stops_the_daemon(
        self, temp_dir, tmp_path
    ) -> None:
        """The Batch A exit criterion: with the pid file deleted by hand,
        `embedding stop` must actually stop the daemon rather than report
        'already stopped'."""
        config = self._config(temp_dir)
        fingerprint = build_llama_cpp_client(config).expected_fingerprint
        self._write_daemon_entry(
            tmp_path, 4242, port=config.llama_cpp.daemon_port, model_alias=fingerprint
        )

        with (
            patch("cementic.supervisor._PROC_ROOT", tmp_path),
            patch("cementic.embedding_runtime.is_managed_process_alive", return_value=True),
            patch("cementic.embedding_runtime.os.kill") as mock_kill,
            patch("cementic.embedding_runtime.wait_for_exit", return_value=[]),
        ):
            stopped = stop_llama_cpp_runtime(config)

        assert stopped is True
        mock_kill.assert_any_call(4242, 15)  # signal.SIGTERM
        assert not config.llama_cpp.daemon_pid_file.exists()


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


class TestDaemonLockIsNotTakenTwice:
    """The restart path holds the daemon lock; the stop it calls must not retake it."""

    def test_stopping_a_mismatched_daemon_under_the_lock_does_not_deadlock(
        self, temp_dir
    ) -> None:
        """Regression: `get_llama_cpp_runtime_client` takes the daemon lock and
        then called the *public* `stop_llama_cpp_runtime`, which takes the same
        lock again. flock is not reentrant even within one process, so every
        runtime-config change waited out the full 180s timeout and then failed
        with "another cementic process holds ..." -- naming itself."""
        from cementic.embedding_runtime import (
            _daemon_lock_path,
            _stop_mismatched_llama_cpp_daemon,
        )
        from cementic.filelock import file_lock

        pid_file = temp_dir / "daemon.pid"
        # A PID that cannot exist, so the stop unlinks the record and returns
        # rather than signalling anything.
        pid_file.write_text(json.dumps({"pid": 4194300, "start_token": "x", "port": 1}))
        config = Config()
        config.llama_cpp.daemon_pid_file = pid_file

        with file_lock(_daemon_lock_path(config), timeout=5.0):
            _stop_mismatched_llama_cpp_daemon(config)

        assert not pid_file.exists()


class TestWorkerLoadCheck:
    """The wedge/busy disambiguation reads the worker's own state file."""

    def _state_file(self, tmp_path, **fields):
        from cementic.state import StateManager, WorkerState

        path = tmp_path / "worker.json"
        manager = StateManager(path)
        manager.save(WorkerState(**fields))
        return path

    def _config_with(self, path):
        config = Config()
        config.pipeline_worker.state_path = path
        return config

    def test_live_worker_with_a_current_file_explains_the_load(self, tmp_path) -> None:
        from cementic.embedding_runtime import _worker_load_explains_slow_embeddings

        path = self._state_file(tmp_path, current_file="/x.pdf", pid=1234, start_token="t")
        with patch(
            "cementic.embedding_runtime.is_managed_process_alive", return_value=True
        ):
            assert _worker_load_explains_slow_embeddings(self._config_with(path)) is True

    def test_a_crashed_workers_leftover_state_does_not_count(self, tmp_path) -> None:
        """The PID is token-checked: state left behind by a dead worker must not
        excuse a wedged daemon."""
        from cementic.embedding_runtime import _worker_load_explains_slow_embeddings

        path = self._state_file(tmp_path, current_file="/x.pdf", pid=1234, start_token="t")
        with patch(
            "cementic.embedding_runtime.is_managed_process_alive", return_value=False
        ):
            assert _worker_load_explains_slow_embeddings(self._config_with(path)) is False

    def test_an_idle_worker_does_not_count(self, tmp_path) -> None:
        from cementic.embedding_runtime import _worker_load_explains_slow_embeddings

        path = self._state_file(tmp_path, pid=1234, start_token="t")
        with patch(
            "cementic.embedding_runtime.is_managed_process_alive", return_value=True
        ):
            assert _worker_load_explains_slow_embeddings(self._config_with(path)) is False

    def test_an_unreadable_state_file_means_no_explanation(self) -> None:
        """Never the reason a probe fails: any error reading state degrades to
        "no explanation", and the caller reports the probe's own result."""
        from cementic.embedding_runtime import _worker_load_explains_slow_embeddings

        config = Config()
        config.pipeline_worker.state_path = None
        assert _worker_load_explains_slow_embeddings(config) is False


class TestProbeEmbedding:
    """The embed probe must respect its budget against a genuinely hung server."""

    def test_a_hung_embeddings_endpoint_fails_the_probe_within_budget(self) -> None:
        import http.server
        import threading
        import time as time_module

        release = threading.Event()

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - http.server API
                release.wait(10)  # hang past any test budget until teardown

            def log_message(self, *args):  # silence
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = RemoteEmbeddingClient(
                host="127.0.0.1",
                port=server.server_address[1],
                embedding_dim=4,
                expected_fingerprint="fp",
            )
            start = time_module.monotonic()
            assert client.probe_embedding(0.3) is False
            assert time_module.monotonic() - start < 2.0
        finally:
            release.set()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


class TestRuntimeFingerprintIsPathSpellingIndependent:
    """One model file must have one alias, however its path was written.

    Indexing builds its runtime spec from config (a relative
    `models/x.gguf`); search builds one from the stored profile, which carries
    the resolved absolute path. Hashing the raw string gave the same file two
    aliases, so each side saw the other's daemon as mismatched and restarted it
    -- the same failure `embedding_dim` is excluded from this fingerprint for.
    """

    def test_relative_and_absolute_spellings_agree(self, tmp_path, monkeypatch) -> None:
        model = tmp_path / "models" / "m.gguf"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"gguf")
        monkeypatch.chdir(tmp_path)

        kwargs = {"n_ctx": 512, "n_batch": 512, "n_gpu_layers": 0, "verbose": False}
        relative = llama_cpp_runtime_fingerprint(model_path="models/m.gguf", **kwargs)
        absolute = llama_cpp_runtime_fingerprint(model_path=str(model), **kwargs)

        assert relative == absolute

    def test_genuinely_different_models_still_differ(self, tmp_path, monkeypatch) -> None:
        for name in ("a.gguf", "b.gguf"):
            (tmp_path / "models").mkdir(exist_ok=True)
            (tmp_path / "models" / name).write_bytes(b"gguf")
        monkeypatch.chdir(tmp_path)

        kwargs = {"n_ctx": 512, "n_batch": 512, "n_gpu_layers": 0, "verbose": False}
        assert llama_cpp_runtime_fingerprint(
            model_path="models/a.gguf", **kwargs
        ) != llama_cpp_runtime_fingerprint(model_path="models/b.gguf", **kwargs)
