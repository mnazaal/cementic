"""Tests for persistent llama.cpp runtime helpers."""

from unittest.mock import patch

from cementic.config import Config
from cementic.embedding_runtime import get_llama_cpp_runtime_client, llama_cpp_runtime_fingerprint


class TestEmbeddingRuntime:
    """Test llama.cpp daemon client helpers."""

    def test_runtime_fingerprint_is_stable(self):
        first = llama_cpp_runtime_fingerprint(
            model_path="model.gguf",
            n_ctx=512,
            n_gpu_layers=0,
            embedding_dim=768,
            verbose=False,
        )
        second = llama_cpp_runtime_fingerprint(
            model_path="model.gguf",
            n_ctx=512,
            n_gpu_layers=0,
            embedding_dim=768,
            verbose=False,
        )
        third = llama_cpp_runtime_fingerprint(
            model_path="other.gguf",
            n_ctx=512,
            n_gpu_layers=0,
            embedding_dim=768,
            verbose=False,
        )

        assert first == second
        assert first != third

    def test_get_runtime_client_reuses_matching_daemon(self):
        config = Config()

        with (
            patch(
                "cementic.embedding_runtime.LlamaCppDaemonClient.matches_expected_runtime",
                return_value=True,
            ),
            patch("cementic.embedding_runtime._restart_llama_cpp_daemon_if_needed") as restart,
            patch("cementic.embedding_runtime._start_llama_cpp_daemon") as start,
            patch("cementic.embedding_runtime._wait_for_daemon_ready") as wait,
        ):
            client = get_llama_cpp_runtime_client(config)

        assert client.embedding_dim == config.llama_cpp.embedding_dim
        restart.assert_not_called()
        start.assert_not_called()
        wait.assert_not_called()

    def test_get_runtime_client_starts_daemon_when_missing(self):
        config = Config()

        with (
            patch(
                "cementic.embedding_runtime.LlamaCppDaemonClient.matches_expected_runtime",
                side_effect=[False, True],
            ),
            patch("cementic.embedding_runtime._restart_llama_cpp_daemon_if_needed") as restart,
            patch("cementic.embedding_runtime._start_llama_cpp_daemon") as start,
            patch("cementic.embedding_runtime._wait_for_daemon_ready") as wait,
        ):
            client = get_llama_cpp_runtime_client(config)

        assert client.embedding_dim == config.llama_cpp.embedding_dim
        restart.assert_called_once_with(config)
        start.assert_called_once_with(config)
        wait.assert_called_once()
