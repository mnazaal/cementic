"""Tests for bootstrap provider branching and pinned build args."""

from unittest.mock import patch

from cementic.bootstrap import Bootstrapper
from cementic.config import Config


class TestBootstrapper:
    """Test runtime bootstrap behavior."""

    def test_ensure_for_index_uses_llama_cpp_without_ollama(self):
        config = Config()
        config.pipeline.embedding_provider = "llama-cpp"

        bootstrapper = Bootstrapper(config)

        with (
            patch.object(bootstrapper, "_ensure_postgres_ready") as ensure_postgres,
            patch.object(bootstrapper, "_ensure_llama_model") as ensure_llama,
            patch.object(bootstrapper, "_ensure_ollama_ready") as ensure_ollama_ready,
            patch.object(bootstrapper, "_ensure_ollama_model") as ensure_ollama_model,
        ):
            bootstrapper.ensure_for_index()

        ensure_postgres.assert_called_once_with()
        ensure_llama.assert_called_once_with()
        ensure_ollama_ready.assert_not_called()
        ensure_ollama_model.assert_not_called()

    def test_ensure_for_index_uses_ollama_without_llama_cpp(self):
        config = Config()
        config.pipeline.embedding_provider = "ollama"
        config.bootstrap.auto_pull_ollama_model = True

        bootstrapper = Bootstrapper(config)

        with (
            patch.object(bootstrapper, "_ensure_postgres_ready") as ensure_postgres,
            patch.object(bootstrapper, "_ensure_llama_model") as ensure_llama,
            patch.object(bootstrapper, "_ensure_ollama_ready") as ensure_ollama_ready,
            patch.object(bootstrapper, "_ensure_ollama_model") as ensure_ollama_model,
        ):
            bootstrapper.ensure_for_index()

        ensure_postgres.assert_called_once_with()
        ensure_ollama_ready.assert_called_once_with()
        ensure_ollama_model.assert_called_once_with()
        ensure_llama.assert_not_called()

    def test_ensure_for_index_skips_ollama_model_pull_when_disabled(self):
        config = Config()
        config.pipeline.embedding_provider = "ollama"
        config.bootstrap.auto_pull_ollama_model = False

        bootstrapper = Bootstrapper(config)

        with (
            patch.object(bootstrapper, "_ensure_postgres_ready"),
            patch.object(bootstrapper, "_ensure_llama_model") as ensure_llama,
            patch.object(bootstrapper, "_ensure_ollama_ready") as ensure_ollama_ready,
            patch.object(bootstrapper, "_ensure_ollama_model") as ensure_ollama_model,
        ):
            bootstrapper.ensure_for_index()

        ensure_ollama_ready.assert_called_once_with()
        ensure_ollama_model.assert_not_called()
        ensure_llama.assert_not_called()

    def test_ensure_postgres_image_passes_pinned_build_args(self):
        config = Config()
        bootstrapper = Bootstrapper(config)

        with (
            patch.object(bootstrapper, "_postgres_image_lock") as image_lock,
            patch.object(bootstrapper, "_image_exists", return_value=False),
            patch.object(bootstrapper, "_run") as run,
        ):
            image_lock.return_value.__enter__.return_value = None
            image_lock.return_value.__exit__.return_value = False
            bootstrapper._ensure_postgres_image()

        run.assert_called_once()
        command = run.call_args.args[0]
        assert f"POSTGRES_BASE_IMAGE={config.bootstrap.postgres_base_image}" in command
        assert f"PGVECTORSCALE_VERSION={config.bootstrap.pgvectorscale_version}" in command
