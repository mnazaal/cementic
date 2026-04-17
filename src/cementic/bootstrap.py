"""Runtime bootstrap helpers for infra and embedding dependencies."""

from __future__ import annotations

import fcntl
import subprocess
import time

# mypy: disable-error-code=import-untyped
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import requests
from sqlalchemy import text

from cementic.config import Config
from cementic.db import get_engine


class Bootstrapper:
    """Ensures required runtime dependencies are available before daemon start."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def ensure_for_convert(self) -> None:
        """Ensure runtime dependencies for conversion."""
        self._ensure_postgres_ready()

    def ensure_for_index(self) -> None:
        """Ensure runtime dependencies for indexing."""
        self._ensure_postgres_ready()
        self._ensure_embedding_runtime()

    def _ensure_embedding_runtime(self) -> None:
        provider = self.config.pipeline.embedding_provider
        if provider == "ollama":
            self._ensure_ollama_ready()
            if self.config.bootstrap.auto_pull_ollama_model:
                self._ensure_ollama_model()
            return

        if provider == "llama-cpp":
            self._ensure_llama_model()
            return

        raise RuntimeError(f"Unsupported embedding provider: {provider}")

    def _ensure_postgres_ready(self) -> None:
        if self._database_ready():
            return

        if self.config.bootstrap.auto_start_infra and self._is_local_database():
            self._start_postgres_container()

        self._wait_until(
            self._database_ready,
            "PostgreSQL did not become ready in time",
        )

    def _ensure_ollama_ready(self) -> None:
        if self._ollama_ready():
            return

        if self.config.bootstrap.auto_start_infra and self._is_local_ollama():
            self._start_ollama_container()

        self._wait_until(
            self._ollama_ready,
            "Ollama did not become ready in time",
        )

    def _ensure_ollama_model(self) -> None:
        model = self.config.ollama.model
        if self._ollama_has_model(model):
            return

        try:
            response = requests.post(
                f"{self.config.ollama.host.rstrip('/')}/api/pull",
                json={"name": model, "stream": False},
                timeout=max(self.config.bootstrap.wait_timeout_seconds, 120),
            )
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to pull Ollama model '{model}': {e}") from e

    def _ensure_llama_model(self) -> None:
        model_path = Path(self.config.llama_cpp.model_path)
        if model_path.exists():
            return

        if not self.config.bootstrap.auto_download_llama_model:
            raise RuntimeError(
                f"llama.cpp model not found at {model_path}. "
                "Set CEMENTIC_LLAMA_MODEL_PATH to an existing file or enable "
                "CEMENTIC_BOOTSTRAP_AUTO_DOWNLOAD_LLAMA_MODEL=true."
            )

        model_path.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(
            self.config.bootstrap.llama_model_url, stream=True, timeout=60
        ) as response:
            response.raise_for_status()
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

    def _database_ready(self) -> bool:
        try:
            engine = get_engine(self.config.database.url)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def _ollama_ready(self) -> bool:
        try:
            response = requests.get(f"{self.config.ollama.host.rstrip('/')}/api/tags", timeout=5)
            return bool(response.status_code == 200)
        except requests.RequestException:
            return False

    def _ollama_has_model(self, model_name: str) -> bool:
        try:
            response = requests.get(f"{self.config.ollama.host.rstrip('/')}/api/tags", timeout=10)
            response.raise_for_status()
            models = response.json().get("models", [])
            return any(m.get("name", "").split(":")[0] == model_name for m in models)
        except requests.RequestException:
            return False

    def _start_postgres_container(self) -> None:
        data_path = self.config.bootstrap.postgres_data_path
        if data_path is None:
            raise RuntimeError("postgres_data_path is not configured")
        data_path.mkdir(parents=True, exist_ok=True)

        if self._container_running(self.config.bootstrap.postgres_container):
            return

        self._ensure_postgres_image()

        if self._container_exists(self.config.bootstrap.postgres_container):
            if (
                self._container_image(self.config.bootstrap.postgres_container)
                != self.config.bootstrap.postgres_image
            ):
                self._run(["podman", "rm", "-f", self.config.bootstrap.postgres_container])
            else:
                try:
                    self._run(["podman", "start", self.config.bootstrap.postgres_container])
                    return
                except RuntimeError:
                    self._run(["podman", "rm", "-f", self.config.bootstrap.postgres_container])

        self._run(
            [
                "podman",
                "run",
                "-d",
                "--name",
                self.config.bootstrap.postgres_container,
                "-e",
                f"POSTGRES_DB={self.config.database.name}",
                "-e",
                f"POSTGRES_USER={self.config.database.user}",
                "-e",
                f"POSTGRES_PASSWORD={self.config.database.password}",
                "-e",
                "PGDATA=/var/lib/postgresql/data/pgdata",
                "-p",
                f"{self.config.database.port}:5432",
                "-v",
                f"{data_path}:/var/lib/postgresql/data",
                self.config.bootstrap.postgres_image,
            ]
        )

    def _ensure_postgres_image(self) -> None:
        image = self.config.bootstrap.postgres_image
        with self._postgres_image_lock():
            if self._image_exists(image):
                return

            if not self.config.bootstrap.auto_build_postgres_image:
                raise RuntimeError(
                    f"Postgres image '{image}' is not available. "
                    "Enable CEMENTIC_BOOTSTRAP_AUTO_BUILD_POSTGRES_IMAGE=true or build it manually."
                )

            build_context = (
                Path(__file__).resolve().parents[2] / "containers" / "postgres-vectorscale"
            )
            if not build_context.exists():
                raise RuntimeError(
                    f"Vectorscale container build context not found: {build_context}"
                )

            self._run(
                [
                    "podman",
                    "build",
                    "-t",
                    image,
                    "--build-arg",
                    f"POSTGRES_BASE_IMAGE={self.config.bootstrap.postgres_base_image}",
                    "--build-arg",
                    f"PGVECTORSCALE_VERSION={self.config.bootstrap.pgvectorscale_version}",
                    str(build_context),
                ]
            )

    def _start_ollama_container(self) -> None:
        if self._container_running(self.config.bootstrap.ollama_container):
            return

        if self._container_exists(self.config.bootstrap.ollama_container):
            self._run(["podman", "start", self.config.bootstrap.ollama_container])
            return

        data_path = self.config.bootstrap.ollama_data_path
        if data_path is None:
            raise RuntimeError("ollama_data_path is not configured")
        data_path.mkdir(parents=True, exist_ok=True)
        self._run(
            [
                "podman",
                "run",
                "-d",
                "--name",
                self.config.bootstrap.ollama_container,
                "-p",
                "11434:11434",
                "-v",
                f"{data_path}:/root/.ollama",
                self.config.bootstrap.ollama_image,
            ]
        )

    def _container_exists(self, name: str) -> bool:
        result = subprocess.run(["podman", "container", "exists", name], capture_output=True)
        return result.returncode == 0

    def _image_exists(self, name: str) -> bool:
        result = subprocess.run(["podman", "image", "exists", name], capture_output=True)
        return result.returncode == 0

    def _container_image(self, name: str) -> str | None:
        result = subprocess.run(
            ["podman", "inspect", "-f", "{{.ImageName}}", name],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None

    def _container_running(self, name: str) -> bool:
        result = subprocess.run(
            ["podman", "inspect", "-f", "{{.State.Running}}", name],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"

    def _wait_until(self, condition: Callable[[], bool], timeout_error: str) -> None:
        deadline = time.time() + self.config.bootstrap.wait_timeout_seconds
        while time.time() < deadline:
            if condition():
                return
            time.sleep(self.config.bootstrap.wait_interval_seconds)
        raise RuntimeError(timeout_error)

    def _is_local_database(self) -> bool:
        return self.config.database.host in {"localhost", "127.0.0.1"}

    def _is_local_ollama(self) -> bool:
        host = self.config.ollama.host
        return "localhost" in host or "127.0.0.1" in host

    def stop_containers(self, include_ollama: bool = True) -> None:
        """Stop infrastructure containers."""
        containers = [self.config.bootstrap.postgres_container]
        if include_ollama:
            containers.append(self.config.bootstrap.ollama_container)

        for name in containers:
            if self._container_running(name):
                self._run(["podman", "stop", name])
            elif self._container_exists(name):
                pass

    def _run(self, command: list[str]) -> None:
        try:
            subprocess.run(
                command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except FileNotFoundError as e:
            raise RuntimeError("Required command not found: podman") from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Command failed: {' '.join(command)}") from e

    @contextmanager
    def _postgres_image_lock(self) -> Iterator[None]:
        data_path = self.config.bootstrap.postgres_data_path
        if data_path is None:
            raise RuntimeError("postgres_data_path is not configured")
        data_path.mkdir(parents=True, exist_ok=True)
        lock_path = data_path.parent / "postgres-image.lock"
        with open(lock_path, "w", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
