"""Runtime bootstrap helpers for infra and embedding dependencies."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Callable

import requests
from sqlalchemy import text

from seman.config import Config
from seman.db import get_engine


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

        if self.config.indexing.embedder == "ollama":
            self._ensure_ollama_ready()
            if self.config.bootstrap.auto_pull_ollama_model:
                self._ensure_ollama_model()
        elif self.config.indexing.embedder == "llama-cpp":
            self._ensure_llama_model()

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
                "Set SEMAN_LLAMA_MODEL_PATH to an existing file or enable "
                "SEMAN_BOOTSTRAP_AUTO_DOWNLOAD_LLAMA_MODEL=true."
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
            return response.status_code == 200
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
        if self._container_running(self.config.bootstrap.postgres_container):
            return

        if self._container_exists(self.config.bootstrap.postgres_container):
            self._run(["podman", "start", self.config.bootstrap.postgres_container])
            return

        data_path = self.config.bootstrap.postgres_data_path
        if data_path is None:
            raise RuntimeError("postgres_data_path is not configured")
        data_path.mkdir(parents=True, exist_ok=True)
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
                "-p",
                f"{self.config.database.port}:5432",
                "-v",
                f"{data_path}:/var/lib/postgresql/data",
                self.config.bootstrap.postgres_image,
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
            subprocess.run(command, check=True)
        except FileNotFoundError as e:
            raise RuntimeError("Required command not found: podman") from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Command failed: {' '.join(command)}") from e
