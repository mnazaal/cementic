"""Read-only runtime diagnostics for cementic."""

from __future__ import annotations

from typing import Any

from sqlalchemy import text

from cementic.bootstrap import llama_model_download_allowed
from cementic.config import (
    Config,
    config_file_error,
    resolve_config_path,
    resolve_llama_model_path,
)
from cementic.db import REQUIRED_DB_EXTENSIONS, get_engine
from cementic.embedding_runtime import (
    DaemonHealth,
    build_llama_cpp_client,
    probe_daemon,
)


def _status(ok: bool, *, warning: bool = False) -> str:
    if ok:
        return "warning" if warning else "ok"
    return "fail"


def _extension_check(conn: Any, name: str) -> dict[str, Any]:
    installed = bool(
        conn.execute(
            text("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = :name)"),
            {"name": name},
        ).scalar()
    )
    available = bool(
        conn.execute(
            text("SELECT EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = :name)"),
            {"name": name},
        ).scalar()
    )
    warning = available and not installed
    return {
        "status": _status(installed or available, warning=warning),
        "installed": installed,
        "available": available,
        "message": (
            "installed"
            if installed
            else "available but not installed; cementic will try CREATE EXTENSION on normal startup"
            if available
            else "not available on this Postgres server"
        ),
    }


def _daemon_state(config: Config) -> tuple[bool, str]:
    """Return (healthy, message) for the embedding daemon.

    Shares one probe with `cementic status`, rather than the bare reachability
    check this used to do. That check asked only whether *something* answered on
    the port, so any llama.cpp server -- serving any model -- reported as ok,
    and the two commands could describe the same daemon differently.
    """
    health = probe_daemon(build_llama_cpp_client(config), config, wait_seconds=0.0)
    if health is DaemonHealth.HEALTHY:
        return True, "reachable"
    if health is DaemonHealth.BUSY:
        # Serializing every request behind one model lock means a daemon that is
        # busy indexing cannot answer; that is not the same as broken, and
        # calling it broken made `status --doctor` exit non-zero mid-build.
        return True, "running but busy (serving a request); not idle enough to answer /v1/models"
    if health is DaemonHealth.WRONG_MODEL:
        return False, "serving a different model than this config expects"
    return False, ""


def collect_doctor_report(config: Config) -> dict[str, Any]:
    """Collect read-only readiness diagnostics.

    This function must not create schemas/extensions, start daemons, download
    models, write state/config, or perform container actions.
    """
    config_path = resolve_config_path()
    model_path = resolve_llama_model_path(config.llama_cpp.model_path)
    # A malformed or unreadable config file is silently discarded and cementic
    # falls back to defaults. Reporting "ok" here -- while pointing at the very
    # file that is not being used -- was the one check that could never fail.
    config_error = config_file_error()

    def _config_message(path: Any, error: str | None) -> str:
        """Say which of the three states this is, rather than always "loaded"."""
        if error is not None:
            return f"{error}; this file is being ignored and defaults are in use"
        if path is None:
            return "no config file; built-in defaults in use (`cementic config init` writes one)"
        return "loaded"

    checks: dict[str, Any] = {
        "config": {
            "status": "ok" if config_error is None else "fail",
            "path": str(config_path) if config_path is not None else None,
            "database_url": config.database.url.render_as_string(hide_password=True),
            "message": (
                _config_message(config_path, config_error)
            ),
        }
    }

    database_ok = False
    extension_ok = False
    try:
        engine = get_engine(config.database.url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            database_ok = True
            extensions = {name: _extension_check(conn, name) for name in REQUIRED_DB_EXTENSIONS}
            extension_ok = all(item["status"] in {"ok", "warning"} for item in extensions.values())
        checks["database"] = {"status": "ok", "reachable": True}
        checks["extensions"] = extensions
    except Exception as error:
        checks["database"] = {
            "status": "fail",
            "reachable": database_ok,
            "message": (
                f"{error}. Run `cementic init postgres ./cementic-postgres` once and "
                "follow its README, or set CEMENTIC_DB_URL to a Postgres with pgvector "
                "and pgvectorscale."
            ),
        }
        checks["extensions"] = {}

    model_exists = model_path.is_file()
    # Promising a download this path forbids is worse than reporting nothing:
    # `status --doctor` said ok and `cementic start` then died on it. The
    # confinement rule is the bootstrapper's own, shared rather than restated.
    model_downloadable = config.bootstrap.auto_download_llama_model and (
        llama_model_download_allowed(model_path)
    )
    model_auto_download = config.bootstrap.auto_download_llama_model
    model_ok = model_exists or model_downloadable
    checks["model"] = {
        "status": "ok" if model_exists else "warning" if model_downloadable else "fail",
        "path": str(model_path),
        "exists": model_exists,
        "auto_download": model_auto_download,
        "url": config.bootstrap.llama_model_url,
        "sha256_configured": bool(config.bootstrap.llama_model_sha256),
        "message": (
            "present"
            if model_exists
            else "missing; cementic will download it automatically when needed"
            if model_downloadable
            else "missing, and auto-download is refused: the path is outside the "
            "cementic data directory, so cementic cannot write it there"
            if model_auto_download
            else "missing and auto_download_llama_model is disabled"
        ),
    }

    daemon_healthy, daemon_message = _daemon_state(config)
    daemon_autostart = config.llama_cpp.daemon_autostart
    daemon_ok = daemon_healthy or daemon_autostart
    checks["daemon"] = {
        "status": "ok" if daemon_healthy else "warning" if daemon_autostart else "fail",
        "reachable": daemon_healthy,
        "autostart": daemon_autostart,
        "message": (
            daemon_message
            or (
                "not running; cementic can autostart it when needed"
                if daemon_autostart
                else "not running and daemon_autostart is disabled"
            )
        ),
    }

    ok = (config_error is None) and database_ok and extension_ok and model_ok and daemon_ok
    return {"ok": ok, "checks": checks}
