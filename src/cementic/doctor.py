"""Read-only runtime diagnostics for cementic."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Sequence
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
    EMBED_PROBE_SECONDS,
    AmbiguousDaemonPidsError,
    DaemonHealth,
    build_llama_cpp_client,
    describe_daemon_health,
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


def _daemon_state(config: Config) -> tuple[bool, bool, str]:
    """Return (healthy, autostart_repairable, message) for the embedding daemon.

    Shares one probe with `cementic status`, rather than the bare reachability
    check this used to do. That check asked only whether *something* answered on
    the port, so any llama.cpp server -- serving any model -- reported as ok,
    and the two commands could describe the same daemon differently.
    """
    try:
        health = probe_daemon(
            build_llama_cpp_client(config),
            config,
            wait_seconds=0.0,
            # Same second stage as `cementic status`: the model list is served
            # without the model lock, so it cannot see a dead embedding path.
            embed_probe_seconds=EMBED_PROBE_SECONDS,
        )
    except AmbiguousDaemonPidsError as error:
        # The diagnostic tool must report the pathological state, not die of
        # it with a traceback. Not repairable by autostart either: recovery
        # refuses to guess between the candidate processes.
        return False, False, str(error)
    if health in (DaemonHealth.HEALTHY, DaemonHealth.BUSY):
        # BUSY: serializing every request behind one model lock means a daemon
        # busy indexing cannot answer; that is not the same as broken, and
        # calling it broken made `cementic doctor` exit non-zero mid-build.
        return True, True, describe_daemon_health(health)
    if health is DaemonHealth.WRONG_MODEL:
        # Autostart does repair a mismatch: the stale daemon is stopped and
        # the configured model spawned on the next embedding use.
        return False, True, describe_daemon_health(health)
    if health is DaemonHealth.WEDGED:
        # Autostart cannot repair a wedge: the daemon still answers /v1/models
        # with the expected fingerprint, so the resolver returns it as-is.
        return False, False, describe_daemon_health(health)
    return False, True, ""


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
        """Say which of the three states this is, rather than always "loaded".

        The error branch is unreachable through the CLI (get_config refuses a
        broken file before this runs, and the CLI reports that refusal as its
        own failing report); it stays for library callers holding a Config
        built while the file is broken.
        """
        if error is not None:
            # Not "ignored and defaults in use": since config load became a
            # hard error, commands refuse to run on a broken file.
            return f"{error}; commands will refuse to run until this is fixed"
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
    active_extractor_profiles: list[tuple[str, dict[str, Any]]] | None = None
    try:
        engine = get_engine(config.database.url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            database_ok = True
            checks["database"] = {"status": "ok", "reachable": True}
            # A failure past this point is about inspecting extensions, not
            # about the database: it used to fall into the handler below, which
            # reported an unreachable database (plus the init-postgres hint) for
            # a server that had just answered SELECT 1.
            try:
                extensions = {
                    name: _extension_check(conn, name) for name in REQUIRED_DB_EXTENSIONS
                }
                extension_ok = all(
                    item["status"] in {"ok", "warning"} for item in extensions.values()
                )
                checks["extensions"] = extensions
                active_extractor_profiles = _active_extractor_profiles(conn)
            except Exception as extension_error:
                checks["extensions"] = {
                    "(inspection)": {
                        "status": "fail",
                        "message": f"could not inspect extensions: {extension_error}",
                    }
                }
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
    # `cementic doctor` said ok and `cementic start` then died on it. The
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

    daemon_healthy, daemon_repairable, daemon_message = _daemon_state(config)
    daemon_autostart = config.llama_cpp.daemon_autostart
    # Autostart only excuses states it can actually repair. Excusing every
    # unhealthy state let a wedged daemon pass `doctor` at ok -- the exact
    # state behind the 21-hour incident the two-stage probe exists to catch.
    daemon_ok = daemon_healthy or (daemon_autostart and daemon_repairable)
    checks["daemon"] = {
        "status": "ok" if daemon_healthy else "warning" if daemon_ok else "fail",
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

    checks["chunk_budget"] = _chunk_budget_check(config)
    checks["embedding_server"] = _embedding_server_check(config)
    server_ok = checks["embedding_server"]["status"] != "fail"
    checks["ocr"] = _ocr_check(config)
    ocr_ok = checks["ocr"]["status"] != "fail"
    checks["extraction_commands"] = _extraction_commands_check(config)
    commands_ok = checks["extraction_commands"]["status"] != "fail"
    checks["extractor_profile"] = _extractor_profile_check(config, active_extractor_profiles)

    ok = (
        (config_error is None)
        and database_ok
        and extension_ok
        and model_ok
        and daemon_ok
        and server_ok
        and ocr_ok
        and commands_ok
    )
    return {"ok": ok, "checks": checks}


#: Long enough for a cold binary to load its libraries and print a version,
#: short enough that `cementic doctor` never appears to hang on a wedged one.
_VERSION_PROBE_TIMEOUT_SECONDS = 10.0


def _embedding_server_check(config: Config) -> dict[str, Any]:
    """Report whether the configured server command names something runnable.

    cementic drives llama.cpp over HTTP and ships no server of its own, so the
    binary is a system prerequisite like Postgres. Checked here because the
    alternative is discovering it at first autostart, inside a background
    worker whose only voice is a log file.

    Only argv[0] is inspected, and only when it is a bare name: a command
    fronted by env(1) or naming a build directly carries setup this check
    cannot reproduce -- an LD_LIBRARY_PATH the operator supplies precisely
    because the binary needs it -- and running it without that would report a
    working setup as broken.

    Being on PATH is not the question, though; being runnable is. A stale
    `llama-server` whose shared library moved resolves fine and then dies on
    exec, which `which` alone reports as ready and autostart discovers later in
    a worker log. So the resolved binary is asked for its version.
    """
    command = config.llama_cpp.daemon_command
    executable = command[0]
    if executable == "env" or os.sep in executable:
        return {
            "status": "ok",
            "command": executable,
            "message": f"daemon_command starts '{executable}'; not resolved further",
        }
    resolved = shutil.which(executable)
    if resolved is None:
        return {
            "status": "fail",
            "command": executable,
            "message": (
                f"'{executable}' is not on PATH, so the embedding server cannot "
                "start. Install llama.cpp (its `llama-server` binary), or point "
                "llama_cpp.daemon_command at a build you already have"
            ),
        }
    problem = _executable_problem(resolved)
    if problem is not None:
        return {
            "status": "fail",
            "command": resolved,
            "message": (
                f"'{resolved}' is on PATH but does not run: {problem}. It would "
                "resolve at autostart and then fail in the daemon log"
            ),
        }
    return {
        "status": "ok",
        "command": resolved,
        "message": f"embedding server found at {resolved}",
    }


def _executable_problem(path: str) -> str | None:
    """Ask a binary for its version; return why it could not answer, or None.

    `--version` because it is the cheapest subcommand that still loads the
    program's shared libraries, which is the failure being looked for. Nothing
    is started or written, so this stays inside `collect_doctor_report`'s
    read-only contract.
    """
    try:
        completed = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            timeout=_VERSION_PROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return f"no answer within {_VERSION_PROBE_TIMEOUT_SECONDS:g}s"
    except OSError as error:
        return str(error)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()
        return detail[-1] if detail else f"exited {completed.returncode}"
    return None


def _extraction_commands_check(config: Config) -> dict[str, Any]:
    """Report each command backend's tool and the version it reports.

    The version string is printed rather than merely checked because the one
    thing that cannot be validated automatically is a *wrong* version flag: a
    command that exits 0 while printing an error records a constant that never
    moves on upgrade, silently disabling the fingerprint guarantee it exists to
    provide. A human reading "I/O Error: Couldn't open file '--version'" here
    sees the problem immediately; no check can.

    Imported here rather than at module scope: `extract` loads the PDF stack.
    """
    from cementic.extract import COMMAND_EXTRACTOR_NAME, command_version

    selected = sorted(
        file_type
        for file_type, backend in config.extraction.backends.items()
        if backend == COMMAND_EXTRACTOR_NAME
    )
    if not selected:
        return {
            "status": "ok",
            "commands": {},
            "message": "no file type uses the command extractor",
        }

    reported: dict[str, str] = {}
    failures: list[str] = []
    for file_type in selected:
        try:
            reported[file_type] = command_version(config.extraction.command_versions[file_type])
        except RuntimeError as error:
            failures.append(f"{file_type}: {error}")
    if failures:
        return {
            "status": "fail",
            "commands": reported,
            "message": (
                "an extraction tool could not report its version, so revisions "
                "cannot record which build produced their text -- "
                + "; ".join(failures)
            ),
        }
    return {
        "status": "ok",
        "commands": reported,
        "message": "; ".join(
            f"{file_type}: {version.splitlines()[0]}" for file_type, version in reported.items()
        ),
    }


def _ocr_check(config: Config) -> dict[str, Any]:
    """Report whether OCR is wanted, reachable, and installed.

    Three states rather than a bare present/absent, because two different
    settings have to agree before OCR runs and each fails quietly on its own.
    A missing backend is a hard failure: with OCR reaching extraction, every
    PDF raises rather than degrading, so reporting it as a warning would call
    an unusable configuration ready.

    Imported here rather than at module scope: `extract` loads the PDF stack.
    """
    from cementic.extract import OCR_INSTALL_HINT, ocr_backend_available, ocr_would_run

    if not config.extraction.use_ocr:
        return {
            "status": "ok",
            "enabled": False,
            "message": "extraction.use_ocr is off; no OCR backend needed",
        }
    if not ocr_would_run(config):
        return {
            "status": "warning",
            "enabled": True,
            "message": (
                "extraction.use_ocr is on, but "
                f"[extraction.backends] pdf = '{config.extraction.backends.get('pdf')}' "
                "reads the text layer and never runs OCR; scanned PDFs will "
                "still extract empty. Set it to 'pymupdf4llm' to use OCR"
            ),
        }
    if not ocr_backend_available():
        return {
            "status": "fail",
            "enabled": True,
            "message": (
                "extraction.use_ocr is on and the pdf backend runs OCR, but no "
                f"OCR backend is installed; every PDF will fail. Either {OCR_INSTALL_HINT}, "
                "or set extraction.use_ocr = false"
            ),
        }
    return {
        "status": "ok",
        "enabled": True,
        "message": "OCR enabled and an OCR backend is installed",
    }


def _describe_profile_drift(recorded: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """Name every payload key whose value moved between two profiles.

    Nested mappings (``extraction_libraries``) are compared entry by entry
    rather than whole, so a pymupdf bump reads as "pymupdf 1.27.1 -> 1.28.2"
    instead of printing both mappings and leaving the reader to diff them.
    """
    descriptions: list[str] = []
    for key in sorted(set(recorded) | set(current)):
        was, now = recorded.get(key), current.get(key)
        if was == now:
            continue
        if isinstance(was, dict) and isinstance(now, dict):
            entries = ", ".join(
                f"{name} {was.get(name)} -> {now.get(name)}"
                for name in sorted(set(was) | set(now))
                if was.get(name) != now.get(name)
            )
            descriptions.append(f"{key}: {entries}")
        else:
            descriptions.append(f"{key}: {was} -> {now}")
    return descriptions


def _extractor_drift_check(
    current_payload: dict[str, Any],
    active_profiles: Sequence[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Report collections this install would rebuild rather than extend.

    The extractor profile is the only one of the three whose inputs come from
    the environment rather than from config or a content digest: it records the
    installed pymupdf/pymupdf4llm/pymupdf-layout versions, so an install that
    resolved different ones opens a new extractor profile and re-extracts the
    whole collection. Exact pins in pyproject stop that within one cementic
    version; this catches what they cannot -- two installs of *different*
    versions on one machine, which is the normal state once a checkout and a
    released snapshot both exist.

    A warning, not a failure: a rebuild is the correct response to a real
    extractor change, and nothing here can know whether this one was intended.
    """
    drifted = {
        collection: _describe_profile_drift(recorded, current_payload)
        for collection, recorded in active_profiles
    }
    drifted = {collection: items for collection, items in drifted.items() if items}
    if not drifted:
        return {
            "status": "ok",
            "drift": {},
            "message": (
                "this install extracts as every active revision did"
                if active_profiles
                else "no active revision to compare against"
            ),
        }
    return {
        "status": "warning",
        "drift": drifted,
        "message": "; ".join(
            f"indexing {collection} from this install would rebuild it ({', '.join(items)})"
            for collection, items in sorted(drifted.items())
        ),
    }


def _extractor_profile_check(
    config: Config, active_profiles: Sequence[tuple[str, dict[str, Any]]] | None
) -> dict[str, Any]:
    """The drift check, plus the two states in which it cannot run.

    Building this install's payload can fail: a command extractor whose version
    probe cannot run raises from here, and doctor is the last place that may
    answer a fault with a traceback. `extraction_commands` reports that fault
    as the failure it is; this reports only that it could not compare.
    """
    if active_profiles is None:
        return {
            "status": "ok",
            "drift": {},
            "message": "database unreachable; extractor profiles not compared",
        }
    # Imported here rather than at module scope for the reason `_ocr_check`
    # gives: building the payload loads the PDF stack.
    from cementic.profiles import build_extractor_profile_payload

    try:
        current_payload = build_extractor_profile_payload(config)
    except Exception as error:
        return {
            "status": "warning",
            "drift": {},
            "message": f"could not build this install's extractor profile to compare: {error}",
        }
    return _extractor_drift_check(current_payload, active_profiles)


def _active_extractor_profiles(conn: Any) -> list[tuple[str, dict[str, Any]]]:
    """(collection, extractor payload) for every collection's active revision."""
    rows = conn.execute(
        text(
            "SELECT r.collection, p.config_json FROM pipeline_revisions r "
            "JOIN extractor_profiles p ON p.id = r.extractor_profile_id "
            "WHERE r.status = 'active' ORDER BY r.collection"
        )
    ).fetchall()
    return [(str(collection), json.loads(payload)) for collection, payload in rows]


def _chunk_budget_check(config: Config) -> dict[str, Any]:
    """Report the chunk_size / n_ctx pairing's effect on the embed fast path.

    Config load refuses only pairings that must truncate. The wider band, where
    a full chunk *might* exceed the window, is a performance matter -- every
    chunk pays an exact tokenize round trip -- so it is reported here rather
    than blocking every command.
    """
    from cementic.embedding_runtime import (
        _TASK_PREFIX_TOKEN_ALLOWANCE,
        _TOKEN_RATIO_UPPER_BOUND,
    )

    chunk_size = config.pipeline.chunk_size
    n_ctx = config.llama_cpp.n_ctx
    worst_case = (chunk_size + _TASK_PREFIX_TOKEN_ALLOWANCE) * _TOKEN_RATIO_UPPER_BOUND
    fast_path = worst_case <= n_ctx
    largest_fast = int(n_ctx / _TOKEN_RATIO_UPPER_BOUND) - _TASK_PREFIX_TOKEN_ALLOWANCE
    return {
        "status": "ok" if fast_path else "warning",
        "chunk_size": chunk_size,
        "n_ctx": n_ctx,
        "message": (
            f"chunk_size {chunk_size} clears the {n_ctx}-token window without a "
            "per-chunk tokenize round trip"
            if fast_path
            else f"chunk_size {chunk_size} is within the {n_ctx}-token window but "
            f"above the cheap-check bound ({largest_fast}), so every chunk pays an "
            "exact tokenize round trip while embedding; nothing is truncated"
        ),
    }
