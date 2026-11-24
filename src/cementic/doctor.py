"""Read-only runtime diagnostics for cementic."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Sequence
from typing import Any

from sqlalchemy import inspect as sa_inspect
from sqlalchemy import text

from cementic.bootstrap import llama_model_download_allowed
from cementic.config import (
    Config,
    resolve_config_path,
    resolve_llama_model_path,
)
from cementic.config_diagnostics import config_file_error
from cementic.db import (
    LEXICAL_INDEX_NAME,
    REQUIRED_DB_EXTENSIONS,
    LexicalIndexFacts,
    get_engine,
    lexical_index_facts,
)
from cementic.embedding_runtime import (
    EMBED_PROBE_SECONDS,
    AmbiguousDaemonPidsError,
    DaemonHealth,
    build_llama_cpp_client,
    describe_daemon_health,
    probe_daemon,
)
from cementic.redact import redact_text, redact_url


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


def daemon_check_status(healthy: bool, repairable: bool, *, autostart: bool) -> str:
    """Doctor's verdict on the embedding daemon: ``ok``, ``warning`` or ``fail``.

    One predicate for `doctor` and `embedding status --check`, so a script and
    the report cannot disagree about whether the runtime is usable. Autostart
    only excuses states it can actually repair: excusing every unhealthy state
    let a wedged daemon pass at ok, the state behind the 21-hour incident the
    two-stage probe exists to catch.
    """
    if healthy:
        return "ok"
    return "warning" if autostart and repairable else "fail"


def _model_check(config: Config) -> dict[str, Any]:
    """Whether the embedding model file is present, or fetchable on demand."""
    model_path = resolve_llama_model_path(config.llama_cpp.model_path)
    model_exists = model_path.is_file()
    # Promising a download this path forbids is worse than reporting nothing:
    # `cementic doctor` said ok and `cementic start` then died on it. The
    # confinement rule is the bootstrapper's own, shared rather than restated.
    model_downloadable = config.bootstrap.auto_download_llama_model and (
        llama_model_download_allowed(model_path)
    )
    model_auto_download = config.bootstrap.auto_download_llama_model
    return {
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


def _daemon_check(config: Config) -> dict[str, Any]:
    """The embedding daemon's check: probe it, then apply the autostart verdict."""
    daemon_healthy, daemon_repairable, daemon_message = _daemon_state(config)
    daemon_autostart = config.llama_cpp.daemon_autostart
    # Autostart only excuses states it can actually repair. Excusing every
    # unhealthy state let a wedged daemon pass `doctor` at ok -- the exact
    # state behind the 21-hour incident the two-stage probe exists to catch.
    return {
        "status": daemon_check_status(
            daemon_healthy, daemon_repairable, autostart=daemon_autostart
        ),
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


def embedding_runtime_checks(config: Config) -> dict[str, dict[str, Any]]:
    """The checks that decide whether embedding can run: model, daemon, server binary.

    One definition for `doctor` and `embedding status --check`, so the script
    gate and the report cannot disagree. Gating on the daemon alone passed a
    stopped daemon with autostart on even when the binary it would autostart
    was missing, a state `doctor` fails.
    """
    return {
        "model": _model_check(config),
        "daemon": _daemon_check(config),
        "embedding_server": _embedding_server_check(config),
    }


def collect_doctor_report(config: Config) -> dict[str, Any]:
    """Collect read-only readiness diagnostics.

    This function must not create schemas/extensions, start daemons, download
    models, write state/config, or perform container actions.
    """
    config_path = resolve_config_path()
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
            "database_url": redact_url(config.database.url),
            "message": (_config_message(config_path, config_error)),
        }
    }

    database_ok = False
    active_extractor_profiles: list[tuple[str, dict[str, Any]]] | None = None
    extractor_profiles_error: str | None = None
    stranded_chunkings: int | None = None
    orphan_vector_tables: list[int] | None = None
    lexical_facts: LexicalIndexFacts | None = None
    #: Set once `lexical_index_facts` returned, so a None result (its query
    #: failed and was swallowed) is told apart from a probe that never ran.
    lexical_probe_ran = False
    active_canaries: list[tuple[str, str, str, str, str | None]] | None = None
    #: The probe in flight once the server has answered, so a raise there can
    #: say which one failed instead of blaming the connection.
    failing_probe = "connecting"
    try:
        engine = get_engine(config.database.url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            database_ok = True
            checks["database"] = {"status": "ok", "reachable": True}
            failing_probe = "the extension inspection"
            # A failure past this point is about inspecting extensions, not
            # about the database: it used to fall into the handler below, which
            # reported an unreachable database (plus the init-postgres hint) for
            # a server that had just answered SELECT 1.
            try:
                extensions = {name: _extension_check(conn, name) for name in REQUIRED_DB_EXTENSIONS}
                checks["extensions"] = extensions
            except Exception as extension_error:
                # A failed statement aborts the PostgreSQL transaction; without
                # this the profile read below reports "transaction is aborted".
                conn.rollback()
                checks["extensions"] = {
                    "(inspection)": {
                        "status": "fail",
                        "message": "could not inspect extensions: "
                        + redact_text(str(extension_error), config.database.url),
                    }
                }
            failing_probe = "the extractor profile read"
            try:
                active_extractor_profiles = _active_extractor_profiles(conn)
            except Exception as profile_error:
                conn.rollback()
                lines = redact_text(str(profile_error), config.database.url).strip().splitlines()
                extractor_profiles_error = lines[0] if lines else type(profile_error).__name__
            failing_probe = "the canary lookup"
            active_canaries = _active_embedding_canaries(conn)
            failing_probe = "the stranded-chunking count"
            stranded_chunkings = _stranded_chunkings(conn)
            failing_probe = "the orphan vector table scan"
            orphan_vector_tables = _orphan_vector_tables(conn)
            failing_probe = "the lexical index inspection"
            lexical_facts = lexical_index_facts(conn)
            lexical_probe_ran = True
    except Exception as error:
        # The driver's text runs to several lines plus a "Background on this
        # error" link. The human report gets its first line, like every other
        # command's one-line "database not reachable"; the JSON keeps it whole.
        error_text = redact_text(str(error), config.database.url)
        first_line = error_text.strip().splitlines()[0] if error_text.strip() else ""
        if database_ok:
            # The server answered `SELECT 1`: "not reachable" and the
            # init-postgres hint would send the reader to the connection when
            # one later probe is what failed. The verdict (`_any_failed`) is
            # unchanged; only the words are.
            # Said as facts only: the usual route here is a probe's own
            # rollback raising -- the connection dropping -- so no diagnosis.
            message = f"the server answered SELECT 1; then {failing_probe} failed: {first_line}"
        else:
            message = (
                f"not reachable ({first_line}). Run `cementic init postgres "
                "./cementic-postgres` once and follow its README, or set "
                "CEMENTIC_DB_URL to a Postgres with pgvector and pgvectorscale."
            )
        checks["database"] = {
            "status": "fail",
            "reachable": database_ok,
            "error": error_text,
            "message": message,
        }
        checks["extensions"] = {}

    runtime = embedding_runtime_checks(config)
    checks["model"] = runtime["model"]
    checks["daemon"] = runtime["daemon"]
    daemon_healthy = bool(runtime["daemon"]["reachable"])

    checks["stranded_chunkings"] = _stranded_chunkings_check(
        stranded_chunkings, database_reachable=database_ok
    )
    checks["orphan_vector_tables"] = _orphan_vector_tables_check(orphan_vector_tables)
    checks["lexical_index"] = _lexical_index_check(
        lexical_facts,
        hybrid=config.search.hybrid,
        database_reachable=database_ok,
        probe_ran=lexical_probe_ran,
    )
    checks["chunk_budget"] = _chunk_budget_check(config)
    checks["tokenizer"] = _tokenizer_check()
    checks["embedding_server"] = runtime["embedding_server"]
    checks["ocr"] = _ocr_check(config)
    checks["extraction_commands"] = _extraction_commands_check(config)
    checks["extractor_profile"] = _extractor_profile_check(
        config,
        active_extractor_profiles,
        read_error=extractor_profiles_error,
        database_reachable=database_ok,
    )
    checks["embedding_canary"] = _embedding_canary_check(
        config, active_canaries, daemon_healthy=daemon_healthy, database_reachable=database_ok
    )

    return {"ok": not _any_failed(checks), "checks": checks}


def _any_failed(checks: dict[str, Any]) -> bool:
    """Whether any check, at any nesting depth, reports ``fail``.

    The verdict is read off the statuses the report prints, not off flags kept
    beside them: a conjunction of flags let `extensions` carry a nested
    ``fail`` while its flag stayed True, so `doctor` said ok and exited 0 over
    a failure it had just printed (C11, 2026-11-18). ``warning`` passes.
    """
    for check in checks.values():
        if not isinstance(check, dict):
            continue
        if check.get("status") == "fail":
            return True
        if _any_failed(check):
            return True
    return False


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
                "cannot record which build produced their text -- " + "; ".join(failures)
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


def _unread_reason(database_reachable: bool) -> str:
    """Why a database-backed probe has no result to report.

    `reachable: True` beside "database unreachable" contradicted itself: after
    the server answered `SELECT 1`, a missing result means the probe did not
    run or could not be read, which the `database` and `extensions` checks
    explain.
    """
    if database_reachable:
        return "the probe did not run or its result could not be read (see the database check)"
    return "database unreachable"


def _extractor_profile_check(
    config: Config,
    active_profiles: Sequence[tuple[str, dict[str, Any]]] | None,
    read_error: str | None = None,
    *,
    database_reachable: bool = False,
) -> dict[str, Any]:
    """The drift check, plus the three states in which it cannot run.

    Building this install's payload can fail: a command extractor whose version
    probe cannot run raises from here, and doctor is the last place that may
    answer a fault with a traceback. `extraction_commands` reports that fault
    as the failure it is; this reports only that it could not compare. A stored
    profile that could not be read (`read_error`) is the same kind of gap.
    """
    if read_error is not None:
        return {
            "status": "warning",
            "drift": {},
            "message": f"could not read the active extractor profiles to compare: {read_error}",
        }
    if active_profiles is None:
        return {
            "status": "ok",
            "drift": {},
            "message": f"{_unread_reason(database_reachable)}; extractor profiles not compared",
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
    """(collection, extractor payload) for every collection's active revision.

    A fresh install has no `pipeline_revisions` until `cementic start` creates
    it, and that is the state the setup docs run `doctor` in, so a missing
    table reads as "no profiles". Only a missing table: any other failure (a
    permission error, a timeout, a payload that is not JSON) raises, and the
    caller reports it as a profile it could not read. Catching every error
    here once printed `extractor_profile: ok` for a comparison that never ran.
    """
    inspector = sa_inspect(conn)
    if not all(inspector.has_table(name) for name in ("pipeline_revisions", "extractor_profiles")):
        return []
    rows = conn.execute(
        text(
            "SELECT r.collection, p.config_json FROM pipeline_revisions r "
            "JOIN extractor_profiles p ON p.id = r.extractor_profile_id "
            "WHERE r.status = 'active' ORDER BY r.collection"
        )
    ).fetchall()
    return [(str(collection), json.loads(payload)) for collection, payload in rows]


def _canary_drift_check(
    config: Config, canaries: Sequence[tuple[str, str, str, str, str | None]]
) -> dict[str, Any]:
    """Replay each active revision's stored canary against the running server.

    Only a server serving *that profile's* model is asked. A daemon loaded with
    another model would fail every canary by construction, which would report a
    deliberate model swap as corruption -- and collections on a legacy profile
    (the usual state once a model has moved on) would warn forever.
    """
    from cementic.canary import (
        compare_canary_vectors,
        decode_texts,
        decode_vectors,
        describe_canary_verdict,
    )
    from cementic.embedding_runtime import runtime_spec_from_profile_json

    findings: dict[str, str] = {}
    checked = 0
    worst_cosine = 1.0
    for collection, profile_json, texts_json, vectors_json, recorded_build in canaries:
        try:
            spec = runtime_spec_from_profile_json(profile_json)
            if spec.provider != "llama-cpp":
                continue
            client = build_llama_cpp_client(config, spec)
            if not client.matches_expected_runtime():
                continue
            observed = client.embed_exact(decode_texts(texts_json))
            verdict = compare_canary_vectors(decode_vectors(vectors_json), observed)
        except Exception as error:
            findings[collection] = f"could not replay this collection's canary: {error}"
            continue
        checked += 1
        worst_cosine = min(worst_cosine, verdict.worst_cosine)
        # Only a changed *function* is a finding. Reporting the server's own
        # scheduling noise as drift would fire on nothing but llama-server
        # batching two requests differently, and a check that cries wolf on
        # normal operation is one nobody reads.
        if not verdict.comparable:
            findings[collection] = describe_canary_verdict(verdict, recorded_build)
    if not findings:
        return {
            "status": "ok",
            "drift": {},
            "message": (
                f"{checked} active revision(s) reproduce their reference vectors "
                f"(worst cosine {worst_cosine:.9f})"
                if checked
                else "no active revision has a reference canary the running server can replay"
            ),
        }
    return {
        "status": "warning",
        "drift": findings,
        "message": "; ".join(
            f"{collection}: {finding}" for collection, finding in sorted(findings.items())
        ),
    }


def _embedding_canary_check(
    config: Config,
    canaries: Sequence[tuple[str, str, str, str, str | None]] | None,
    *,
    daemon_healthy: bool,
    database_reachable: bool = False,
) -> dict[str, Any]:
    """The canary replay, plus the two states in which it cannot run.

    Skipped rather than failed when the daemon is not answering: this check
    reads a running server, and `doctor` may not start one.
    """
    if canaries is None:
        return {
            "status": "ok",
            "drift": {},
            "message": f"{_unread_reason(database_reachable)}; reference vectors not compared",
        }
    if not daemon_healthy:
        return {
            "status": "ok",
            "drift": {},
            "message": "embedding server not answering; reference vectors not compared",
        }
    return _canary_drift_check(config, canaries)


def _active_embedding_canaries(
    conn: Any,
) -> list[tuple[str, str, str, str, str | None]]:
    """(collection, embedding runtime json, canary texts, canary vectors, build)."""
    try:
        rows = conn.execute(
            text(
                "SELECT r.collection, p.config_json, c.texts_json, c.vectors_json, "
                "c.server_build FROM pipeline_revisions r "
                "JOIN embedding_profiles p ON p.id = r.embedding_profile_id "
                "JOIN embedding_profile_canaries c ON c.embedding_profile_id = p.id "
                "WHERE r.status = 'active' ORDER BY r.collection"
            )
        ).fetchall()
    except Exception:
        # A database created before this table exists is the normal state on
        # upgrade, and `doctor` may not create schema to fix it -- the next
        # indexing run does. Reading that as "none recorded" keeps a diagnostic
        # from failing on the absence of the thing it diagnoses.
        conn.rollback()
        return []
    return [(str(row[0]), str(row[1]), str(row[2]), str(row[3]), row[4]) for row in rows]


#: Chunkings a revision count calls done that own no chunks. Every such
#: document is unsearchable: its text was chunked once, the chunks (and their
#: vectors, by cascade) were deleted, and the chunking kept a `source_content_hash`
#: matching its extraction -- which is exactly the condition `_step_chunk` reads
#: as "already done". `total_chunks > 0` keeps a legitimately empty document out
#: of the count. Deliberately unscoped: every collection at once, and every
#: chunk profile, so a chunking emptied under a profile no revision uses any
#: more still counts. For a diagnostic that is the right side to err on -- the
#: alternative hides a real loss behind a profile the reader has forgotten.
_STRANDED_CHUNKINGS_SQL = """
SELECT count(*)
FROM chunked_documents cd
JOIN extracted_documents ed ON cd.extracted_document_id = ed.id
JOIN source_documents sd ON ed.document_id = sd.id
WHERE cd.status = 'done'
  AND cd.total_chunks > 0
  AND cd.source_content_hash = ed.content_hash
  AND sd.status <> 'deleted'
  AND NOT EXISTS (SELECT 1 FROM chunks_v2 c WHERE c.chunked_document_id = cd.id)
"""


def _stranded_chunkings(conn: Any) -> int | None:
    """Count them, or None when the pipeline tables are not there to count.

    Guarded like `_active_embedding_canaries` below, and for the same reason: a
    database that has never indexed anything has no `chunked_documents`, which
    is the ordinary state of a fresh install running `doctor` for the first
    time. Unguarded, the raise lands in the caller's `except` and reports a
    server that just answered `SELECT 1` as an unreachable database, telling
    the user to go and create one.
    """
    try:
        return int(conn.execute(text(_STRANDED_CHUNKINGS_SQL)).scalar() or 0)
    except Exception:
        conn.rollback()
        return None


def _stranded_chunkings_check(
    count: int | None, *, database_reachable: bool = False
) -> dict[str, Any]:
    """Report documents that are counted as chunked but hold no chunks.

    This is the class of failure that has now been found twice -- once from a
    failed re-extraction, once from a watcher purge -- and both times the cost
    was not the bug but the silence: `cementic status` read 100% chunked while
    the documents returned nothing, for weeks. Any future purge that forgets to
    invalidate the chunking lands here instead of nowhere.

    A warning rather than a failure: the installation is fit to run, a corpus
    inside it needs repair, and doctor's overall verdict is about the former.
    """
    if count is None:
        return {
            "status": "warning",
            "message": (
                "could not be checked; the database has never indexed anything or the "
                "probe did not run"
                if database_reachable
                else "could not be checked; the database is unreachable or has "
                "never indexed anything"
            ),
        }
    return {
        "status": "ok" if count == 0 else "warning",
        "stranded": count,
        "message": (
            "every chunking counted as done owns chunks"
            if count == 0
            else f"{count} document(s) are counted as chunked but hold no chunks, so they "
            "match nothing in search; clear their chunkings' source_content_hash to "
            "re-open the work and let the worker re-chunk them"
        ),
    }


def _orphan_vector_tables(conn: Any) -> list[int] | None:
    """Profile ids with a vector table no revision references, or None if unreadable."""
    from cementic.vector_store import unreferenced_vector_profile_ids

    try:
        return unreferenced_vector_profile_ids(conn)
    except Exception:
        conn.rollback()
        return None


def _orphan_vector_tables_check(profile_ids: list[int] | None) -> dict[str, Any]:
    """Warn about vector tables left behind by a cleanup that timed out or never ran.

    The next promote or `collection remove` sweeps them; the warning says
    so rather than leaving full-size tables unexplained.
    """
    if profile_ids is None:
        return {"status": "warning", "message": "could not be checked"}
    if not profile_ids:
        return {"status": "ok", "orphans": 0, "message": "every vector table has a revision"}
    return {
        "status": "warning",
        "orphans": len(profile_ids),
        "profile_ids": profile_ids,
        "message": f"{len(profile_ids)} vector table(s) have no revision (profiles "
        f"{', '.join(map(str, profile_ids))}); the next promote or "
        "`collection remove` drops them",
    }


def _lexical_index_check(
    facts: LexicalIndexFacts | None,
    *,
    hybrid: bool,
    database_reachable: bool = False,
    probe_ran: bool = False,
) -> dict[str, Any]:
    """Report whether exact-word search has the index it silently depends on.

    Same class of failure as `_stranded_chunkings_check` above, one layer down.
    Without this index the lexical half of hybrid search matches nothing and
    `cementic search` still answers, from the vector arm alone -- so a corpus
    indexed before hybrid landed loses exactly the queries hybrid exists to
    serve (rare author surnames, acronyms, equation labels) while every command
    reports ok. Nothing else in cementic looks at it: `ensure_lexical_index`
    deliberately skips a table that already holds rows, because building there
    blocks worker startup for minutes.

    A warning rather than a failure, like the stranded-chunkings check: the
    installation is fit to run and a corpus inside it needs one command.
    """
    repair = "`cementic collection reindex <collection>` builds it"
    if facts is None:
        if not database_reachable:
            reason = "the database is unreachable or is not PostgreSQL"
        elif probe_ran:
            # The probe ran and its pg_index query failed: the database check is
            # ok, so pointing at it would send the reader nowhere.
            reason = "the index catalog could not be read; the database may not be PostgreSQL"
        else:
            reason = (
                "the probe did not run or its result could not be read (see the database check)"
            )
        return {"status": "warning", "message": f"could not be checked; {reason}"}
    if facts.present and facts.valid:
        return {
            "status": "ok",
            "present": True,
            "message": (
                f"{LEXICAL_INDEX_NAME} is present and valid"
                if hybrid
                else f"{LEXICAL_INDEX_NAME} is present; search.hybrid is off, so "
                "nothing reads it"
            ),
        }
    if facts.present:
        return {
            "status": "warning",
            "present": True,
            "message": (
                f"{LEXICAL_INDEX_NAME} exists but is INVALID, left by an interrupted "
                "build. The planner ignores it, so every exact-word query falls back "
                f"to a sequential scan of every chunk; {repair} again from scratch"
            ),
        }
    if not facts.has_chunks:
        return {
            "status": "ok",
            "present": False,
            "message": "no chunks yet; the index is created with the first ones",
        }
    if not hybrid:
        return {
            "status": "ok",
            "present": False,
            "message": (
                f"{LEXICAL_INDEX_NAME} is missing, and search.hybrid is off so "
                f"nothing reads it; {repair} if you turn hybrid on"
            ),
        }
    return {
        "status": "warning",
        "present": False,
        "message": (
            f"{LEXICAL_INDEX_NAME} is missing on a corpus that already holds chunks, "
            "so the exact-word half of every hybrid search matches nothing and only "
            f"the vector arm answers; {repair} (about 6 minutes for 2.3M chunks)"
        ),
    }


def _tokenizer_check() -> dict[str, Any]:
    """Report whether the chunking tokenizer is cached locally.

    tiktoken downloads it on first use. A warning rather than a failure when
    absent: a machine with network fetches it once and is fine, and `doctor`
    must not download anything itself. Offline, every chunk and every search
    fails until it is cached, which is what the message says.
    """
    from cementic.chunk import TOKENIZER, tokenizer_cache_file

    path = tokenizer_cache_file()
    cached = path.is_file()
    return {
        "status": "ok" if cached else "warning",
        "path": str(path),
        "cached": cached,
        "message": (
            f"{TOKENIZER} cached"
            if cached
            else f"{TOKENIZER} is not cached; the first chunk or search downloads it "
            "(about 1.7 MB) and fails without network. To fetch it now: "
            "printf x | cementic chunk"
        ),
    }


def _chunk_budget_check(config: Config) -> dict[str, Any]:
    """Report the chunk_size / n_ctx pairing's effect on the embed fast path.

    Config load refuses only pairings that must truncate. The wider band, where
    a full chunk *might* exceed the window, is a performance matter -- every
    chunk pays an exact tokenize round trip -- so it is reported here rather
    than blocking every command.
    """
    from cementic.embedding_runtime import (
        TASK_PREFIX_TOKEN_ALLOWANCE,
        TOKEN_RATIO_UPPER_BOUND,
    )

    chunk_size = config.pipeline.chunk_size
    n_ctx = config.llama_cpp.n_ctx
    worst_case = (chunk_size + TASK_PREFIX_TOKEN_ALLOWANCE) * TOKEN_RATIO_UPPER_BOUND
    fast_path = worst_case <= n_ctx
    largest_fast = int(n_ctx / TOKEN_RATIO_UPPER_BOUND) - TASK_PREFIX_TOKEN_ALLOWANCE
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
