"""Diagnostics for the config file and config errors.

Everything here explains *why* a configuration cannot be used or what cementic
would silently ignore; ``config`` builds the values. Kept apart so ``config``
stays the value layer and ``get_config`` is the only seam between the two.
"""

import difflib
import os
import tomllib
from pathlib import Path

from pydantic import ValidationError
from pydantic_settings import SettingsError
from rich.markup import escape

from cementic import config
from cementic.config import Config


def config_file_error() -> str | None:
    """Why the active config file could not be loaded, or None if it is fine.

    Returns None both when there is no config file (nothing to load) and when
    one loaded cleanly; the distinction is ``config.resolve_config_path()``'s job.

    Exists so `cementic doctor` can report an unusable config. A
    malformed or unreadable file is silently discarded and cementic runs on
    defaults -- the wrong database, the wrong model -- which is exactly the
    situation a diagnostic command must not describe as "ok".
    """
    path = config.resolve_config_path()
    if path is None:
        return None
    try:
        with open(path, "rb") as handle:
            tomllib.load(handle)
    except tomllib.TOMLDecodeError as error:
        return f"malformed TOML: {error}"
    except UnicodeDecodeError as error:
        # tomllib decodes the bytes itself, and UnicodeDecodeError is a
        # ValueError, not an OSError -- so a config saved in any non-UTF-8
        # encoding escaped both handlers below and reached the user as a
        # traceback, including from the command that exists to explain this.
        return f"not valid UTF-8: {error}"
    except OSError as error:
        return f"unreadable: {error}"
    return None


class ConfigError(Exception):
    """A configuration problem stated in one line, for a CLI to print.

    Distinct from pydantic's ``ValidationError``: this covers the problems
    pydantic cannot see, because they happen before or around the model --
    an explicitly requested config file that is missing, or a section name
    that matches nothing and is therefore dropped in silence.
    """


def config_path_error() -> str | None:
    """Why an explicitly requested config file cannot be used, or None.

    ``CEMENTIC_CONFIG`` naming a missing file or a directory used to fall
    through to ``./cementic.toml`` and then the user config, so a stale path in
    a service unit ran against entirely different settings -- and ``config
    path`` printed the fallback, never mentioning that the request was ignored.
    """
    explicit = os.environ.get("CEMENTIC_CONFIG")
    if not explicit:
        return None
    candidate = config.expand_user_path(explicit)
    if candidate is None:
        return (
            f"CEMENTIC_CONFIG points at {explicit}, whose home directory cannot be "
            "resolved (no such user)"
        )
    if candidate.is_file():
        return None
    if candidate.is_dir():
        return f"CEMENTIC_CONFIG points at {candidate}, which is a directory"
    return f"CEMENTIC_CONFIG points at {candidate}, which does not exist"


def _known_sections() -> dict[str, type]:
    """Map TOML section name -> the settings model that owns it.

    Derived from ``Config``'s own fields rather than a hand-kept list, so a new
    section is recognised the moment it is added.
    """
    sections: dict[str, type] = {}
    for field in Config.model_fields.values():
        model = field.annotation
        section = getattr(model, "_toml_section", None)
        if isinstance(section, str) and isinstance(model, type):
            sections[section] = model
    return sections


def _valid_keys(model: type) -> set[str]:
    """Field names and validation aliases accepted by a settings model."""
    keys: set[str] = set()
    for name, field in getattr(model, "model_fields", {}).items():
        keys.add(name)
        alias = getattr(field, "validation_alias", None)
        for choice in getattr(alias, "choices", []) or ([alias] if alias else []):
            if isinstance(choice, str):
                keys.add(choice)
    return keys


def config_file_problems() -> list[str]:
    """Parts of the active config file that cementic would silently ignore.

    A typo *inside* a known section is caught hard by ``extra="forbid"``, but a
    typo in the section *name* was invisible: the section was dropped and the
    affected settings fell back to plausible defaults. That is the expensive
    direction -- ``[databse] host`` does not error, it quietly points at
    localhost, and a dropped ``[pipeline] chunk_size`` changes the embedding
    profile, recoverable only by a full re-index.

    Reported together rather than one per run, because pydantic stops at the
    first broken section.
    """
    path = config.resolve_config_path()
    if path is None:
        return []
    parsed = config.load_config_file()
    sections = _known_sections()
    problems: list[str] = []
    for key, value in parsed.items():
        if key in sections:
            if not isinstance(value, dict):
                # `database = 5` names a real section but is not a table; the
                # TOML source drops it silently -- the exact silent-drop
                # failure this function exists to report.
                problems.append(f"[{key}] must be a table of settings, not a bare value")
                continue
            unknown_keys = sorted(set(value) - _valid_keys(sections[key]))
            problems.extend(
                f"[{key}] has no setting {name!r}{_suggest(name, _valid_keys(sections[key]))}"
                for name in unknown_keys
            )
        elif isinstance(value, dict):
            problems.append(f"unknown section [{key}]{_suggest(key, set(sections))}")
        else:
            problems.append(
                f"{key!r} is at the top level, which cementic ignores; "
                "settings live under a section"
            )
    return problems


def _suggest(name: str, candidates: set[str]) -> str:
    """A ' — did you mean X?' hint, or empty when nothing is close."""
    close = difflib.get_close_matches(name, sorted(candidates), n=1)
    return f" — did you mean {close[0]!r}?" if close else ""


def format_config_error(error: ValidationError, path: Path | None) -> str:
    """Render a pydantic failure as one line naming the source, section and key.

    Deliberately built from ``loc``/``msg``/``type`` only. ``str(error)`` and
    ``err["input"]`` both embed the offending *value*, so formatting either one
    prints a mistyped ``[database] passwrd`` straight into the terminal and into
    the worker log files the CLI points users at.

    When the failing key is set by a ``CEMENTIC_*`` variable, the message names
    the variable: env overrides the file, so prefixing the innocent file path
    (as every error here used to) sent the user to edit a file whose value was
    never read.
    """
    sections = _known_sections()
    section_by_model = {model.__name__: name for name, model in sections.items()}
    section = section_by_model.get(error.title)
    env_prefix = None
    if section is not None:
        # _known_sections types its values as bare `type`; every section model
        # is a BaseSettings subclass carrying model_config.
        env_prefix = getattr(sections[section], "model_config", {}).get("env_prefix")
    lines: list[str] = []
    for item in error.errors():
        parts = [str(part) for part in item["loc"]]
        # The section can come from the error's own model (a section validated
        # on its own) or from the first location part (the same section reached
        # through Config). Without the second case no env_prefix was available
        # for anything raised through Config(), so nothing was ever attributed.
        item_section = section
        if item_section is None and parts and parts[0] in sections:
            item_section = parts[0]
            parts = parts[1:]
        item_prefix = env_prefix
        if item_section is not None and item_section in sections:
            item_prefix = getattr(sections[item_section], "model_config", {}).get("env_prefix")
        location = ".".join(parts)
        where = item_section or (parts[0] if parts else "config")
        field = location or "<section>"
        hint = ""
        if item["type"] == "extra_forbidden" and item_section in sections:
            hint = _suggest(location, _valid_keys(sections[item_section]))
        env_name = _blaming_env_var(location, item_prefix)
        if env_name is not None:
            lines.append(f"{env_name} (environment variable): {item['msg']}{hint}")
            continue
        # Prefix per line, not once around the joined string: a file-caused and
        # an env-caused error in the same section used to be joined and then
        # prefixed together, so the env line read as if it came from the file.
        location_text = f"{path}: " if path is not None else ""
        line = f"{location_text}[{where}] {field}: {item['msg']}{hint}"
        if not location:
            # A whole-section or whole-config validator has no key to blame, so
            # nothing above can attribute it -- and its inputs may well have come
            # from the environment. Name the variables actually in effect rather
            # than sending the user to a file whose values may be overridden.
            active = _active_env_vars(item_prefix)
            if active:
                line += f" (in effect: {', '.join(active)})"
        lines.append(line)
    return "; ".join(lines)


def render_config_error(error: Exception) -> str | None:
    """Render a config-loading exception as one markup-escaped line, or None.

    ``None`` means ``error`` is not one of the three exceptions
    ``get_config()`` can raise -- ``ConfigError``, pydantic's
    ``ValidationError``, or pydantic-settings' ``SettingsError`` -- so the
    caller knows to let it propagate rather than mistaking "nothing to say"
    for "no error". The CLI and the runner both caught the same three
    exceptions and built the same "config error: ..." line from them; this is
    that logic in one place. The CLI additionally prints a hint for
    ``SettingsError`` (env vars that take a list or table must be JSON) that
    the runner never carried, so callers still branch on
    ``isinstance(error, SettingsError)`` for that hint rather than getting it
    from here.
    """
    if isinstance(error, ConfigError):
        detail = str(error)
    elif isinstance(error, ValidationError):
        detail = format_config_error(error, config.resolve_config_path())
    elif isinstance(error, SettingsError):
        detail = str(error)
    else:
        return None
    return f"config error: {escape(detail)}"


def _blaming_env_var(location: str, env_prefix: str | None) -> str | None:
    """The CEMENTIC_* variable responsible for this error location, if any.

    Two shapes reach here: a plain field name, whose variable is the section
    prefix plus the name, and a field whose validation alias *is* the variable
    (``CEMENTIC_DB_URL``), where prefixing again produced the nonexistent
    ``CEMENTIC_DB_CEMENTIC_DB_URL`` and so never attributed the most-documented
    variable in the project.
    """
    if not location:
        return None
    if location.startswith("CEMENTIC_") and os.environ.get(location) is not None:
        return location
    if env_prefix:
        candidate = f"{env_prefix}{location.upper()}"
        if os.environ.get(candidate) is not None:
            return candidate
    return None


def _active_env_vars(env_prefix: str | None) -> list[str]:
    """Every set CEMENTIC_* variable belonging to this section, sorted."""
    if not env_prefix:
        return []
    return sorted(name for name in os.environ if name.startswith(env_prefix))
