"""Configuration management for cementic."""

import difflib
import os
import re
import sys
import tomllib
from pathlib import Path
from typing import Any, ClassVar

from platformdirs import user_config_dir, user_data_dir
from pydantic import (
    AliasChoices,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    SettingsError,
)
from rich.markup import escape
from sqlalchemy.engine import URL, make_url

from cementic.index_strategies import supported_index_methods
from cementic.vector_store import HNSW_ITERATIVE_SCAN_MODES

#: Directory names the watcher skips by default. Pointing `cementic start` at a
#: project directory otherwise indexes every README and note inside dependency,
#: build and VCS trees -- thousands of files no one meant to search, each costing
#: a hash, an extraction and an embedding. Names, not globs: matching is by exact
#: directory name at any depth.
DEFAULT_IGNORED_DIRECTORIES: tuple[str, ...] = (
    ".bzr",
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "site-packages",
    "target",
    "venv",
)


def _expand_user_path(raw: str) -> Path | None:
    """Expand a leading ``~``, or None when the home directory is unknowable.

    ``Path.expanduser()`` raises RuntimeError for ``~nosuchuser`` (and for ``~``
    with no HOME). Both callers below are the *diagnostic* path for a bad
    ``CEMENTIC_CONFIG``, so letting that escape turned "your config path is
    wrong" into a raw traceback from every command.
    """
    try:
        return Path(raw).expanduser()
    except RuntimeError:
        return None


def resolve_config_path() -> Path | None:
    """Resolve the active config file path, or None if there isn't one.

    Pure given the environment: ``CEMENTIC_CONFIG`` env var, then a project-local
    ``./cementic.toml``, then ``<user_config_dir>/cementic/config.toml``.
    """
    candidates: list[Path] = []
    explicit = os.environ.get("CEMENTIC_CONFIG")
    if explicit:
        # expanduser to match config_path_error's check: without it,
        # CEMENTIC_CONFIG=~/cementic.toml never matched is_file() (the ~ stays
        # literal), so the variable was silently ignored while the guard that
        # exists to report exactly that judged the same path usable.
        expanded = _expand_user_path(explicit)
        if expanded is not None:
            candidates.append(expanded)
    candidates.append(Path.cwd() / "cementic.toml")
    candidates.append(Path(user_config_dir("cementic")) / "config.toml")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def default_config_path() -> Path:
    """The canonical location `cementic config init` writes to."""
    return Path(user_config_dir("cementic")) / "config.toml"


def cementic_data_dir(*, ensure_exists: bool = False) -> Path:
    """Return the cementic user-data directory (where artifacts/models live)."""
    return Path(user_data_dir("cementic", ensure_exists=ensure_exists))


def resolve_llama_model_path(model_path: str) -> Path:
    """Resolve a configured llama.cpp model path to a concrete file location.

    The same rule decides where the bootstrapper *downloads* the model and where
    the runtime *loads* it, so the two always agree:

    - an absolute path is honored as-is;
    - a relative path that already exists from the current directory is used as
      given (the in-repo ``./models/...`` development convenience);
    - otherwise it resolves under the cementic data directory, which is where an
      auto-downloaded model is written.
    """
    # expanduser first: "~/models/x.gguf" is neither absolute nor existent as
    # written, so it used to resolve to a literal "~" directory *inside* the
    # data dir -- the configured model was never found and auto-download was
    # attempted against a nonsense path.
    raw = Path(model_path).expanduser()
    if raw.is_absolute():
        return raw
    if raw.exists():
        return raw.resolve()
    return (cementic_data_dir() / raw).resolve()


def config_file_error() -> str | None:
    """Why the active config file could not be loaded, or None if it is fine.

    Returns None both when there is no config file (nothing to load) and when
    one loaded cleanly; the distinction is ``resolve_config_path()``'s job.

    Exists so `cementic doctor` can report an unusable config. A
    malformed or unreadable file is silently discarded and cementic runs on
    defaults -- the wrong database, the wrong model -- which is exactly the
    situation a diagnostic command must not describe as "ok".
    """
    path = resolve_config_path()
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
    candidate = _expand_user_path(explicit)
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
    path = resolve_config_path()
    if path is None:
        return []
    parsed = load_config_file()
    sections = _known_sections()
    problems: list[str] = []
    for key, value in parsed.items():
        if key in sections:
            unknown_keys = sorted(set(value) - _valid_keys(sections[key])) if isinstance(
                value, dict
            ) else []
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
        detail = format_config_error(error, resolve_config_path())
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


#: Config-file problems already reported, so the warning is not repeated once
#: per section. Each sub-model reads the file through its own settings source
#: (nine of them), which otherwise printed the identical warning nine times.
_warned_config_problems: set[tuple[str, str]] = set()


def _warn_once(path: Path, problem: str) -> None:
    key = (str(path), problem)
    if key in _warned_config_problems:
        return
    _warned_config_problems.add(key)
    print(f"warning: ignoring {problem} config file {path}", file=sys.stderr)


def load_config_file() -> dict[str, Any]:
    """Read the active TOML config file into a dict (empty if none/invalid)."""
    path = resolve_config_path()
    if path is None:
        return {}
    try:
        with open(path, "rb") as handle:
            return tomllib.load(handle)
    except (tomllib.TOMLDecodeError, UnicodeDecodeError):
        _warn_once(path, "malformed")
        return {}
    except OSError:
        # Previously silent: an unreadable config left no trace at all.
        _warn_once(path, "unreadable")
        return {}


class _SectionedTomlSource(PydanticBaseSettingsSource):
    """Feed one ``[section]`` of the TOML config file into a settings model.

    Scoped per sub-model and ranked *below* env, so env always overrides the
    file while the file overrides built-in defaults.
    """

    def __init__(self, settings_cls: type[BaseSettings], section: str) -> None:
        super().__init__(settings_cls)
        self._section = section

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        section = load_config_file().get(self._section)
        return dict(section) if isinstance(section, dict) else {}


class _SectionSettings(BaseSettings):
    """Base for config sections: env > config-file section > defaults."""

    #: TOML table this section reads from (e.g. "database").
    _toml_section: ClassVar[str] = ""

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        sources: list[PydanticBaseSettingsSource] = [
            init_settings,
            env_settings,
            dotenv_settings,
        ]
        if cls._toml_section:
            sources.append(_SectionedTomlSource(settings_cls, cls._toml_section))
        sources.append(file_secret_settings)
        return tuple(sources)


class DatabaseConfig(_SectionSettings):
    """PostgreSQL database configuration."""

    model_config = SettingsConfigDict(env_prefix="CEMENTIC_DB_", populate_by_name=True)
    _toml_section = "database"

    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, ge=1, le=65535, description="Database port")
    name: str = Field(default="cementic", description="Database name")
    user: str = Field(default="cementic", description="Database user")
    password: SecretStr = Field(
        default=SecretStr("cementic"),
        description="Database password (override CEMENTIC_DB_PASSWORD outside local dev)",
    )
    url_override: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("CEMENTIC_DB_URL"),
        description="Full SQLAlchemy URL; when set it wins over the discrete fields above",
    )

    @field_validator("url_override")
    @classmethod
    def _validate_url_override(cls, value: SecretStr | None) -> SecretStr | None:
        """Reject an unparseable URL here, where it is still a config error.

        Left to `url`, the parse failure surfaced from whichever command
        happened to touch the property first -- as a raw traceback out of
        `cementic doctor`, and out of the very error message `start` builds to
        explain it. An empty value keeps meaning "unset", which the discrete
        host/port/name fields depend on.
        """
        if value is None or not value.get_secret_value():
            return value
        try:
            make_url(value.get_secret_value())
        except Exception as error:
            # str(error) is safe: SQLAlchemy reports the failure without echoing
            # the URL, which would carry the password.
            raise ValueError(f"is not a valid SQLAlchemy URL ({error})") from None
        return value

    @property
    def url(self) -> URL:
        """SQLAlchemy database URL (password redacted in repr).

        A ``CEMENTIC_DB_URL`` override takes precedence over the discrete
        host/port/name/user/password fields.
        """
        if self.url_override is not None and self.url_override.get_secret_value():
            return make_url(self.url_override.get_secret_value())
        return URL.create(
            "postgresql",
            username=self.user,
            password=self.password.get_secret_value(),
            host=self.host,
            port=self.port,
            database=self.name,
        )


class LlamaCppConfig(_SectionSettings):
    """llama.cpp configuration."""

    model_config = SettingsConfigDict(env_prefix="CEMENTIC_LLAMA_")
    _toml_section = "llama_cpp"

    model_path: str = Field(
        default="models/nomic-embed-text-v2-moe.Q8_0.gguf",
        description="Path to .gguf model file",
    )
    # ge=1: n_ctx <= 0 silently disabled the token-budget guard (its cheap
    # path treats a non-positive window as "no budget"), so every over-long
    # chunk went back to embedding truncated.
    n_ctx: int = Field(default=512, ge=1, description="Context window size")
    n_gpu_layers: int = Field(
        default=0,
        ge=-1,
        description="Number of layers to offload to GPU (-1 for all)",
    )
    embedding_dim: int = Field(
        default=768,
        ge=1,
        description="Fallback embedding dimension; the live model's dimension is "
        "probed and stored in the profile when available",
    )
    verbose: bool = Field(default=False, description="Enable verbose output")
    daemon_host: str = Field(default="127.0.0.1", description="llama.cpp daemon host")
    daemon_port: int = Field(
        default=11555, ge=1, le=65535, description="llama.cpp daemon port"
    )
    daemon_start_timeout_seconds: int = Field(
        default=120,
        ge=1,
        description="Seconds to wait for the llama.cpp daemon to become ready. "
        "A cold start loads a multi-GB model from disk; the previous 30s default "
        "contradicted cementic's own 'can take 30s+' warning and gave up on "
        "daemons that were still starting successfully. Startup now fails fast "
        "when the process dies, so this budget only bounds a genuinely slow load.",
    )
    llama_embed_timeout_seconds: int = Field(
        default=120,
        ge=1,
        description="Seconds to wait for an embedding HTTP request to complete; "
        "separate from daemon_start_timeout_seconds since a large batch can "
        "legitimately run far longer than a startup probe should ever wait",
    )
    daemon_autostart: bool = Field(
        default=True,
        description="Automatically start/restart the llama.cpp daemon when a client needs it",
    )
    daemon_pid_file: Path | None = Field(default=None, description="llama.cpp daemon PID path")
    daemon_log_file: Path | None = Field(default=None, description="llama.cpp daemon log path")


class PipelineConfig(_SectionSettings):
    """Pipeline configuration."""

    model_config = SettingsConfigDict(
        env_prefix="CEMENTIC_PIPELINE_",
        populate_by_name=True,
    )
    _toml_section = "pipeline"

    chunk_size: int = Field(
        default=320,
        ge=1,
        description="Tokens per chunk, counted with the tiktoken encoding named "
        "in chunk.TOKENIZER -- not the embedding model's own tokenizer, which "
        "is what llama_cpp.n_ctx (512) bounds. The two disagree, and for the "
        "default Nomic model they disagree in the dangerous direction: measured "
        "over real indexed chunks, one cl100k token is a median of 1.14 model "
        "tokens, p95 1.24, max 1.33. At the old default of 512 that put 96% of "
        "full-size chunks (160 of 167) over the window, and the server drops "
        "the overflow silently. 320 leaves headroom to a ratio of 1.45 -- "
        "counting the task prefix the embedding client prepends, which is what "
        "is actually measured, so the runtime guard's cheap path still applies "
        "and a full chunk costs no extra round trip. "
        "Raising n_ctx is not an alternative: the model architecture caps at "
        "512 (nomic-bert-moe.context_length in the GGUF metadata). "
        "This is now enforced at runtime rather than assumed -- the embedding "
        "client counts with the model's own tokenizer and fails an over-budget "
        "chunk instead of truncating it. Re-measure with "
        "scripts/measure_chunk_context_fit.py before raising this.",
    )
    chunk_overlap: int = Field(default=80, ge=0, description="Token overlap between chunks")
    embedding_provider: str = Field(
        default="llama-cpp",
        description="Embedding provider to use (currently llama-cpp)",
    )

    @field_validator("embedding_provider")
    @classmethod
    def _validate_provider(cls, value: str) -> str:
        """Check the name against the registry, as `index.method` already does.

        Unvalidated, a near-miss like "llama_cpp" loaded fine and `cementic
        doctor` reported ok, because doctor runs the llama.cpp checks
        regardless; the first `cementic start` then died on it.
        """
        # Imported here, not at module scope: embedding_runtime imports config.
        from cementic.embedding_runtime import supported_embedding_providers

        supported = supported_embedding_providers()
        if value not in supported:
            raise ValueError(f"must be one of {', '.join(sorted(supported))}")
        return value

    @model_validator(mode="after")
    def _validate_chunk_window(self) -> "PipelineConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class IndexConfig(_SectionSettings):
    """ANN index configuration.

    ``method`` is a serving choice (which index to build/query), not an embedding
    fact — changing it rebuilds an index, never re-embeds. Query-time knobs
    (``hnsw_ef_search``, ``diskann_query_rescore``) are applied per query; the
    others are build-time.
    """

    model_config = SettingsConfigDict(env_prefix="CEMENTIC_INDEX_")
    _toml_section = "index"

    method: str = Field(default="hnsw", description="ANN index method: hnsw or diskann")
    # pgvector HNSW
    hnsw_m: int = Field(default=16, ge=1, description="HNSW max connections per layer (build)")
    hnsw_ef_construction: int = Field(
        default=64, ge=1, description="HNSW build-time candidate list size"
    )
    hnsw_ef_search: int = Field(default=40, ge=1, description="HNSW query-time candidate list size")
    hnsw_iterative_scan: str = Field(
        default="relaxed_order",
        description="HNSW iterative scan mode: relaxed_order (default), "
        "strict_order, or off. Search filters candidates during the index scan, "
        "so without this a query over a collection holding a small share of a "
        "shared vector table can return fewer rows than asked for, or none. "
        "Needs pgvector 0.8+; ignored on older servers.",
    )
    # pgvectorscale DiskANN
    diskann_num_neighbors: int = Field(default=50, ge=1, description="DiskANN graph degree (build)")
    diskann_search_list_size: int = Field(
        default=100, ge=1, description="DiskANN build-time search list size"
    )
    diskann_query_rescore: int = Field(
        default=50, ge=1, description="DiskANN query-time rescore count"
    )
    build_memory: str = Field(
        default="2GB",
        description="maintenance_work_mem for the session that builds an ANN "
        "index. PostgreSQL defaults to 64MB; an HNSW graph that does not fit "
        "spills to disk and the build slows sharply — 100k 768-dim vectors took "
        "1454s at 64MB and 345s at 2GB. Applies only while an index is being "
        "built, one at a time. Lower it on a memory-constrained server.",
    )

    @field_validator("build_memory")
    @classmethod
    def _validate_build_memory(cls, value: str) -> str:
        # This goes into a SET statement as a literal, so the grammar is closed
        # deliberately rather than passed through to the server to judge: a
        # config file is not a trusted source of SQL.
        if not re.fullmatch(r"\d+\s*(kB|MB|GB|TB)?", value.strip(), flags=re.IGNORECASE):
            raise ValueError(
                "build_memory must be a PostgreSQL memory size such as "
                "'512MB' or '2GB' (a bare number is kilobytes)"
            )
        return value.strip()

    @field_validator("hnsw_iterative_scan")
    @classmethod
    def _validate_iterative_scan(cls, value: str) -> str:
        # Refused here rather than at the server: an invalid value aborts the
        # SET LOCAL, and with it the search query it was tuning.
        if value not in HNSW_ITERATIVE_SCAN_MODES:
            raise ValueError(
                f"hnsw_iterative_scan must be one of: {', '.join(HNSW_ITERATIVE_SCAN_MODES)}"
            )
        return value

    @field_validator("method")
    @classmethod
    def _validate_method(cls, value: str) -> str:
        # Ask the index-strategy registry rather than repeating its contents, so
        # adding a method stays a single-entry change as its module claims.
        supported = supported_index_methods()
        if value not in supported:
            raise ValueError(f"index method must be one of: {', '.join(sorted(supported))}")
        return value


class ExtractionConfig(_SectionSettings):
    """Document extraction configuration."""

    model_config = SettingsConfigDict(env_prefix="CEMENTIC_EXTRACT_")
    _toml_section = "extraction"

    backends: dict[str, str] = Field(
        default_factory=dict,
        description="Per-file-type extractor choice, e.g. {'pdf': 'pymupdf4llm'}. "
        "Keys are file types, with or without a leading dot and in any case; "
        "values are registered extractor names. Unset types fall back to the "
        "registry default.",
    )
    use_ocr: bool = Field(default=False, description="Enable OCR when supported")

    @field_validator("backends")
    @classmethod
    def _validate_backends(cls, value: dict[str, str]) -> dict[str, str]:
        """Normalise keys and check each entry against the extractor registry.

        Both halves were unvalidated. A key like `PDF` or `.pdf` matched nothing
        at lookup time and was silently ignored -- while still entering the
        extractor profile's fingerprint, so it forced a full re-extraction that
        produced exactly what the previous one did. A misspelt extractor name
        surfaced only when a matching file eventually arrived, one failed
        document at a time, reported as a capability problem rather than a typo.

        Imported here rather than at module scope: `extract` imports this module.
        """
        from cementic.extract import backend_choice_error, normalize_backend_file_type

        normalized: dict[str, str] = {}
        for file_type, extractor_name in value.items():
            key = normalize_backend_file_type(file_type)
            if not key:
                raise ValueError(f"empty file type in extraction backends: {file_type!r}")
            if key in normalized and normalized[key] != extractor_name:
                raise ValueError(
                    f"file type '{key}' is configured twice with different backends: "
                    f"'{normalized[key]}' and '{extractor_name}'"
                )
            problem = backend_choice_error(key, extractor_name)
            if problem is not None:
                raise ValueError(problem)
            normalized[key] = extractor_name
        return normalized


class StorageConfig(_SectionSettings):
    """Artifact storage configuration."""

    model_config = SettingsConfigDict(env_prefix="CEMENTIC_STORAGE_")
    _toml_section = "storage"

    artifacts_path: Path | None = Field(
        default=None, description="Root path for cached artifacts"
    )


class SourceWatcherConfig(_SectionSettings):
    """Source watcher configuration."""

    model_config = SettingsConfigDict(env_prefix="CEMENTIC_SOURCE_WATCHER_")
    _toml_section = "source_watcher"

    log_file: Path | None = Field(default=None, description="Log file path")
    state_path: Path | None = Field(default=None, description="Source watcher state file path")
    ignore_directories: list[str] = Field(
        default_factory=lambda: list(DEFAULT_IGNORED_DIRECTORIES),
        description="Directory names never descended into. Matching is by exact "
        "name at any depth, so 'node_modules' skips every such directory in the "
        "tree. Set to an empty list to index everything.",
    )


class PipelineWorkerConfig(_SectionSettings):
    """Pipeline worker configuration."""

    model_config = SettingsConfigDict(
        env_prefix="CEMENTIC_PIPELINE_WORKER_",
        populate_by_name=True,
    )
    _toml_section = "pipeline_worker"

    log_file: Path | None = Field(default=None, description="Log file path")
    state_path: Path | None = Field(default=None, description="Pipeline worker state file path")
    batch_size: int = Field(
        default=32,
        ge=1,
        le=128,
        description="Number of chunks to embed in one batch",
    )
    # gt=0: zero or negative turned the idle wait into a hot spin pinning a
    # core, with nothing anywhere reporting why.
    poll_interval: float = Field(
        default=1.0,
        gt=0,
        description="Seconds between polling for pending chunks",
    )


class BootstrapConfig(_SectionSettings):
    """Runtime bootstrap configuration for the embedding model.

    cementic does not manage containers. Postgres (with pgvector + vectorscale)
    is provisioned externally -- e.g. via ``cementic init postgres
    ./cementic-postgres`` and its generated setup, or any Postgres pointed at by
    ``CEMENTIC_DB_URL``.
    """

    model_config = SettingsConfigDict(env_prefix="CEMENTIC_BOOTSTRAP_")
    _toml_section = "bootstrap"

    auto_download_llama_model: bool = Field(
        default=True,
        description="Automatically download the llama.cpp model file if missing",
    )
    llama_model_url: str = Field(
        default=(
            "https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe-GGUF/resolve/main/"
            "nomic-embed-text-v2-moe.Q8_0.gguf"
        ),
        description="Default llama.cpp model download URL",
    )
    llama_model_sha256: str | None = Field(
        default="06e7a7e594a26985523c18383aba4aad39fe6e14f08ffc6ab5b554e1ccdc3cff",
        description="Expected SHA-256 of the llama.cpp model file. Defaults to the digest "
        "of the bundled Nomic model; set empty to disable verification when pointing "
        "llama_model_url at a different file.",
    )


class Config(BaseSettings):
    """Main configuration class."""

    model_config = SettingsConfigDict(
        env_prefix="CEMENTIC_",
    )

    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    llama_cpp: LlamaCppConfig = Field(default_factory=LlamaCppConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    source_watcher: SourceWatcherConfig = Field(default_factory=SourceWatcherConfig)
    pipeline_worker: PipelineWorkerConfig = Field(default_factory=PipelineWorkerConfig)
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)

    @model_validator(mode="after")
    def _chunks_must_fit_the_context_window(self) -> "Config":
        """Refuse a chunk_size / n_ctx pairing that *must* embed truncated.

        The invariant was previously only *tested* at the shipped defaults, so
        a user changing either knob got no check at all -- the exact drift that
        once truncated 93% of full-size chunks in silence.

        The bound here is deliberately the one that cannot be argued with: a
        chunk plus its task prefix that exceeds the window even at one model
        token per tiktoken token is too long however the text tokenizes. The
        *likely*-truncation band above it is not refused, because refusing it
        made `chunk_size = 352` -- a value measured to produce zero over-budget
        chunks on the live corpus, and the documented way to keep an index
        already built at it -- fail every command including `config show`.
        That band costs a tokenize round trip per chunk, not correctness, and
        `cementic doctor` reports it; truncation itself is caught exactly, per
        chunk, against the model's own tokenizer by
        ``RemoteEmbeddingClient.over_budget_tokens``.
        """
        if self.pipeline.embedding_provider != "llama-cpp":
            return self
        # Imported here, not at module scope: embedding_runtime imports config.
        from cementic.embedding_runtime import _TASK_PREFIX_TOKEN_ALLOWANCE

        smallest_possible = self.pipeline.chunk_size + _TASK_PREFIX_TOKEN_ALLOWANCE
        if smallest_possible > self.llama_cpp.n_ctx:
            raise ValueError(
                f"pipeline.chunk_size={self.pipeline.chunk_size} cannot fit "
                f"llama_cpp.n_ctx={self.llama_cpp.n_ctx}: a full chunk plus its "
                f"task prefix is at least {smallest_possible} tokens even if the "
                "model tokenizes as coarsely as tiktoken, so it would embed "
                "truncated. Use chunk_size <= "
                f"{self.llama_cpp.n_ctx - _TASK_PREFIX_TOKEN_ALLOWANCE}, or raise "
                "n_ctx only if the model's real context length allows it (the "
                "default Nomic model caps at 512)."
            )
        return self

    def _normalize_paths(self) -> None:
        """Expand ``~`` and anchor relative paths to the current directory.

        TOML and systemd ``Environment=`` do no shell expansion, so a
        hand-written ``~/Documents`` stayed a literal ``~`` directory created
        under the working directory. A relative path was worse than wrong: the
        artifacts root is used both to build the paths stored in the database
        and to resolve them again later, so a reader in a different directory
        checked containment against its own root and "removed" a file that was
        never there -- deleting the rows while the bytes stayed on disk.

        ``Path.cwd() / p`` rather than ``resolve()``: resolving also follows
        symlinks, which would rewrite paths under a symlinked temp or home
        directory into something the caller never configured.
        """

        def anchored(path: Path) -> Path:
            expanded = path.expanduser()
            return expanded if expanded.is_absolute() else Path.cwd() / expanded

        for section, field in (
            (self.source_watcher, "log_file"),
            (self.source_watcher, "state_path"),
            (self.pipeline_worker, "log_file"),
            (self.pipeline_worker, "state_path"),
            (self.llama_cpp, "daemon_pid_file"),
            (self.llama_cpp, "daemon_log_file"),
            (self.storage, "artifacts_path"),
        ):
            current = getattr(section, field)
            if current is not None:
                setattr(section, field, anchored(Path(current)))

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Set default paths based on data directory. Computing these paths must
        # not create the directory — read-only commands (e.g. `config show`)
        # should not write to disk; writers create it lazily when they need it.
        data_dir = Path(user_data_dir("cementic", ensure_exists=False))

        if self.source_watcher.log_file is None:
            self.source_watcher.log_file = data_dir / "source_watcher.log"

        if self.source_watcher.state_path is None:
            self.source_watcher.state_path = data_dir / "source_watcher_state.json"

        if self.pipeline_worker.log_file is None:
            self.pipeline_worker.log_file = data_dir / "pipeline_worker.log"

        if self.pipeline_worker.state_path is None:
            self.pipeline_worker.state_path = data_dir / "pipeline_worker_state.json"

        if self.llama_cpp.daemon_pid_file is None:
            self.llama_cpp.daemon_pid_file = data_dir / "llama_cpp_daemon.pid"

        if self.llama_cpp.daemon_log_file is None:
            self.llama_cpp.daemon_log_file = data_dir / "llama_cpp_daemon.log"

        if self.storage.artifacts_path is None:
            self.storage.artifacts_path = data_dir / "artifacts"

        # After the defaults are filled, so it covers both what the user set and
        # what we derived.
        self._normalize_paths()


def get_config() -> Config:
    """Build a Config, refusing the problems pydantic cannot see.

    ``Config()`` itself stays permissive so the library and the test suite can
    construct one freely; this is the entry point every command goes through,
    and the only place that can tell an ignored file or a dropped section from
    a deliberate default.
    """
    path_problem = config_path_error()
    if path_problem is not None:
        raise ConfigError(path_problem)
    # A file that cannot be parsed at all was the one config fault cementic
    # tolerated: load_config_file() swallowed it, returned {}, and every setting
    # in the file went with it, so a single stray character silently moved the
    # whole run onto built-in defaults -- a different database, a different
    # chunk_size -- behind one line of stderr that scrolls past under
    # `cementic start`. Every other fault here already exits non-zero.
    file_problem = config_file_error()
    if file_problem is not None:
        raise ConfigError(f"{resolve_config_path()}: {file_problem}")
    problems = config_file_problems()
    if problems:
        path = resolve_config_path()
        raise ConfigError(f"{path}: " + "; ".join(problems))
    return Config()
