"""Document extraction: a content-type registry over per-format extractors."""

from __future__ import annotations

import contextlib
import importlib.util
import shlex
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# mypy: disable-error-code=import-untyped
from cementic.config import Config


def _get_pymupdf() -> tuple[Any, Any]:
    """Import the PDF stack on first use, returning ``(pymupdf, pymupdf4llm)``.

    Deliberately not at module scope. ``pymupdf.layout`` loads an ONNX layout
    model and networkx, which costs about half a second warm and over a second
    cold -- and `cli.py` imports this module, so `cementic --version` paid it too.
    The runtime-budget tests could not see it: they time dispatch after the test
    module has already imported the CLI.
    """
    import pymupdf

    try:
        import pymupdf.layout  # noqa: F401
    except ImportError:
        pass

    import pymupdf4llm

    return pymupdf, pymupdf4llm


def _get_pymupdf_bare() -> Any:
    """Import PyMuPDF *without* ``pymupdf.layout``.

    Not a duplicate of ``_get_pymupdf``: the layout import is the entire cost
    the raw backend exists to avoid, and it loads its ONNX model on import,
    before any page is analysed. The ``Any`` return is load-bearing under
    strict mypy -- pymupdf is untyped, and this seam is also what the
    never-loads-the-layout-model test patches.
    """
    import pymupdf

    return pymupdf


#: Where the OCR backend comes from. rapidocr is an opt-in dependency rather
#: than a declared one (see pyproject.toml), so the version floor lives here,
#: next to the check that needs it, instead of in a resolver that never sees
#: it. Named once and used by both the extraction error and `cementic doctor`.
OCR_INSTALL_HINT = (
    "install it into the same environment: "
    "`uv tool install --with 'rapidocr>=3.6.0' <cementic source>` "
    "-- re-listing any other --with pins, which uv replaces rather than "
    "merges -- or `uv pip install 'rapidocr>=3.6.0'` for a plain virtualenv"
)


#: The packages that can actually recognise text. pymupdf4llm's adapter accepts
#: either, and asking about them by name is what makes the check survive the
#: adapter being rewritten between pymupdf4llm releases.
_OCR_BACKEND_PACKAGES = ("rapidocr", "rapidocr_onnxruntime")


def _ocr_backend_installed() -> bool:
    """Whether an OCR engine package is importable, without importing it.

    `find_spec` rather than an import: importing rapidocr loads onnxruntime and
    its models, which is seconds of work and megabytes of memory to answer a
    yes/no question `cementic doctor` asks on every run.
    """
    for name in _OCR_BACKEND_PACKAGES:
        try:
            if importlib.util.find_spec(name) is not None:
                return True
        except (ImportError, ValueError):
            continue
    return False


def _get_rapidocr_api() -> Any | None:
    """Return pymupdf4llm's RapidOCR adapter, or None when OCR cannot run.

    cementic never imports rapidocr itself: pymupdf4llm owns the adapter and
    this hands its ``exec_ocr`` over as a callback. That indirection is what
    lets rapidocr be an opt-in dependency -- but it is also why the engine is
    checked for separately rather than inferred from whether the adapter
    imports.

    Both readings of that import are wrong on some pymupdf4llm version. Up to
    0.3.4 the adapter does a top-level `import rapidocr`, so ImportError does
    mean "no OCR". By 1.28.2 it resolves its engine at import time and swallows
    the absence, so the module imports *successfully* with nothing behind it --
    and extraction then writes un-OCR'd text into an immutable artifact, the
    very failure `extract_pdf_markdown` raises to prevent. Asking about the
    engine package by name is the one question both versions answer the same
    way.

    The adapter also prints its chosen engine to stdout as it loads, and
    `cementic extract` puts document text on stdout, so the import runs under
    the same redirect extraction uses.
    """
    if not _ocr_backend_installed():
        return None
    with contextlib.redirect_stdout(sys.stderr):
        try:
            from pymupdf4llm.ocr import rapidocr_api
        except ImportError:
            return None
    return rapidocr_api


def ocr_backend_available() -> bool:
    """Whether OCR could actually run right now (no side effects)."""
    return _get_rapidocr_api() is not None


def extract_pdf_markdown(pdf_path: str, use_ocr: bool = False) -> str:
    """Extract PDF content as Markdown using pymupdf4llm.

    Backend selection happens upstream in ``extractor_for`` (via the
    ``[extraction.backends]`` config map); this function is the resolved
    pymupdf4llm extractor, so it takes no backend argument.

    Args:
        pdf_path: Path to PDF file
        use_ocr: Whether to run OCR over the pages. Defaults to off, matching
            ``extraction.use_ocr``; the two disagreeing meant the only honest
            reading of a bare call was "whichever default you happen to hit".

    Returns:
        Markdown content as string
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pymupdf, pymupdf4llm = _get_pymupdf()
    if pymupdf._get_layout is None:
        raise RuntimeError(
            "pymupdf_layout is required for improved page layout analysis. "
            "Install it with: uv pip install pymupdf-layout"
        )

    ocr_kwargs: dict[str, Any] = {"use_ocr": use_ocr}
    if use_ocr:
        rapidocr_api = _get_rapidocr_api()
        if rapidocr_api is None:
            raise RuntimeError(
                "extraction.use_ocr is enabled but no OCR backend is "
                "installed. Extraction would silently fall back to no OCR and "
                "write the result into an immutable artifact, so the whole "
                "corpus would carry it. Either set extraction.use_ocr = false, "
                f"or {OCR_INSTALL_HINT}."
            )
        ocr_kwargs["ocr_function"] = rapidocr_api.exec_ocr

    # pymupdf4llm writes its own OCR notices to stdout. `cementic extract` puts
    # the markdown on stdout, so in `extract | chunk` an upstream print becomes
    # document content. Send anything it prints to stderr instead.
    with contextlib.redirect_stdout(sys.stderr):
        md_text = pymupdf4llm.to_markdown(
            str(path),
            header=False,
            footer=False,
            **ocr_kwargs,
        )

    if not isinstance(md_text, str):
        # Only reachable if the upstream contract changes: page chunks come back
        # as dicts, and the previous code str()-joined them, so their reprs
        # would have been stored and embedded as though they were the document.
        raise RuntimeError(
            "pymupdf4llm returned page chunks rather than markdown text; "
            "cementic never requests page_chunks"
        )
    return md_text


def extract_pdf_text(pdf_path: str) -> str:
    """Extract a PDF's text layer with no layout analysis.

    Measured over 153 academic papers against the pymupdf4llm backend: 37 ms
    per document against 10.2 s, and 5 s of CPU in total against 4.3 core-hours
    -- the difference being an ONNX layout model run over every page. It is
    also the more predictable of the two: the worst document here took 0.54 s,
    where the layout path spends 42 s on a four-page paper.

    What it gives up is Markdown structure -- headings, tables, header/footer
    stripping. ``chunk_text`` reads none of that: it slices a fixed token
    window regardless. On 38 metadata-title queries the two backends landed
    within one document of each other (n is small; that rules out a large
    quality gap, not a small one), so pymupdf4llm remains the default and this
    is the cheap path for a born-digital corpus.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pymupdf = _get_pymupdf_bare()
    with pymupdf.open(str(path)) as document:
        return "\n".join(page.get_text() for page in document)


@dataclass(frozen=True)
class ExtractorSpec:
    """Self-describing identity of an extractor (mirrors EmbeddingRuntimeSpec).

    ``extensions`` is None for an extractor whose file types come from config
    rather than from the registry. Such an extractor is never a fallback --
    `extractor_for` skips it unless `[extraction.backends]` names it -- because
    "handles anything" would otherwise make it the default for everything.
    """

    name: str
    version: int
    extensions: tuple[str, ...] | None


# An extractor turns a file path into Markdown/plain text. Pure w.r.t. the file.
ExtractorFn = Callable[[str, Config], str]


def _pdf_extractor(path: str, config: Config) -> str:
    return extract_pdf_markdown(path, use_ocr=config.extraction.use_ocr)


def _raw_pdf_extractor(path: str, config: Config) -> str:
    return extract_pdf_text(path)


def _plaintext_extractor(path: str, config: Config) -> str:
    return Path(path).read_text(encoding="utf-8")


#: How long an extraction command may run before it is killed. Generous: OCR
#: over a long scanned document is minutes of work, and the alternative to
#: waiting is a half-extracted corpus.
COMMAND_TIMEOUT_SECONDS = 900.0


def _command_extractor(path: str, config: Config) -> str:
    """Run the configured command for this file type and take its stdout.

    The whole contract is argv in, text on stdout. Anything richer -- a tool
    that writes files, or one needing a pipeline -- belongs in a wrapper script
    the operator owns, which is the composition boundary this extractor exists
    to offer rather than to erase.
    """
    file_type = normalize_backend_file_type(Path(path).suffix.lower())
    argv = config.extraction.commands.get(file_type)
    if not argv:
        # Unreachable through config (validation pairs backends with commands),
        # so this is for a Config assembled in code.
        raise ValueError(
            f"no [extraction.commands] entry for '{file_type}'; the 'command' "
            "extractor cannot run without one"
        )
    rendered = [argument.format_map({"path": path}) for argument in argv]
    try:
        completed = subprocess.run(
            rendered,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"extraction command timed out after {COMMAND_TIMEOUT_SECONDS:g}s: "
            f"{shlex.join(rendered)}"
        )
    except OSError as error:
        raise RuntimeError(f"extraction command could not run ({error}): {rendered[0]}")
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip().splitlines()
        raise RuntimeError(
            f"extraction command exited {completed.returncode}"
            + (f": {detail[-1]}" if detail else "")
            + f" [{shlex.join(rendered)}]"
        )
    return completed.stdout


#: The registry name that selects the external-command extractor. Named so
#: profiles.py can ask "is this file type extracted by a command" without
#: repeating the literal.
COMMAND_EXTRACTOR_NAME = "command"


#: A version probe answers in milliseconds or it is broken. Short, unlike
#: COMMAND_TIMEOUT_SECONDS, because this one runs during revision creation and
#: a hang there stalls indexing before any document is touched.
VERSION_PROBE_TIMEOUT_SECONDS = 30.0


def command_version(argv: list[str]) -> str:
    """What the extraction tool reports about itself, for the fingerprint.

    stdout and stderr are both taken, and a non-zero exit is not an error: tools
    disagree about all of it -- `mutool -v` prints to stderr, `gs --version` to
    stdout -- and the string only has to *change when the tool changes*, not to
    be well-formed. A tool that cannot answer at all is a hard error, because
    the alternative is a profile that cannot notice its extractor was upgraded.

    This is also why the version command is operator-supplied. `pdftotext
    --version` reads the flag as a filename and exits 0 with an I/O error, so a
    guessed flag records a constant that never moves on upgrade -- worse than
    no probe, because it looks like one.
    """
    try:
        completed = subprocess.run(
            argv, capture_output=True, text=True, timeout=VERSION_PROBE_TIMEOUT_SECONDS
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"version command timed out after {VERSION_PROBE_TIMEOUT_SECONDS:g}s: "
            f"{shlex.join(argv)}"
        )
    except OSError as error:
        raise RuntimeError(f"version command could not run ({error}): {argv[0]}")
    reported = (completed.stdout + completed.stderr).strip()
    if not reported:
        raise RuntimeError(
            f"version command printed nothing: {shlex.join(argv)}. Its output is "
            "the extraction fingerprint's only record of which build produced "
            "the text, so an empty answer would let a tool upgrade rewrite the "
            "corpus under an unchanged revision"
        )
    return reported


# The one place that maps content type -> extractor. Adding a document type is a
# single entry here; the watcher, worker, and CLI never branch on file type.
_EXTRACTORS: dict[str, tuple[ExtractorSpec, ExtractorFn]] = {
    "pymupdf4llm": (
        ExtractorSpec(name="pymupdf4llm", version=1, extensions=(".pdf",)),
        _pdf_extractor,
    ),
    # Order matters: `extractor_for` falls back to the first entry handling a
    # suffix, so this must stay below pymupdf4llm to leave the .pdf default
    # alone. Selected with `[extraction.backends] pdf = "pymupdf-raw"`.
    "pymupdf-raw": (
        ExtractorSpec(name="pymupdf-raw", version=1, extensions=(".pdf",)),
        _raw_pdf_extractor,
    ),
    "plaintext": (
        ExtractorSpec(name="plaintext", version=1, extensions=(".txt", ".md", ".markdown")),
        _plaintext_extractor,
    ),
    # extensions=None: this one handles whatever [extraction.commands] names,
    # which is what lets it add a file type no built-in extractor knows.
    COMMAND_EXTRACTOR_NAME: (
        ExtractorSpec(name=COMMAND_EXTRACTOR_NAME, version=1, extensions=None),
        _command_extractor,
    ),
}


#: The one registered extractor that runs OCR. `use_ocr` is inert under any
#: other backend -- pymupdf-raw reads the text layer and never rasterises a
#: page -- and an unset pdf backend falls through to this one, per the ordering
#: `_EXTRACTORS` documents above.
_OCR_CAPABLE_EXTRACTOR = "pymupdf4llm"


def ocr_would_run(config: Config) -> bool:
    """Whether the configured PDF backend would actually invoke OCR (pure).

    `use_ocr` alone does not mean OCR happens: it reaches extraction only
    through the pymupdf4llm backend. Enabling the flag while the configured
    pdf backend is something else is a silent no-op, which is worth reporting
    as its own state rather than reading as "OCR is on".
    """
    if not config.extraction.use_ocr:
        return False
    chosen = config.extraction.backends.get("pdf")
    return chosen is None or chosen == _OCR_CAPABLE_EXTRACTOR


def supported_extensions(config: Config | None = None) -> frozenset[str]:
    """Every file extension some extractor can handle (lowercase).

    Takes a Config because the ``command`` extractor's file types are config,
    not registry: without it, a command backend could only re-handle types a
    built-in extractor already claims, which forecloses adding a new one. None
    means the built-in extractors alone -- the right answer for callers asking
    what cementic can do before any config is in hand.
    """
    extensions = {
        ext for spec, _ in _EXTRACTORS.values() if spec.extensions for ext in spec.extensions
    }
    if config is not None:
        extensions |= {f".{file_type}" for file_type in config.extraction.commands}
    return frozenset(extensions)


def normalize_backend_file_type(file_type: str) -> str:
    """Canonical form of a ``[extraction.backends]`` key (pure).

    Keys are looked up as a bare lowercase suffix, so ``PDF`` and ``.pdf`` used
    to match nothing and be silently ignored -- while still changing the
    extractor profile's fingerprint, forcing a re-extraction that did exactly
    what the old one did.
    """
    return file_type.strip().lstrip(".").lower()


def backend_choice_error(file_type: str, extractor_name: str) -> str | None:
    """Why ``extractor_name`` cannot serve ``file_type``, or None (pure).

    Two separate mistakes, which used to share one message: naming an extractor
    that does not exist reported that it "does not handle '.pdf'", as though a
    typo'd name were a capability problem.
    """
    entry = _EXTRACTORS.get(extractor_name)
    if entry is None:
        return (
            f"unknown extraction backend '{extractor_name}' for '{file_type}'. "
            f"Available: {', '.join(sorted(_EXTRACTORS))}"
        )
    if entry[0].extensions is None:
        # A config-driven extractor: whether it handles this type is decided by
        # its own config table, which ExtractionConfig validates as a pair with
        # `commands`. Checking a static extension list here would reject every
        # legitimate use.
        return None
    suffix = f".{normalize_backend_file_type(file_type)}"
    if suffix not in entry[0].extensions:
        return (
            f"extraction backend '{extractor_name}' does not handle '{suffix}'. "
            f"It handles: {', '.join(sorted(entry[0].extensions))}"
        )
    return None


def extractor_for(path: str, config: Config) -> tuple[str, ExtractorFn] | None:
    """Resolve a path to its extractor; None when no extractor handles the type.

    A `[extraction.backends]` entry for the file type (e.g. ``pdf = "docling"``)
    selects a specific extractor; otherwise the first registered one that handles
    the suffix is used. Config validation rejects an unusable entry up front, so
    the check here is a backstop for a Config assembled in code.
    """
    suffix = Path(path).suffix.lower()
    file_type = normalize_backend_file_type(suffix)
    chosen = config.extraction.backends.get(file_type)
    if chosen is not None:
        problem = backend_choice_error(file_type, chosen)
        if problem is not None:
            raise ValueError(problem)
        return chosen, _EXTRACTORS[chosen][1]
    for name, (spec, fn) in _EXTRACTORS.items():
        if spec.extensions is not None and suffix in spec.extensions:
            return name, fn
    return None


def strip_unstorable(content: str) -> str:
    """Remove characters PostgreSQL cannot hold in a text column (pure).

    PostgreSQL rejects NUL (0x00) in `text`/`varchar` outright, so a document
    carrying one takes down the write rather than the document: the pipeline
    worker's chunk step raises `ValueError: A string literal cannot contain NUL
    (0x00) characters`, retries, and the whole collection stalls on it.

    20% of a 153-paper sample hit this via `pymupdf-raw` -- PDF text layers
    carry stray NULs and `page.get_text()` passes them through, where
    pymupdf4llm happens to filter them. Applied at the single dispatch rather
    than in one extractor, because the constraint belongs to the storage
    layer every extractor feeds, not to any one of them.

    Only NUL is removed. Other control characters store fine, and stripping
    them would be a content decision this function has no business making.
    """
    return content.replace("\x00", "")


def extraction_is_empty(content: str) -> bool:
    """Whether an extraction produced nothing usable (pure)."""
    return not content.strip()


def _empty_extraction_reason(path: str, config: Config) -> str:
    """Explain an empty extraction, naming the knob most likely responsible."""
    if Path(path).suffix.lower() == ".pdf" and not config.extraction.use_ocr:
        return (
            "extracted no text: the PDF has no text layer (a scan or images), "
            "and extraction.use_ocr is off"
        )
    return "extracted no text: the file has no readable text content"


def extract_document(path: str, config: Config) -> str:
    """Extract any supported document to Markdown/text — the single dispatch.

    An empty result is an error, not an empty success. Recorded as success it
    became a `done` extraction with zero chunks, which satisfies every
    completeness check -- so a directory of scanned PDFs reported no failures,
    reached `ready`, and promoted an index with nothing in it.
    """
    resolved = extractor_for(path, config)
    if resolved is None:
        raise ValueError(f"no extractor for '{Path(path).suffix.lower() or '(none)'}'")
    _name, fn = resolved
    content = strip_unstorable(fn(path, config))
    if extraction_is_empty(content):
        raise ValueError(_empty_extraction_reason(path, config))
    return content


def extractor_registry_payload(config: Config | None = None) -> list[dict[str, object]]:
    """Registry identity for profile fingerprinting (sorted by name).

    Adding or revving a built-in extractor re-versions every revision, and
    should: `extractor_for` falls back to the first registered extractor
    handling a suffix, so the *set* of them decides what an unconfigured corpus
    extracts with.

    A config-driven extractor (``extensions is None``) is excluded unless
    `backends` names it, because it takes no part in that fallback -- its
    presence cannot change what any other corpus extracted. Including it would
    re-version every existing revision on the day it was added, forcing a
    rebuild to record an extractor none of them can have used.
    """
    chosen = set(config.extraction.backends.values()) if config is not None else set()
    return sorted(
        (
            {"name": spec.name, "version": spec.version}
            for spec, _ in _EXTRACTORS.values()
            if spec.extensions is not None or spec.name in chosen
        ),
        key=lambda item: str(item["name"]),
    )
