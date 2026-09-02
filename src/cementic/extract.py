"""Document extraction: a content-type registry over per-format extractors."""

from __future__ import annotations

import contextlib
import importlib.util
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


def extract_pdf_markdown(pdf_path: str, use_ocr: bool = True) -> str:
    """Extract PDF content as Markdown using pymupdf4llm.

    Backend selection happens upstream in ``extractor_for`` (via the
    ``[extraction.backends]`` config map); this function is the resolved
    pymupdf4llm extractor, so it takes no backend argument.

    Args:
        pdf_path: Path to PDF file
        use_ocr: Whether to run OCR over the pages

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
    """Self-describing identity of an extractor (mirrors EmbeddingRuntimeSpec)."""

    name: str
    version: int
    extensions: tuple[str, ...]


# An extractor turns a file path into Markdown/plain text. Pure w.r.t. the file.
ExtractorFn = Callable[[str, Config], str]


def _pdf_extractor(path: str, config: Config) -> str:
    return extract_pdf_markdown(path, use_ocr=config.extraction.use_ocr)


def _raw_pdf_extractor(path: str, config: Config) -> str:
    return extract_pdf_text(path)


def _plaintext_extractor(path: str, config: Config) -> str:
    return Path(path).read_text(encoding="utf-8")


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


def supported_extensions() -> frozenset[str]:
    """Every file extension some registered extractor can handle (lowercase)."""
    return frozenset(ext for spec, _ in _EXTRACTORS.values() for ext in spec.extensions)


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
        if suffix in spec.extensions:
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


def extractor_registry_payload() -> list[dict[str, object]]:
    """Stable registry identity for profile fingerprinting (sorted by name)."""
    return sorted(
        ({"name": spec.name, "version": spec.version} for spec, _ in _EXTRACTORS.values()),
        key=lambda item: str(item["name"]),
    )
