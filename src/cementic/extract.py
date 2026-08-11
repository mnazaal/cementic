"""Document extraction: a content-type registry over per-format extractors."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# mypy: disable-error-code=import-untyped
import pymupdf

from cementic.config import Config

try:
    import pymupdf.layout  # noqa: F401
except ImportError:
    pass

import pymupdf4llm


def _get_rapidocr_api() -> Any | None:
    """Lazily import RapidOCR API adapter."""
    try:
        from pymupdf4llm.ocr import rapidocr_api

        return rapidocr_api
    except ImportError:
        return None


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

    if pymupdf._get_layout is None:
        raise RuntimeError(
            "pymupdf_layout is required for improved page layout analysis. "
            "Install it with: uv pip install pymupdf-layout"
        )

    ocr_kwargs: dict[str, Any] = {"use_ocr": use_ocr}
    if use_ocr:
        rapidocr_api = _get_rapidocr_api()
        if rapidocr_api is not None:
            ocr_kwargs["ocr_function"] = rapidocr_api.exec_ocr

    md_text = pymupdf4llm.to_markdown(
        str(path),
        header=False,
        footer=False,
        **ocr_kwargs,
    )

    if isinstance(md_text, str):
        return md_text

    return "\n\n".join(str(page_chunk) for page_chunk in md_text)


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


def _plaintext_extractor(path: str, config: Config) -> str:
    return Path(path).read_text(encoding="utf-8")


# The one place that maps content type -> extractor. Adding a document type is a
# single entry here; the watcher, worker, and CLI never branch on file type.
_EXTRACTORS: dict[str, tuple[ExtractorSpec, ExtractorFn]] = {
    "pymupdf4llm": (
        ExtractorSpec(name="pymupdf4llm", version=1, extensions=(".pdf",)),
        _pdf_extractor,
    ),
    "plaintext": (
        ExtractorSpec(name="plaintext", version=1, extensions=(".txt", ".md", ".markdown")),
        _plaintext_extractor,
    ),
}


def supported_extensions() -> frozenset[str]:
    """Every file extension some registered extractor can handle (lowercase)."""
    return frozenset(ext for spec, _ in _EXTRACTORS.values() for ext in spec.extensions)


def extractor_for(path: str, config: Config) -> tuple[str, ExtractorFn] | None:
    """Resolve a path to its extractor; None when no extractor handles the type.

    A `[extraction.backends]` entry for the file type (e.g. ``pdf = "docling"``)
    selects a specific extractor; otherwise the first registered one that handles
    the suffix is used. A configured backend that can't handle the type is a
    misconfiguration and raises.
    """
    suffix = Path(path).suffix.lower()
    chosen = config.extraction.backends.get(suffix.removeprefix("."))
    if chosen is not None:
        entry = _EXTRACTORS.get(chosen)
        if entry is None or suffix not in entry[0].extensions:
            raise ValueError(f"extraction backend '{chosen}' does not handle '{suffix}'")
        return chosen, entry[1]
    for name, (spec, fn) in _EXTRACTORS.items():
        if suffix in spec.extensions:
            return name, fn
    return None


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
    content = fn(path, config)
    if extraction_is_empty(content):
        raise ValueError(_empty_extraction_reason(path, config))
    return content


def extractor_registry_payload() -> list[dict[str, object]]:
    """Stable registry identity for profile fingerprinting (sorted by name)."""
    return sorted(
        ({"name": spec.name, "version": spec.version} for spec, _ in _EXTRACTORS.values()),
        key=lambda item: str(item["name"]),
    )
