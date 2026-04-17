"""PDF extraction helpers using pymupdf4llm."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# mypy: disable-error-code=import-untyped
import pymupdf

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


def extract_pdf_markdown(
    pdf_path: str,
    pages: tuple[int, int] | None = None,
    backend: str = "pymupdf4llm",
    use_ocr: bool = True,
) -> str:
    """Extract PDF content as Markdown using pymupdf4llm.

    Args:
        pdf_path: Path to PDF file
        pages: Optional tuple of (start_page, end_page) for partial conversion

    Returns:
        Markdown content as string
    """
    if backend != "pymupdf4llm":
        raise ValueError(f"Unsupported extraction backend: {backend}")

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
        pages=pages,
        header=False,
        footer=False,
        **ocr_kwargs,
    )

    if isinstance(md_text, str):
        return md_text

    return "\n\n".join(str(page_chunk) for page_chunk in md_text)
