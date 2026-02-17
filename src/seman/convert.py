"""PDF to Markdown conversion using pymupdf4llm."""

from pathlib import Path
from typing import Optional, Tuple

import pymupdf

try:
    import pymupdf.layout  # noqa: F401
except ImportError:
    pass

import pymupdf4llm

try:
    from pymupdf4llm.ocr import rapidocr_api
except ImportError:
    rapidocr_api = None


def convert_pdf_to_markdown(
    pdf_path: str,
    pages: Optional[Tuple[int, int]] = None,
) -> str:
    """Convert PDF to Markdown using pymupdf4llm.

    Args:
        pdf_path: Path to PDF file
        pages: Optional tuple of (start_page, end_page) for partial conversion

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

    ocr_kwargs = {}
    if rapidocr_api is not None:
        ocr_kwargs = {
            "use_ocr": True,
            "ocr_function": rapidocr_api.exec_ocr,
        }

    # Convert PDF to markdown
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
