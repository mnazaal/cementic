"""PDF to Markdown conversion using pymupdf4llm."""

from pathlib import Path
from typing import Optional, Tuple

import pymupdf4llm


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

    # Convert PDF to markdown
    md_text = pymupdf4llm.to_markdown(
        str(path),
        pages=pages,
    )

    return md_text
