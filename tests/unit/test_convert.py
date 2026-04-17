"""Tests for PDF conversion helpers."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cementic.extract import extract_pdf_markdown


def test_convert_uses_layout_and_disables_header_footer(temp_dir: Path) -> None:
    """Conversion enables layout support and disables headers/footers."""
    pdf_path = temp_dir / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    rapidocr = MagicMock()
    with patch("cementic.extract.pymupdf._get_layout", object()):
        with patch("cementic.extract._get_rapidocr_api", return_value=rapidocr):
            with patch(
                "cementic.extract.pymupdf4llm.to_markdown", return_value="markdown"
            ) as mock_md:
                result = extract_pdf_markdown(str(pdf_path))

    assert result == "markdown"
    mock_md.assert_called_once_with(
        str(pdf_path),
        pages=None,
        header=False,
        footer=False,
        use_ocr=True,
        ocr_function=rapidocr.exec_ocr,
    )


def test_convert_without_rapidocr_uses_default_ocr(temp_dir: Path) -> None:
    """Conversion still works when rapidocr bindings are unavailable."""
    pdf_path = temp_dir / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    with patch("cementic.extract.pymupdf._get_layout", object()):
        with patch("cementic.extract._get_rapidocr_api", return_value=None):
            with patch(
                "cementic.extract.pymupdf4llm.to_markdown", return_value="markdown"
            ) as mock_md:
                result = extract_pdf_markdown(str(pdf_path))

    assert result == "markdown"
    mock_md.assert_called_once_with(
        str(pdf_path),
        pages=None,
        header=False,
        footer=False,
        use_ocr=True,
    )


def test_convert_requires_pymupdf_layout(temp_dir: Path) -> None:
    """Conversion fails fast when layout package is unavailable."""
    pdf_path = temp_dir / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    with patch("cementic.extract.pymupdf._get_layout", None):
        with pytest.raises(RuntimeError, match="pymupdf_layout"):
            extract_pdf_markdown(str(pdf_path))
