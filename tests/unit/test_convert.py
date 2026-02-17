"""Tests for PDF conversion helpers."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from seman.convert import convert_pdf_to_markdown


def test_convert_uses_layout_and_disables_header_footer(temp_dir: Path) -> None:
    """Conversion enables layout support and disables headers/footers."""
    pdf_path = temp_dir / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    rapidocr = MagicMock()
    with patch("seman.convert.pymupdf._get_layout", object()):
        with patch("seman.convert.rapidocr_api", rapidocr):
            with patch("seman.convert.pymupdf4llm.to_markdown", return_value="markdown") as mock_md:
                result = convert_pdf_to_markdown(str(pdf_path))

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

    with patch("seman.convert.pymupdf._get_layout", object()):
        with patch("seman.convert.rapidocr_api", None):
            with patch("seman.convert.pymupdf4llm.to_markdown", return_value="markdown") as mock_md:
                result = convert_pdf_to_markdown(str(pdf_path))

    assert result == "markdown"
    mock_md.assert_called_once_with(
        str(pdf_path),
        pages=None,
        header=False,
        footer=False,
    )


def test_convert_requires_pymupdf_layout(temp_dir: Path) -> None:
    """Conversion fails fast when layout package is unavailable."""
    pdf_path = temp_dir / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    with patch("seman.convert.pymupdf._get_layout", None):
        with pytest.raises(RuntimeError, match="pymupdf_layout"):
            convert_pdf_to_markdown(str(pdf_path))
