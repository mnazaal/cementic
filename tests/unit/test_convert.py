"""Tests for PDF conversion helpers."""

import builtins
import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import cementic.extract as extract_module
from cementic.extract import extract_pdf_markdown


def test_extract_module_tolerates_missing_layout_import() -> None:
    """Import-time layout hook is optional; runtime check reports missing layout."""
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "pymupdf.layout":
            raise ImportError("missing layout")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=guarded_import):
        reloaded = importlib.reload(extract_module)

    assert reloaded.extract_pdf_markdown is not None
    importlib.reload(extract_module)


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


def test_convert_joins_iterable_markdown_chunks(temp_dir: Path) -> None:
    """Non-string markdown result is joined into one string."""
    pdf_path = temp_dir / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    with patch("cementic.extract.pymupdf._get_layout", object()):
        with patch("cementic.extract.pymupdf4llm.to_markdown", return_value=["one", "two"]):
            result = extract_pdf_markdown(str(pdf_path), use_ocr=False)

    assert result == "one\n\ntwo"
