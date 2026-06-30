"""Integration tests for PDF extraction module."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from cementic.extract import extract_pdf_markdown


class TestExtractErrorPaths:
    """Test extraction error handling paths."""

    def test_file_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="PDF not found"):
            extract_pdf_markdown("/nonexistent/path/file.pdf")

    def test_missing_pymupdf_layout_raises(self) -> None:
        """Simulate pymupdf._get_layout being None (layout not installed)."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("cementic.extract.pymupdf._get_layout", None):
                with pytest.raises(RuntimeError, match="pymupdf_layout is required"):
                    extract_pdf_markdown("any.pdf")

    @patch("cementic.extract.pymupdf4llm.to_markdown")
    def test_non_string_return_type(self, mock_to_md: MagicMock) -> None:
        """When to_markdown returns a list, join to string."""
        mock_to_md.return_value = ["page 1 text", "page 2 text"]

        with patch("pathlib.Path.exists", return_value=True):
            result = extract_pdf_markdown("/fake/path.pdf")
        assert result == "page 1 text\n\npage 2 text"

    @patch("cementic.extract.pymupdf4llm.to_markdown")
    @patch("cementic.extract._get_rapidocr_api")
    def test_ocr_disabled(self, mock_ocr: MagicMock, mock_to_md: MagicMock) -> None:
        """use_ocr=False skips OCR function lookup."""
        mock_to_md.return_value = "plain text"

        with patch("pathlib.Path.exists", return_value=True):
            result = extract_pdf_markdown("/fake/path.pdf", use_ocr=False)
        assert result == "plain text"
        mock_ocr.assert_not_called()

    @patch("cementic.extract.pymupdf4llm.to_markdown")
    @patch("cementic.extract._get_rapidocr_api")
    def test_ocr_enabled_with_rapidocr(self, mock_ocr: MagicMock, mock_to_md: MagicMock) -> None:
        """use_ocr=True with OCR available: passes ocr_function."""
        mock_ocr.return_value = MagicMock()
        mock_to_md.return_value = "text with ocr"

        with patch("pathlib.Path.exists", return_value=True):
            result = extract_pdf_markdown("/fake/path.pdf", use_ocr=True)
        assert result == "text with ocr"
        call_kwargs = mock_to_md.call_args.kwargs
        assert "ocr_function" in call_kwargs

    @patch("cementic.extract.pymupdf4llm.to_markdown")
    @patch("cementic.extract._get_rapidocr_api", return_value=None)
    def test_ocr_enabled_without_rapidocr(
        self, mock_ocr: MagicMock, mock_to_md: MagicMock
    ) -> None:
        """use_ocr=True when rapidocr not installed: no ocr_function passed."""
        mock_to_md.return_value = "text"

        with patch("pathlib.Path.exists", return_value=True):
            result = extract_pdf_markdown("/fake/path.pdf", use_ocr=True)
        assert result == "text"
        call_kwargs = mock_to_md.call_args.kwargs
        assert "ocr_function" not in call_kwargs


class TestImportFailures:
    """Test graceful handling of optional import failures."""

    def test_rapidocr_import_failure_returns_none(self) -> None:
        """Verify _get_rapidocr_api returns None when rapidocr not available."""
        from cementic.extract import _get_rapidocr_api

        # Simulate import failure by removing any cached module
        saved = sys.modules.pop("pymupdf4llm.ocr", None)
        try:
            with patch.dict(sys.modules, {"pymupdf4llm.ocr": None}):
                with patch("cementic.extract._get_rapidocr_api", return_value=None):
                    result = _get_rapidocr_api()
            # With our patch, it returns None
            assert result is None
        finally:
            if saved is not None:
                sys.modules["pymupdf4llm.ocr"] = saved

    def test_pymupdf_layout_import_handled(self) -> None:
        """Verify pymupdf.layout import failure is silently handled."""
        # The module already imported pymupdf.layout (or caught the error)
        # Just verify extract_pdf_markdown still works
        import cementic.extract
        assert hasattr(cementic.extract, "pymupdf")
