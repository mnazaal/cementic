"""Integration tests for PDF extraction module."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cementic.extract import extract_pdf_markdown


def _fake_pdf_stack(*, layout: object = object(), to_markdown: object = "markdown"):
    """Stand-ins for the modules ``_get_pymupdf`` resolves.

    Patched at that seam rather than on the real modules, because importing
    ``pymupdf.layout`` runs an ``activate()`` that rebinds
    ``pymupdf4llm.to_markdown`` -- so a patch applied before the first lazy
    import was silently replaced by the real function mid-test.
    """
    pymupdf = SimpleNamespace(_get_layout=layout)
    pymupdf4llm = SimpleNamespace(
        to_markdown=to_markdown
        if callable(to_markdown)
        else MagicMock(return_value=to_markdown)
    )
    return pymupdf, pymupdf4llm


class TestExtractErrorPaths:
    """Test extraction error handling paths."""

    def test_file_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="PDF not found"):
            extract_pdf_markdown("/nonexistent/path/file.pdf")

    def test_missing_pymupdf_layout_raises(self) -> None:
        """Layout not installed must be reported, not worked around."""
        stack = _fake_pdf_stack(layout=None)
        with patch("pathlib.Path.exists", return_value=True):
            with patch("cementic.extract._get_pymupdf", return_value=stack):
                with pytest.raises(RuntimeError, match="pymupdf_layout is required"):
                    extract_pdf_markdown("any.pdf")
        stack[1].to_markdown.assert_not_called()

    def test_non_string_return_type(self) -> None:
        """Page chunks are refused, not stringified.

        Unreachable today (page_chunks is never requested), but if the upstream
        contract changes they arrive as dicts, and joining their str() would
        store and embed the reprs as though they were the document.
        """
        stack = _fake_pdf_stack(to_markdown=["page 1 text", "page 2 text"])

        with patch("pathlib.Path.exists", return_value=True):
            with patch("cementic.extract._get_pymupdf", return_value=stack):
                with pytest.raises(RuntimeError, match="page chunks"):
                    extract_pdf_markdown("/fake/path.pdf")

    @patch("cementic.extract._get_rapidocr_api")
    def test_ocr_disabled(self, mock_ocr: MagicMock) -> None:
        """use_ocr=False skips OCR function lookup."""
        stack = _fake_pdf_stack(to_markdown="plain text")

        with patch("pathlib.Path.exists", return_value=True):
            with patch("cementic.extract._get_pymupdf", return_value=stack):
                result = extract_pdf_markdown("/fake/path.pdf", use_ocr=False)
        assert result == "plain text"
        mock_ocr.assert_not_called()

    @patch("cementic.extract._get_rapidocr_api")
    def test_ocr_enabled_with_rapidocr(self, mock_ocr: MagicMock) -> None:
        """use_ocr=True with OCR available: passes ocr_function."""
        mock_ocr.return_value = MagicMock()
        stack = _fake_pdf_stack(to_markdown="text with ocr")

        with patch("pathlib.Path.exists", return_value=True):
            with patch("cementic.extract._get_pymupdf", return_value=stack):
                result = extract_pdf_markdown("/fake/path.pdf", use_ocr=True)
        assert result == "text with ocr"
        assert "ocr_function" in stack[1].to_markdown.call_args.kwargs

    @patch("cementic.extract._get_rapidocr_api", return_value=None)
    def test_ocr_enabled_without_rapidocr(self, mock_ocr: MagicMock) -> None:
        """Configured OCR with no rapidocr is refused rather than degraded.

        It used to proceed without OCR, and the un-OCR'd text went into an
        immutable artifact -- so the corpus carried the degradation while the
        config still said OCR was on.
        """
        stack = _fake_pdf_stack(to_markdown="text")

        with patch("pathlib.Path.exists", return_value=True):
            with patch("cementic.extract._get_pymupdf", return_value=stack):
                with pytest.raises(RuntimeError, match="rapidocr"):
                    extract_pdf_markdown("/fake/path.pdf", use_ocr=True)
        stack[1].to_markdown.assert_not_called()


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
