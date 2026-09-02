"""Integration tests for PDF extraction module."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cementic.extract import (
    _get_rapidocr_api,
    _ocr_backend_installed,
    extract_pdf_markdown,
    ocr_backend_available,
)


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

        # OCR is stubbed rather than left to the environment: rapidocr is an
        # opt-in dependency now, so the default use_ocr=True would otherwise
        # make this test's outcome depend on whether it happens to be installed.
        with patch("pathlib.Path.exists", return_value=True):
            with patch("cementic.extract._get_rapidocr_api", return_value=MagicMock()):
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


class TestOcrBackendDetection:
    """Availability is decided by the engine package, not by the adapter."""

    def test_no_engine_package_means_no_ocr(self) -> None:
        with patch("cementic.extract._ocr_backend_installed", return_value=False):
            assert _get_rapidocr_api() is None
            assert ocr_backend_available() is False

    def test_an_importable_adapter_is_not_evidence_of_ocr(self) -> None:
        """The bug an ImportError check could not see.

        From pymupdf4llm 1.28 the adapter resolves its engine at import time and
        swallows the absence, so `from pymupdf4llm.ocr import rapidocr_api`
        succeeds with nothing behind it. Reading that as "OCR available" let
        extraction run without OCR and write the result into an immutable
        artifact -- the corpus then carried the degradation permanently.
        """
        with patch("cementic.extract.importlib.util.find_spec", return_value=None):
            with patch.dict(sys.modules, {"pymupdf4llm.ocr.rapidocr_api": MagicMock()}):
                assert _get_rapidocr_api() is None
                assert ocr_backend_available() is False

    def test_a_missing_adapter_module_is_still_handled(self) -> None:
        """Older pymupdf4llm raises ImportError instead; both must read alike."""
        with patch("cementic.extract._ocr_backend_installed", return_value=True):
            with patch.dict(sys.modules, {"pymupdf4llm.ocr": None}):
                assert _get_rapidocr_api() is None

    def test_either_engine_package_counts(self) -> None:
        found = {"rapidocr_onnxruntime"}
        with patch(
            "cementic.extract.importlib.util.find_spec",
            side_effect=lambda name: object() if name in found else None,
        ):
            assert _ocr_backend_installed() is True

    def test_engine_is_probed_without_importing_it(self) -> None:
        """`find_spec`, not `import`: importing rapidocr loads ONNX models."""
        asked: list[str] = []

        def fake_find_spec(name: str) -> None:
            asked.append(name)
            return None

        with patch("cementic.extract.importlib.util.find_spec", side_effect=fake_find_spec):
            assert _ocr_backend_installed() is False
        assert "rapidocr" in asked
