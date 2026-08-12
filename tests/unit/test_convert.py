"""Tests for PDF conversion helpers.

The PDF stack is imported on first use rather than at module scope, and that
first import runs an ``activate()`` which rebinds ``pymupdf4llm.to_markdown``.
Patching the real modules is therefore unstable -- the patch survives or not
depending on whether some earlier test already triggered the import. These patch
``_get_pymupdf``, the seam the lazy import goes through.
"""

import builtins
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cementic.extract import _get_pymupdf, extract_pdf_markdown


def _fake_pdf_stack(*, layout: object = object(), to_markdown: object = "markdown"):
    """Stand-ins for the modules ``_get_pymupdf`` resolves."""
    pymupdf = SimpleNamespace(_get_layout=layout)
    pymupdf4llm = SimpleNamespace(
        to_markdown=to_markdown
        if callable(to_markdown)
        else MagicMock(return_value=to_markdown)
    )
    return pymupdf, pymupdf4llm


def test_layout_import_stays_optional() -> None:
    """The layout hook is optional; the runtime check reports it missing."""
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "pymupdf.layout":
            raise ImportError("missing layout")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=guarded_import):
        pymupdf, pymupdf4llm = _get_pymupdf()

    assert pymupdf is not None
    assert pymupdf4llm is not None


def test_convert_uses_layout_and_disables_header_footer(temp_dir: Path) -> None:
    """Conversion enables layout support and disables headers/footers."""
    pdf_path = temp_dir / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    rapidocr = MagicMock()
    stack = _fake_pdf_stack()
    with patch("cementic.extract._get_pymupdf", return_value=stack):
        with patch("cementic.extract._get_rapidocr_api", return_value=rapidocr):
            result = extract_pdf_markdown(str(pdf_path))

    assert result == "markdown"
    stack[1].to_markdown.assert_called_once_with(
        str(pdf_path),
        header=False,
        footer=False,
        use_ocr=True,
        ocr_function=rapidocr.exec_ocr,
    )


def test_configured_ocr_without_rapidocr_is_refused(temp_dir: Path) -> None:
    """Silently falling back to no OCR wrote the un-OCR'd result into an
    immutable artifact, so the whole corpus carried the degradation with the
    config still claiming OCR was on."""
    pdf_path = temp_dir / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    with patch("cementic.extract._get_pymupdf", return_value=_fake_pdf_stack()):
        with patch("cementic.extract._get_rapidocr_api", return_value=None):
            with pytest.raises(RuntimeError, match="rapidocr"):
                extract_pdf_markdown(str(pdf_path), use_ocr=True)


def test_convert_requires_pymupdf_layout(temp_dir: Path) -> None:
    """Conversion fails fast when layout package is unavailable."""
    pdf_path = temp_dir / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    with patch("cementic.extract._get_pymupdf", return_value=_fake_pdf_stack(layout=None)):
        with pytest.raises(RuntimeError, match="pymupdf_layout"):
            extract_pdf_markdown(str(pdf_path))


def test_page_chunks_are_refused_rather_than_stringified(temp_dir: Path) -> None:
    """Unreachable today, since page_chunks is never requested. If the upstream
    contract changes, page chunks arrive as dicts and the old code str()-joined
    them -- storing and embedding their reprs as though they were the document."""
    pdf_path = temp_dir / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    stack = _fake_pdf_stack(to_markdown=["one", "two"])
    with patch("cementic.extract._get_pymupdf", return_value=stack):
        with pytest.raises(RuntimeError, match="page chunks"):
            extract_pdf_markdown(str(pdf_path), use_ocr=False)
