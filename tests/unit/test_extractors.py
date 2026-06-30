"""Tests for the content-type extractor registry (Move 4)."""

from unittest.mock import patch

import pytest

from cementic import extract as extract_mod
from cementic.config import Config
from cementic.extract import (
    ExtractorSpec,
    extract_document,
    extractor_for,
    extractor_registry_payload,
    supported_extensions,
)
from cementic.profiles import build_extractor_profile_payload
from cementic.source_watcher import DocumentEventHandler


@pytest.fixture
def config() -> Config:
    return Config()


def test_supported_extensions_covers_pdf_and_text():
    exts = supported_extensions()
    assert {".pdf", ".txt", ".md", ".markdown"} <= exts


def test_extractor_for_resolves_by_suffix(config):
    assert extractor_for("/docs/a.pdf", config)[0] == "pymupdf4llm"
    assert extractor_for("/notes/b.MD", config)[0] == "plaintext"  # case-insensitive
    assert extractor_for("/x/c.docx", config) is None


def test_extract_document_reads_plain_text(tmp_path, config):
    f = tmp_path / "note.md"
    f.write_text("# Heading\n\nbody", encoding="utf-8")
    assert extract_document(str(f), config) == "# Heading\n\nbody"


def test_extract_document_dispatches_pdf(config):
    with patch("cementic.extract.extract_pdf_markdown", return_value="PDF MD") as mock_pdf:
        assert extract_document("/docs/paper.pdf", config) == "PDF MD"
    assert mock_pdf.call_args.kwargs["use_ocr"] == config.extraction.use_ocr


def test_extract_document_raises_for_unsupported(config):
    with pytest.raises(ValueError, match="no extractor"):
        extract_document("/x/archive.zip", config)


def test_registry_payload_is_sorted_and_named():
    payload = extractor_registry_payload()
    names = [entry["name"] for entry in payload]
    assert names == sorted(names)
    assert {"plaintext", "pymupdf4llm"} <= set(names)


def test_extractor_profile_payload_records_registry(config):
    payload = build_extractor_profile_payload(config)
    assert payload["backends"] == config.extraction.backends
    assert payload["extractors"] == extractor_registry_payload()


def test_config_selects_a_specific_backend(monkeypatch, config):
    """`[extraction.backends]` chooses which extractor handles a file type."""
    monkeypatch.setitem(
        extract_mod._EXTRACTORS,
        "docling",
        (ExtractorSpec(name="docling", version=1, extensions=(".pdf",)), lambda p, c: "DOCLING"),
    )
    # default picks the first registered pdf extractor (pymupdf4llm)
    assert extractor_for("/docs/a.pdf", config)[0] == "pymupdf4llm"
    # an explicit choice wins
    config.extraction.backends = {"pdf": "docling"}
    assert extractor_for("/docs/a.pdf", config)[0] == "docling"
    assert extract_document("/docs/a.pdf", config) == "DOCLING"


def test_misconfigured_backend_raises(config):
    config.extraction.backends = {"pdf": "plaintext"}  # plaintext can't read pdf
    with pytest.raises(ValueError, match="does not handle"):
        extract_document("/docs/a.pdf", config)


def test_adding_one_registry_entry_is_all_it_takes(monkeypatch, config):
    """A new content type = one _EXTRACTORS entry; watcher + dispatch follow."""
    monkeypatch.setitem(
        extract_mod._EXTRACTORS,
        "fake",
        (ExtractorSpec(name="fake", version=1, extensions=(".xyz",)), lambda p, c: "FAKE"),
    )
    assert ".xyz" in supported_extensions()
    assert extract_document("/some/file.xyz", config) == "FAKE"
    # the watcher picks it up with no change of its own
    handler = DocumentEventHandler(lambda _p: None)
    assert handler._should_process("/some/file.xyz") is True
