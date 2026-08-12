"""Tests for the content-type extractor registry (Move 4)."""

from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from cementic import extract as extract_mod
from cementic.config import Config, ExtractionConfig
from cementic.extract import (
    ExtractorSpec,
    backend_choice_error,
    extract_document,
    extraction_is_empty,
    extractor_for,
    extractor_registry_payload,
    normalize_backend_file_type,
    supported_extensions,
)
from cementic.profiles import build_extractor_profile_payload
from cementic.source_watcher import DocumentEventHandler, _is_missing


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


class TestBackendKeyHandling:
    """`[extraction.backends]` keys were taken bare, lowercase and unchecked.

    A key like `PDF` or `.pdf` matched nothing at lookup time and was silently
    ignored -- while still entering the extractor profile's fingerprint, so it
    forced a re-extraction that produced exactly what the previous one did.
    """

    @pytest.mark.parametrize("written", ["pdf", ".pdf", "PDF", ".PDF", "  pdf  "])
    def test_the_ways_a_file_type_can_be_written_all_normalise(self, written):
        assert normalize_backend_file_type(written) == "pdf"

    def test_an_unknown_name_is_not_reported_as_a_capability_problem(self):
        problem = backend_choice_error("pdf", "pymypdf4llm")
        assert problem is not None
        assert "unknown extraction backend" in problem
        # Naming the alternatives is the point: the mistake is a typo.
        assert "pymupdf4llm" in problem

    def test_a_real_extractor_for_the_wrong_type_says_what_it_handles(self):
        problem = backend_choice_error("txt", "pymupdf4llm")
        assert problem is not None
        assert "does not handle '.txt'" in problem
        assert ".pdf" in problem

    def test_a_usable_pairing_has_no_complaint(self):
        assert backend_choice_error("pdf", "pymupdf4llm") is None

    def test_a_dotted_key_now_actually_selects_the_backend(self, config):
        """Previously ignored in silence, falling back to the registry default.

        Normalisation lives in the config validator, so the entry has to arrive
        through it -- assigning the dict afterwards skips validation entirely.
        """
        validated = ExtractionConfig.model_validate({"backends": {".PDF": "pymupdf4llm"}})
        config.extraction = validated

        assert validated.backends == {"pdf": "pymupdf4llm"}
        assert extractor_for("/docs/a.pdf", config)[0] == "pymupdf4llm"

    def test_an_unusable_entry_is_refused_at_config_time(self):
        """One error at load, rather than one failed document at a time."""
        with pytest.raises(ValidationError, match="does not handle"):
            ExtractionConfig.model_validate({"backends": {"txt": "pymupdf4llm"}})

    def test_the_same_type_written_two_ways_is_a_conflict(self):
        """`pdf` and `.pdf` used to be separate keys, so one silently won."""
        with pytest.raises(ValidationError, match="configured twice"):
            ExtractionConfig.model_validate(
                {"backends": {"pdf": "pymupdf4llm", ".pdf": "plaintext"}}
            )


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


class TestEmptyExtractionIsAFailure:
    """An empty extraction must not be recorded as a successful one.

    Recorded as success it becomes a `done` extraction with zero chunks, which
    satisfies every completeness check: a directory of scanned PDFs then reports
    no failures, reaches `ready`, and promotes an index with nothing in it.
    """

    @pytest.mark.parametrize(
        ("content", "empty"),
        [("", True), ("   \n\t ", True), ("x", False), ("# Heading", False)],
    )
    def test_emptiness_predicate(self, content, empty):
        assert extraction_is_empty(content) is empty

    def test_whitespace_only_document_is_rejected(self, tmp_path, config):
        f = tmp_path / "blank.md"
        f.write_text("   \n\n\t", encoding="utf-8")

        with pytest.raises(ValueError, match="extracted no text"):
            extract_document(str(f), config)

    def test_pdf_message_names_the_ocr_setting(self, config):
        """The likeliest cause is a scan with OCR off, so say which knob to change."""
        config.extraction.use_ocr = False

        with patch("cementic.extract.extract_pdf_markdown", return_value=""):
            with pytest.raises(ValueError, match="extraction.use_ocr is off"):
                extract_document("/docs/scan.pdf", config)

    def test_non_pdf_message_does_not_blame_ocr(self, tmp_path, config):
        f = tmp_path / "blank.txt"
        f.write_text("", encoding="utf-8")

        with pytest.raises(ValueError) as excinfo:
            extract_document(str(f), config)
        assert "use_ocr" not in str(excinfo.value)


class TestWatcherIgnoresRelativeToTheRoot:
    """Ignored names must be matched below the watched root, not above it.

    Testing the whole absolute path also tested the root's own ancestors, so
    watching a directory that happens to live under an ignored name indexed
    everything on the initial scan -- which walks down from the root and never
    looks up -- then silently dropped every create, modify and delete event for
    the life of the process.
    """

    IGNORED = "node_" + "modules"

    def _handler(self, roots):
        return DocumentEventHandler(
            lambda _p: None,
            ignore_directories={self.IGNORED, "build", ".git"},
            watched_roots=roots,
        )

    def test_a_root_under_an_ignored_name_still_processes_its_files(self):
        handler = self._handler([Path("/srv/build/docs")])

        assert handler._should_process("/srv/build/docs/paper.pdf") is True

    def test_an_ignored_directory_below_the_root_is_still_skipped(self):
        handler = self._handler([Path("/srv/build/docs")])

        assert handler._should_process(f"/srv/build/docs/{self.IGNORED}/readme.md") is False

    def test_paths_outside_every_root_fall_back_to_checking_the_whole_path(self):
        handler = self._handler([Path("/srv/docs")])

        assert handler._should_process(f"/elsewhere/{self.IGNORED}/readme.md") is False


class TestDeletionNeedsProofOfAbsence:
    def test_a_missing_file_counts_as_deleted(self, tmp_path):
        assert _is_missing(str(tmp_path / "gone.pdf")) is True

    def test_an_existing_file_does_not(self, tmp_path):
        present = tmp_path / "here.pdf"
        present.write_text("x", encoding="utf-8")

        assert _is_missing(str(present)) is False

    def test_an_unreadable_parent_is_not_proof_of_deletion(self, tmp_path):
        """Path.exists() raises on a permission failure rather than returning
        False, so this used to take the watcher down with a traceback; treating
        it as deletion would instead drop every document under the subtree."""
        locked = tmp_path / "locked"
        locked.mkdir()
        target = locked / "doc.pdf"
        target.write_text("x", encoding="utf-8")
        locked.chmod(0o000)
        try:
            assert _is_missing(str(target)) is False
        finally:
            locked.chmod(0o755)
