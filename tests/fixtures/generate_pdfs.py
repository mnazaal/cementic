"""Generate the PDF fixtures used by the test suite.

``*.pdf`` is gitignored (binaries don't belong in version control), so these
are regenerated on demand -- either by running this script directly, or
automatically by the ``_ensure_fixture_pdfs`` autouse fixture in
``tests/conftest.py`` before the integration tests that need them run.

Usage:
    python tests/fixtures/generate_pdfs.py

Generated PDFs:
    - test_doc_a.pdf      (generic pipeline-worker/e2e fixture)
    - test_doc_b.pdf      (generic pipeline-worker/e2e fixture, distinct content)
"""

from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

OUT_DIR = Path(__file__).resolve().parent

PDF_SPECS = {
    "test_doc_a": {
        "title": "Test Document A: Symbiosis and Time",
        "paragraphs": [
            """This document discusses symbiosis between organisms over time. A computer """
            """can model these relationships. Man has studied symbiotic time-dependent """
            """systems for centuries, using both observation and machine assistance.""",
            """Time-series analysis of symbiotic relationships reveals patterns that a """
            """computer or man can both learn to recognize, given enough data.""",
        ],
    },
    "test_doc_b": {
        "title": "Test Document B: Machines and Learning",
        "paragraphs": [
            """This document discusses machine learning as it relates to vector spaces. """
            """A vector represents semantic meaning that a neural network can learn from """
            """training data, building an internal model of the underlying structure.""",
            """Semantic search relies on a network of learned vectors, where a machine """
            """compares queries against stored representations to find relevant results.""",
        ],
    },
}


def generate_pdf(filename: str, title: str, paragraphs: list[str]) -> None:
    """Generate a single PDF with title and paragraphs."""
    output_path = OUT_DIR / filename
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=LETTER,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72,
    )
    styles = getSampleStyleSheet()
    story: list = []

    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 24))

    for para_text in paragraphs:
        story.append(Paragraph(para_text, styles["BodyText"]))
        story.append(Spacer(1, 12))

    doc.build(story)
    print(f"Generated: {output_path} ({output_path.stat().st_size} bytes)")


def ensure_fixture_pdfs() -> None:
    """Generate any fixture PDF that doesn't already exist on disk.

    ``*.pdf`` is gitignored, so a fresh clone has none of these; this makes
    the integration test suite self-sufficient without tracking binaries.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, spec in PDF_SPECS.items():
        output_path = OUT_DIR / f"{name}.pdf"
        if not output_path.exists():
            generate_pdf(f"{name}.pdf", spec["title"], spec["paragraphs"])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, spec in PDF_SPECS.items():
        generate_pdf(f"{name}.pdf", spec["title"], spec["paragraphs"])
    print(f"\nDone. {len(PDF_SPECS)} PDFs generated in tests/fixtures/")


if __name__ == "__main__":
    main()
