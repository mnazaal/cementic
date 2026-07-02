"""Generate the PDF fixtures used by the test suite.

``*.pdf`` is gitignored (binaries don't belong in version control), so these
are regenerated on demand -- either by running this script directly, or
automatically by the ``_ensure_fixture_pdfs`` autouse fixture in
``tests/conftest.py`` before the integration tests that need them run.

Usage:
    python tests/fixtures/generate_pdfs.py

Generated PDFs:
    - cs_neural_nets.pdf  (neural networks / deep learning; search-quality demos)
    - bio_cell.pdf        (cell biology / organelles; search-quality demos)
    - hist_rome.pdf       (Roman Republic history; search-quality demos)
    - test_doc_a.pdf      (generic pipeline-worker/e2e fixture)
    - test_doc_b.pdf      (generic pipeline-worker/e2e fixture, distinct content)
"""

from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

OUT_DIR = Path(__file__).resolve().parent

PDF_SPECS = {
    "cs_neural_nets": {
        "title": "A Survey of Neural Network Architectures",
        "paragraphs": [
            """Neural networks are computational models inspired by biological neural systems. """
            """The fundamental building block is the artificial neuron, which computes a weighted sum """
            """of its inputs and applies a nonlinear activation function. Common activation functions """
            """include the rectified linear unit (ReLU), sigmoid, and hyperbolic tangent. Training """
            """these networks involves minimizing a loss function through gradient-based optimization, """
            """with backpropagation serving as the primary algorithm for computing gradients across """
            """layers. Stochastic gradient descent and its adaptive variants, such as Adam and RMSProp, """
            """remain the workhorses of modern deep learning optimization.""",

            """Convolutional neural networks (CNNs) revolutionized computer vision by introducing """
            """spatially-localized filters that scan across input images to detect visual features """
            """like edges, textures, and eventually complex patterns at higher layers. Pooling layers """
            """downsample feature maps, providing translation invariance and reducing computational """
            """cost. Meanwhile, recurrent neural networks (RNNs) and their gated variants, including """
            """long short-term memory (LSTM) and gated recurrent units (GRU), process sequential data """
            """by maintaining hidden states that capture temporal dependencies. The transformer """
            """architecture later displaced RNNs for many sequence tasks by replacing recurrence with """
            """self-attention mechanisms that can capture long-range dependencies in parallel.""",

            """The training of deep neural networks presents significant computational challenges. """
            """Vanishing and exploding gradients can destabilize training in deep architectures, """
            """motivating techniques like batch normalization, residual connections, and careful """
            """weight initialization. Overfitting is managed through regularization methods including """
            """dropout, weight decay, and early stopping. Data augmentation artificially expands """
            """training sets through transformations like rotation, cropping, and color jittering. """
            """Transfer learning leverages pretrained models on large datasets, fine-tuning them """
            """for specialized downstream tasks with limited labeled data.""",
        ],
    },
    "bio_cell": {
        "title": "Fundamentals of Cell Biology: Organelles and Membrane Dynamics",
        "paragraphs": [
            """The eukaryotic cell is a highly organized structure containing membrane-bound organelles """
            """that compartmentalize specialized biochemical functions. The nucleus houses the cell's """
            """genetic material, enclosed by a double membrane perforated with nuclear pores that """
            """regulate molecular traffic. The endoplasmic reticulum forms an extensive network of """
            """interconnected tubules and flattened sacs, with the rough ER studded with ribosomes """
            """synthesizing membrane and secretory proteins, while the smooth ER participates in lipid """
            """synthesis and calcium homeostasis. The Golgi apparatus receives vesicles from the ER, """
            """modifies their cargo through glycosylation and other post-translational modifications, """
            """and sorts proteins for delivery to their final destinations.""",

            """Mitochondria are the primary energy-producing organelles, converting nutrients into """
            """adenosine triphosphate (ATP) through oxidative phosphorylation. These double-membrane """
            """organelles contain their own circular DNA genome and ribosomes, supporting the """
            """endosymbiotic theory of their evolutionary origin from ancient bacteria. The inner """
            """mitochondrial membrane is highly folded into cristae, dramatically increasing the """
            """surface area available for electron transport chain complexes and ATP synthase. """
            """Chloroplasts in plant cells perform photosynthesis, capturing light energy to fix """
            """carbon dioxide into carbohydrates, sustaining nearly all life on Earth through """
            """the production of oxygen and organic compounds.""",

            """Cellular membranes are dynamic structures composed primarily of phospholipid bilayers """
            """with embedded proteins that mediate transport, signaling, and cell recognition. The """
            """fluid mosaic model describes membranes as two-dimensional fluids where lipids and """
            """proteins diffuse laterally. Membrane transport occurs through passive diffusion, """
            """facilitated diffusion via channel and carrier proteins, and active transport powered """
            """by ATP hydrolysis. Endocytosis and exocytosis enable bulk transport of materials """
            """across the membrane through vesicle formation and fusion. Cell signaling pathways """
            """involve membrane receptors that transduce extracellular signals into intracellular """
            """responses through cascades of protein phosphorylation and second messenger systems.""",
        ],
    },
    "hist_rome": {
        "title": "The Roman Republic: Institutions, Expansion, and Decline",
        "paragraphs": [
            """The Roman Republic, established around 509 BCE after the overthrow of the last king, """
            """developed a sophisticated system of governance based on checks and balances among """
            """competing institutions. The Senate, composed of aristocratic patricians, served as """
            """the primary deliberative body, controlling finances, foreign policy, and advising """
            """magistrates. The popular assemblies, including the Comitia Centuriata and Comitia """
            """Tributa, elected magistrates and passed legislation, though their power was constrained """
            """by senatorial influence. Two annually elected consuls shared executive authority, """
            """each possessing veto power over the other's decisions, while tribunes of the plebs """
            """could block actions harmful to the common people.""",

            """Roman expansion transformed the Republic from a small city-state into a Mediterranean """
            """empire through a series of pivotal conflicts. The Punic Wars against Carthage """
            """(264-146 BCE) represented the defining struggle for Mediterranean supremacy. The First """
            """Punic War gave Rome control of Sicily and its first overseas province. The Second """
            """Punic War featured Hannibal's legendary crossing of the Alps with war elephants and """
            """his devastating victories at Lake Trasimene and Cannae before Scipio Africanus """
            """ultimately defeated Carthage at the Battle of Zama. The Third Punic War ended with """
            """the complete destruction of Carthage, its territory annexed as the province of Africa.""",

            """The late Republic was marked by profound social and political crises that ultimately """
            """undermined republican institutions. The Gracchi brothers attempted land reforms to """
            """address growing inequality but were killed by senatorial opponents, establishing a """
            """precedent for political violence. Military reforms by Gaius Marius created professional """
            """armies loyal to their commanders rather than the state, enabling generals like Sulla """
            """and Julius Caesar to march on Rome itself. The First Triumvirate of Caesar, Pompey, """
            """and Crassus operated as an informal alliance dominating Roman politics. Caesar's """
            """crossing of the Rubicon in 49 BCE triggered a civil war that ended with his """
            """dictatorship and assassination, setting the stage for Octavian's rise and the """
            """transition from Republic to Empire.""",
        ],
    },
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
