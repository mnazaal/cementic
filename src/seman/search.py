"""Search functionality using pgvectorscale."""

from typing import List, Optional, TypedDict

from seman.config import Config, get_config
from seman.db import Chunk, Document, get_engine, get_session_factory
from seman.embedders import get_embedder
from seman.embedding_text import format_query_text


class SearchResult(TypedDict):
    """Type for search results."""

    collection: str
    source_path: str
    content: str
    score: float
    page_start: int
    page_end: int


class Searcher:
    """Searcher for semantic search over indexed documents."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialize searcher with config."""
        self.config = config or get_config()
        engine = get_engine(self.config.database.url)
        self.Session = get_session_factory(engine)

        # Initialize embedder based on config
        if self.config.indexing.embedder == "llama-cpp":
            self.embedder = get_embedder(
                "llama-cpp",
                model_path=self.config.llama_cpp.model_path,
                n_ctx=self.config.llama_cpp.n_ctx,
                n_gpu_layers=self.config.llama_cpp.n_gpu_layers,
                embedding_dim=self.config.llama_cpp.embedding_dim,
                verbose=self.config.llama_cpp.verbose,
            )
        else:
            self.embedder = get_embedder(
                "ollama",
                host=self.config.ollama.host,
                model=self.config.ollama.model,
                embedding_dim=self.config.ollama.embedding_dim,
            )

    def search(
        self,
        query: str,
        top_k: int = 10,
        collections: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """Search for documents similar to query.

        Args:
            query: Search query text
            top_k: Number of top results to return
            collections: Optional collection names to filter by

        Returns:
            List of search results sorted by relevance
        """
        # Generate query embedding
        query_embedding = self.embedder.embed(format_query_text(query, self.config))

        with self.Session() as session:
            # Perform vector similarity search
            query_stmt = session.query(
                Chunk,
                Chunk.embedding.cosine_distance(query_embedding).label("distance"),
            ).join(Document, Chunk.document_id == Document.id)

            query_stmt = query_stmt.filter(Chunk.embedding.isnot(None))

            if collections:
                query_stmt = query_stmt.filter(Document.collection.in_(collections))

            results = (
                query_stmt.order_by(Chunk.embedding.cosine_distance(query_embedding))
                .limit(top_k)
                .all()
            )

            # Convert to SearchResult format
            search_results = []
            for chunk, distance in results:
                score = 1.0 - distance

                search_results.append(
                    SearchResult(
                        collection=chunk.document.collection,
                        source_path=chunk.document.source_path,
                        content=chunk.content,
                        score=score,
                        page_start=chunk.page_start or 0,
                        page_end=chunk.page_end or 0,
                    )
                )

            return search_results
