"""Database models for seman using SQLAlchemy."""

from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


class Document(Base):
    """Document model for tracking PDF files."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    collection: Mapped[str] = mapped_column(String(100), default="default", nullable=False)
    source_path: Mapped[str] = mapped_column(String, nullable=False)
    file_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    # Status values:
    # - 'pending': PDF detected, not yet converted
    # - 'converted': Markdown extracted, chunks created
    # - 'completed': All chunks have embeddings
    # - 'failed': Error during processing
    total_chunks: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship to chunks
    chunks: Mapped[List["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_documents_collection", "collection"),
        Index("ix_documents_collection_source", "collection", "source_path", unique=True),
    )

    def __repr__(self) -> str:
        return (
            f"<Document(id={self.id}, collection={self.collection}, "
            f"path={self.source_path}, status={self.status})>"
        )


class Chunk(Base):
    """Chunk model for storing text segments and embeddings."""

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(768), nullable=True)
    embedding_status: Mapped[str] = mapped_column(String(20), default="pending")
    # embedding_status values:
    # - 'pending': Needs embedding
    # - 'processing': Currently being embedded
    # - 'done': Has embedding
    # - 'failed': Failed to generate embedding
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    page_start: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    page_end: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship to document
    document: Mapped["Document"] = relationship(back_populates="chunks")

    # Indexes for efficient querying
    __table_args__ = (
        Index("ix_chunks_document_chunk", "document_id", "chunk_index", unique=True),
        Index("ix_chunks_embedding_status", "embedding_status"),
        Index(
            "ix_chunks_pending",
            "embedding_status",
            postgresql_where="embedding_status = 'pending'",
        ),
    )

    def __repr__(self) -> str:
        return f"<Chunk(id={self.id}, doc_id={self.document_id}, index={self.chunk_index})>"


def get_engine(database_url: str) -> "Engine":
    """Create database engine."""
    return create_engine(database_url, echo=False)


def ensure_vector_extensions(engine: "Engine") -> None:
    """Ensure vector-related PostgreSQL extensions are enabled."""
    with engine.connect() as conn:
        try:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE"))
        except Exception:
            conn.rollback()
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()


def create_tables(engine: "Engine") -> None:
    """Create all tables and vector indexes in the database."""
    ensure_vector_extensions(engine)
    Base.metadata.create_all(engine)

    with engine.connect() as conn:
        has_vectorscale = conn.execute(
            text("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vectorscale')")
        ).scalar()

        if has_vectorscale:
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_chunks_embedding_diskann "
                    "ON chunks USING diskann (embedding vector_cosine_ops)"
                )
            )
        else:
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw "
                    "ON chunks USING hnsw (embedding vector_cosine_ops)"
                )
            )
        conn.commit()


def get_session_factory(engine: "Engine") -> sessionmaker:
    """Get session factory bound to engine."""
    return sessionmaker(bind=engine, expire_on_commit=False)
