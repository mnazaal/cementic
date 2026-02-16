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
    source_path: Mapped[str] = mapped_column(String, unique=True, nullable=False)
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

    def __repr__(self) -> str:
        return f"<Document(id={self.id}, path={self.source_path}, status={self.status})>"


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
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    def __repr__(self) -> str:
        return f"<Chunk(id={self.id}, doc_id={self.document_id}, index={self.chunk_index})>"


def get_engine(database_url: str) -> "Engine":
    """Create database engine with pgvector extension."""
    engine = create_engine(database_url, echo=False)

    # Enable pgvector extension
    with engine.connect() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()

    return engine


def create_tables(engine: "Engine") -> None:
    """Create all tables in the database."""
    Base.metadata.create_all(engine)


def get_session_factory(engine: "Engine") -> sessionmaker:
    """Get session factory bound to engine."""
    return sessionmaker(bind=engine)
