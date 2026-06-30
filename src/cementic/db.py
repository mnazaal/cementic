"""Database models for cementic using versioned pipeline artifacts."""

from __future__ import annotations

# mypy: disable-error-code=import-untyped
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from sqlalchemy import ForeignKey, Index, Integer, String, Text, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from cementic.index_strategies import IndexParams, build_index_ddl
from cementic.vector_store import index_access_method, vector_index_name, vector_table_name

if TYPE_CHECKING:
    from sqlalchemy.engine import URL, Engine
    from sqlalchemy.orm import Session


_ENGINE_CACHE: dict[str, Engine] = {}


def _url_cache_key(database_url: str | URL) -> str:
    """Return a stable cache key for the database URL."""
    if hasattr(database_url, "render_as_string"):
        return database_url.render_as_string(hide_password=False)
    return database_url


class Base(DeclarativeBase):
    """Base class for all models."""


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class SourceDocument(Base):
    """Stable source-of-truth for discovered source documents."""

    __tablename__ = "source_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    collection: Mapped[str] = mapped_column(String(100), default="default", nullable=False)
    source_path: Mapped[str] = mapped_column(String, nullable=False)
    file_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(default=utc_now, onupdate=utc_now)

    extracted_documents: Mapped[list["ExtractedDocument"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_source_documents_collection", "collection"),
        Index(
            "ix_source_documents_collection_source",
            "collection",
            "source_path",
            unique=True,
        ),
    )


class ExtractorProfile(Base):
    """Immutable extractor configuration definition."""

    __tablename__ = "extractor_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fingerprint: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    config_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=utc_now)

    extracted_documents: Mapped[list["ExtractedDocument"]] = relationship(back_populates="profile")
    revisions: Mapped[list["PipelineRevision"]] = relationship(back_populates="extractor_profile")


class ExtractedDocument(Base):
    """Persisted extracted text artifact for one document/profile pair."""

    __tablename__ = "extracted_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("source_documents.id", ondelete="CASCADE"), nullable=False
    )
    extractor_profile_id: Mapped[int] = mapped_column(
        ForeignKey("extractor_profiles.id", ondelete="CASCADE"), nullable=False
    )
    source_file_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    artifact_path: Mapped[str | None] = mapped_column(String, nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(default=utc_now, onupdate=utc_now)

    document: Mapped[SourceDocument] = relationship(back_populates="extracted_documents")
    profile: Mapped[ExtractorProfile] = relationship(back_populates="extracted_documents")
    chunked_documents: Mapped[list["ChunkedDocument"]] = relationship(
        back_populates="extracted_document", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index(
            "ix_extracted_documents_document_profile",
            "document_id",
            "extractor_profile_id",
            unique=True,
        ),
        Index("ix_extracted_documents_profile_status", "extractor_profile_id", "status"),
    )


class ChunkProfile(Base):
    """Immutable chunking configuration definition."""

    __tablename__ = "chunk_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fingerprint: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    config_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=utc_now)

    chunked_documents: Mapped[list["ChunkedDocument"]] = relationship(back_populates="profile")
    revisions: Mapped[list["PipelineRevision"]] = relationship(back_populates="chunk_profile")


class ChunkedDocument(Base):
    """Chunked view of an extracted document for one chunk profile."""

    __tablename__ = "chunked_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    extracted_document_id: Mapped[int] = mapped_column(
        ForeignKey("extracted_documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_profile_id: Mapped[int] = mapped_column(
        ForeignKey("chunk_profiles.id", ondelete="CASCADE"), nullable=False
    )
    source_content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False)
    total_chunks: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(default=utc_now, onupdate=utc_now)

    extracted_document: Mapped[ExtractedDocument] = relationship(back_populates="chunked_documents")
    profile: Mapped[ChunkProfile] = relationship(back_populates="chunked_documents")
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="chunked_document", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index(
            "ix_chunked_documents_extracted_profile",
            "extracted_document_id",
            "chunk_profile_id",
            unique=True,
        ),
        Index("ix_chunked_documents_profile_status", "chunk_profile_id", "status"),
    )


class Chunk(Base):
    """Chunk text artifact produced from one chunked document."""

    __tablename__ = "chunks_v2"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("source_documents.id", ondelete="CASCADE"), nullable=False
    )
    chunked_document_id: Mapped[int] = mapped_column(
        ForeignKey("chunked_documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(default=utc_now, onupdate=utc_now)

    document: Mapped[SourceDocument] = relationship(back_populates="chunks")
    chunked_document: Mapped[ChunkedDocument] = relationship(back_populates="chunks")
    embeddings: Mapped[list["ChunkEmbedding"]] = relationship(
        back_populates="chunk", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index(
            "ix_chunks_v2_chunked_document_chunk",
            "chunked_document_id",
            "chunk_index",
            unique=True,
        ),
        Index("ix_chunks_v2_document", "document_id"),
    )


class EmbeddingProfile(Base):
    """Immutable embedding configuration definition."""

    __tablename__ = "embedding_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fingerprint: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    model_identifier: Mapped[str] = mapped_column(String, nullable=False)
    embedding_dim: Mapped[int] = mapped_column(Integer, nullable=False)
    distance_metric: Mapped[str] = mapped_column(String(20), default="cosine", nullable=False)
    config_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=utc_now)

    embeddings: Mapped[list["ChunkEmbedding"]] = relationship(back_populates="profile")
    revisions: Mapped[list["PipelineRevision"]] = relationship(back_populates="embedding_profile")


class ChunkEmbedding(Base):
    """Embedding work-tracking for one chunk/profile pair.

    Tracks status/error and supports the per-chunk row-claim; the vector itself
    lives in the per-profile ``embedding_vectors_p{id}`` table (see vector_store).
    """

    __tablename__ = "chunk_embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chunk_id: Mapped[int] = mapped_column(
        ForeignKey("chunks_v2.id", ondelete="CASCADE"), nullable=False
    )
    embedding_profile_id: Mapped[int] = mapped_column(
        ForeignKey("embedding_profiles.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(default=utc_now, onupdate=utc_now)

    chunk: Mapped[Chunk] = relationship(back_populates="embeddings")
    profile: Mapped[EmbeddingProfile] = relationship(back_populates="embeddings")

    __table_args__ = (
        Index(
            "ix_chunk_embeddings_chunk_profile",
            "chunk_id",
            "embedding_profile_id",
            unique=True,
        ),
        Index("ix_chunk_embeddings_profile_status", "embedding_profile_id", "status"),
    )


class PipelineRevision(Base):
    """Search-visible release composed from extractor/chunk/embedding profiles."""

    __tablename__ = "pipeline_revisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    collection: Mapped[str] = mapped_column(String(100), nullable=False)
    label: Mapped[str | None] = mapped_column(String(255), nullable=True)
    extractor_profile_id: Mapped[int] = mapped_column(
        ForeignKey("extractor_profiles.id", ondelete="RESTRICT"), nullable=False
    )
    chunk_profile_id: Mapped[int] = mapped_column(
        ForeignKey("chunk_profiles.id", ondelete="RESTRICT"), nullable=False
    )
    embedding_profile_id: Mapped[int] = mapped_column(
        ForeignKey("embedding_profiles.id", ondelete="RESTRICT"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(20), default="building", nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=utc_now)
    promoted_at: Mapped[datetime | None] = mapped_column(nullable=True)

    extractor_profile: Mapped[ExtractorProfile] = relationship(back_populates="revisions")
    chunk_profile: Mapped[ChunkProfile] = relationship(back_populates="revisions")
    embedding_profile: Mapped[EmbeddingProfile] = relationship(back_populates="revisions")

    __table_args__ = (
        Index("ix_pipeline_revisions_collection_status", "collection", "status"),
        Index(
            "ix_pipeline_revisions_collection_profiles",
            "collection",
            "extractor_profile_id",
            "chunk_profile_id",
            "embedding_profile_id",
            unique=True,
        ),
    )


def get_engine(database_url: str | URL) -> Engine:
    """Create database engine."""
    cache_key = _url_cache_key(database_url)
    engine = _ENGINE_CACHE.get(cache_key)
    if engine is None:
        url_str = (
            database_url if isinstance(database_url, str)
            else database_url.render_as_string(hide_password=False)
        )
        connect_args: dict[str, object] = {"connect_timeout": 5}
        if url_str.startswith("postgresql://"):
            connect_args["gssencmode"] = "disable"
        engine = create_engine(
            database_url,
            echo=False,
            pool_pre_ping=True,
            connect_args=connect_args,
        )
        _ENGINE_CACHE[cache_key] = engine
    return engine


def ensure_vector_extensions(engine: Engine) -> None:
    """Ensure vector-related PostgreSQL extensions are enabled."""
    with engine.connect() as conn:

        def extension_exists(name: str) -> bool:
            return bool(
                conn.execute(
                    text("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = :name)"),
                    {"name": name},
                ).scalar()
            )

        if not extension_exists("vectorscale"):
            try:
                conn.execute(text("CREATE EXTENSION vectorscale CASCADE"))
                conn.commit()
            except Exception:
                conn.rollback()

        if extension_exists("vector"):
            return

        try:
            conn.execute(text("CREATE EXTENSION vector"))
            conn.commit()
        except Exception:
            conn.rollback()
            if not extension_exists("vector"):
                raise


def create_tables(engine: Engine) -> None:
    """Create all tables required by the versioned pipeline schema."""
    ensure_vector_extensions(engine)
    Base.metadata.create_all(engine)


def get_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Get session factory bound to engine."""
    return sessionmaker(bind=engine, expire_on_commit=False)


def _ensure_ann_access_method(conn: Any, method: str) -> None:
    """Fail early if PostgreSQL cannot build the requested ANN index method."""
    if method != "diskann":
        return
    exists = bool(
        conn.execute(
            text("SELECT EXISTS (SELECT 1 FROM pg_am WHERE amname = :method)"),
            {"method": method},
        ).scalar()
    )
    if not exists:
        raise RuntimeError(
            "DiskANN indexing requires the vectorscale PostgreSQL extension with the "
            "diskann access method available"
        )


def ensure_embedding_ann_index(
    engine: Engine,
    *,
    profile_id: int,
    method: str,
    params: IndexParams,
    distance_metric: str = "cosine",
) -> None:
    """Ensure the chosen ANN index exists on the profile's vector table.

    The method (`hnsw`/`diskann`) is resolved through the index-strategy
    registry; this is the thin imperative shell that executes its pure DDL.
    """
    index_name = vector_index_name(profile_id)
    ddl = build_index_ddl(
        method=method,
        index_name=index_name,
        table=vector_table_name(profile_id),
        column="embedding",
        metric=distance_metric,
        params=params,
    )
    with engine.connect() as conn:
        _ensure_ann_access_method(conn, method)
        # If an index already exists under a different method, drop it first so a
        # method switch actually takes effect (CREATE INDEX IF NOT EXISTS alone
        # would silently keep the old one).
        existing_method = index_access_method(conn, index_name)
        if existing_method is not None and existing_method != method:
            conn.execute(text(f"DROP INDEX IF EXISTS {index_name}"))
        conn.execute(text(ddl))
        conn.commit()
