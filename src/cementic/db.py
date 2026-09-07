"""Database models for cementic using versioned pipeline artifacts."""

from __future__ import annotations

# mypy: disable-error-code=import-untyped
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from sqlalchemy import ForeignKey, Index, Integer, String, Text, create_engine, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError, OperationalError, ProgrammingError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from cementic.index_strategies import IndexParams, build_index_ddl
from cementic.vector_store import (
    index_access_method,
    vector_index_name,
    vector_table_exists,
    vector_table_name,
)

if TYPE_CHECKING:
    from sqlalchemy.engine import URL, Engine
    from sqlalchemy.orm import Session


_ENGINE_CACHE: dict[str, Engine] = {}
REQUIRED_DB_EXTENSIONS = ("vector", "vectorscale")


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
    #: "pending" (present) or "deleted" (removed from a watched directory).
    #: Per-artifact progress/error state lives on ExtractedDocument et al.
    status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False)
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


class EmbeddingProfileCanary(Base):
    """Reference vectors proving which server produced a profile's embeddings.

    One row per embedding profile, written once and then only re-stamped
    deliberately. The profile itself cannot carry this: anything inside
    `build_embedding_profile_payload` moves the fingerprint and forks the
    corpus, which is the outcome the canary exists to avoid (see `canary.py`).
    """

    __tablename__ = "embedding_profile_canaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    embedding_profile_id: Mapped[int] = mapped_column(
        ForeignKey("embedding_profiles.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    #: The exact strings sent, so a replay re-sends the same request.
    texts_json: Mapped[str] = mapped_column(Text, nullable=False)
    vectors_json: Mapped[str] = mapped_column(Text, nullable=False)
    #: llama.cpp `build_info` when the vectors were taken, where the server
    #: reported one. Evidence for reading a later difference, never identity.
    server_build: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utc_now)


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

    # Denormalised claim filters, mirroring the vector tables' FILTER_COLUMNS and
    # for the same reason: on a joined table the planner drives from the
    # collection instead of the work queue, walks every finished chunk to reach
    # the unfinished tail, and the claim costs O(corpus) per batch. Measured at
    # 300k chunks, 90% embedded: 1,895,124 buffers joined against 2,804 here.
    # Safe to denormalise because they never change for a row -- a chunk cannot
    # move collection, and a deleted document's chunks (and these rows, by
    # cascade) are removed outright rather than filtered at claim time.
    collection: Mapped[str | None] = mapped_column(String(100), nullable=True)
    extractor_profile_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_profile_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

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
        # Covers the whole claim predicate so it never leaves the index.
        Index(
            "ix_chunk_embeddings_claim",
            "embedding_profile_id",
            "status",
            "collection",
            "extractor_profile_id",
            "chunk_profile_id",
        ),
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


class SearchActivity(Base):
    """Single-row lease recording when an interactive search last ran.

    Written best-effort by ``Searcher.search`` before it embeds a query; read
    by the pipeline worker between embed sub-batches, which yields the shared
    embedding server while the lease is fresh so interactive queries do not
    queue behind bulk indexing (notes/design-embed-scheduling.html).
    """

    __tablename__ = "search_activity"

    #: Always 1 -- the table holds one row, upserted in place.
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    last_search_at: Mapped[datetime] = mapped_column(default=utc_now, nullable=False)


def get_engine(database_url: str | URL) -> Engine:
    """Create database engine."""
    cache_key = _url_cache_key(database_url)
    engine = _ENGINE_CACHE.get(cache_key)
    if engine is None:
        connect_args: dict[str, object] = {"connect_timeout": 5}
        # Parsed, not prefix-matched. `gssencmode` is a psycopg2 connect arg, so
        # the question is which driver this URL resolves to -- and
        # `postgresql+psycopg2://`, which a user setting CEMENTIC_DB_URL may
        # well write, does not start with `postgresql://`. It got no
        # `gssencmode` and the connection hung on the GSSAPI probe the option
        # exists to skip.
        if make_url(database_url).get_backend_name() == "postgresql":
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
    if engine.dialect.name != "postgresql":
        return
    with engine.connect() as conn:
        try:
            for extension in REQUIRED_DB_EXTENSIONS:
                conn.execute(text(f"CREATE EXTENSION IF NOT EXISTS {extension}"))
            conn.commit()
        except Exception as error:
            conn.rollback()
            details = (
                "Postgres must provide the pgvector (`vector`) and pgvectorscale "
                "(`vectorscale`) extensions. Use `cementic init postgres ./cementic-postgres` "
                "for a local setup, install the extensions on your Postgres server, or connect "
                "as a user with CREATE EXTENSION privileges."
            )
            raise RuntimeError(
                f"Failed to enable required Postgres extensions: {error}. {details}"
            ) from error


#: One active revision per collection is an invariant promote_revision
#: maintains procedurally, but two concurrent promotes under READ COMMITTED can
#: each miss the other's newly-activated row and leave two actives -- which
#: search and status then disagree about. This partial unique index makes the
#: database refuse the second activation outright. It lives here rather than on
#: the model because create_all cannot retrofit an index onto existing
#: databases; IF NOT EXISTS makes it idempotent on both PostgreSQL and SQLite.
_ACTIVE_REVISION_UNIQUE_DDL = (
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_pipeline_revisions_one_active "
    "ON pipeline_revisions (collection) WHERE status = 'active'"
)


#: Name of the full-text index the lexical half of hybrid search reads.
LEXICAL_INDEX_NAME = "ix_chunks_v2_fts"

#: Text search configuration for that index. It must match the one the query
#: uses, or the planner silently ignores the index and every lexical search
#: becomes a sequential scan of 2.3M rows.
LEXICAL_TEXT_CONFIG = "english"


def _lexical_index_exists(engine: Engine) -> bool:
    """Whether the full-text index is present."""
    return LEXICAL_INDEX_NAME in {
        index["name"] for index in inspect(engine).get_indexes(Chunk.__tablename__)
    }


def _table_has_rows(conn: Any, table: str) -> bool:
    return conn.execute(text(f"SELECT 1 FROM {table} LIMIT 1")).first() is not None


def ensure_lexical_index(engine: Engine) -> None:
    """Create the full-text index hybrid search reads, if it is absent.

    PostgreSQL-only, like ``ensure_vector_extensions``: sqlite has no
    ``tsvector``, the unit suite runs on sqlite, and a no-op keeps it that way.

    One index serves every collection and every revision. Vectors need a table
    per embedding profile because two models produce incomparable vectors;
    chunk *text* does not depend on the model, so this belongs beside
    ``ix_chunks_v2_document`` as an ordinary table-level index and takes no part
    in revision or profile machinery.

    Built up front rather than after import. Measured 2026-09-07: the index
    makes chunk inserts about 4.5x slower (19,184 -> 4,166 rows/s), which is
    ~7 minutes across the whole corpus against ~115 hours of embedding -- 0.1%
    of import wall clock. Paying it continuously buys an index that is correct
    at every moment, with no post-import step to forget and no window where
    exact-match search silently returns nothing.

    On a table that already holds rows the build takes minutes under an ACCESS
    EXCLUSIVE lock, so it runs CONCURRENTLY on its own autocommit connection and
    indexing and search keep working throughout.
    """
    if engine.dialect.name != "postgresql":
        return
    if _lexical_index_exists(engine):
        return

    table = Chunk.__tablename__
    with engine.connect() as conn:
        if _table_has_rows(conn, table):
            # Deliberately not built here. This runs at worker startup, and on a
            # populated table the build takes minutes: a plain CREATE INDEX
            # holds ACCESS EXCLUSIVE for all of it, and CONCURRENTLY waits for
            # every open transaction on the table before it even begins -- so
            # `cementic start` would block for minutes, or indefinitely against
            # a long-running reader. An existing corpus upgrades by calling
            # `build_lexical_index` explicitly instead.
            return

    try:
        with engine.begin() as conn:
            conn.execute(text(_lexical_index_ddl(table)))
    except (IntegrityError, ProgrammingError, OperationalError):
        # Same race as the active-revision index below: `IF NOT EXISTS` checks
        # the catalog before taking its lock, so the two workers `cementic
        # start` spawns can both pass and one loses on pg_class. Re-check rather
        # than guess -- an unhandled raise here kills the worker at startup.
        if not _lexical_index_exists(engine):
            raise


def _lexical_index_ddl(table: str, *, concurrently: bool = False) -> str:
    concurrent = "CONCURRENTLY " if concurrently else ""
    return (
        f"CREATE INDEX {concurrent}IF NOT EXISTS {LEXICAL_INDEX_NAME} ON {table} "
        f"USING gin (to_tsvector('{LEXICAL_TEXT_CONFIG}', content))"
    )


def build_lexical_index(engine: Engine) -> None:
    """Build the full-text index on a corpus that already has chunks.

    The upgrade path, kept out of `ensure_lexical_index` because it blocks:
    minutes of work on a large corpus, and CONCURRENTLY additionally waits for
    every open transaction on the table before starting. Fine as something a
    person runs and watches; not fine at worker startup.

    CONCURRENTLY so indexing and search keep working throughout, which also
    means it cannot run inside a transaction block.
    """
    if engine.dialect.name != "postgresql":
        return
    if _lexical_index_exists(engine) and _lexical_index_is_valid(engine):
        return
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text(_lexical_index_ddl(Chunk.__tablename__, concurrently=True)))

    # A failed CONCURRENTLY build leaves an INVALID index behind that the
    # planner ignores, so the catalog says "present" while every lexical search
    # silently falls back to a sequential scan of every chunk. Drop it, so a
    # retry builds rather than inheriting a permanently useless index.
    if not _lexical_index_is_valid(engine):
        with engine.begin() as conn:
            conn.execute(text(f"DROP INDEX IF EXISTS {LEXICAL_INDEX_NAME}"))
        raise RuntimeError(
            f"Concurrent build of {LEXICAL_INDEX_NAME} did not complete; the "
            "invalid index it left behind has been dropped. Re-run to retry."
        )


def _lexical_index_is_valid(engine: Engine) -> bool:
    """Whether the full-text index finished building (PostgreSQL only)."""
    with engine.connect() as conn:
        valid = conn.execute(
            text(
                "SELECT indisvalid FROM pg_index "
                "WHERE indexrelid = to_regclass(:name)"
            ),
            {"name": LEXICAL_INDEX_NAME},
        ).scalar()
    return bool(valid)


def create_tables(engine: Engine) -> None:
    """Create all tables required by the versioned pipeline schema."""
    ensure_vector_extensions(engine)
    Base.metadata.create_all(engine)
    ensure_lexical_index(engine)
    try:
        with engine.begin() as conn:
            conn.execute(text(_ACTIVE_REVISION_UNIQUE_DDL))
    except (IntegrityError, ProgrammingError, OperationalError):
        # `CREATE UNIQUE INDEX IF NOT EXISTS` checks the catalog before taking
        # its lock, so the two workers `cementic start` spawns can both pass the
        # check and race: the loser gets a duplicate-key error on pg_class. The
        # index either exists now or the failure was real -- re-check rather
        # than guess, since an unhandled raise here kills the worker at startup.
        if not _active_revision_index_exists(engine):
            raise


def _active_revision_index_exists(engine: Engine) -> bool:
    """Whether the one-active-per-collection index is present."""
    return "uq_pipeline_revisions_one_active" in {
        index["name"] for index in inspect(engine).get_indexes("pipeline_revisions")
    }


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
    build_memory: str | None = None,
    force_rebuild: bool = False,
) -> None:
    """Ensure the chosen ANN index exists on the profile's vector table.

    The method (`hnsw`/`diskann`) is resolved through the index-strategy
    registry; this is the thin imperative shell that executes its pure DDL.

    No-ops when the vector table does not exist yet. A revision can legitimately
    complete with zero vectors (an empty watch directory, or one where every
    document failed to extract), and `CREATE INDEX ... IF NOT EXISTS` guards only
    the index name, not the table -- so indexing one unconditionally would raise
    and leave the revision stuck in `building` forever.

    ``force_rebuild`` drops the existing index even when the method is unchanged,
    which is the only way to pick up build-time parameters like ``hnsw_m`` and
    ``ef_construction``. It belongs here, on the same connection as the create,
    rather than in the caller: a drop committed separately leaves the collection
    with no index at all if the rebuild then fails, and searches keep working
    (via sequential scan) so nothing surfaces it.
    """
    if engine.dialect.name != "postgresql":
        return
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
        if not vector_table_exists(conn, profile_id):
            return
        _ensure_ann_access_method(conn, method)
        if build_memory is not None:
            # An HNSW graph that does not fit in maintenance_work_mem spills and
            # the build slows sharply -- measured 1454s at the 64MB default
            # versus 345s at 2GB for 100k 768-dim vectors. SET LOCAL, not SET:
            # the connection returns to the *pool* afterwards (engine.connect()
            # is pooled, not discarded), and a session-level SET would leave
            # every later borrower running with gigabytes of
            # maintenance_work_mem. LOCAL scopes it to this transaction, which
            # still covers the CREATE INDEX below -- the commit() is after it.
            # The value's grammar is validated in config, not here.
            conn.execute(text(f"SET LOCAL maintenance_work_mem = '{build_memory}'"))
        # If an index already exists under a different method, drop it first so a
        # method switch actually takes effect (CREATE INDEX IF NOT EXISTS alone
        # would silently keep the old one).
        existing_method = index_access_method(conn, index_name)
        if existing_method is not None and (force_rebuild or existing_method != method):
            conn.execute(text(f"DROP INDEX IF EXISTS {index_name}"))
        conn.execute(text(ddl))
        conn.commit()
