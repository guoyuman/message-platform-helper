"""SQLAlchemy models for the PostgreSQL + pgvector RAG store."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, MetaData, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Declarative base for RAG database metadata."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class DocumentRecord(TimestampMixin, Base):
    """Stored source document."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, default=dict)

    chunks: Mapped[list[ChunkRecord]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class ChunkRecord(TimestampMixin, Base):
    """Searchable child chunk with vector and full-text search fields."""

    __tablename__ = "chunks"
    __table_args__ = (
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_chunks_content_tsv", "content_tsv", postgresql_using="gin"),
        Index("ix_chunks_document_id", "document_id"),
        Index("ix_chunks_parent_id", "parent_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chunks.id", ondelete="SET NULL"),
        nullable=True,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(1024), nullable=False)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, default=dict)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content_tsv: Mapped[str | None] = mapped_column(TSVECTOR, nullable=True)

    document: Mapped[DocumentRecord] = relationship(back_populates="chunks")
    parent: Mapped[ChunkRecord | None] = relationship(
        remote_side="ChunkRecord.id",
        back_populates="children",
    )
    children: Mapped[list[ChunkRecord]] = relationship(back_populates="parent")


class SessionMemoryRecord(TimestampMixin, Base):
    """Conversation memory persisted in PostgreSQL."""

    __tablename__ = "session_memory"

    session_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    memory_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)


class HelperRunRecord(Base):
    """Auditable request/response history for helper runs."""

    __tablename__ = "helper_runs"
    __table_args__ = (Index("ix_helper_runs_session", "session_id", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(128), nullable=False)
    request_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    response_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class CounterRecord(Base):
    """Frequency-control counters persisted in PostgreSQL."""

    __tablename__ = "counters"

    key: Mapped[str] = mapped_column(String(512), primary_key=True)
    value: Mapped[int] = mapped_column(Integer, nullable=False)
    expires_at: Mapped[float] = mapped_column(Float, nullable=False)
