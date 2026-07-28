"""Database session helpers for PostgreSQL RAG."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from .models import Base


DEFAULT_DATABASE_URL = "postgresql+psycopg://message_helper:message_helper@127.0.0.1:5432/message_helper"


def rag_database_url() -> str:
    return os.environ.get("MESSAGE_HELPER_RAG_DATABASE_URL") or DEFAULT_DATABASE_URL


def build_session_factory(database_url: str | None = None, *, create_schema: bool = True) -> sessionmaker[Session]:
    engine = create_engine(database_url or rag_database_url(), pool_pre_ping=True)
    if create_schema:
        try:
            with engine.begin() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.execute(
                    text(
                        """
                        CREATE OR REPLACE FUNCTION rag_chunks_content_tsv_update()
                        RETURNS trigger AS $$
                        BEGIN
                          NEW.content_tsv := to_tsvector('simple', coalesce(NEW.content, ''));
                          NEW.updated_at := now();
                          RETURN NEW;
                        END
                        $$ LANGUAGE plpgsql
                        """
                    )
                )
            Base.metadata.create_all(engine)
            with engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        DO $$
                        BEGIN
                          IF NOT EXISTS (
                            SELECT 1 FROM pg_trigger WHERE tgname = 'trg_chunks_content_tsv_update'
                          ) THEN
                            CREATE TRIGGER trg_chunks_content_tsv_update
                            BEFORE INSERT OR UPDATE OF content
                            ON chunks
                            FOR EACH ROW
                            EXECUTE FUNCTION rag_chunks_content_tsv_update();
                          END IF;
                        END
                        $$;
                        """
                    )
                )
        except SQLAlchemyError as exc:
            raise RuntimeError("PostgreSQL RAG database requires the pgvector extension. Use the bundled docker-compose service or install pgvector on the target database.") from exc
    return sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def rag_session(database_url: str | None = None) -> Iterator[Session]:
    factory = build_session_factory(database_url)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
