from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from alembic.config import Config
from message_platform_helper.rag.db import Base


ROOT = Path(__file__).resolve().parents[1]


class RagMigrationTests(unittest.TestCase):
    def test_alembic_config_points_to_migrations(self) -> None:
        config = Config(str(ROOT / "alembic.ini"))

        self.assertEqual(config.get_main_option("script_location"), "migrations")

    def test_initial_migration_matches_declared_tables(self) -> None:
        migration = _load_initial_migration()

        self.assertEqual(migration.revision, "0001_rag_postgres_schema")
        self.assertIsNone(migration.down_revision)
        self.assertEqual({"documents", "chunks", "session_memory", "helper_runs", "counters"}, set(Base.metadata.tables))

    def test_initial_migration_declares_pgvector_and_fts_operations(self) -> None:
        text = (ROOT / "migrations" / "versions" / "0001_rag_postgres_schema.py").read_text(encoding="utf-8")

        self.assertIn("CREATE EXTENSION IF NOT EXISTS vector", text)
        self.assertIn("Vector(1536)", text)
        self.assertIn("postgresql.TSVECTOR", text)
        self.assertIn("postgresql_using=\"hnsw\"", text)
        self.assertIn("postgresql_using=\"gin\"", text)
        self.assertIn("rag_chunks_content_tsv_update", text)


def _load_initial_migration() -> object:
    path = ROOT / "migrations" / "versions" / "0001_rag_postgres_schema.py"
    spec = importlib.util.spec_from_file_location("initial_rag_migration", path)
    if spec is None or spec.loader is None:
        raise AssertionError("Could not load initial migration module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
