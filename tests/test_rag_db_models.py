from __future__ import annotations

import unittest

from message_platform_helper.rag.db import Base, ChunkRecord, DocumentRecord


class RagDatabaseModelTests(unittest.TestCase):
    def test_documents_and_chunks_tables_are_declared(self) -> None:
        self.assertIn("documents", Base.metadata.tables)
        self.assertIn("chunks", Base.metadata.tables)

    def test_document_columns_match_target_schema(self) -> None:
        columns = DocumentRecord.__table__.columns

        self.assertEqual(set(columns.keys()), {"id", "title", "content", "source", "metadata", "created_at", "updated_at"})
        self.assertTrue(columns["id"].primary_key)
        self.assertEqual(columns["metadata"].type.__class__.__name__, "JSONB")

    def test_chunk_columns_and_indexes_match_target_schema(self) -> None:
        columns = ChunkRecord.__table__.columns
        indexes = {index.name: index for index in ChunkRecord.__table__.indexes}

        self.assertEqual(
            set(columns.keys()),
            {
                "id",
                "document_id",
                "parent_id",
                "content",
                "embedding",
                "metadata",
                "chunk_index",
                "content_tsv",
                "created_at",
                "updated_at",
            },
        )
        self.assertTrue(columns["id"].primary_key)
        self.assertEqual(str(columns["embedding"].type), "VECTOR(1536)")
        self.assertEqual(columns["content_tsv"].type.__class__.__name__, "TSVECTOR")
        self.assertEqual(indexes["ix_chunks_embedding_hnsw"].dialect_options["postgresql"]["using"], "hnsw")
        self.assertEqual(indexes["ix_chunks_content_tsv"].dialect_options["postgresql"]["using"], "gin")


if __name__ == "__main__":
    unittest.main()
