from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from message_platform_helper.rag import MarkdownChunkStrategy
from message_platform_helper.rag import MarkdownLoader, MarkdownParser

from tests.fakes import InMemoryKnowledgeBase


class RagIngestionPipelineTests(unittest.TestCase):
    def test_markdown_loader_returns_document_with_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "email_failure.md"
            path.write_text(
                "\n".join(
                    [
                        "---",
                        "title: Email failure",
                        "tags: mail, ops",
                        "---",
                        "# Email failure",
                        "",
                        "Check delivery logs.",
                    ]
                ),
                encoding="utf-8",
            )

            document = MarkdownLoader().load(path)

        self.assertEqual(document.title, "Email failure")
        self.assertEqual(document.metadata.format, "markdown")
        self.assertIn("mail", document.metadata.tags)
        self.assertIn("Check delivery logs.", document.content)

    def test_markdown_parser_preserves_heading_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guide.md"
            path.write_text("# Mail\n\nIntro.\n\n## Troubleshooting\n\nKeep paragraphs whole.", encoding="utf-8")
            parsed = MarkdownParser().parse(MarkdownLoader().load(path))

        self.assertEqual([section.title for section in parsed], ["Mail", "Troubleshooting"])
        self.assertEqual(parsed[1].level, 2)
        self.assertEqual(parsed[1].content, "Keep paragraphs whole.")

    def test_knowledge_base_ingests_chunks_with_parent_child_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guide.md"
            path.write_text("# Mail\n\nIntro.\n\n## Troubleshooting\n\nCheck logs.", encoding="utf-8")
            kb = InMemoryKnowledgeBase()

            chunks = kb.ingest_file(path, tags=["mail"], strategy=MarkdownChunkStrategy())

        self.assertEqual(len(chunks), 2)
        self.assertEqual(kb.count(), 2)
        self.assertTrue(any(chunk.title == "Troubleshooting" for chunk in chunks))
        self.assertTrue(all(chunk.source.endswith("guide.md") for chunk in chunks))
        self.assertTrue(all(chunk.tags == ["mail"] for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
