from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from message_platform_helper.rag import Document, DocumentMetadata, DocumentSection, MarkdownChunkStrategy, ParentChildChunkStrategy
from message_platform_helper.rag import MarkdownLoader, MarkdownParser
from message_platform_helper.rag.loader.docx_loader import read_docx_text
from message_platform_helper.rag.parser.docx_parser import DocxParser
from message_platform_helper.rag.parser.pdf_parser import PdfParser
from message_platform_helper.rag import XlsxLoader

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

    def test_markdown_chunking_uses_paragraph_boundaries_before_size_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guide.md"
            path.write_text("# Mail\n\nIntro one.\n\nIntro two.\n\n## Troubleshooting\n\nCheck logs.", encoding="utf-8")
            kb = InMemoryKnowledgeBase()

            chunks = kb.ingest_file(path, tags=["mail"], strategy=MarkdownChunkStrategy())

        self.assertEqual([chunk.title for chunk in chunks], ["Mail", "Mail", "Troubleshooting"])
        self.assertEqual([chunk.content for chunk in chunks], ["Intro one.", "Intro two.", "Check logs."])

    def test_xlsx_loader_extracts_sheet_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "roles.xlsx"
            with ZipFile(path, "w") as workbook:
                workbook.writestr(
                    "xl/workbook.xml",
                    """<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
                    xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
                    <sheets><sheet name="岗位表" sheetId="1" r:id="rId1"/></sheets></workbook>""",
                )
                workbook.writestr(
                    "xl/_rels/workbook.xml.rels",
                    """<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
                    <Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>""",
                )
                workbook.writestr(
                    "xl/worksheets/sheet1.xml",
                    """<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
                    <sheetData><row><c t="inlineStr"><is><t>单位</t></is></c>
                    <c t="inlineStr"><is><t>岗位</t></is></c></row></sheetData></worksheet>""",
                )

            document = XlsxLoader().load(path)

        self.assertEqual(document.metadata.format, "xlsx")
        self.assertIn("# 岗位表", document.content)
        self.assertIn("单位 | 岗位", document.content)

    def test_parent_child_chunker_keeps_semantic_table_rows_together(self) -> None:
        document = Document(
            id="doc-table",
            title="岗位表",
            content="",
            metadata=DocumentMetadata(format="docx", source="table.docx"),
        )
        section = DocumentSection(
            title="岗位表",
            level=1,
            kind="table",
            content="\n".join(["单位 | 岗位", *[f"单位{i} | 岗位{i}" for i in range(12)]]),
        )

        chunks = ParentChildChunkStrategy(child_chunk_size=70).chunk(document, [section])

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.content.startswith("单位 | 岗位") for chunk in chunks))
        self.assertTrue(all(chunk.metadata["section_kind"] == "table" for chunk in chunks))

    def test_pdf_parser_preserves_page_table_and_image_sections(self) -> None:
        document = Document(
            id="pdf",
            title="Guide",
            content="[Page 2]\n正文\n\n[Table 2.1]\nA | B\n1 | 2\n\n[Image 2.1: page=2, size=20x30]",
            metadata=DocumentMetadata(format="pdf", source="guide.pdf"),
        )

        sections = PdfParser().parse(document)

        self.assertEqual([section.kind for section in sections], ["paragraph", "table", "image"])
        self.assertEqual(sections[1].metadata["page"], 2)
        self.assertEqual(sections[2].title, "Guide Page 2 Image 2.1")

    def test_docx_loader_preserves_paragraph_table_and_image_markers_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mixed.docx"
            _write_mixed_docx(path)

            content = read_docx_text(path)
            sections = DocxParser().parse(
                Document(id="docx", title="Mixed", content=content, metadata=DocumentMetadata(format="docx", source=str(path)))
            )

        self.assertLess(content.index("Intro"), content.index("A | B"))
        self.assertLess(content.index("A | B"), content.index("[Image 1:"))
        self.assertEqual([section.kind for section in sections], ["paragraph", "table", "image"])


def _write_mixed_docx(path: Path) -> None:
    import base64

    from docx import Document as DocxDocument  # type: ignore[import-untyped]

    image_path = path.with_suffix(".png")
    image_path.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
    )
    document = DocxDocument()
    document.add_paragraph("Intro")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "A"
    table.cell(0, 1).text = "B"
    table.cell(1, 0).text = "1"
    table.cell(1, 1).text = "2"
    paragraph = document.add_paragraph()
    paragraph.add_run().add_picture(str(image_path))
    document.save(path)


if __name__ == "__main__":
    unittest.main()
