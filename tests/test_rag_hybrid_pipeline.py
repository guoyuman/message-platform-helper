from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from message_platform_helper.models import KnowledgeChunk
from message_platform_helper.rag import (
    ContextBuilder,
    DeterministicEmbeddingProvider,
    DocumentIngestionPipeline,
    HybridRetriever,
    QueryAnalyzer,
    RetrievalConfig,
    RuleBasedReranker,
    build_rag_service,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)


class HybridRagPipelineTests(unittest.TestCase):
    def test_deterministic_embedding_has_configured_dimension(self) -> None:
        provider = DeterministicEmbeddingProvider(dimensions=1536)

        vectors = provider.embed(["\u90ae\u4ef6\u53d1\u9001\u5931\u8d25\u600e\u4e48\u529e"])

        self.assertEqual(len(vectors), 1)
        self.assertEqual(len(vectors[0]), 1536)
        self.assertAlmostEqual(sum(value * value for value in vectors[0]), 1.0, places=6)

    def test_document_pipeline_uses_structure_first_chunking(self) -> None:
        pipeline = DocumentIngestionPipeline()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mail.md"
            path.write_text("# Mail\n\nIntro.\n\n## Troubleshooting\n\nCheck logs and channel config.", encoding="utf-8")
            result = pipeline.ingest_path(path, title="Mail Guide", tags=["mail"])

        self.assertEqual(result.document.title, "Mail Guide")
        self.assertEqual(len(result.chunks), 2)
        self.assertTrue(all(chunk.metadata["source"].endswith("mail.md") for chunk in result.chunks))
        self.assertTrue(all(chunk.metadata["tags"] == ["mail"] for chunk in result.chunks))

    def test_query_analyzer_and_context_builder_apply_intent_limits(self) -> None:
        question = "\u90ae\u4ef6\u53d1\u9001\u5931\u8d25\u600e\u4e48\u529e"
        analysis = QueryAnalyzer().analyze(question)
        chunks = [
            KnowledgeChunk(id=str(index), title=f"chunk {index}", content="content", source="test", score=1.0)
            for index in range(5)
        ]

        built = ContextBuilder().build(question, chunks, analysis)

        self.assertEqual(analysis.intent, "troubleshooting")
        self.assertIn("mail", analysis.filters["tags"])
        self.assertEqual(len(built.chunks), 3)
        self.assertEqual(len(built.citations), 3)

    def test_rrf_and_hybrid_retriever_merge_keyword_and_vector_results(self) -> None:
        keyword = [
            KnowledgeChunk(id="a", title="A", content="alpha", source="kw", score=3.0),
            KnowledgeChunk(id="b", title="B", content="beta", source="kw", score=2.0),
        ]
        vector = [
            KnowledgeChunk(id="b", title="B", content="beta", source="vec", score=0.9),
            KnowledgeChunk(id="c", title="C", content="gamma", source="vec", score=0.8),
        ]

        fused = reciprocal_rank_fusion([keyword, vector], limit=3)
        hybrid = HybridRetriever(FakeRetriever(keyword), FakeRetriever(vector)).retrieve("query", limit=3)

        self.assertEqual([chunk.id for chunk in fused], ["b", "a", "c"])
        self.assertEqual([chunk.id for chunk in hybrid], ["a", "b", "c"])

    def test_weighted_fusion_can_prefer_semantic_score_when_configured(self) -> None:
        keyword = [
            KnowledgeChunk(id="keyword", title="Keyword", content="literal match", source="kw", score=1.0),
            KnowledgeChunk(id="shared", title="Shared", content="both", source="kw", score=0.8),
        ]
        vector = [
            KnowledgeChunk(id="semantic", title="Semantic", content="meaning match", source="vec", score=1.0),
            KnowledgeChunk(id="shared", title="Shared", content="both", source="vec", score=0.9),
        ]

        fused = weighted_score_fusion(keyword, vector, config=RetrievalConfig(keyword_weight=0.2, vector_weight=0.8), limit=3)

        self.assertEqual([chunk.id for chunk in fused], ["semantic", "shared", "keyword"])

    def test_weighted_fusion_filters_low_similarity_vector_noise(self) -> None:
        vector = [
            KnowledgeChunk(id="noise", title="Noise", content="unrelated semantic neighbor", source="vec", score=0.1),
        ]

        fused = weighted_score_fusion([], vector, config=RetrievalConfig(), limit=3)

        self.assertEqual(fused, [])

    def test_rule_based_reranker_boosts_title_tag_and_keyword_matches(self) -> None:
        chunks = [
            KnowledgeChunk(id="1", title="Other", content="plain text", source="test", tags=[], score=1.0),
            KnowledgeChunk(id="2", title="Mail failure", content="check logs", source="test", tags=["mail"], score=0.1),
        ]

        ranked = RuleBasedReranker().rerank("mail failure", chunks, limit=2)

        self.assertEqual(ranked[0].id, "2")

    def test_rule_based_reranker_boosts_chinese_partial_title_matches(self) -> None:
        chunks = [
            KnowledgeChunk(
                id="1",
                title="\u6a21\u677f\u6e32\u67d3\u5931\u8d25\u6392\u67e5",
                content="\u68c0\u67e5\u6a21\u677f\u53d8\u91cf",
                source="test",
                tags=["template"],
                score=0.0164,
            ),
            KnowledgeChunk(
                id="2",
                title="\u6d88\u606f\u6a21\u677f\u914d\u7f6e",
                content="\u6253\u5f00\u6d88\u606f\u6a21\u677f\u8282\u70b9",
                source="test",
                tags=["template", "config"],
                score=0.0161,
            ),
        ]

        ranked = RuleBasedReranker().rerank("\u6d88\u606f\u6a21\u677f\u5982\u4f55\u914d\u7f6e", chunks, limit=2)

        self.assertEqual(ranked[0].id, "2")

    def test_query_analyzer_uses_readable_chinese_signals(self) -> None:
        analysis = QueryAnalyzer().analyze("\u6d88\u606f\u6a21\u677f\u5982\u4f55\u914d\u7f6e")

        self.assertEqual(analysis.intent, "template")
        self.assertIn("template", analysis.filters["tags"])
        self.assertIn("\u6d88\u606f\u6a21\u677f\u5982\u4f55\u914d\u7f6e", analysis.terms)

    def test_rag_service_falls_back_when_inferred_tags_are_too_narrow(self) -> None:
        chunks = [
            KnowledgeChunk(
                id="wrong-tag",
                title="\u6d88\u606f\u6a21\u677f\u914d\u7f6e",
                content="\u6d88\u606f\u6a21\u677f\u9700\u5148\u7ef4\u62a4\u6807\u9898\u3001\u5185\u5bb9\u548c\u53d8\u91cf\u3002",
                source="test",
                tags=["config"],
                score=1.0,
            )
        ]
        service = build_rag_service(TagAwareRetriever(chunks))

        results = service.retrieve("\u6d88\u606f\u6a21\u677f\u5982\u4f55\u914d\u7f6e", limit=1)

        self.assertEqual(results[0].id, "wrong-tag")

    def test_hybrid_retriever_logs_backend_failures(self) -> None:
        vector = [KnowledgeChunk(id="vec", title="Vector", content="semantic match", source="vec", score=0.9)]

        with self.assertLogs("message_platform_helper.rag.retrieval.hybrid_retriever", level="INFO") as logs:
            results = HybridRetriever(FailingRetriever(), FakeRetriever(vector)).retrieve("query", limit=1)

        self.assertEqual([chunk.id for chunk in results], ["vec"])
        self.assertTrue(any('"event": "rag.retrieval.backend.failed"' in line and '"stage": "keyword"' in line for line in logs.output))
        self.assertTrue(any('"event": "rag.retrieval.hybrid.fused"' in line for line in logs.output))


class FakeRetriever:
    def __init__(self, chunks: list[KnowledgeChunk]) -> None:
        self.chunks = chunks

    def retrieve(self, query: str, *, limit: int = 5, tags: list[str] | None = None, filters: dict | None = None) -> list[KnowledgeChunk]:
        return self.chunks[:limit]


class TagAwareRetriever:
    def __init__(self, chunks: list[KnowledgeChunk]) -> None:
        self.chunks = chunks

    def retrieve(self, query: str, *, limit: int = 5, tags: list[str] | None = None, filters: dict | None = None) -> list[KnowledgeChunk]:
        tag_filter = set(tags or [])
        if tag_filter:
            return [chunk for chunk in self.chunks if tag_filter.intersection(chunk.tags)][:limit]
        return self.chunks[:limit]


class FailingRetriever:
    def retrieve(self, query: str, *, limit: int = 5, tags: list[str] | None = None, filters: dict | None = None) -> list[KnowledgeChunk]:
        raise RuntimeError("backend unavailable")


if __name__ == "__main__":
    unittest.main()
