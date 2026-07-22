"""Composable RAG service."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..application.security import TenantContext
from ..models import JsonDict, KnowledgeChunk
from .citation_builder import CitationBuilder
from .context_builder import ContextBuilder
from .prompt_builder import RagPromptBuilder
from .query_analyzer import QueryAnalyzer
from .reranker import Reranker, RuleBasedReranker, ScoreReranker
from .retriever import Retriever


@dataclass
class RagService:
    retriever: Retriever
    reranker: Reranker
    prompt_builder: RagPromptBuilder
    citation_builder: CitationBuilder
    query_analyzer: QueryAnalyzer = field(default_factory=QueryAnalyzer)
    context_builder: ContextBuilder = field(default_factory=ContextBuilder)

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
        tags: list[str] | None = None,
        tenant_context: TenantContext | None = None,
    ) -> list[KnowledgeChunk]:
        analysis = self.query_analyzer.analyze(query)
        merged_tags = _merge_tags(tags, analysis.filters.get("tags"))
        retrieval_limit = max(limit, 20)
        retrieved = self.retriever.retrieve(query, limit=retrieval_limit, tags=merged_tags, filters=analysis.filters)
        if not retrieved and tags is None and merged_tags:
            retrieved = self.retriever.retrieve(query, limit=retrieval_limit, tags=None, filters=None)
        return self.reranker.rerank(query, retrieved, limit=limit)

    def answer(
        self,
        question: str,
        *,
        chunks: list[KnowledgeChunk] | None = None,
        limit: int = 5,
        tenant_context: TenantContext | None = None,
    ) -> JsonDict:
        analysis = self.query_analyzer.analyze(question)
        selected = chunks if chunks is not None else self.retrieve(question, limit=limit, tenant_context=tenant_context)
        built_context = self.context_builder.build(question, selected, analysis)
        selected = built_context.chunks
        if not selected:
            return {
                "ok": False,
                "summary": "knowledge not found",
                "answer": "知识库里还没有检索到可用内容。请先写入使用说明或排查文档，再重试这个问题。",
                "citations": built_context.citations,
                "prompt": self.prompt_builder.build(question, []),
                "issues": [
                    {
                        "code": "knowledge.not_found",
                        "message": "No knowledge chunks matched the question.",
                        "severity": "warning",
                    }
                ],
            }
        return {
            "ok": True,
            "summary": f"answered from {len(selected)} knowledge chunk(s)",
            "answer": _extractive_answer(question, selected),
            "citations": built_context.citations,
            "context": built_context.context,
            "prompt": self.prompt_builder.build(question, selected),
        }


def build_rag_service(retriever: Retriever, reranker: Reranker | None = None) -> RagService:
    return RagService(
        retriever=retriever,
        reranker=reranker or ScoreReranker(),
        prompt_builder=RagPromptBuilder(),
        citation_builder=CitationBuilder(),
    )


def build_postgres_rag_service(session: object, embedding_provider: object | None = None, reranker: Reranker | None = None) -> RagService:
    from .embedding import DeterministicEmbeddingProvider
    from .repository import ChunkRepository, VectorRepository
    from .retrieval import HybridRetriever, KeywordRetriever, VectorRetriever

    provider = embedding_provider if embedding_provider is not None else DeterministicEmbeddingProvider()
    keyword_retriever = KeywordRetriever(ChunkRepository(session))  # type: ignore[arg-type]
    vector_retriever = VectorRetriever(VectorRepository(session), provider)  # type: ignore[arg-type]
    return build_rag_service(HybridRetriever(keyword_retriever, vector_retriever), reranker or RuleBasedReranker())


def _merge_tags(explicit: list[str] | None, inferred: list[str] | None) -> list[str] | None:
    if explicit is None and inferred is None:
        return None
    result: list[str] = []
    for tag in [*(explicit or []), *(inferred or [])]:
        if tag not in result:
            result.append(tag)
    return result


def _extractive_answer(question: str, chunks: list[KnowledgeChunk]) -> str:
    lowered = question.lower()
    is_failure = any(keyword in question for keyword in ("失败", "报错", "排查", "错误", "异常")) or any(
        keyword in lowered for keyword in ("fail", "error", "troubleshoot")
    )
    lines: list[str] = []
    if is_failure:
        lines.append("可以按这条链路排查：先定位失败阶段，再验证配置、模板、接收人、策略和通道。")
    else:
        lines.append("可以按知识库里的流程执行：先配置基础对象，再预览校验，最后保存发布并验证。")
    for index, chunk in enumerate(chunks, start=1):
        excerpt = " ".join(chunk.content.split())
        if len(excerpt) > 260:
            excerpt = excerpt[:259].rstrip() + "..."
        lines.append(f"{index}. {chunk.title}: {excerpt}")
    lines.append("建议实际处理时保留发送记录里的错误码、通道、接收人、模板编码和请求 payload，便于继续追踪。")
    return "\n".join(lines)


__all__ = ["RagService", "build_postgres_rag_service", "build_rag_service"]
