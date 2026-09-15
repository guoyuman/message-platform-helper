"""Composable RAG service."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from ..application.security import TenantContext
from ..infrastructure.observability import elapsed_ms, start_span, update_observation
from ..llm import LLMClient
from ..models import JsonDict, KnowledgeChunk
from .citation_builder import CitationBuilder
from .context_builder import ContextBuilder
from .prompt_builder import RagPromptBuilder
from .query_analyzer import QueryAnalyzer
from .reranker import BGEReranker, Reranker
from .retriever import Retriever


LOGGER = logging.getLogger(__name__)


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
        started = time.perf_counter()
        with start_span(
            "rag.retrieve",
            as_type="retriever",
            input={"query": query, "limit": limit, "tags": tags},
            metadata={"tenant_namespace": getattr(tenant_context, "knowledge_namespace", "")},
        ) as observation:
            analysis = self.query_analyzer.analyze(query)
            # Explicit tags are hard filters. Inferred intent tags are only hints;
            # making them hard filters is a common recall killer for Chinese queries.
            merged_tags = tags
            backend_filters = {key: value for key, value in analysis.filters.items() if key != "tags"}
            retrieval_limit = max(limit, 20)
            _log_rag_event(
                "rag.retrieve.start",
                query=query,
                limit=limit,
                retrieval_limit=retrieval_limit,
                explicit_tags=tags,
                inferred_filters=analysis.filters,
                merged_tags=merged_tags,
                tenant_namespace=getattr(tenant_context, "knowledge_namespace", ""),
            )
            retrieved = self.retriever.retrieve(query, limit=retrieval_limit, tags=merged_tags, filters=backend_filters)
            _log_rag_event(
                "rag.retrieve.recalled",
                query=query,
                count=len(retrieved),
                top_chunks=_chunk_snapshots(retrieved),
            )
            reranked = self.reranker.rerank(query, retrieved, limit=limit)
            selected_ids = {chunk.id for chunk in reranked}
            _log_rag_event(
                "rag.rerank.completed",
                query=query,
                candidate_count=len(retrieved),
                selected_count=len(reranked),
                before_top_chunks=_chunk_snapshots(retrieved),
                after_top_chunks=_chunk_snapshots(reranked),
                dropped_chunk_ids=[chunk.id for chunk in retrieved if chunk.id not in selected_ids][:10],
            )
            update_observation(
                observation,
                output={"chunks": _chunk_snapshots(reranked), "chunk_ids": [chunk.id for chunk in reranked]},
                metadata={
                    "intent": analysis.intent,
                    "terms": analysis.terms,
                    "filters": analysis.filters,
                    "candidate_count": len(retrieved),
                    "selected_count": len(reranked),
                    "latency_ms": round(elapsed_ms(started), 3),
                },
            )
            return reranked

    def answer(
        self,
        question: str,
        *,
        chunks: list[KnowledgeChunk] | None = None,
        limit: int = 5,
        tenant_context: TenantContext | None = None,
        llm: LLMClient | None = None,
    ) -> JsonDict:
        with start_span("rag.answer", input={"question": question, "limit": limit}) as observation:
            analysis = self.query_analyzer.analyze(question)
            selected = chunks if chunks is not None else self.retrieve(question, limit=limit, tenant_context=tenant_context)
            _log_rag_event(
                "rag.answer.input",
                question=question,
                supplied_chunks=chunks is not None,
                chunk_count=len(selected),
                chunks=_chunk_snapshots(selected),
                analysis={"intent": analysis.intent, "terms": analysis.terms, "filters": analysis.filters},
            )
            built_context = self.context_builder.build(question, selected, analysis)
            selected = built_context.chunks
            prompt = self.prompt_builder.build(question, selected)
            _log_rag_event(
                "rag.prompt.built",
                question=question,
                selected_count=len(selected),
                selected_chunks=_chunk_snapshots(selected),
                context_chars=len(built_context.context),
                prompt_chars=len(prompt),
                context_preview=built_context.context[:500],
                prompt_preview=prompt[:500],
                citation_count=len(built_context.citations),
            )
            if not selected:
                result = {
                    "ok": False,
                    "summary": "knowledge not found",
                    "answer": "\u77e5\u8bc6\u5e93\u91cc\u8fd8\u6ca1\u6709\u68c0\u7d22\u5230\u53ef\u7528\u5185\u5bb9\u3002\u8bf7\u5148\u5199\u5165\u4f7f\u7528\u8bf4\u660e\u6216\u6392\u67e5\u6587\u6863\uff0c\u518d\u91cd\u8bd5\u8fd9\u4e2a\u95ee\u9898\u3002",
                    "citations": built_context.citations,
                    "prompt": prompt,
                    "issues": [
                        {
                            "code": "knowledge.not_found",
                            "message": "No knowledge chunks matched the question.",
                            "severity": "warning",
                        }
                    ],
                }
                update_observation(observation, output=result, level="WARNING", metadata={"selected_count": 0})
                return result
            generated_answer = _llm_answer(llm, question, prompt) if llm is not None else ""
            result = {
                "ok": True,
                "summary": f"answered from {len(selected)} knowledge chunk(s)",
                "answer": generated_answer or _extractive_answer(question, selected),
                "citations": built_context.citations,
                "context": built_context.context,
                "prompt": prompt,
            }
            update_observation(
                observation,
                output=result,
                metadata={
                    "selected_count": len(selected),
                    "citation_count": len(built_context.citations),
                    "context_chars": len(built_context.context),
                    "prompt_chars": len(prompt),
                },
            )
            return result


def build_rag_service(retriever: Retriever, reranker: Reranker | None = None) -> RagService:
    return RagService(
        retriever=retriever,
        reranker=reranker or BGEReranker(),
        prompt_builder=RagPromptBuilder(),
        citation_builder=CitationBuilder(),
    )


def build_postgres_rag_service(
    session: object,
    embedding_provider: object | None = None,
    reranker: Reranker | None = None,
    config: dict | None = None,
) -> RagService:
    from .embedding import DeterministicEmbeddingProvider
    from .repository import ChunkRepository, VectorRepository
    from .retrieval import HybridRetriever, KeywordRetriever, RetrievalConfig, VectorRetriever

    provider = embedding_provider if embedding_provider is not None else DeterministicEmbeddingProvider()
    retrieval = dict((config or {}).get("retrieval") or {})
    retrieval_config = RetrievalConfig(
        keyword_weight=float(retrieval.get("keyword_weight", 0.5)),
        vector_weight=float(retrieval.get("vector_weight", 0.5)),
        similarity_threshold=float(retrieval.get("similarity_threshold", 0.0)),
    )
    keyword_retriever = KeywordRetriever(ChunkRepository(session))  # type: ignore[arg-type]
    vector_retriever = VectorRetriever(VectorRepository(session), provider)  # type: ignore[arg-type]
    retriever = HybridRetriever(
        keyword_retriever,
        vector_retriever,
        keyword_top_k=int(retrieval.get("keyword_top_k", 50)),
        vector_top_k=int(retrieval.get("vector_top_k", 50)),
        final_top_k=int(retrieval.get("candidate_top_k", 50)),
        config=retrieval_config,
    )
    rerank = dict((config or {}).get("reranker") or {})
    reranker = reranker or BGEReranker(
        model_name=str(rerank.get("model") or "BAAI/bge-reranker-v2-m3"),
        model_weight=float(rerank.get("model_weight", 0.85)),
        metadata_weight=float(rerank.get("metadata_weight", 0.10)),
        rule_weight=float(rerank.get("rule_weight", 0.05)),
    )
    return build_rag_service(retriever, reranker)


def _log_rag_event(event: str, **payload: object) -> None:
    LOGGER.info(json.dumps({"event": event, **payload}, ensure_ascii=False, default=str))


def _chunk_snapshots(chunks: list[KnowledgeChunk], *, limit: int = 8) -> list[JsonDict]:
    return [_chunk_snapshot(chunk) for chunk in chunks[:limit]]


def _chunk_snapshot(chunk: KnowledgeChunk) -> JsonDict:
    return {
        "id": chunk.id,
        "title": chunk.title,
        "source": chunk.source,
        "tags": chunk.tags,
        "score": round(float(chunk.score or 0.0), 6),
        "content_preview": " ".join(chunk.content.split())[:180],
    }


def _extractive_answer(question: str, chunks: list[KnowledgeChunk]) -> str:
    lowered = question.lower()
    is_failure = any(keyword in question for keyword in ("\u5931\u8d25", "\u62a5\u9519", "\u6392\u67e5", "\u9519\u8bef", "\u5f02\u5e38")) or any(
        keyword in lowered for keyword in ("fail", "error", "troubleshoot")
    )
    lines: list[str] = []
    if is_failure:
        lines.append("\u53ef\u4ee5\u6309\u8fd9\u6761\u94fe\u8def\u6392\u67e5\uff1a\u5148\u5b9a\u4f4d\u5931\u8d25\u9636\u6bb5\uff0c\u518d\u9a8c\u8bc1\u914d\u7f6e\u3001\u6a21\u677f\u3001\u63a5\u6536\u4eba\u3001\u7b56\u7565\u548c\u901a\u9053\u3002")
    else:
        lines.append("\u53ef\u4ee5\u6309\u77e5\u8bc6\u5e93\u91cc\u7684\u6d41\u7a0b\u6267\u884c\uff1a\u5148\u914d\u7f6e\u57fa\u7840\u5bf9\u8c61\uff0c\u518d\u9884\u89c8\u6821\u9a8c\uff0c\u6700\u540e\u4fdd\u5b58\u53d1\u5e03\u5e76\u9a8c\u8bc1\u3002")
    for index, chunk in enumerate(chunks, start=1):
        excerpt = " ".join(chunk.content.split())
        if len(excerpt) > 260:
            excerpt = excerpt[:259].rstrip() + "..."
        lines.append(f"{index}. {chunk.title}: {excerpt}")
    lines.append("\u5efa\u8bae\u5b9e\u9645\u5904\u7406\u65f6\u4fdd\u7559\u53d1\u9001\u8bb0\u5f55\u91cc\u7684\u9519\u8bef\u7801\u3001\u901a\u9053\u3001\u63a5\u6536\u4eba\u3001\u6a21\u677f\u7f16\u7801\u548c\u8bf7\u6c42 payload\uff0c\u4fbf\u4e8e\u7ee7\u7eed\u8ffd\u8e2a\u3002")
    return "\n".join(lines)


def _llm_answer(llm: LLMClient, question: str, prompt: str) -> str:
    response = llm.answer(
        prompt,
        system_prompt=(
            "你是企业消息平台知识库助手。"
            "请只依据用户问题和给定知识片段回答；信息不足时说明缺口。"
            "回答要直接、可执行，并保留关键配置名、错误码和步骤。"
        ),
    )
    return str(response.get("answer") or "").strip()


__all__ = ["RagService", "build_postgres_rag_service", "build_rag_service"]
