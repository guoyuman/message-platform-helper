"""RAG package exports."""

_EXPORT_MODULES = {
    "CitationBuilder": ".citation_builder",
    "BGEEmbeddingProvider": ".embedding",
    "BGEReranker": ".reranker",
    "Chunk": ".models",
    "ChunkStrategy": ".chunker",
    "ContextBuilder": ".context_builder",
    "CrossEncoderReranker": ".reranker",
    "DeterministicEmbeddingProvider": ".embedding",
    "Document": ".models",
    "DocumentIngestionPipeline": ".ingestion",
    "DocumentLoader": ".loader",
    "DocumentMetadata": ".models",
    "DocumentSection": ".models",
    "EmbeddingProvider": ".embedding",
    "HybridRetriever": ".retrieval",
    "IngestionResult": ".ingestion",
    "KeywordRetriever": ".retrieval",
    "KnowledgeBase": ".knowledge_base",
    "KnowledgeBaseRetriever": ".retriever",
    "KnowledgeDocument": ".knowledge_base",
    "LoaderFactory": ".loader",
    "MarkdownChunkStrategy": ".chunker",
    "MarkdownLoader": ".loader",
    "MarkdownParser": ".parser",
    "ParentChildChunkStrategy": ".chunker",
    "ParentDocument": ".models",
    "OpenAIEmbeddingProvider": ".embedding",
    "QueryAnalysis": ".query_analyzer",
    "QueryAnalyzer": ".query_analyzer",
    "RagPromptBuilder": ".prompt_builder",
    "RagService": ".service",
    "Reranker": ".reranker",
    "Retriever": ".retriever",
    "RecursiveChunkStrategy": ".chunker",
    "RetrievalConfig": ".retrieval",
    "RuleBasedReranker": ".reranker",
    "ScoreReranker": ".reranker",
    "VectorRetriever": ".retrieval",
    "XlsxLoader": ".loader",
    "bm25_like": ".knowledge_base",
    "build_rag_service": ".service",
    "build_postgres_rag_service": ".service",
    "load_default_knowledge_documents": ".knowledge_base",
    "seed_default_knowledge": ".knowledge_base",
    "split_text": ".knowledge_base",
    "tokenize": ".knowledge_base",
    "reciprocal_rank_fusion": ".retrieval",
    "weighted_score_fusion": ".retrieval",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str):
    if name not in _EXPORT_MODULES:
        raise AttributeError(name)
    from importlib import import_module

    module = import_module(_EXPORT_MODULES[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
