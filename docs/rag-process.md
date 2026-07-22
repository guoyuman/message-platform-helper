# RAG 知识库流程

消息平台助手的 RAG 知识库统一使用 PostgreSQL + pgvector。完整链路是：

1. 通过 `/api/knowledge/ingest` 或 CLI `ingest` 写入文本、Markdown、PDF、DOCX。
2. loader 提取文本，并记录文件名、文件类型、文本长度。
3. parser 按标题、段落、页文本和表格生成结构化 section。
4. chunker 使用中文友好的递归切片，默认 `chunk_size=900`、`chunk_overlap=120`。
5. embedding provider 为每个 chunk 生成固定维度向量。
6. PostgreSQL 表 `documents`、`chunks` 保存文档、chunk、metadata、embedding。
7. 检索时并行使用 PostgreSQL FTS、中文 substring fallback 和 pgvector 相似度，再通过 RRF 与 reranker 排序。

## 配置

- `MESSAGE_HELPER_RAG_DATABASE_URL`: PostgreSQL SQLAlchemy URL，默认 `postgresql+psycopg://message_helper:message_helper@127.0.0.1:5432/message_helper`。
- `MESSAGE_HELPER_EMBEDDING_PROVIDER`: `deterministic`、`openai` 或 `bge`。
- `MESSAGE_HELPER_EMBEDDING_DIMENSIONS`: 默认 `1536`，必须和数据库 `vector(1536)` 一致。

## 写入

```http
POST /api/knowledge/ingest
Content-Type: application/json

{
  "path": "knowledge/计算机网络.docx",
  "tags": ["network"],
  "replace": true
}
```

文本写入仍可使用：

```json
{
  "title": "邮件发送失败排查",
  "content": "先检查 SMTP 主机、端口、SSL/TLS、授权码、发件账号和网络连通性。",
  "source": "manual",
  "tags": ["mail", "failure"]
}
```

响应会返回 `ingestion.file_name`、`ingestion.file_type`、`ingestion.text_length`、`ingestion.chunk_count` 和 `ingestion.embedding_dimensions`。

## 检索

```http
POST /api/knowledge/search
Content-Type: application/json

{"query": "消息发送失败原因如何排查", "limit": 5}
```

中文查询主要依赖向量检索和 substring fallback，PostgreSQL FTS 用于英文、数字、接口名和混合文本增强召回。
