# Message Platform Helper

`message-platform-helper` 是消息平台的多 Agent 编排层，用于把模板翻译同步、邮件通道配置和知识库问答串成可审计的助手流程。

核心能力：

- ReAct 编排：每个 Agent 按计划、工具调用、观察结果执行，保留审计轨迹。
- 多 Agent 协作：模板翻译同步、通道配置和知识问答由统一 Orchestrator 调度。
- PostgreSQL 持久化：长期记忆、运行记录、频控计数、RAG 文档和向量统一写入 PostgreSQL。
- RAG 知识库：支持 Markdown、文本、PDF、DOCX 入库，使用 PostgreSQL FTS + pgvector 混合检索。
- LLM通过环境变量接入。

## Quick Start

```powershell
docker compose up -d 

copy .env.example .env
# Edit .env locally, keep real keys out of git.

uvicorn message_platform_helper.api.app:create_app --factory --reload --host 127.0.0.1 --port 8790
python -m unittest discover -s tests
python -m message_platform_helper.cli demo

```

前端单独启动：

```powershell
cd web
npm install
npm run dev
```

打开页面：

```text
http://127.0.0.1:5173
```

后端仅提供 `/api/*` 接口，前端开发服务通过 Vite proxy 转发到 `http://127.0.0.1:8790`。

## Runtime Env

- `MESSAGE_HELPER_RAG_DATABASE_URL`: PostgreSQL SQLAlchemy URL，默认 `postgresql+psycopg://message_helper:message_helper@127.0.0.1:5432/message_helper`。
- `MESSAGE_HELPER_REDIS_URL`: 配置后优先使用 Redis 保存记忆缓存和频控计数。
- `MESSAGE_HELPER_LLM_BASE_URL`, `MESSAGE_HELPER_LLM_API_KEY`, `MESSAGE_HELPER_LLM_MODEL`: OpenAI-compatible LLM。
- `MESSAGE_HELPER_PLATFORM_BASE_URL`: Java 消息平台 HTTP 地址。

[//]: # (- `MESSAGE_HELPER_BUSINESS_AGENT_URL`: 已有 `message-agent` 地址，例如 `http://127.0.0.1:8787`。)
- `MESSAGE_HELPER_PLATFORM_HEADERS_JSON`: 透传给 Java 平台的租户、鉴权、网关 Header。
- `MESSAGE_HELPER_EMBEDDING_PROVIDER`: `deterministic`、`openai` 或 `bge`。
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`: 同时配置后启用 Langfuse trace。
- `LANGFUSE_BASE_URL`: Langfuse 地址，云端默认 `https://cloud.langfuse.com`，自部署时改为实例地址。
- `LANGFUSE_TRACING_ENVIRONMENT`: Langfuse environment 标签，默认建议使用 `development`、`staging` 或 `production`。
- `MESSAGE_HELPER_LANGFUSE_ENABLED`: 显式开启/关闭 Langfuse；未配置 key 时仍保持 no-op。

Langfuse 观测覆盖：

- 每个请求一个 trace，关联现有 `trace_id`、`request_id`、session、租户/用户 hash。
- LLM `complete_json` 和 function-calling 记录 generation、模型、输入输出、usage 和耗时。
- RAG 检索记录 retriever span、候选数、rerank 后 chunk id、引用数和耗时。
- 每次工具执行记录 tool span、工具名、Agent、成功状态和耗时。
- 请求结束记录 `task_success` 与 `request_latency_ms` score。密码、API key、Authorization、Cookie 和邮箱会脱敏/哈希。

安装观测依赖：

```bash
pip install -e '.[server,observability]'
```

## HTTP API

- `POST /api/chat/stream`: 统一入口，自动判断是否检索、调哪些 Agent。
- `POST /api/knowledge/ingest`: 写入文本或通过 `path` 写入 Markdown/PDF/DOCX 文件。
- `POST /api/knowledge/search`: 知识库检索。
- `GET /api/knowledge/stats`: 知识库统计。
- `GET /api/memory?sessionId=...`: 查看长期记忆。
- `POST /api/strategy/evaluate`: 本地评估发送策略拦截规则。
- `GET /api/health`: 健康检查。

## RAG 知识库

RAG 全流程、写入/统计/检索/聊天问答接口见 [docs/rag-process.md](docs/rag-process.md)。启动默认知识来自 [knowledge/](knowledge/) 下的 Markdown 文档，服务启动时会按 RAG 入库流程切分、生成 embedding 并写入 PostgreSQL。
