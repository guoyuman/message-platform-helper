# Message Platform Helper

`message-platform-helper` 是消息平台的多 Agent 编排层，目标是把模板翻译同步、业务消息配置、邮箱通道配置、发送策略配置串成企业级上线闭环。

核心能力：

- ReAct 编排：每个 Agent 按「推理摘要 -> 工具调用 -> 观察结果」执行，保留可审计轨迹，不暴露原始隐式思维链。
- 多 Agent 协作：模板翻译 Agent、业务消息配置 Agent、通道配置 Agent、发送策略 Agent 由统一 Orchestrator 调度。
- 长期记忆：SQLite 默认落盘，Redis 可选；保存用户画像、对话摘要、最近上下文和运行记录。
- RAG 知识库：内置轻量检索，推理层先判断是否需要检索，再把检索结果交给相关 Agent。
- 可上线边界：LLM、Redis、Java 消息平台、已有 `message-agent` 都通过环境变量接入，本地 mock 可直接跑测试。

## Quick Start

```powershell
cd "C:\Users\yuman\Desktop\workspace\message platform agent\message-platform-helper"
$env:PYTHONPATH = "src"
python -m unittest discover -s tests
python -m message_platform_helper.cli demo
python -m message_platform_helper.cli serve --port 8790
```

打开页面：

```text
http://127.0.0.1:8790
```

## Runtime Env

- `MESSAGE_HELPER_DATA_DIR`: SQLite 数据目录，默认使用系统临时目录。
- `MESSAGE_HELPER_REDIS_URL`: 配置后优先使用 Redis 保存记忆缓存和频控计数。
- `MESSAGE_HELPER_LLM_BASE_URL`, `MESSAGE_HELPER_LLM_API_KEY`, `MESSAGE_HELPER_LLM_MODEL`: OpenAI-compatible LLM。
- `MESSAGE_HELPER_PLATFORM_BASE_URL`: Java 消息平台 HTTP 地址。
- `MESSAGE_HELPER_BUSINESS_AGENT_URL`: 已有 `message-agent` 地址，例如 `http://127.0.0.1:8787`。
- `MESSAGE_HELPER_PLATFORM_HEADERS_JSON`: 透传给 Java 平台的租户、鉴权、网关 Header。

## HTTP API

- `POST /api/chat`: 统一入口，自动判断是否检索、调哪些 Agent。
- `POST /api/knowledge/ingest`: 写入知识库。
- `GET /api/memory?sessionId=...`: 查看长期记忆。
- `POST /api/strategy/evaluate`: 发送策略拦截规则本地评估。
- `GET /api/health`: 健康检查。

更多上线说明见 [docs/architecture.md](docs/architecture.md) 和 [docs/deployment-checklist.md](docs/deployment-checklist.md)。
## RAG 知识库

RAG 全流程、数据库位置、写入/统计/检索/聊天问答接口见 [docs/rag-process.md](docs/rag-process.md)。
启动默认知识来自 [knowledge/](knowledge/) 下的 Markdown 文档，服务启动时会按 RAG 入库流程切分并写入 SQLite。
