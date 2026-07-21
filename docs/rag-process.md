# RAG 知识库流程

## 目标

消息平台助手的 RAG 知识库用于回答消息平台使用说明、发送失败原因排查、通道配置规范、模板变量问题等指导类问题。完整链路是：

1. 通过 `/api/knowledge/ingest` 或 CLI `ingest` 写入文档。
2. 服务按段落切分为 chunk，提取中英文 token。
3. chunk 落盘到 SQLite 表 `knowledge_chunks`。
4. `/api/knowledge/search` 或 `/api/chat` 根据问题检索相关 chunk。
5. `KnowledgeAgent` 将检索结果整理成回答，并在响应里返回引用来源。

## 数据库位置

未设置 `MESSAGE_HELPER_DATA_DIR` 时，源码运行默认使用工作区根目录的 `knowledge.sqlite3`、`memory.sqlite3`、`counters.sqlite3`。如果工作区根目录还没有这些库，则回退到 `message-platform-helper/data`。

每个知识库接口都会返回 `db_path`，验收时以响应里的路径为准。旧版本曾默认写入系统临时目录 `%TEMP%/message-platform-helper/knowledge.sqlite3`，因此手工查看工作区根目录数据库时会看到仍是默认三条。新版本启动时会把旧临时知识库中的 chunk 导入到当前默认库。

## 接口

### 写入知识

```http
POST /api/knowledge/ingest
Content-Type: application/json
```

```json
{
  "title": "消息发送失败排查补充",
  "content": "发送失败时先看发送记录错误码，再检查模板变量、接收人、策略拦截和通道测试结果。",
  "source": "ops-manual",
  "tags": ["troubleshooting", "send"],
  "replace": true
}
```

默认 `replace=true`，同一 `source + title` 会替换旧 chunk，避免重复入库。响应里的 `before_chunks`、`after_chunks`、`delta_chunks` 可以直接验证数据库条数变化。

### 查看统计

```http
GET /api/knowledge/stats
```

返回 `db_path`、`total_chunks`、按 source/tag 汇总和最新 chunk。

### 检索知识

```http
GET /api/knowledge/search?query=消息发送失败原因如何排查&limit=5
```

也支持：

```http
POST /api/knowledge/search
Content-Type: application/json
```

```json
{
  "query": "邮件发送失败如何排查",
  "limit": 5,
  "tags": ["mail"]
}
```

### 聊天问答

```http
POST /api/chat
Content-Type: application/json
```

```json
{
  "sessionId": "demo",
  "text": "消息发送失败原因如何排查？",
  "payload": {},
  "dryRun": true
}
```

响应中：

- `decision.need_retrieval=true` 表示触发 RAG。
- `decision.selected_agents` 包含 `knowledge`。
- `retrieved` 是检索到的知识 chunk。
- `results[].output.answer` 是整理后的指导答案。
- `results[].output.citations` 是引用来源。

## 默认知识

启动时会从 `knowledge/*.md` 读取默认知识文档，再通过 `ingest_text` 切分、生成 token、写入 SQLite。默认 source 使用 `default:<文档相对路径>`，同一文档重复启动会替换原 chunk，不会不断追加重复数据。

当前默认文档覆盖：

- 消息平台接口映射。
- 发送策略 pipeline。
- 邮件通道必填字段。
- 业务消息配置到发送的使用说明。
- 消息发送失败通用排查清单。
- 邮件发送失败排查。
- 模板渲染失败排查。
