"""Knowledge-answer agent backed by retrieved RAG chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from ..models import AgentResult, AgentStep, JsonDict, KnowledgeChunk, Tool
from ..react import AgentContext, ReActAgent


@dataclass
class KnowledgeAgent(ReActAgent):
    name = "knowledge"

    def tools(self, context: AgentContext) -> Dict[str, Tool]:
        return {
            "knowledge.answer": Tool(
                "knowledge.answer",
                "Answer a product usage or troubleshooting question from retrieved knowledge.",
                lambda payload: self._answer(context, payload),
            )
        }

    def plan(self, context: AgentContext) -> List[JsonDict]:
        return [
            {
                "thought": "The user is asking for guidance, so summarize the retrieved knowledge with citations.",
                "tool": "knowledge.answer",
                "input": {"question": context.request.text},
            }
        ]

    def finalize(self, context: AgentContext, observations: List[JsonDict], steps: List[AgentStep]) -> AgentResult:
        output = observations[-1] if observations else {}
        issues = output.get("issues") or []
        return AgentResult(agent=self.name, ok=not any(issue.get("severity") == "error" for issue in issues), output=output, issues=issues, steps=steps)

    def _answer(self, context: AgentContext, payload: JsonDict) -> JsonDict:
        question = str(payload.get("question") or context.request.text)
        chunks = context.retrieved[:5]
        if context.rag_service is not None:
            return context.rag_service.answer(question, chunks=chunks, tenant_context=context.tenant_context, llm=context.llm)
        if not chunks:
            return {
                "ok": False,
                "summary": "knowledge not found",
                "answer": "知识库里还没有检索到可用内容。请先通过 /api/knowledge/ingest 写入使用说明或排查文档，再重试这个问题。",
                "citations": [],
                "issues": [
                    {
                        "code": "knowledge.not_found",
                        "message": "No knowledge chunks matched the question.",
                        "severity": "warning",
                    }
                ],
            }

        answer = build_answer(question, chunks)
        citations = [
            {
                "id": chunk.id,
                "title": chunk.title,
                "source": chunk.source,
                "tags": chunk.tags,
                "score": round(chunk.score, 4),
            }
            for chunk in chunks
        ]
        return {
            "ok": True,
            "summary": f"answered from {len(chunks)} knowledge chunk(s)",
            "answer": answer,
            "citations": citations,
        }


def build_answer(question: str, chunks: List[KnowledgeChunk]) -> str:
    lowered = question.lower()
    is_failure = any(keyword in question for keyword in ("失败", "报错", "排查", "错误", "异常")) or any(
        keyword in lowered for keyword in ("fail", "error", "troubleshoot")
    )
    lines: List[str] = []
    if is_failure:
        lines.append("可以按这条链路排查：先定位失败阶段，再验证配置、模板、接收人、策略和通道。")
    else:
        lines.append("可以按知识库里的流程执行：先配置基础对象，再预览校验，最后保存发布并验证。")

    for index, chunk in enumerate(chunks, start=1):
        excerpt = compact_text(chunk.content, 260)
        lines.append(f"{index}. {chunk.title}: {excerpt}")

    lines.append("建议实际处理时保留发送记录里的错误码、通道、接收人、模板编码和请求 payload，便于继续追踪。")
    return "\n".join(lines)


def compact_text(text: str, limit: int) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "..."
