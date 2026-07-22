from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from message_platform_helper.llm import RuleBasedLLMClient
from message_platform_helper.memory import InMemoryMemoryStore, MemoryManager
from message_platform_helper.models import AssistantRequest
from message_platform_helper.rag import KnowledgeBaseRetriever, build_rag_service, seed_default_knowledge
from message_platform_helper.reasoning import ReasoningLayer

from tests.fakes import InMemoryKnowledgeBase


class ReasoningRagMemoryTests(unittest.TestCase):
    def test_reasoning_routes_channel_question_to_retrieval(self) -> None:
        llm = RuleBasedLLMClient()
        with tempfile.TemporaryDirectory() as tmp:
            memory = InMemoryMemoryStore({}).load("s1")
            request = AssistantRequest(text="如何配置 ops@example.com 的邮件通道？", payload={"email": "ops@example.com"})
            decision = ReasoningLayer(llm).decide(request, memory)
        self.assertTrue(decision.need_retrieval)
        self.assertEqual(decision.request_type, "query")
        self.assertIn("knowledge", decision.selected_agents)

    def test_rag_ingest_and_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            kb = InMemoryKnowledgeBase()
            kb.ingest("mail endpoint", "Email channel saves through /msgTemplateChannel/saveChannelConfig", tags=["mail"])
            results = kb.search("Email channel save endpoint")
        self.assertEqual(results[0].title, "mail endpoint")

    def test_rag_service_builds_answer_and_citations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            kb = InMemoryKnowledgeBase()
            kb.ingest("mail endpoint", "Email channel saves through /msgTemplateChannel/saveChannelConfig", tags=["mail"])
            service = build_rag_service(KnowledgeBaseRetriever(kb))

            answer = service.answer("Email channel save endpoint")

        self.assertTrue(answer["ok"])
        self.assertIn("citations", answer)
        self.assertIn("prompt", answer)
        self.assertEqual(answer["citations"][0]["title"], "mail endpoint")

    def test_rag_ingest_text_replaces_same_source_title(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            kb = InMemoryKnowledgeBase()
            first = kb.ingest_text("发送失败排查", "先检查模板变量。", source="manual", tags=["send"])
            second = kb.ingest_text("发送失败排查", "先检查发送记录错误码，再检查通道。", source="manual", tags=["send"])
            results = kb.search("发送记录错误码")
            self.assertEqual(len(first), 1)
            self.assertEqual(len(second), 1)
            self.assertEqual(kb.count(), 1)
            self.assertIn("发送记录错误码", results[0].content)

    def test_knowledge_question_routes_to_rag_agent(self) -> None:
        llm = RuleBasedLLMClient()
        with tempfile.TemporaryDirectory() as tmp:
            memory = InMemoryMemoryStore({}).load("s1")
            request = AssistantRequest(text="消息发送失败原因如何排查？")
            decision = ReasoningLayer(llm).decide(request, memory)
        self.assertTrue(decision.need_retrieval)
        self.assertIn("knowledge", decision.selected_agents)

    def test_default_knowledge_answers_failure_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            kb = InMemoryKnowledgeBase()
            seed_default_knowledge(kb)
            results = kb.search("消息发送失败原因如何排查", limit=3)
        self.assertTrue(results)
        self.assertTrue(any("排查" in chunk.title for chunk in results))

    def test_default_knowledge_is_seeded_from_markdown_documents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            knowledge_dir = root / "knowledge"
            knowledge_dir.mkdir()
            (knowledge_dir / "ops.md").write_text(
                "\n".join(
                    [
                        "---",
                        "title: Webhook retry guide",
                        "tags: ops, webhook",
                        "---",
                        "",
                        "# Webhook retry guide",
                        "",
                        "Webhook retry failures should be checked by requestId and upstream status.",
                    ]
                ),
                encoding="utf-8",
            )
            kb = InMemoryKnowledgeBase()
            kb.ingest("Legacy hardcoded seed", "old hardcoded endpoint", source="default", tags=["legacy"])

            seed_default_knowledge(kb, knowledge_dir)
            first_count = kb.count()
            seed_default_knowledge(kb, knowledge_dir)
            second_count = kb.count()
            results = kb.search("Webhook retry requestId", limit=1)
            legacy_results = kb.search("old hardcoded endpoint")

        self.assertGreaterEqual(first_count, 1)
        self.assertEqual(second_count, first_count)
        self.assertTrue(results)
        self.assertEqual(results[0].title, "Webhook retry guide")
        self.assertEqual(results[0].source, "default:ops.md")
        self.assertIn("ops", results[0].tags)
        self.assertFalse(legacy_results)

    def test_memory_updates_profile_and_facts(self) -> None:
        llm = RuleBasedLLMClient()
        with tempfile.TemporaryDirectory() as tmp:
            manager = MemoryManager(InMemoryMemoryStore({}), llm)
            memory = manager.load("s1")
            updated = manager.remember_request(memory, "请给 ops@example.com 配置邮件通道", {"email": "ops@example.com"})
            loaded = manager.load("s1")
        self.assertEqual(updated.facts["lastEmail"], "ops@example.com")
        self.assertIn("mail", loaded.profile.preferred_channels)


if __name__ == "__main__":
    unittest.main()
