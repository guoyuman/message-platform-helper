from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from message_platform_helper.config import Settings
from message_platform_helper.llm import RuleBasedLLMClient
from message_platform_helper.memory import InMemoryMemoryStore, MemoryManager
from message_platform_helper.models import AssistantRequest
from message_platform_helper.orchestrator import MessagePlatformHelper
from message_platform_helper.platform import PlatformGateway
from message_platform_helper.rag import KnowledgeBaseRetriever, build_rag_service, seed_default_knowledge
from message_platform_helper.rate_limit import MemoryCounterStore

from tests.fakes import InMemoryKnowledgeBase


def sample_template_detail() -> dict:
    return {
        "messageTempVO": {
            "id": "tpl-001",
            "templateName": "Purchase order notice",
            "templateCode": "purchase_order_notice",
            "description": "Purchase order approval notice",
            "groupCode": "businessprocess",
            "documentId": "purchase_order",
            "languageDict": {
                "templateName": {"en_US": {"id": "", "value": "Purchase order notice"}},
                "description": {"en_US": {"id": "", "value": "Purchase order approval notice"}},
            },
        },
        "messageTemplateContentVOList": [
            {
                "id": "content-001",
                "channelType": "mail",
                "language": "en_US",
                "title": "Business approval notification",
                "content": "Hello, you have a business message to process. {{orderNo}}",
                "raw": "Hello, you have a business message to process. {{orderNo}}",
                "contentType": "html",
            }
        ],
    }


class OrchestratorTests(unittest.TestCase):
    def test_end_to_end_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            llm = RuleBasedLLMClient()
            kb = InMemoryKnowledgeBase()
            seed_default_knowledge(kb)
            helper = MessagePlatformHelper(
                settings=Settings(data_dir=root),
                llm=llm,
                memory_manager=MemoryManager(InMemoryMemoryStore({}), llm),
                knowledge_base=kb,
                platform=PlatformGateway(),
                counter_store=MemoryCounterStore({}),
                rag_service=build_rag_service(KnowledgeBaseRetriever(kb)),
            )
            response = helper.handle(
                AssistantRequest(
                    text="给 ops@example.com 配置邮件通道并测试，同步采购订单下所有模板到阿拉伯语，设置频次每小时最多 2 次",
                    session_id="s1",
                    payload={
                        "email": "ops@example.com",
                        "template": {
                            "templates": [sample_template_detail()],
                            "sourceLanguage": "en_US",
                            "targetLanguage": "ar_SA",
                            "documentId": "purchase_order",
                            "groupCode": "businessprocess",
                        },
                    },
                    dry_run=True,
                )
            )
        agents = [result.agent for result in response.results]
        self.assertTrue(response.ok)
        self.assertIn("channel_config", agents)
        self.assertIn("template", agents)
        self.assertTrue(response.memory.summary)

    def test_knowledge_question_returns_rag_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            llm = RuleBasedLLMClient()
            kb = InMemoryKnowledgeBase()
            seed_default_knowledge(kb)
            helper = MessagePlatformHelper(
                settings=Settings(data_dir=root),
                llm=llm,
                memory_manager=MemoryManager(InMemoryMemoryStore({}), llm),
                knowledge_base=kb,
                platform=PlatformGateway(),
                counter_store=MemoryCounterStore({}),
                rag_service=build_rag_service(KnowledgeBaseRetriever(kb)),
            )
            response = helper.handle(AssistantRequest(text="消息发送失败原因如何排查？", session_id="s-knowledge", payload={}, dry_run=True))
        self.assertTrue(response.ok)
        self.assertIn("knowledge", [result.agent for result in response.results])
        self.assertTrue(response.retrieved)
        answer = response.results[0].output.get("answer", "")
        self.assertIn("排查", answer)
        stored = helper.memory_manager.load("s-knowledge")
        self.assertIn(answer, stored.recent_messages[-1]["text"])
        self.assertNotEqual(stored.recent_messages[-1]["text"], "knowledge:ok")

    def test_chat_recall_uses_operation_history_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            llm = RuleBasedLLMClient()
            kb = InMemoryKnowledgeBase()
            helper = MessagePlatformHelper(
                settings=Settings(data_dir=root),
                llm=llm,
                memory_manager=MemoryManager(InMemoryMemoryStore({}), llm),
                knowledge_base=kb,
                platform=PlatformGateway(),
                counter_store=MemoryCounterStore({}),
                rag_service=build_rag_service(KnowledgeBaseRetriever(kb)),
            )
            helper.handle(
                AssistantRequest(
                    text="\u5e2e\u6211\u540c\u6b65\u91c7\u8d2d\u8ba2\u5355\u6a21\u677f\u7ffb\u8bd1",
                    session_id="s-recall",
                    payload={"template": {"templates": [sample_template_detail()], "sourceLanguage": "en_US", "targetLanguage": "ar_SA"}},
                    dry_run=True,
                )
            )

            response = helper.handle(
                AssistantRequest(
                    text="\u4f60\u8fd8\u8bb0\u5f97\u6211\u540c\u6b65\u8fc7\u54ea\u4e9b\u5355\u636e\u7684\u6a21\u677f\u7ffb\u8bd1\u5417",
                    session_id="s-recall",
                    payload={},
                    dry_run=True,
                )
            )

        self.assertTrue(response.ok)
        self.assertFalse(response.decision.need_retrieval)
        self.assertTrue(response.memory.facts.get("operationHistory"))
        answer = response.results[0].output.get("answer", "")
        self.assertIn("Purchase order notice", answer)

    def test_display_memory_uses_saved_run_response_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            llm = RuleBasedLLMClient()
            store = InMemoryMemoryStore({})
            helper = MessagePlatformHelper(
                settings=Settings(data_dir=root),
                llm=llm,
                memory_manager=MemoryManager(store, llm),
                knowledge_base=InMemoryKnowledgeBase(),
                platform=PlatformGateway(),
                counter_store=MemoryCounterStore({}),
                rag_service=build_rag_service(KnowledgeBaseRetriever(InMemoryKnowledgeBase())),
            )
            store.save_run(
                "s-display",
                {"text": "question"},
                {"results": [{"agent": "knowledge", "ok": True, "output": {"answer": "real answer"}}]},
            )

            memory = helper.load_memory_for_display("s-display")

        self.assertEqual([item["text"] for item in memory.recent_messages], ["question", "real answer"])

    def test_memory_history_is_tenant_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            llm = RuleBasedLLMClient()
            store = InMemoryMemoryStore({})
            helper = MessagePlatformHelper(
                settings=Settings(data_dir=root),
                llm=llm,
                memory_manager=MemoryManager(store, llm),
                knowledge_base=InMemoryKnowledgeBase(),
                platform=PlatformGateway(),
                counter_store=MemoryCounterStore({}),
                rag_service=build_rag_service(KnowledgeBaseRetriever(InMemoryKnowledgeBase())),
            )
            helper.handle(AssistantRequest(text="tenant a question", session_id="shared", tenant_id="tenant-a", payload={}, dry_run=True))
            helper.handle(AssistantRequest(text="tenant b question", session_id="shared", tenant_id="tenant-b", payload={}, dry_run=True))

            tenant_a_memory = helper.load_memory_for_display("shared", "tenant-a")
            tenant_b_memory = helper.load_memory_for_display("shared", "tenant-b")
            tenant_a_sessions = helper.list_memory_sessions_for_display("tenant-a")
            tenant_b_sessions = helper.list_memory_sessions_for_display("tenant-b")

        self.assertIn("tenant a question", [item["text"] for item in tenant_a_memory.recent_messages])
        self.assertNotIn("tenant b question", [item["text"] for item in tenant_a_memory.recent_messages])
        self.assertIn("tenant b question", [item["text"] for item in tenant_b_memory.recent_messages])
        self.assertEqual([item["sessionId"] for item in tenant_a_sessions], ["shared"])
        self.assertEqual([item["sessionId"] for item in tenant_b_sessions], ["shared"])

    def test_chat_recall_hydrates_operation_history_from_saved_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            llm = RuleBasedLLMClient()
            store = InMemoryMemoryStore({})
            kb = InMemoryKnowledgeBase()
            helper = MessagePlatformHelper(
                settings=Settings(data_dir=root),
                llm=llm,
                memory_manager=MemoryManager(store, llm),
                knowledge_base=kb,
                platform=PlatformGateway(),
                counter_store=MemoryCounterStore({}),
                rag_service=build_rag_service(KnowledgeBaseRetriever(kb)),
            )
            store.save_run(
                "s-history",
                {"text": "sync"},
                {
                    "results": [
                        {
                            "agent": "template",
                            "ok": True,
                            "output": {
                                "summary": "template language sync translated 1 template(s), skipped 0 existing template(s)",
                                "sourceLanguage": "en_US",
                                "targetLanguage": "ar_SA",
                                "translatedTemplates": [
                                    {
                                        "templateId": "tpl-001",
                                        "templateCode": "purchase_order_notice",
                                        "templateName": "Purchase order notice",
                                    }
                                ],
                            },
                        }
                    ]
                },
            )

            response = helper.handle(
                AssistantRequest(
                    text="\u4f60\u8fd8\u8bb0\u5f97\u6211\u540c\u6b65\u8fc7\u54ea\u4e9b\u5355\u636e\u7684\u6a21\u677f\u7ffb\u8bd1\u5417",
                    session_id="s-history",
                    payload={},
                    dry_run=True,
                )
            )

        self.assertTrue(response.ok)
        self.assertTrue(response.memory.facts.get("operationHistory"))
        answer = response.results[0].output.get("answer", "")
        self.assertIn("Purchase order notice", answer)


if __name__ == "__main__":
    unittest.main()
