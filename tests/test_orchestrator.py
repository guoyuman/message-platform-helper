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
                        "strategy": {},
                    },
                    dry_run=True,
                )
            )
        agents = [result.agent for result in response.results]
        self.assertTrue(response.ok)
        self.assertIn("channel_config", agents)
        self.assertIn("template", agents)
        self.assertIn("send_strategy", agents)
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


if __name__ == "__main__":
    unittest.main()
