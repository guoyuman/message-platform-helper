from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from message_platform_helper.application import response_to_streaming_events
from message_platform_helper.config import Settings
from message_platform_helper.llm import RuleBasedLLMClient
from message_platform_helper.memory import MemoryManager, SQLiteMemoryStore
from message_platform_helper.models import AssistantRequest
from message_platform_helper.orchestrator import MessagePlatformHelper
from message_platform_helper.platform import PlatformGateway
from message_platform_helper.rag import KnowledgeBase, seed_default_knowledge
from message_platform_helper.rate_limit import MemoryCounterStore


class StreamingObservabilityTests(unittest.TestCase):
    def test_response_contains_trace_metrics_and_stream_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            llm = RuleBasedLLMClient()
            kb = KnowledgeBase(root / "kb.sqlite3")
            seed_default_knowledge(kb)
            helper = MessagePlatformHelper(
                settings=Settings(data_dir=root),
                llm=llm,
                memory_manager=MemoryManager(SQLiteMemoryStore(root / "memory.sqlite3"), llm),
                knowledge_base=kb,
                platform=PlatformGateway(),
                counter_store=MemoryCounterStore({}),
            )

            response = helper.handle(
                AssistantRequest(
                    text="消息发送失败原因如何排查？",
                    session_id="s-observe",
                    request_id="req-test",
                    trace_id="trace-test",
                )
            )

        events = list(response_to_streaming_events(response))

        self.assertEqual(response.request_id, "req-test")
        self.assertEqual(response.trace_id, "trace-test")
        self.assertGreaterEqual(response.metrics["latency_ms"], 0)
        self.assertEqual(events[0].type, "Reasoning")
        self.assertIn("Source", [event.type for event in events])
        self.assertEqual(events[-1].type, "Done")


if __name__ == "__main__":
    unittest.main()
