from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from message_platform_helper.agents.registry import AgentRegistry, build_default_agent_registry
from message_platform_helper.application.security import PermissionPolicy, TenantContext
from message_platform_helper.config import Settings
from message_platform_helper.llm import RuleBasedLLMClient
from message_platform_helper.memory import MemoryManager, SQLiteMemoryStore
from message_platform_helper.models import (
    AgentResult,
    AgentSpec,
    AssistantRequest,
    ConversationMemory,
    DecisionResult,
    JsonDict,
    Tool,
    ToolSpec,
    WorkflowDefinition,
    WorkflowStep,
    StreamingEvent,
    to_jsonable,
)
from message_platform_helper.orchestrator import MessagePlatformHelper
from message_platform_helper.platform import PlatformGateway
from message_platform_helper.rag import KnowledgeBase
from message_platform_helper.rate_limit import MemoryCounterStore
from message_platform_helper.react import AgentContext, ReActAgent
from message_platform_helper.tools import ToolRegistry


@dataclass
class NoopAgent(ReActAgent):
    name = "noop"

    def tools(self, context: AgentContext) -> Dict[str, Tool]:
        return {}

    def plan(self, context: AgentContext) -> List[JsonDict]:
        return []

    def finalize(self, context: AgentContext, observations: List[JsonDict], steps: list) -> AgentResult:
        return AgentResult(agent=self.name, ok=True, output={"handled": True}, steps=steps)


@dataclass
class RegistryOnlyAgent(ReActAgent):
    name = "registry_only"

    def tools(self, context: AgentContext) -> Dict[str, Tool]:
        return {}

    def plan(self, context: AgentContext) -> List[JsonDict]:
        return [{"tool": "registry.echo", "input": {"text": context.request.text}}]


class ContractAndRegistryTests(unittest.TestCase):
    def test_phase_one_contracts_are_jsonable(self) -> None:
        decision = DecisionResult(
            intent="knowledge_query",
            request_type="query",
            domain="business_message",
            operation="explain",
            workflow="knowledge_answer",
            need_rag=True,
            need_tools=True,
            selected_agents=["knowledge"],
            selected_tools=["knowledge.search"],
            confidence=0.91,
            search_query="消息发送失败",
        )
        workflow = WorkflowDefinition(
            name="knowledge_answer",
            steps=[WorkflowStep(name="answer", agent="knowledge", tools=["knowledge.search"])],
        )
        event = StreamingEvent(type="ToolCall", data={"tool": "knowledge.search"}, trace_id="trace-1", sequence=1)

        payload = to_jsonable({"decision": decision, "workflow": workflow, "event": event})

        self.assertEqual(payload["decision"]["intent"], "knowledge_query")
        self.assertEqual(payload["workflow"]["steps"][0]["agent"], "knowledge")
        self.assertEqual(payload["event"]["type"], "ToolCall")

    def test_default_agent_registry_resolves_existing_agents_in_requested_order(self) -> None:
        registry = build_default_agent_registry(MemoryCounterStore({}))
        agents = registry.resolve(["missing", "knowledge", "template"])

        self.assertEqual([agent.name for agent in agents], ["knowledge", "template"])
        self.assertIn("knowledge", registry.names())
        self.assertEqual(registry.select_by_capability("rag.answer")[0].name, "knowledge")

    def test_agent_registry_supports_custom_injected_agents(self) -> None:
        registry = AgentRegistry()
        registry.register(AgentSpec(name="noop", capabilities=["demo.noop"], priority=1), NoopAgent)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            llm = RuleBasedLLMClient()
            helper = MessagePlatformHelper(
                settings=Settings(data_dir=root),
                llm=llm,
                memory_manager=MemoryManager(SQLiteMemoryStore(root / "memory.sqlite3"), llm),
                knowledge_base=KnowledgeBase(root / "kb.sqlite3"),
                platform=PlatformGateway(),
                counter_store=MemoryCounterStore({}),
                agent_registry=registry,
            )

            resolved = helper._agent_sequence(["unknown", "noop"])

        self.assertEqual([agent.name for agent in resolved], ["noop"])

    def test_tool_registry_registers_resolves_and_selects_tools(self) -> None:
        registry = ToolRegistry()
        tool = Tool("echo", "Echo input.", lambda payload: {"ok": True, "payload": payload})
        registry.register_tool(
            tool,
            ToolSpec(name="echo", description="Echo input.", capabilities=["demo.echo"], priority=5),
        )

        resolved = registry.resolve(["missing", "echo"])
        selected = registry.select_by_capability("demo.echo")

        self.assertEqual(registry.names(), ["echo"])
        self.assertEqual(resolved[0].run({"value": 1})["payload"]["value"], 1)
        self.assertEqual(selected[0].name, "echo")
        with self.assertRaises(ValueError):
            registry.register_tool(tool)
        with self.assertRaises(KeyError):
            registry.resolve(["missing"], strict=True)

    def test_react_agent_can_call_registry_only_tool(self) -> None:
        registry = ToolRegistry()
        registry.register_tool(
            Tool(
                "registry.echo",
                "Echo from registry.",
                lambda payload, context: {"ok": True, "summary": "echoed", "echo": payload["text"], "sessionId": context.request.session_id},
                requires_context=True,
            )
        )
        context = AgentContext(
            request=AssistantRequest(text="hello registry", session_id="s-registry"),
            memory=ConversationMemory(session_id="s-registry"),
            retrieved=[],
            llm=RuleBasedLLMClient(),
            platform=PlatformGateway(),
            tool_registry=registry,
        )

        result = RegistryOnlyAgent().run(context)

        self.assertTrue(result.ok)
        self.assertEqual(result.output["echo"], "hello registry")
        self.assertEqual(result.steps[0].action, "registry.echo")

    def test_tool_registry_enforces_tenant_permissions(self) -> None:
        registry = ToolRegistry(permission_policy=PermissionPolicy(tool_permissions={"registry.echo": ["tool:echo"]}))
        registry.register_tool(
            Tool(
                "registry.echo",
                "Echo from registry.",
                lambda payload, context: {"ok": True, "summary": "echoed"},
                requires_context=True,
            )
        )
        base_context = dict(
            request=AssistantRequest(text="hello", session_id="s-auth"),
            memory=ConversationMemory(session_id="s-auth"),
            retrieved=[],
            llm=RuleBasedLLMClient(),
            platform=PlatformGateway(),
            tool_registry=registry,
        )

        denied = RegistryOnlyAgent().run(AgentContext(**base_context, tenant_context=TenantContext(tenant_id="t1", permissions=[])))
        allowed = RegistryOnlyAgent().run(AgentContext(**base_context, tenant_context=TenantContext(tenant_id="t1", permissions=["tool:echo"])))

        self.assertFalse(denied.ok)
        self.assertIn("Missing permission", denied.issues[0]["message"])
        self.assertTrue(allowed.ok)


if __name__ == "__main__":
    unittest.main()
