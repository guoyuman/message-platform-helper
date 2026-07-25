from __future__ import annotations

import unittest
from dataclasses import dataclass

from message_platform_helper.agents.registry import build_default_agent_registry
from message_platform_helper.decision import DecisionEngine, KnowledgePolicy, LLMIntentClassifier
from message_platform_helper.decision.routers import AgentSelector, ToolRouter, WorkflowRouter
from message_platform_helper.llm import RuleBasedLLMClient
from message_platform_helper.models import AssistantRequest, ConversationMemory, IntentResult, WorkflowDefinition, WorkflowStep
from message_platform_helper.rate_limit import MemoryCounterStore
from message_platform_helper.workflow import WorkflowExecutor, WorkflowRegistry, build_default_workflow_registry


@dataclass
class FakeClassifier:
    intent: IntentResult

    def classify(self, request: AssistantRequest, memory: ConversationMemory) -> IntentResult:
        return self.intent


class DecisionWorkflowTests(unittest.TestCase):
    def _engine(self, intent: IntentResult, knowledge_policy: KnowledgePolicy | None = None) -> DecisionEngine:
        workflows = build_default_workflow_registry()
        return DecisionEngine(
            classifier=FakeClassifier(intent),
            knowledge_policy=knowledge_policy or KnowledgePolicy(),
            workflow_router=WorkflowRouter(),
            agent_selector=AgentSelector(registry=build_default_agent_registry(MemoryCounterStore({}))),
            tool_router=ToolRouter(registry=None),
            workflow_registry=workflows,
        )

    def test_decision_engine_uses_policy_not_llm_for_rag(self) -> None:
        workflows = build_default_workflow_registry()
        engine = DecisionEngine(
            classifier=FakeClassifier(IntentResult(intent="translation", confidence=0.9, metadata={"selectedAgents": ["template"]})),
            knowledge_policy=KnowledgePolicy(intent_rules={"translation": False}),
            workflow_router=WorkflowRouter(),
            agent_selector=AgentSelector(registry=build_default_agent_registry(MemoryCounterStore({}))),
            tool_router=ToolRouter(registry=None),
            workflow_registry=workflows,
        )

        decision = engine.decide(AssistantRequest(text="翻译模板"), ConversationMemory(session_id="s1"))

        self.assertEqual(decision.intent, "translation")
        self.assertFalse(decision.need_rag)
        self.assertEqual(decision.workflow, "template_workflow")
        self.assertEqual(decision.selected_agents, ["template"])

    def test_business_message_configuration_question_routes_to_knowledge_query(self) -> None:
        engine = self._engine(
            IntentResult(
                intent="business_config",
                request_type="query",
                domain="business_message",
                operation="explain",
                confidence=0.86,
                metadata={"selectedAgents": ["business_config"]},
            )
        )

        decision = engine.decide(
            AssistantRequest(text="\u4e1a\u52a1\u6d88\u606f\u5982\u4f55\u914d\u7f6e"),
            ConversationMemory(session_id="s1"),
        )

        self.assertEqual(decision.request_type, "query")
        self.assertEqual(decision.domain, "business_message")
        self.assertEqual(decision.operation, "explain")
        self.assertEqual(decision.intent, "knowledge_query")
        self.assertTrue(decision.need_rag)
        self.assertEqual(decision.workflow, "knowledge_answer")
        self.assertEqual(decision.selected_agents, ["knowledge"])
        self.assertEqual(decision.selected_tools, ["knowledge.search", "knowledge.answer"])

    def test_business_message_configuration_action_routes_to_business_agent(self) -> None:
        engine = self._engine(
            IntentResult(
                intent="business_config",
                request_type="action",
                domain="business_message",
                operation="create",
                confidence=0.88,
            )
        )

        decision = engine.decide(
            AssistantRequest(text="\u5e2e\u6211\u914d\u7f6e\u4e1a\u52a1\u6d88\u606f"),
            ConversationMemory(session_id="s1"),
        )

        self.assertEqual(decision.request_type, "action")
        self.assertEqual(decision.domain, "business_message")
        self.assertEqual(decision.operation, "create")
        self.assertEqual(decision.intent, "implementation")
        self.assertFalse(decision.need_rag)
        self.assertEqual(decision.workflow, "implementation_workflow")
        self.assertEqual(decision.selected_agents, ["business_config"])
        self.assertEqual(decision.selected_tools, ["business.build_payload", "platform.business.preview"])

    def test_memory_recall_question_does_not_route_to_knowledge_query(self) -> None:
        engine = self._engine(
            IntentResult(
                intent="knowledge_query",
                request_type="query",
                domain="template",
                operation="explain",
                confidence=0.82,
            )
        )

        decision = engine.decide(
            AssistantRequest(text="\u8fd8\u8bb0\u5f97\u6211\u540c\u6b65\u7ffb\u8bd1\u8fc7\u54ea\u4e9b\u5355\u636e\u7684\u6a21\u677f\u5417"),
            ConversationMemory(session_id="s1"),
        )

        self.assertEqual(decision.request_type, "chat")
        self.assertEqual(decision.intent, "casual_chat")
        self.assertFalse(decision.need_rag)
        self.assertEqual(decision.workflow, "general_chat")
        self.assertEqual(decision.selected_agents, [])
        self.assertEqual(decision.selected_tools, [])

    def test_rule_based_classifier_splits_business_query_and_action(self) -> None:
        workflows = build_default_workflow_registry()
        engine = DecisionEngine(
            classifier=LLMIntentClassifier(RuleBasedLLMClient()),
            knowledge_policy=KnowledgePolicy(),
            workflow_router=WorkflowRouter(),
            agent_selector=AgentSelector(registry=build_default_agent_registry(MemoryCounterStore({}))),
            tool_router=ToolRouter(registry=None),
            workflow_registry=workflows,
        )

        query = engine.decide(
            AssistantRequest(text="\u4e1a\u52a1\u6d88\u606f\u5982\u4f55\u914d\u7f6e"),
            ConversationMemory(session_id="s1"),
        )
        action = engine.decide(
            AssistantRequest(text="\u5e2e\u6211\u914d\u7f6e\u4e1a\u52a1\u6d88\u606f"),
            ConversationMemory(session_id="s1"),
        )

        self.assertEqual(query.request_type, "query")
        self.assertEqual(query.selected_agents, ["knowledge"])
        self.assertTrue(query.need_rag)
        self.assertEqual(action.request_type, "action")
        self.assertEqual(action.selected_agents, ["business_config"])
        self.assertFalse(action.need_rag)

    def test_rule_based_classifier_maps_payload_to_multi_agent_workflow(self) -> None:
        workflows = build_default_workflow_registry()
        engine = DecisionEngine(
            classifier=LLMIntentClassifier(RuleBasedLLMClient()),
            knowledge_policy=KnowledgePolicy(),
            workflow_router=WorkflowRouter(),
            agent_selector=AgentSelector(registry=build_default_agent_registry(MemoryCounterStore({}))),
            tool_router=ToolRouter(registry=None),
            workflow_registry=workflows,
        )

        decision = engine.decide(
            AssistantRequest(
                text="\u5e2e\u6211\u5b8c\u6210\u6d88\u606f\u914d\u7f6e",
                payload={
                    "template": {"targetLanguage": "en_US"},
                    "email": "ops@example.com",
                    "strategy": {"maxPerHour": 2},
                },
            ),
            ConversationMemory(session_id="s1"),
        )

        self.assertEqual(decision.request_type, "action")
        self.assertEqual(decision.domain, "implementation")
        self.assertEqual(decision.workflow, "message_platform_workflow")
        self.assertFalse(decision.need_rag)
        self.assertEqual(decision.selected_agents, ["template", "channel_config", "send_strategy"])

    def test_rule_based_classifier_does_not_route_keyword_question_to_business_agent(self) -> None:
        raw = RuleBasedLLMClient().complete_json(
            "Classify user intent and request taxonomy for the message platform helper.",
            {
                "text": "\u5982\u4f55\u914d\u7f6e\u4e1a\u52a1\u6d88\u606f\u6a21\u677f\uff1f",
                "payload": {},
            },
        )

        self.assertEqual(raw["request_type"], "query")
        self.assertEqual(raw["intent"], "knowledge_query")
        self.assertEqual(raw["selectedAgents"], ["knowledge"])
        self.assertTrue(raw["needRag"])

    def test_new_workflow_can_be_registered_without_changing_decision_engine(self) -> None:
        workflows = WorkflowRegistry()
        workflows.register(WorkflowDefinition(name="custom_workflow", steps=[WorkflowStep(name="first", agent="knowledge")]))
        engine = DecisionEngine(
            classifier=FakeClassifier(IntentResult(intent="custom_intent", confidence=0.8, metadata={"workflow": "custom_workflow"})),
            knowledge_policy=KnowledgePolicy(intent_rules={"custom_intent": True}),
            workflow_router=WorkflowRouter(intent_workflows={"custom_intent": "custom_workflow"}),
            agent_selector=AgentSelector(registry=build_default_agent_registry(MemoryCounterStore({}))),
            tool_router=ToolRouter(registry=None),
            workflow_registry=workflows,
        )

        decision = engine.decide(AssistantRequest(text="custom"), ConversationMemory(session_id="s1"))

        self.assertTrue(decision.need_rag)
        self.assertEqual(decision.workflow, "custom_workflow")
        self.assertEqual(decision.selected_agents, ["knowledge"])

    def test_workflow_executor_orders_selected_agents_by_workflow_steps(self) -> None:
        executor = WorkflowExecutor(
            build_default_agent_registry(MemoryCounterStore({})),
            build_default_workflow_registry(),
        )

        ordered = executor.resolve_execution_order("message_platform_workflow", ["send_strategy", "template", "channel_config"])

        self.assertEqual(ordered, ["template", "channel_config", "send_strategy"])


if __name__ == "__main__":
    unittest.main()
