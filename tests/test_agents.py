from __future__ import annotations

import unittest

from message_platform_helper.agents.channel_config import infer_email_config
from message_platform_helper.agents.send_strategy import build_strategy
from message_platform_helper.agents.template import TemplateAgent, extract_variables
from message_platform_helper.llm import RuleBasedLLMClient
from message_platform_helper.models import AssistantRequest, ConversationMemory
from message_platform_helper.platform import PlatformGateway
from message_platform_helper.rate_limit import MemoryCounterStore
from message_platform_helper.react import AgentContext
from message_platform_helper.strategy import SendStrategyEvaluator


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


class AgentUnitTests(unittest.TestCase):
    def test_email_config_infers_provider(self) -> None:
        config = infer_email_config("ops@gmail.com", {})
        self.assertEqual(config.mail_host, "smtp.gmail.com")
        self.assertEqual(config.mail_port, 587)
        self.assertTrue(config.smtp_tls)

    def test_template_variable_extraction_preserves_tokens(self) -> None:
        variables = extract_variables("订单 {{orderNo}} 金额 ${amount} 字段 [md#supplier.name]")
        self.assertIn("orderNo", variables)
        self.assertIn("amount", variables)
        self.assertIn("supplier.name", variables)

    def test_template_sync_adds_missing_target_language(self) -> None:
        context = AgentContext(
            request=AssistantRequest(
                text="同步采购订单下所有模板到阿拉伯语",
                payload={"template": {"templates": [sample_template_detail()], "sourceLanguage": "en_US", "targetLanguage": "阿拉伯语"}},
                dry_run=True,
            ),
            memory=ConversationMemory(session_id="s-template"),
            retrieved=[],
            llm=RuleBasedLLMClient(),
            platform=PlatformGateway(),
        )
        result = TemplateAgent().run(context)
        self.assertTrue(result.ok)
        translated = result.output["translatedTemplates"][0]
        payload = translated["platformPayload"]
        languages = [item["language"] for item in payload["messageTemplateContentVOList"]]
        self.assertIn("ar_SA", languages)
        self.assertEqual(payload["messageTempVO"]["languageDict"]["templateName"]["ar_SA"]["id"], "")

    def test_template_sync_skips_existing_target_language(self) -> None:
        detail = sample_template_detail()
        detail["messageTemplateContentVOList"].append(
            {
                "id": "content-002",
                "channelType": "mail",
                "language": "ar_SA",
                "title": "اشعار الموافقة على الأعمال",
                "content": "مرحبا {{orderNo}}",
                "raw": "مرحبا {{orderNo}}",
                "contentType": "html",
            }
        )
        context = AgentContext(
            request=AssistantRequest(
                text="同步采购订单下所有模板到阿拉伯语",
                payload={"template": {"templates": [detail], "sourceLanguage": "en_US", "targetLanguage": "ar_SA"}},
                dry_run=True,
            ),
            memory=ConversationMemory(session_id="s-template"),
            retrieved=[],
            llm=RuleBasedLLMClient(),
            platform=PlatformGateway(),
        )
        result = TemplateAgent().run(context)
        self.assertTrue(result.ok)
        self.assertFalse(result.output["translatedTemplates"])
        self.assertEqual(result.output["skippedTemplates"][0]["reason"], "already exists")

    def test_strategy_frequency_blocks_after_limit(self) -> None:
        strategy = build_strategy("每小时最多 1 次，只允许邮件通道", {})
        evaluator = SendStrategyEvaluator(MemoryCounterStore({}))
        first = evaluator.evaluate(strategy, {"channels": ["mail"], "receivers": ["u001"]})
        second = evaluator.evaluate(strategy, {"channels": ["mail"], "receivers": ["u001"]})
        self.assertTrue(first.allowed)
        self.assertFalse(second.allowed)
        self.assertIn("frequency", second.matched_rules)

    def test_strategy_channel_limit_blocks_sms(self) -> None:
        strategy = build_strategy("只允许邮件通道", {})
        decision = SendStrategyEvaluator(MemoryCounterStore({})).evaluate(strategy, {"channels": ["sms"], "receivers": ["u001"]})
        self.assertFalse(decision.allowed)


if __name__ == "__main__":
    unittest.main()
