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


def sample_multichannel_template_detail() -> dict:
    detail = sample_template_detail()
    detail["messageTempVO"]["templateName"] = "Purchase requisition notice"
    detail["messageTempVO"]["templateCode"] = "purchase_requisition_notice"
    detail["messageTemplateContentVOList"] = [
        {
            "id": "content-mail-cn",
            "channelType": "mail",
            "language": "zh_CN",
            "title": "请购单${billNo}已保存",
            "content": '<div style="color:red">单号：<b>${billNo}</b><a href="https://example.test">查看</a></div>',
            "raw": '<div style="color:red">单号：<b>${billNo}</b><a href="https://example.test">查看</a></div>',
            "contentType": "html",
        },
        {
            "id": "content-mail-id",
            "channelType": "mail",
            "language": "id_ID",
            "title": "existing",
            "content": "existing",
            "raw": "existing",
            "contentType": "html",
        },
        {
            "id": "content-uspace-cn",
            "channelType": "uspace",
            "language": "zh_CN",
            "title": "请购单#{billNo}待审批",
            "content": "请处理 {{userName}} 的请购单 #{billNo}",
            "raw": "请处理 {{userName}} 的请购单 #{billNo}",
            "contentType": "text",
        },
    ]
    return detail


class TemplateSyncPlatform(PlatformGateway):
    def __init__(self) -> None:
        super().__init__()
        self.find_payloads: list[dict] = []
        self.saved_payloads: list[dict] = []

    def get_domain_tree(self, payload: dict | None = None, dry_run: bool = True) -> dict:
        return {
            "ok": True,
            "tree": [
                {
                    "name": "采购领域",
                    "code": "purchase",
                    "children": [{"name": "请购单", "code": "PR001", "documentId": "PR001", "children": []}],
                }
            ],
        }

    def find_templates(self, payload: dict, dry_run: bool = True) -> dict:
        self.find_payloads.append(payload)
        return {"ok": True, "data": {"templates": [sample_multichannel_template_detail()]}}

    def save_template(self, payload: dict, dry_run: bool = True) -> dict:
        self.saved_payloads.append(payload)
        return {"ok": True, "status": "saved", "payload": payload}


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

    def test_template_sync_resolves_business_object_and_syncs_missing_channels(self) -> None:
        platform = TemplateSyncPlatform()
        context = AgentContext(
            request=AssistantRequest(
                text="将请购单下的所有模板信息同步到印尼语种下",
                dry_run=True,
            ),
            memory=ConversationMemory(session_id="s-template"),
            retrieved=[],
            llm=RuleBasedLLMClient(),
            platform=platform,
        )
        result = TemplateAgent().run(context)
        self.assertTrue(result.ok)
        self.assertEqual(result.output["businessObject"]["businessObjectCode"], "PR001")
        self.assertEqual(platform.find_payloads[0]["businessObjectCode"], "PR001")
        saved_contents = platform.saved_payloads[0]["messageTemplateContentVOList"]
        id_contents = [item for item in saved_contents if item["language"] == "id_ID"]
        self.assertEqual(len(id_contents), 2)
        self.assertEqual([item["channelType"] for item in id_contents].count("mail"), 1)
        self.assertEqual([item["channelType"] for item in id_contents].count("uspace"), 1)
        translated_uspace = [item for item in id_contents if item["channelType"] == "uspace"][0]
        self.assertIn("#{billNo}", translated_uspace["title"])
        self.assertIn("{{userName}}", translated_uspace["content"])
        self.assertIn("检查目标语种是否存在", result.output["executionTrace"])



if __name__ == "__main__":
    unittest.main()
