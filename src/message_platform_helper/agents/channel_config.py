"""Email channel configuration agent."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List

from ..models import AgentResult, AgentStep, EmailChannelConfig, JsonDict, Tool, to_jsonable
from ..react import AgentContext, FunctionCallingAgent


SMTP_PROVIDERS = {
    "qq.com": ("smtp.qq.com", 465, True, False),
    "163.com": ("smtp.163.com", 465, True, False),
    "126.com": ("smtp.126.com", 465, True, False),
    "gmail.com": ("smtp.gmail.com", 587, False, True),
    "outlook.com": ("smtp.office365.com", 587, False, True),
    "hotmail.com": ("smtp.office365.com", 587, False, True),
    "office365.com": ("smtp.office365.com", 587, False, True),
}


@dataclass
class ChannelConfigAgent(FunctionCallingAgent):
    name = "channel_config"

    def tools(self, context: AgentContext) -> Dict[str, Tool]:
        return {
            "channel.infer_email": Tool(
                "channel.infer_email",
                "Infer SMTP config from an email address. Returns the platform channel payload with mailHost/mailPort/mailUsername from the address domain.",
                lambda payload: self._infer(context, payload),
                parameters={
                    "type": "object",
                    "properties": {
                        "email": {"type": "string", "description": "Email address, e.g. ops@example.com. May be omitted when present in the user text."},
                        "configName": {"type": "string", "description": "Channel config name. Defaults to mail-<domain>."},
                        "mailHost": {"type": "string", "description": "SMTP host override. Inferred from the address domain when omitted."},
                        "mailPort": {"type": "integer", "description": "SMTP port override. Inferred from the address domain when omitted."},
                        "mailUsername": {"type": "string", "description": "SMTP username. Defaults to the email address."},
                        "mailPwd": {"type": "string", "description": "SMTP password or authorization code."},
                        "smtpSSL": {"type": "boolean", "description": "Use SSL. Inferred from the address domain when omitted."},
                        "smtpTLS": {"type": "boolean", "description": "Use STARTTLS. Inferred from the address domain when omitted."},
                        "verifyUser": {"type": "string", "description": "Recipient used to verify the channel. Defaults to the email address."},
                    },
                    "required": [],
                },
            ),
            "channel.save": Tool(
                "channel.save",
                "Save or dry-run save the email channel config to the message platform.",
                lambda payload: self._save(context, payload),
                parameters={
                    "type": "object",
                    "properties": {
                        "configName": {"type": "string", "description": "Channel config name."},
                        "defaultStatus": {"type": "integer", "description": "1 to enable by default, 0 otherwise."},
                        "sysId": {"type": "string", "description": "System id. Defaults to diwork."},
                        "tenantId": {"type": "string", "description": "Tenant id."},
                    },
                    "required": [],
                },
            ),
            "channel.test": Tool(
                "channel.test",
                "Test the email channel config by sending a verification message.",
                lambda payload: self._test(context, payload),
                parameters={
                    "type": "object",
                    "properties": {
                        "verifyUser": {"type": "string", "description": "Recipient email that receives the verification message."},
                    },
                    "required": [],
                },
            ),
        }

    def system_prompt_for(self, context: AgentContext) -> str:
        return (
            "你是企业消息平台的邮件通道配置 agent。"
            "配置邮件通道的流程：先用 channel.infer_email 从邮箱地址推断 SMTP 配置并生成通道 payload，"
            "再用 channel.save 保存配置，最后用 channel.test 发送验证消息测试通道。"
            "邮箱地址优先从用户输入提取；用户没有给出邮箱时必须报告缺失（missingSlots），不得编造邮箱。"
            "不要编造 SMTP 主机、端口、密码或验证结果；推断和保存结果以工具返回为准。"
        )

    def plan(self, context: AgentContext) -> List[JsonDict]:
        payload = _channel_payload(context)
        steps = [
            {
                "thought": "Email channel setup starts by inferring SMTP host and platform config from the address.",
                "tool": "channel.infer_email",
                "input": payload,
            },
            {
                "thought": "Once the config is built, persist it through the message platform boundary.",
                "tool": "channel.save",
                "input": payload,
            },
        ]
        if payload.get("test", True):
            steps.append(
                {
                    "thought": "The channel should be verified after configuration, so run a send test boundary.",
                    "tool": "channel.test",
                    "input": payload,
                }
            )
        return steps

    def finalize(self, context: AgentContext, observations: List[JsonDict], steps: List[AgentStep]) -> AgentResult:
        output: JsonDict = {}
        issues: List[JsonDict] = []
        for item in observations:
            output.update(item)
            issues.extend(item.get("issues") or [])
        return AgentResult(agent=self.name, ok=not any(i.get("severity") == "error" for i in issues), output=output, issues=issues, steps=steps)

    def _infer(self, context: AgentContext, payload: JsonDict) -> JsonDict:
        email = str(payload.get("email") or context.memory.facts.get("lastEmail") or extract_email(context.request.text) or "")
        if not email:
            return {
                "ok": False,
                "summary": "email missing",
                "issues": [{"code": "channel.email.required", "message": "Email address is required.", "severity": "error"}],
            }
        config = infer_email_config(email, payload)
        platform_payload = email_platform_payload(config, payload)
        context.scratch["emailChannelPayload"] = platform_payload
        return {"ok": True, "summary": "email channel inferred", "emailConfig": to_jsonable(config), "channelPayload": platform_payload}

    def _save(self, context: AgentContext, payload: JsonDict) -> JsonDict:
        channel_payload = context.scratch.get("emailChannelPayload")
        if not channel_payload:
            inferred = self._infer(context, payload)
            channel_payload = inferred.get("channelPayload") or {}
        response = context.platform.save_channel_config(channel_payload, dry_run=context.request.dry_run)
        return {"ok": True, "summary": "channel save boundary called", "channelSaveResponse": response}

    def _test(self, context: AgentContext, payload: JsonDict) -> JsonDict:
        channel_payload = context.scratch.get("emailChannelPayload")
        if not channel_payload:
            inferred = self._infer(context, payload)
            channel_payload = inferred.get("channelPayload") or {}
        response = context.platform.test_email_channel(channel_payload, dry_run=context.request.dry_run)
        issues: List[JsonDict] = []
        if response.get("ok") is False:
            issues.append({"code": "channel.test.failed", "message": str(response.get("message") or response), "severity": "warning"})
        return {"ok": response.get("ok", True), "summary": "channel test boundary called", "channelTestResponse": response, "issues": issues}


def _channel_payload(context: AgentContext) -> JsonDict:
    payload = dict(context.request.payload.get("channel") or context.request.payload)
    if "email" not in payload:
        found = extract_email(context.request.text)
        if found:
            payload["email"] = found
    return payload


def extract_email(text: str) -> str:
    match = re.search(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}", text or "")
    return match.group(0) if match else ""


def infer_email_config(email: str, payload: JsonDict) -> EmailChannelConfig:
    domain = email.split("@", 1)[1].lower()
    host, port, smtp_ssl, smtp_tls = SMTP_PROVIDERS.get(domain, (f"smtp.{domain}", 587, False, True))
    return EmailChannelConfig(
        email=email,
        config_name=str(payload.get("configName") or f"mail-{domain}"),
        mail_host=str(payload.get("mailHost") or host),
        mail_port=int(payload.get("mailPort") or port),
        username=str(payload.get("mailUsername") or payload.get("username") or email),
        password=str(payload.get("mailPwd") or payload.get("password") or ""),
        smtp_ssl=bool(payload.get("smtpSSL", smtp_ssl)),
        smtp_tls=bool(payload.get("smtpTLS", smtp_tls)),
        verify_user=str(payload.get("verifyUser") or payload.get("receiver") or email),
        default_status=int(payload.get("defaultStatus") or 1),
    )


def email_platform_payload(config: EmailChannelConfig, hints: JsonDict) -> JsonDict:
    config_info = {
        "mailHost": config.mail_host,
        "mailPort": str(config.mail_port),
        "mailUsername": config.username,
        "mailPwd": config.password,
        "verifyUser": config.verify_user,
        "mailName": config.config_name,
        "mailSender": config.email,
        "smtpSSL": config.smtp_ssl,
        "smtpTLS": config.smtp_tls,
        "mailRouteType": "MAIL_PROTOCOL",
    }
    return {
        "id": hints.get("id") or "",
        "configName": config.config_name,
        "channelType": "mail",
        "defaultStatus": config.default_status,
        "sysId": hints.get("sysId") or "diwork",
        "tenantId": hints.get("tenantId") or "",
        "domain": hints.get("domain") or "",
        "configInfo": json.dumps(config_info, ensure_ascii=False),
    }
