"""Business message configuration agent integrating existing message-agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from ..models import AgentResult, AgentStep, JsonDict, Tool
from ..react import AgentContext, ReActAgent


@dataclass
class BusinessMessageConfigAgent(ReActAgent):
    name = "business_config"

    def tools(self, context: AgentContext) -> Dict[str, Tool]:
        return {
            "business.build_payload": Tool("business.build_payload", "Build message-agent compatible payload.", lambda payload: self._build_payload(context, payload)),
            "business.preview_or_publish": Tool("business.preview_or_publish", "Call existing message-agent or Java save boundary.", lambda payload: self._publish(context, payload)),
        }

    def plan(self, context: AgentContext) -> List[JsonDict]:
        payload = dict(context.request.payload.get("businessConfig") or context.request.payload)
        return [
            {
                "thought": "Business configuration should first be normalized into the existing message-agent contract.",
                "tool": "business.build_payload",
                "input": payload,
            },
            {
                "thought": "After normalization, use the existing business config boundary for preview or publish.",
                "tool": "business.preview_or_publish",
                "input": payload,
            },
        ]

    def finalize(self, context: AgentContext, observations: List[JsonDict], steps: List[AgentStep]) -> AgentResult:
        output: JsonDict = {}
        issues: List[JsonDict] = []
        for item in observations:
            output.update(item)
            issues.extend(item.get("issues") or [])
        ok = not any(issue.get("severity") == "error" for issue in issues)
        return AgentResult(agent=self.name, ok=ok, output=output, issues=issues, steps=steps)

    def _build_payload(self, context: AgentContext, payload: JsonDict) -> JsonDict:
        if "businessMessageConfigVO" in payload:
            normalized = payload
        else:
            normalized = normalize_business_payload(context, payload)
        issues = validate_business_payload(normalized)
        context.scratch["businessPayload"] = normalized
        ok = not any(issue.get("severity") == "error" for issue in issues)
        return {"ok": ok, "summary": "business payload normalized", "businessPayload": normalized, "issues": issues}

    def _publish(self, context: AgentContext, payload: JsonDict) -> JsonDict:
        normalized = context.scratch.get("businessPayload") or normalize_business_payload(context, payload)
        response = context.platform.run_business_config(normalized, dry_run=context.request.dry_run)
        return {"ok": True, "summary": "business config boundary called", "businessResponse": response}


def normalize_business_payload(context: AgentContext, payload: JsonDict) -> JsonDict:
    business = payload.get("business") or payload
    config = {
        "id": business.get("id") or "",
        "name": business.get("name") or business.get("configName") or "AI generated business message",
        "channelType": business.get("channels") or business.get("channelType") or ["mail"],
        "documentId": business.get("documentId") or business.get("billId") or "",
        "documentNo": business.get("documentNo") or business.get("billNo") or "",
        "documentName": business.get("documentName") or "",
        "actionId": business.get("actionId") or business.get("action") or "",
        "templateId": business.get("templateId") or "",
        "state": 1 if business.get("enabled", True) else 0,
        "description": business.get("description") or context.request.text[:200],
        "triggerCondition": business.get("triggerCondition") or 2,
        "receivers": business.get("receivers") or [],
        "messageType": business.get("messageType") or "notice",
    }
    return {
        "sessionId": context.request.session_id,
        "name": config["name"],
        "channels": config["channelType"],
        "businessMessageConfigVO": config,
        "businessMessageConditionVOList": business.get("conditions") or business.get("businessMessageConditionVOList") or [],
    }


def validate_business_payload(payload: JsonDict) -> List[JsonDict]:
    config = payload.get("businessMessageConfigVO") or {}
    checks = [
        ("name", "business.name.required", "Business config name is required."),
        ("channelType", "business.channels.required", "At least one channel is required."),
        ("documentNo", "business.document_no.required", "Document number is required for business config."),
        ("actionId", "business.action.required", "Action id/code is required."),
        ("templateId", "business.template.required", "Template id is required before publishing."),
        ("receivers", "business.receivers.required", "At least one receiver is required."),
    ]
    issues: List[JsonDict] = []
    for key, code, message in checks:
        if not config.get(key):
            severity = "warning" if key in {"templateId", "receivers", "actionId", "documentNo"} else "error"
            issues.append({"code": code, "message": message, "severity": severity})
    return issues
