"""Adapters for Java message platform and existing message-agent."""

from __future__ import annotations

import json
import smtplib
import ssl
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from email.message import EmailMessage
from typing import Any, Dict, Mapping

from .models import JsonDict


class PlatformError(RuntimeError):
    pass


@dataclass
class PlatformGateway:
    platform_base_url: str = ""
    business_agent_url: str = ""
    headers: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: int = 20

    def save_template(self, payload: JsonDict, dry_run: bool = True) -> JsonDict:
        if dry_run or not self.platform_base_url:
            return {"ok": True, "status": "dry-run", "endpoint": "/msgTemplate/saveTempInfo", "payload": payload}
        return self._post_platform("/msgTemplate/saveTempInfo", payload)

    def find_templates(self, payload: JsonDict, dry_run: bool = True) -> JsonDict:
        if not self.platform_base_url:
            return {"ok": True, "status": "dry-run", "endpoint": "/msgTemplate/findMsgTemps", "payload": payload}
        return self._get_platform("/msgTemplate/findMsgTemps", payload)

    def get_template_detail(self, template_id: str, scope: JsonDict | None = None, dry_run: bool = True) -> JsonDict:
        scope = dict(scope or {})
        app_code = str(scope.get("appCode") or scope.get("app_code") or "")
        bill_no = str(scope.get("billNo") or scope.get("documentId") or scope.get("msgDocumentId") or scope.get("billId") or "")
        src_type = str(scope.get("srcType") or "")
        uri = str(scope.get("uri") or "")
        use_business_path = bool(app_code and bill_no)
        path = f"/msgTemplate/getBizProcessTemplateInfoById/{template_id}" if use_business_path else f"/msgTemplate/getTemplateInfoById/{template_id}"
        params: JsonDict = {}
        if use_business_path:
            params["appCode"] = app_code
            params["billNo"] = bill_no
        if src_type:
            params["srcType"] = src_type
        if uri:
            params["uri"] = uri
        if not self.platform_base_url:
            return {"ok": True, "status": "dry-run", "endpoint": path, "payload": params, "templateId": template_id}
        return self._get_platform(path, params)

    def save_channel_config(self, payload: JsonDict, dry_run: bool = True) -> JsonDict:
        if dry_run or not self.platform_base_url:
            return {"ok": True, "status": "dry-run", "endpoint": "/msgTemplateChannel/saveChannelConfig", "payload": payload}
        return self._post_platform("/msgTemplateChannel/saveChannelConfig", payload)

    def test_email_channel(self, payload: JsonDict, dry_run: bool = True) -> JsonDict:
        if dry_run:
            return {"ok": True, "status": "dry-run", "endpoint": "/test/sendTestWithChannelConfig", "payload": payload}
        if self.platform_base_url:
            return self._post_platform("/test/sendTestWithChannelConfig", payload)
        config = payload.get("configInfo")
        if isinstance(config, str):
            config = json.loads(config)
        return smtp_send_test(config or {})

    def run_business_config(self, payload: JsonDict, dry_run: bool = True) -> JsonDict:
        if self.business_agent_url:
            path = "/api/config/preview" if dry_run else "/api/config/publish"
            return self._post_url(self.business_agent_url.rstrip("/") + path, payload)
        if self.platform_base_url and not dry_run:
            return self._post_platform("/bizMessageConfig/save", payload)
        return {"ok": True, "status": "dry-run", "endpoint": "message-agent:/api/config/preview", "payload": payload}

    def save_strategy(self, payload: JsonDict, dry_run: bool = True) -> JsonDict:
        endpoint = "/msgSendStrategy/save"
        if dry_run or not self.platform_base_url:
            return {"ok": True, "status": "dry-run", "endpoint": endpoint, "payload": payload}
        return self._post_platform(endpoint, payload)

    def _post_platform(self, path: str, payload: JsonDict) -> JsonDict:
        return self._post_url(self.platform_base_url.rstrip("/") + "/" + path.lstrip("/"), payload)

    def _get_platform(self, path: str, params: JsonDict | None = None) -> JsonDict:
        url = self.platform_base_url.rstrip("/") + "/" + path.lstrip("/")
        if params:
            query = urllib.parse.urlencode({key: value for key, value in params.items() if value not in (None, "")}, doseq=True)
            if query:
                url = f"{url}?{query}"
        return self._get_url(url)

    def _get_url(self, url: str) -> JsonDict:
        headers = {"Content-Type": "application/json", **dict(self.headers)}
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise PlatformError(f"HTTP request failed: {exc}") from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PlatformError(f"HTTP response is not JSON: {raw[:200]}") from exc
        if data.get("ok") is False or data.get("status") in (0, "0", "failed"):
            raise PlatformError(str(data.get("msg") or data.get("message") or data))
        return data

    def _post_url(self, url: str, payload: JsonDict) -> JsonDict:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", **dict(self.headers)}
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise PlatformError(f"HTTP request failed: {exc}") from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PlatformError(f"HTTP response is not JSON: {raw[:200]}") from exc
        if data.get("ok") is False or data.get("status") in (0, "0", "failed"):
            raise PlatformError(str(data.get("msg") or data.get("message") or data))
        return data


def smtp_send_test(config: JsonDict) -> JsonDict:
    required = ["mailHost", "mailPort", "mailUsername", "mailPwd", "verifyUser"]
    missing = [key for key in required if not config.get(key)]
    if missing:
        return {
            "ok": False,
            "status": "missing-config",
            "missing": missing,
            "message": "Real SMTP test requires host, port, username, password and verify receiver.",
        }
    host = str(config["mailHost"])
    port = int(config["mailPort"])
    username = str(config["mailUsername"])
    password = str(config["mailPwd"])
    receiver = str(config["verifyUser"])
    sender = str(config.get("mailSender") or username)
    message = EmailMessage()
    message["Subject"] = "Message platform helper email test"
    message["From"] = sender
    message["To"] = receiver
    message.set_content("This is a test email from message-platform-helper.")
    if bool(config.get("smtpSSL")):
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(host, port, timeout=15, context=context) as client:
            client.login(username, password)
            client.send_message(message)
    else:
        with smtplib.SMTP(host, port, timeout=15) as client:
            if bool(config.get("smtpTLS", True)):
                client.starttls(context=ssl.create_default_context())
            client.login(username, password)
            client.send_message(message)
    return {"ok": True, "status": "sent", "receiver": receiver}
