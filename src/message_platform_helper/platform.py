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

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from .models import JsonDict


class PlatformError(RuntimeError):
    pass


@dataclass
class PlatformGateway:
    platform_base_url: str = ""
    business_agent_url: str = ""
    mock_database_url: str = ""
    headers: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: int = 20

    def save_template(self, payload: JsonDict, dry_run: bool = True) -> JsonDict:
        if dry_run or not self.platform_base_url:
            template_id = _mock_template_id(payload)
            persistence = _save_mock_template_snapshot(payload, self.mock_database_url)
            return {
                "ok": True,
                "status": 1,
                "msg": "success",
                "data": {"templateId": template_id, "saved": True, **persistence},
                "endpoint": "/msgTemplate/saveTempInfo",
                "payload": payload,
            }
        return self._post_platform("/msgTemplate/saveTempInfo", payload)

    def find_templates(self, payload: JsonDict, dry_run: bool = True) -> JsonDict:
        if dry_run or not self.platform_base_url:
            stored = _find_mock_template_snapshots(payload, self.mock_database_url)
            return {
                "ok": True,
                "status": 1,
                "data": {"templates": stored} if stored else _mock_template_list(payload),
                "endpoint": "/msgTemplate/findMsgTemps",
                "payload": payload,
            }
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
        if dry_run or not self.platform_base_url:
            stored = _load_mock_template_snapshot(template_id, self.mock_database_url)
            return {
                "ok": True,
                "status": 1,
                "data": stored or _mock_template_detail(template_id, params),
                "endpoint": path,
                "payload": params,
                "templateId": template_id,
            }
        return self._get_platform(path, params)

    def get_domain_tree(self, payload: JsonDict | None = None, dry_run: bool = True) -> JsonDict:
        payload = dict(payload or {})
        group_code = str(payload.get("groupCode") or payload.get("categorycode") or "businessprocess")
        params = {"bizObjectSystem": payload.get("bizObjectSystem", True)}
        endpoint = f"/console/categorytree/{group_code}"
        if dry_run or not self.platform_base_url:
            return {
                "ok": True,
                "status": 1,
                "tool": "domain.tree.get",
                "endpoint": endpoint,
                "payload": params,
                "tree": _mock_domain_tree(),
            }
        response = self._get_platform(endpoint, params)
        return {"ok": True, "tool": "domain.tree.get", "endpoint": endpoint, "payload": params, "tree": _extract_domain_tree(response), "raw": response}

    def save_channel_config(self, payload: JsonDict, dry_run: bool = True) -> JsonDict:
        if dry_run or not self.platform_base_url:
            return {
                "ok": True,
                "status": 1,
                "msg": "success",
                "data": {"saved": True},
                "endpoint": "/msgTemplateChannel/saveChannelConfig",
                "payload": payload,
            }
        return self._post_platform("/msgTemplateChannel/saveChannelConfig", payload)

    def test_email_channel(self, payload: JsonDict, dry_run: bool = True) -> JsonDict:
        if dry_run:
            return {
                "ok": True,
                "status": 1,
                "msg": "success",
                "data": {"receiver": _mock_verify_user(payload)},
                "endpoint": "/test/sendTestWithChannelConfig",
                "payload": payload,
            }
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
        return {
            "ok": True,
            "status": 1,
            "msg": "success",
            "data": {"saved": True},
            "endpoint": "message-agent:/api/config/preview",
            "payload": payload,
        }

    def save_strategy(self, payload: JsonDict, dry_run: bool = True) -> JsonDict:
        endpoint = "/msgSendStrategy/save"
        if dry_run or not self.platform_base_url:
            return {
                "ok": True,
                "status": 1,
                "msg": "success",
                "data": {"saved": True},
                "endpoint": endpoint,
                "payload": payload,
            }
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


def _extract_domain_tree(response: JsonDict) -> list[JsonDict]:
    data = response.get("data") if isinstance(response, dict) else response
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    children = data.get("children")
    if isinstance(children, list):
        return children
    tree = data.get("tree")
    return tree if isinstance(tree, list) else [data]


def _mock_domain_tree() -> list[JsonDict]:
    return [
        {
            "name": "采购供应",
            "code": "purchase_supply",
            "children": [
                {
                    "name": "采购管理",
                    "code": "purchase_manage",
                    "children": [
                        {"name": "采购订单", "documentId": "purchase_order", "children": []},
                        {"name": "请购单",  "documentId": "requisition_order", "children": []}
                    ],
                }
            ],
        }
    ]


def _mock_template_id(payload: JsonDict) -> str:
    message_temp = payload.get("messageTempVO") if isinstance(payload.get("messageTempVO"), dict) else {}
    for key in ("id", "templateId", "template_id"):
        value = payload.get(key) or message_temp.get(key)
        if value:
            return str(value)
    return "tpl-mock-001"


def _mock_verify_user(payload: JsonDict) -> str:
    config = payload.get("configInfo")
    if isinstance(config, str):
        try:
            config = json.loads(config)
        except json.JSONDecodeError:
            config = {}
    if isinstance(config, dict):
        return str(config.get("verifyUser") or config.get("mailUsername") or "")
    return ""


def _mock_template_detail(template_id: str, scope: JsonDict) -> JsonDict:
    template_code = f"{template_id}-code"
    message_temp = {
        "id": template_id,
        "templateName": "请购单保存通知",
        "templateCode": template_code,
        "description": "请购单保存通知",
        "groupCode": "businessprocess",
        "documentId": "requisition_order",
        "domain": "",
        "orgId": "",
        "transId": "",
        "typeId": "type-001",
        "wechatTempId": "",
        "firstRegion": "",
        "languageDict": {
            "templateName": {
                "zh_CN": {"id": "", "value": "请购单保存通知"},
            },
            "description": {
                "zh_CN": {"id": "", "value": "请购单保存通知"},
            },
        },
        "metaDataTypeMapper": "{}",
    }
    return {
        "messageTempVO": message_temp,
        "messageTemplateContentVOList": [
            {
                "id": f"{template_id}-content-mail-zh",
                "channelType": "mail",
                "language": "zh_CN",
                "title": "请购单保存通知",
                "content": "你有一条请购单保存的通知，单据编号为：{billNo}",
                "contentType": "text",
            }
        ],
    }


def _mock_template_list(payload: JsonDict) -> JsonDict:
    templates_vo = [
        {
            "id": "tpl-001",
            "templateName": "Purchase requisition notice",
            "typeName": "请购单保存",
            "templateCode": "purchase_requisition_notice",
            "enable": 1,
            "tenantDefault": 0,
            "templateLevel": 3,
            "firstRegion": "",
            "firstRegionName": [],
            "groupCode": "businessprocess",
            "documentId": "PR001",
            "appCode": payload.get("appCode") or "",
            "orgId": payload.get("orgId") or "666666",
            "domain": payload.get("domain") or "",
            "typeId": "type-001",
            "typeCode": "businessprocess",
            "currentOrgTemps": [
                {
                    "id": "tpl-001",
                    "templateName": "Purchase requisition notice",
                    "typeName": "请购单保存",
                    "templateCode": "purchase_requisition_notice",
                    "enable": 1,
                    "tenantDefault": 0,
                    "templateLevel": 3,
                    "firstRegion": "",
                    "firstRegionName": [],
                    "groupCode": "businessprocess",
                    "documentId": "PR001",
                    "appCode": payload.get("appCode") or "",
                    "orgId": payload.get("orgId") or "666666",
                    "uri": payload.get("uri") or "",
                    "srcType": payload.get("srcType") or "",
                    "metadataId": payload.get("metadataId") or "",
                    "domain": payload.get("domain") or "",
                    "typeId": "type-001",
                    "typeCode": "businessprocess",
                }
            ],
        }
    ]
    return {
        "templatesVOList": templates_vo,
        "msgTempTypeCatalogVOList": [
            {
                "id": "type-001",
                "typeCode": "businessprocess",
                "typeName": "\u4e1a\u52a1\u6d41\u7a0b",
            }
        ],
    }


def _save_mock_template_snapshot(payload: JsonDict, database_url: str) -> JsonDict:
    if not database_url:
        return {"persisted": False}
    try:
        engine = create_engine(database_url, pool_pre_ping=True, connect_args={"connect_timeout": 3})
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS mock_template_snapshots (
                        template_id text PRIMARY KEY,
                        template_code text NOT NULL DEFAULT '',
                        template_name text NOT NULL DEFAULT '',
                        scope jsonb NOT NULL DEFAULT '{}'::jsonb,
                        payload jsonb NOT NULL,
                        created_at timestamptz NOT NULL DEFAULT now(),
                        updated_at timestamptz NOT NULL DEFAULT now()
                    )
                    """
                )
            )
            message_temp = payload.get("messageTempVO") if isinstance(payload.get("messageTempVO"), dict) else {}
            scope = {key: value for key, value in {**payload, **message_temp}.items() if key in {"orgId", "groupCode", "domain", "appCode", "documentId", "msgDocumentId", "billNo", "billId", "templateTypeId", "msgTemplateTypeId", "typeId", "transId", "uri", "srcType", "businessObjectCode"} and value not in (None, "")}
            conn.execute(
                text(
                    """
                    INSERT INTO mock_template_snapshots (template_id, template_code, template_name, scope, payload, created_at, updated_at)
                    VALUES (:template_id, :template_code, :template_name, CAST(:scope AS jsonb), CAST(:payload AS jsonb), now(), now())
                    ON CONFLICT (template_id) DO UPDATE
                    SET template_code = EXCLUDED.template_code,
                        template_name = EXCLUDED.template_name,
                        scope = EXCLUDED.scope,
                        payload = EXCLUDED.payload,
                        updated_at = now()
                    """
                ),
                {
                    "template_id": _mock_template_id(payload),
                    "template_code": _template_code_from_payload(payload),
                    "template_name": _template_name_from_payload(payload),
                    "scope": json.dumps(scope, ensure_ascii=False),
                    "payload": json.dumps(payload, ensure_ascii=False),
                },
            )
        return {"persisted": True}
    except SQLAlchemyError as exc:
        return {"persisted": False, "error": str(exc)}


def _find_mock_template_snapshots(payload: JsonDict, database_url: str) -> list[JsonDict]:
    if not database_url:
        return []
    try:
        engine = create_engine(database_url, pool_pre_ping=True, connect_args={"connect_timeout": 3})
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS mock_template_snapshots (
                        template_id text PRIMARY KEY,
                        template_code text NOT NULL DEFAULT '',
                        template_name text NOT NULL DEFAULT '',
                        scope jsonb NOT NULL DEFAULT '{}'::jsonb,
                        payload jsonb NOT NULL,
                        created_at timestamptz NOT NULL DEFAULT now(),
                        updated_at timestamptz NOT NULL DEFAULT now()
                    )
                    """
                )
            )
            rows = conn.execute(
                text(
                    """
                    SELECT payload
                    FROM mock_template_snapshots
                    ORDER BY updated_at DESC
                    LIMIT 50
                    """
                )
            ).all()
    except SQLAlchemyError:
        return []
    result: list[JsonDict] = []
    for row in rows:
        payload_value = row[0]
        if isinstance(payload_value, str):
            try:
                payload_value = json.loads(payload_value)
            except json.JSONDecodeError:
                continue
        if isinstance(payload_value, dict):
            result.append(payload_value)
    return result


def _load_mock_template_snapshot(template_id: str, database_url: str) -> JsonDict:
    if not database_url or not template_id:
        return {}
    try:
        engine = create_engine(database_url, pool_pre_ping=True, connect_args={"connect_timeout": 3})
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS mock_template_snapshots (
                        template_id text PRIMARY KEY,
                        template_code text NOT NULL DEFAULT '',
                        template_name text NOT NULL DEFAULT '',
                        scope jsonb NOT NULL DEFAULT '{}'::jsonb,
                        payload jsonb NOT NULL,
                        created_at timestamptz NOT NULL DEFAULT now(),
                        updated_at timestamptz NOT NULL DEFAULT now()
                    )
                    """
                )
            )
            row = conn.execute(
                text(
                    """
                    SELECT payload
                    FROM mock_template_snapshots
                    WHERE template_id = :template_id
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """
                ),
                {"template_id": template_id},
            ).first()
    except SQLAlchemyError:
        return {}
    if not row:
        return {}
    payload_value = row[0]
    if isinstance(payload_value, str):
        try:
            payload_value = json.loads(payload_value)
        except json.JSONDecodeError:
            return {}
    return payload_value if isinstance(payload_value, dict) else {}


def _template_code_from_payload(payload: JsonDict) -> str:
    message_temp = payload.get("messageTempVO") if isinstance(payload.get("messageTempVO"), dict) else {}
    return str(payload.get("templateCode") or payload.get("template_code") or message_temp.get("templateCode") or "")


def _template_name_from_payload(payload: JsonDict) -> str:
    message_temp = payload.get("messageTempVO") if isinstance(payload.get("messageTempVO"), dict) else {}
    return str(payload.get("templateName") or payload.get("title") or message_temp.get("templateName") or "")
