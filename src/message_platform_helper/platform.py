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
                "data": {"templates": stored or _mock_template_details_for_scope(payload)} if self.mock_database_url else _mock_template_list(payload),
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
                        {"name": "采购订单", "code": "purchase_order", "documentId": "purchase_order", "children": []},
                        {"name": "请购单", "code": "requisition_order", "documentId": "requisition_order", "children": []},
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
    default_detail = _mock_template_detail_by_id(template_id)
    if default_detail:
        return default_detail
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
    templates_vo = [_template_list_item_from_detail(detail, payload) for detail in _mock_template_details_for_scope(payload)]
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


def seed_mock_templates(database_url: str) -> JsonDict:
    """Restore local template mock snapshots used by dry-run template sync."""

    if not database_url:
        return {"ok": False, "seeded": 0, "error": "database_url is required"}
    seeded = 0
    errors: list[str] = []
    for payload in _default_mock_template_snapshots():
        result = _save_mock_template_snapshot(payload, database_url)
        if result.get("persisted"):
            seeded += 1
        elif result.get("error"):
            errors.append(str(result["error"]))
    return {"ok": not errors, "seeded": seeded, "errors": errors}


def _default_mock_template_snapshots() -> list[JsonDict]:
    return [
        _template_detail(
            template_id="tpl-requisition-submit",
            template_name="请购单提交通知",
            template_code="requisition_submit_notice",
            description="请购单提交后通知审批人处理",
            document_id="requisition_order",
            business_object_code="requisition_order",
            app_code="purchase_manage",
            domain="purchase_supply",
            contents=[
                _content(
                    "tpl-requisition-submit-mail-zh",
                    "mail",
                    "zh_CN",
                    "请购单 ${billNo} 待审批",
                    '<p>您好，{{approveUser}}：</p><p>请购单 <b>${billNo}</b> 已由 {{applyUser}} 提交，金额为 ${amount}，请及时审批。</p><p>采购组织：[md#purchaseOrg.name]</p>',
                    "html",
                    extend_content={
                        "editorVersion": "v2.0",
                        "lastEditUser": "mock-user",
                        "channelSetting": {"id": "mail-channel-default", "channelName": "默认邮件通道"},
                        "printAttachmentLocale": "zh_CN",
                        "variableDescription": [
                            {"code": "billNo", "name": "单据编号"},
                            {"code": "applyUser", "name": "申请人"},
                            {"code": "amount", "name": "金额"},
                        ],
                    },
                ),
                _content(
                    "tpl-requisition-submit-mail-en",
                    "mail",
                    "en_US",
                    "Purchase requisition ${billNo} requires approval",
                    "<p>Hello {{approveUser}},</p><p>purchase requisition <b>${billNo}</b> was submitted by {{applyUser}} for ${amount}. Please review it.</p><p>Purchase organization: [md#purchaseOrg.name]</p>",
                    "html",
                    extend_content={
                        "editorVersion": "v2.0",
                        "lastEditUser": "mock-user",
                        "channelSetting": {"id": "mail-channel-default", "channelName": "Default mail channel"},
                        "printAttachmentLocale": "en_US",
                    },
                ),
                _content(
                    "tpl-requisition-submit-uspace-zh",
                    "uspace",
                    "zh_CN",
                    "请购单 ${billNo} 待审批",
                    "请处理 {{applyUser}} 提交的请购单 #{billNo}，金额 ${amount}。",
                    "text",
                    extend_content={
                        "webUrl": "/yonbip/purchase/requisition/detail?billNo=${billNo}",
                        "mUrl": "/mobile/purchase/requisition/${billNo}",
                        "messageLevel": "normal",
                        "editorVersion": "v2.0",
                    },
                ),
                _content(
                    "tpl-requisition-submit-uspace-en",
                    "uspace",
                    "en_US",
                    "Purchase requisition ${billNo} requires approval",
                    "Please process purchase requisition #{billNo} submitted by {{applyUser}} for ${amount}.",
                    "text",
                    extend_content={
                        "webUrl": "/yonbip/purchase/requisition/detail?billNo=${billNo}",
                        "mUrl": "/mobile/purchase/requisition/${billNo}",
                        "messageLevel": "normal",
                        "editorVersion": "v2.0",
                    },
                ),
                _content(
                    "tpl-requisition-submit-sms-zh",
                    "sms",
                    "zh_CN",
                    "【用友】",
                    "请购单${billNo}已提交，请{{approveUser}}审批。",
                    "text",
                    extend_content={"smsType": "notice", "smsCategory": "mock-sms"},
                ),
                _content(
                    "tpl-requisition-submit-sms-en",
                    "sms",
                    "en_US",
                    "[Yonyou]",
                    "Purchase requisition ${billNo} was submitted. Please approve it, {{approveUser}}.",
                    "text",
                    extend_content={"smsType": "notice", "smsCategory": "mock-sms"},
                ),
            ],
        ),
        _template_detail(
            template_id="tpl-purchase-order-approved",
            template_name="采购订单审批通过通知",
            template_code="purchase_order_approved_notice",
            description="采购订单审批通过后通知申请人",
            document_id="purchase_order",
            business_object_code="purchase_order",
            app_code="purchase_manage",
            domain="purchase_supply",
            contents=[
                _content(
                    "tpl-purchase-order-approved-mail-zh",
                    "mail",
                    "zh_CN",
                    "采购订单 ${orderNo} 已审批通过",
                    '<p>您好 {{applyUser}}，采购订单 <b>${orderNo}</b> 已审批通过。</p><p>供应商：[md#supplier.name]，含税金额：${taxInclusiveAmount}</p>',
                    "html",
                    extend_content={
                        "editorVersion": "v2.0",
                        "lastEditUser": "mock-user",
                        "channelSetting": {"id": "mail-channel-default", "channelName": "默认邮件通道"},
                    },
                ),
                _content(
                    "tpl-purchase-order-approved-mail-en",
                    "mail",
                    "en_US",
                    "Purchase order ${orderNo} was approved",
                    "<p>Hello {{applyUser}}, purchase order <b>${orderNo}</b> was approved.</p><p>Supplier: [md#supplier.name], amount with tax: ${taxInclusiveAmount}</p>",
                    "html",
                    extend_content={
                        "editorVersion": "v2.0",
                        "lastEditUser": "mock-user",
                        "channelSetting": {"id": "mail-channel-default", "channelName": "Default mail channel"},
                    },
                ),
                _content(
                    "tpl-purchase-order-approved-uspace-zh",
                    "uspace",
                    "zh_CN",
                    "采购订单 ${orderNo} 已通过",
                    "采购订单 #{orderNo} 已审批通过，供应商：[md#supplier.name]。",
                    "text",
                    extend_content={
                        "webUrl": "/yonbip/purchase/order/detail?orderNo=${orderNo}",
                        "mUrl": "/mobile/purchase/order/${orderNo}",
                        "messageLevel": "normal",
                        "editorVersion": "v2.0",
                    },
                ),
                _content(
                    "tpl-purchase-order-approved-uspace-en",
                    "uspace",
                    "en_US",
                    "Purchase order ${orderNo} was approved",
                    "Purchase order #{orderNo} was approved. Supplier: [md#supplier.name].",
                    "text",
                    extend_content={
                        "webUrl": "/yonbip/purchase/order/detail?orderNo=${orderNo}",
                        "mUrl": "/mobile/purchase/order/${orderNo}",
                        "messageLevel": "normal",
                        "editorVersion": "v2.0",
                    },
                ),
            ],
        ),
    ]


def _template_detail(
    *,
    template_id: str,
    template_name: str,
    template_code: str,
    description: str,
    document_id: str,
    business_object_code: str,
    app_code: str,
    domain: str,
    contents: list[JsonDict],
) -> JsonDict:
    language_dict = {
        "templateName": {
            "zh_CN": {"id": f"{template_id}-name-zh", "value": template_name},
            "en_US": {"id": f"{template_id}-name-en", "value": _english_template_name(template_code)},
            "zh_TW": {"id": "", "value": ""},
        },
        "description": {
            "zh_CN": {"id": f"{template_id}-desc-zh", "value": description},
            "en_US": {"id": f"{template_id}-desc-en", "value": _english_description(template_code)},
            "zh_TW": {"id": "", "value": ""},
        },
    }
    message_temp = {
        "id": template_id,
        "templateName": template_name,
        "templateCode": template_code,
        "description": description,
        "groupCode": "businessprocess",
        "documentId": document_id,
        "msgDocumentId": document_id,
        "businessObjectCode": business_object_code,
        "billNo": document_id,
        "appCode": app_code,
        "domain": domain,
        "orgId": "666666",
        "transId": "",
        "uri": "",
        "srcType": "businessprocess",
        "typeId": "type-businessprocess",
        "typeCode": "businessprocess",
        "typeName": "业务流程",
        "enable": 1,
        "tenantDefault": 0,
        "templateLevel": 3,
        "wechatTempId": "",
        "firstRegion": "",
        "firstRegionName": [],
        "languageDict": language_dict,
        "metaDataTypeMapper": json.dumps(
            {
                "billNo": "单据编号",
                "orderNo": "订单编号",
                "applyUser": "申请人",
                "approveUser": "审批人",
                "amount": "金额",
                "purchaseOrg.name": "采购组织",
                "supplier.name": "供应商",
            },
            ensure_ascii=False,
        ),
    }
    return {
        "messageTempVO": message_temp,
        "messageTemplateContentVOList": contents,
    }


def _content(
    content_id: str,
    channel_type: str,
    language: str,
    title: str,
    content: str,
    content_type: str,
    *,
    extend_content: JsonDict | None = None,
) -> JsonDict:
    extend_content = dict(extend_content or {})
    return {
        "id": content_id,
        "key": content_id,
        "channelType": channel_type,
        "language": language,
        "title": title,
        "content": content,
        "raw": content,
        "contentType": content_type,
        "extendContent": json.dumps(extend_content, ensure_ascii=False),
        "messageTemplateAnnexPOList": [],
        "annexIdList": [],
        "status": "",
        "target": False,
    }


def _english_template_name(template_code: str) -> str:
    names = {
        "requisition_submit_notice": "Purchase requisition submission notice",
        "purchase_order_approved_notice": "Purchase order approval notice",
    }
    return names.get(template_code, template_code.replace("_", " ").title())


def _english_description(template_code: str) -> str:
    descriptions = {
        "requisition_submit_notice": "Notify approvers after a purchase requisition is submitted",
        "purchase_order_approved_notice": "Notify applicants after a purchase order is approved",
    }
    return descriptions.get(template_code, "")


def _mock_template_detail_by_id(template_id: str) -> JsonDict:
    for detail in _default_mock_template_snapshots():
        if _mock_template_id(detail) == template_id:
            return detail
    return {}


def _mock_template_details_for_scope(payload: JsonDict) -> list[JsonDict]:
    templates = _default_mock_template_snapshots()
    query_values = _mock_template_document_values(payload)
    if not query_values:
        return templates
    return [detail for detail in templates if _mock_template_scope_matches(payload, detail, detail.get("messageTempVO") or {})]


def _template_list_item_from_detail(detail: JsonDict, payload: JsonDict) -> JsonDict:
    message_temp = detail.get("messageTempVO") if isinstance(detail.get("messageTempVO"), dict) else {}
    item = {
        "id": message_temp.get("id") or _mock_template_id(detail),
        "templateName": message_temp.get("templateName") or "",
        "typeName": message_temp.get("typeName") or "业务流程",
        "templateCode": message_temp.get("templateCode") or "",
        "enable": message_temp.get("enable", 1),
        "tenantDefault": message_temp.get("tenantDefault", 0),
        "templateLevel": message_temp.get("templateLevel", 3),
        "firstRegion": message_temp.get("firstRegion") or "",
        "firstRegionName": message_temp.get("firstRegionName") or [],
        "groupCode": message_temp.get("groupCode") or "businessprocess",
        "documentId": message_temp.get("documentId") or "",
        "msgDocumentId": message_temp.get("msgDocumentId") or message_temp.get("documentId") or "",
        "businessObjectCode": message_temp.get("businessObjectCode") or "",
        "appCode": payload.get("appCode") or message_temp.get("appCode") or "",
        "orgId": payload.get("orgId") or message_temp.get("orgId") or "666666",
        "uri": payload.get("uri") or message_temp.get("uri") or "",
        "srcType": payload.get("srcType") or message_temp.get("srcType") or "",
        "metadataId": payload.get("metadataId") or message_temp.get("metadataId") or "",
        "domain": payload.get("domain") or message_temp.get("domain") or "",
        "typeId": message_temp.get("typeId") or "type-businessprocess",
        "typeCode": message_temp.get("typeCode") or "businessprocess",
    }
    return {**item, "detail": detail, "currentOrgTemps": [{**item, "detail": detail}]}


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
                    SELECT payload, scope
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
        scope_value = row[1] if len(row) > 1 else {}
        if isinstance(payload_value, str):
            try:
                payload_value = json.loads(payload_value)
            except json.JSONDecodeError:
                continue
        if isinstance(scope_value, str):
            try:
                scope_value = json.loads(scope_value)
            except json.JSONDecodeError:
                scope_value = {}
        if isinstance(payload_value, dict) and _mock_template_scope_matches(payload, payload_value, scope_value):
            result.append(payload_value)
    return result


def _mock_template_scope_matches(query: JsonDict, template_payload: JsonDict, template_scope: object) -> bool:
    query_values = _mock_template_document_values(query)
    if not query_values:
        return False
    candidate_values = _mock_template_document_values(template_payload)
    if isinstance(template_scope, dict):
        candidate_values.update(_mock_template_document_values(template_scope))
    message_temp = template_payload.get("messageTempVO") if isinstance(template_payload.get("messageTempVO"), dict) else {}
    candidate_values.update(_mock_template_document_values(message_temp))
    return bool(candidate_values & query_values)


def _mock_template_document_values(payload: JsonDict) -> set[str]:
    keys = ("documentId", "msgDocumentId", "billNo", "billId", "businessObjectCode")
    return {str(payload.get(key)).strip() for key in keys if payload.get(key) not in (None, "")}


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
