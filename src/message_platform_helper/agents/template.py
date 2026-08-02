"""Template translation and localization sync agent."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List
from threading import Lock

from .business_object_resolver import BusinessObjectResolver
from ..models import AgentResult, AgentStep, JsonDict, Tool
from ..react import AgentContext, ReActAgent


DEFAULT_SOURCE_LANGUAGE = "en_US"
DEFAULT_TEMPLATE_SYNC_BATCH_SIZE = 20
DEFAULT_TEMPLATE_SYNC_MAX_CONCURRENCY = 4

LANGUAGE_ALIASES = {
    "en": "en_US",
    "en_us": "en_US",
    "en-us": "en_US",
    "english": "en_US",
    "英语": "en_US",
    "英文": "en_US",
    "zh": "zh_CN",
    "zh_cn": "zh_CN",
    "zh-cn": "zh_CN",
    "chinese": "zh_CN",
    "simplified chinese": "zh_CN",
    "简体中文": "zh_CN",
    "中文": "zh_CN",
    "zh_tw": "zh_TW",
    "zh-tw": "zh_TW",
    "traditional chinese": "zh_TW",
    "繁体中文": "zh_TW",
    "ar": "ar_SA",
    "ar_sa": "ar_SA",
    "ar-sa": "ar_SA",
    "arabic": "ar_SA",
    "阿拉伯语": "ar_SA",
    "阿拉伯文": "ar_SA",
    "阿拉伯": "ar_SA",
}

LANGUAGE_ALIASES.update(
    {
        "id": "id_ID",
        "id_id": "id_ID",
        "id-id": "id_ID",
        "indonesian": "id_ID",
        "\u5370\u5c3c": "id_ID",
        "\u5370\u5c3c\u8bed": "id_ID",
        "\u5370\u5c3c\u6587": "id_ID",
    }
)

LANGUAGE_HINTS = (
    ("\u5370\u5c3c\u8bed", "id_ID"),
    ("\u5370\u5c3c\u6587", "id_ID"),
    ("\u5370\u5c3c", "id_ID"),
    ("indonesian", "id_ID"),
    ("阿拉伯语", "ar_SA"),
    ("阿拉伯文", "ar_SA"),
    ("阿拉伯", "ar_SA"),
    ("arabic", "ar_SA"),
    ("英语", "en_US"),
    ("英文", "en_US"),
    ("english", "en_US"),
    ("繁体中文", "zh_TW"),
    ("简体中文", "zh_CN"),
)

SCOPE_KEYS = (
    "orgId",
    "groupCode",
    "domain",
    "appCode",
    "documentId",
    "msgDocumentId",
    "billNo",
    "billId",
    "templateTypeId",
    "msgTemplateTypeId",
    "typeId",
    "transId",
    "uri",
    "srcType",
    "businessObjectCode",
)

TEMPLATE_LIST_KEYS = ("templates", "templateList", "templateDetails", "sourceTemplates")

MOCK_DOMAIN_TREE: List[JsonDict] = [
    {
        "nodeName": "供应链云",
        "nodeCode": "supply-cloud",
        "nodeId": "supply-cloud",
        "nodeType": "supply-cloud",
        "children": [
            {
                "nodeName": "采购供应",
                "nodeCode": "purchase_supply",
                "nodeId": "purchase_supply",
                "nodeType": "domain",
                "children": [
                    {
                        "nodeName": "采购管理",
                        "nodeCode": "purchase_manage",
                        "nodeId": "purchase_manage",
                        "nodeType": "application",
                        "domain": "purchase_supply",
                        "appCode": "purchase_manage",
                        "children": [
                            {
                                "nodeName": "采购订单",
                                "nodeCode": "purchase_order",
                                "nodeId": "purchase_order",
                                "nodeType": "document",
                                "domain": "purchase_supply",
                                "appCode": "purchase_manage",
                            },
                            {
                                "nodeName": "销售订单",
                                "nodeCode": "sales_order",
                                "nodeId": "sales_order",
                                "nodeType": "document",
                                "domain": "purchase_supply",
                                "appCode": "purchase_manage",
                            },
                            {
                                "nodeName": "库存调拨单",
                                "nodeCode": "inventory_transfer",
                                "nodeId": "inventory_transfer",
                                "nodeType": "document",
                                "domain": "purchase_supply",
                                "appCode": "purchase_manage",
                            },
                        ],
                    }
                ],
            }
        ],
    }
]


@dataclass
class TemplateAgent(ReActAgent):
    name = "template"

    def tools(self, context: AgentContext) -> Dict[str, Tool]:
        return {
            "template.translate": Tool(
                "template.translate",
                "Translate an existing template to a target language.",
                lambda payload: self._translate(context, payload),
            ),
            "template.sync_language": Tool(
                "template.sync_language",
                "Add a missing target-language version to templates in a scope.",
                lambda payload: self._sync_language(context, payload),
            ),
            "domain.tree.get": Tool(
                "domain.tree.get",
                "Fetch the full domain tree for business-object matching.",
                lambda payload: self._get_domain_tree(context, payload),
            ),
            "business_object.resolve": Tool(
                "business_object.resolve",
                "Resolve a business object code from the fetched domain tree.",
                lambda payload: self._resolve_business_object(context, payload),
            ),
        }

    def plan(self, context: AgentContext) -> List[JsonDict]:
        payload = _normalize_request(context, _template_payload(context))
        if _needs_business_object_resolution(payload):
            return [
                {
                    "thought": "Fetch the complete domain tree before matching the requested business object.",
                    "tool": "domain.tree.get",
                    "input": payload,
                },
                {
                    "thought": "Resolve the business object name to a platform business object code.",
                    "tool": "business_object.resolve",
                    "input": payload,
                },
                {
                    "thought": "Use the resolved business object code to sync missing target-language template versions.",
                    "tool": "template.sync_language",
                    "input": payload,
                },
            ]
        if _is_batch_sync_request(payload, context.request.text):
            return [
                {
                    "thought": "The request targets a template scope, so fetch existing templates and add only missing language versions.",
                    "tool": "template.sync_language",
                    "input": payload,
                }
            ]
        return [
            {
                "thought": "The template agent only translates existing templates, so prepare a translated language version.",
                "tool": "template.translate",
                "input": payload,
            }
        ]

    def finalize(self, context: AgentContext, observations: List[JsonDict], steps: List[AgentStep]) -> AgentResult:
        output: JsonDict = {}
        issues: List[JsonDict] = []
        for item in observations:
            output.update({key: value for key, value in item.items() if key != "issues"})
            issues.extend(item.get("issues") or [])
        ok = all(item.get("ok", True) for item in observations) and not any(issue.get("severity") == "error" for issue in issues)
        return AgentResult(agent=self.name, ok=ok, output=output, issues=issues, steps=steps)

    def _get_domain_tree(self, context: AgentContext, payload: JsonDict) -> JsonDict:
        _trace(context, "\u8bc6\u522b\u6a21\u677f\u7ffb\u8bd1\u540c\u6b65\u4efb\u52a1")
        _trace(context, "\u8c03\u7528\u9886\u57df\u6811 MCP")
        response = context.platform.get_domain_tree(payload, dry_run=context.request.dry_run)
        tree = _extract_domain_tree(response) or MOCK_DOMAIN_TREE
        context.scratch["domainTreeResponse"] = response
        context.scratch["domainTree"] = tree
        return {
            "ok": bool(response.get("ok", True)),
            "summary": f"domain tree fetched with {len(tree)} root node(s)",
            "tree": tree,
            "domainTreeResponse": response,
            "executionTrace": list(context.scratch.get("templateExecutionTrace") or []),
        }

    def _resolve_business_object(self, context: AgentContext, payload: JsonDict) -> JsonDict:
        payload = _normalize_request(context, payload)
        name = str(payload.get("businessObject") or payload.get("businessObjectName") or "")
        _trace(context, f"\u8bc6\u522b\u4e1a\u52a1\u5bf9\u8c61\uff1a{name}")
        tree = payload.get("tree") if isinstance(payload.get("tree"), list) else context.scratch.get("domainTree") or []
        result = BusinessObjectResolver(context.llm).resolve(name, tree)
        if result.get("ok"):
            _trace(context, "\u5339\u914d\u4e1a\u52a1\u5bf9\u8c61\u7f16\u7801")
            context.scratch["businessObjectResolution"] = result
        return {**result, "executionTrace": list(context.scratch.get("templateExecutionTrace") or [])}

    def _translate(self, context: AgentContext, payload: JsonDict) -> JsonDict:
        payload = _normalize_request(context, payload)
        scope = _scope_payload(payload)
        template_candidates = _candidate_templates(payload)
        template_ref = template_candidates[0] if template_candidates else payload
        detail = _resolve_template_detail(context, template_ref, payload, scope)
        if not detail:
            return _error_result("template.source.required", "Existing template detail or templateId is required for translation.")

        translated = _translate_detail(context, detail, payload)
        if translated.get("status") == "skipped":
            return translated
        save_response = context.platform.save_template(translated["platformPayload"], dry_run=context.request.dry_run)
        translated["saveResponse"] = save_response
        context.scratch["lastTemplateTranslationPayload"] = translated["platformPayload"]
        return translated

    def _sync_language(self, context: AgentContext, payload: JsonDict) -> JsonDict:
        payload = _normalize_request(context, payload)
        _trace(context, "\u8bc6\u522b\u6a21\u677f\u7ffb\u8bd1\u540c\u6b65\u4efb\u52a1")
        resolution = _business_object_resolution(context, payload)
        if resolution.get("ambiguous"):
            return {
                "ok": False,
                "summary": "Multiple business object candidates require user confirmation.",
                "businessObjectCandidates": resolution.get("candidates") or [],
                "issues": resolution.get("issues") or [],
                "executionTrace": list(context.scratch.get("templateExecutionTrace") or []),
            }
        if resolution.get("ok"):
            payload.setdefault("businessObjectCode", resolution.get("businessObjectCode"))
            payload.setdefault("documentId", resolution.get("documentId"))
            payload.setdefault("msgDocumentId", resolution.get("documentId"))
            _trace(context, "\u5339\u914d\u4e1a\u52a1\u5bf9\u8c61\u7f16\u7801")
        elif payload.get("businessObject") and not (payload.get("documentId") or payload.get("msgDocumentId")):
            return {
                "ok": False,
                "summary": resolution.get("summary") or "Business object could not be resolved.",
                "issues": resolution.get("issues")
                or [
                    {
                        "code": "business_object.not_resolved",
                        "message": "Business object could not be resolved; refusing to query templates without a document id.",
                        "severity": "error",
                    }
                ],
                "businessObject": resolution,
                "executionTrace": list(context.scratch.get("templateExecutionTrace") or []),
            }
        scope = _scope_payload(payload)
        batch_size, max_concurrency = _sync_limits(payload)
        issues: List[JsonDict] = []
        translated_templates: List[JsonDict] = []
        skipped_templates: List[JsonDict] = []
        failed_templates: List[JsonDict] = []

        templates = _candidate_templates(payload)
        list_response: JsonDict | None = None
        if not templates:
            _trace(context, "\u83b7\u53d6\u6a21\u677f\u5217\u8868")
            list_response = context.platform.find_templates(scope, dry_run=context.request.dry_run)
            templates = _extract_template_candidates(list_response)

        if not templates:
            return _error_result(
                "template.scope.empty",
                "No templates were found. Provide template.templates or configure platform access for the requested scope.",
                extra={"scope": scope, "templateListResponse": list_response or {}},
            )

        for batch in _chunked(templates, batch_size):
            if len(batch) == 1 or max_concurrency <= 1:
                batch_results = [_sync_template_entry(context, batch[0], payload, scope)]
            else:
                with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
                    futures = {
                        executor.submit(_sync_template_entry, context, template_ref, payload, scope): index
                        for index, template_ref in enumerate(batch)
                    }
                    batch_results = []
                    for future in as_completed(futures):
                        batch_results.append((futures[future], future.result()))
                batch_results.sort(key=lambda item: item[0])
                batch_results = [item[1] for item in batch_results]

            for result in batch_results:
                issues.extend(result.get("issues") or [])
                kind = result.get("kind")
                if kind == "translated":
                    translated_templates.append(result["translated"])
                elif kind == "skipped":
                    skipped_templates.append(result["skipped"])
                elif kind == "failed":
                    failed_templates.append(result["failed"])

        has_errors = any(issue.get("severity") == "error" for issue in issues)
        return {
            "ok": not has_errors,
            "summary": f"template language sync translated {len(translated_templates)} template(s), skipped {len(skipped_templates)} existing template(s)",
            "scope": scope,
            "sourceLanguage": payload["sourceLanguage"],
            "targetLanguage": payload["targetLanguage"],
            "translatedTemplates": translated_templates,
            "skippedTemplates": skipped_templates,
            "failedTemplates": failed_templates,
            "templateListResponse": list_response or {},
            "issues": issues,
            "businessObject": resolution if resolution.get("ok") else {},
            "executionTrace": list(context.scratch.get("templateExecutionTrace") or []),
        }


def _template_payload(context: AgentContext) -> JsonDict:
    raw_payload = context.request.payload
    payload = dict(raw_payload.get("template") or {})
    for key in (
        "scope",
        "templates",
        "templateList",
        "templateDetails",
        "sourceTemplates",
        "sourceTemplate",
        "templateId",
        "templateIds",
        "targetLanguage",
        "target_language",
        "sourceLanguage",
        "source_language",
        "baseLanguage",
        "language",
        "syncAll",
        "batch",
        "batchSync",
        "businessObject",
        "businessObjectName",
        "business_object",
        "business_object_name",
        *SCOPE_KEYS,
    ):
        if key in raw_payload and key not in payload:
            payload[key] = raw_payload[key]
    if isinstance(raw_payload.get("scope"), dict):
        for key, value in raw_payload["scope"].items():
            payload.setdefault(key, value)
    if isinstance(payload.get("scope"), dict):
        for key, value in payload["scope"].items():
            payload.setdefault(key, value)
    payload.setdefault("businessObject", _detect_business_object(context.request.text))
    return payload


def _normalize_request(context: AgentContext, payload: JsonDict) -> JsonDict:
    normalized = dict(payload)
    target_language = (
        normalized.get("targetLanguage")
        or normalized.get("target_language")
        or _detect_target_language(context.request.text)
        or (normalized.get("language") if not _looks_like_template_detail(normalized) else "")
        or "en_US"
    )
    source_language = normalized.get("sourceLanguage") or normalized.get("source_language") or normalized.get("baseLanguage") or DEFAULT_SOURCE_LANGUAGE
    normalized["targetLanguage"] = normalize_language_code(target_language, "en_US")
    normalized["sourceLanguage"] = normalize_language_code(source_language, DEFAULT_SOURCE_LANGUAGE)
    normalized["businessObject"] = (
        normalized.get("businessObject")
        or normalized.get("businessObjectName")
        or normalized.get("business_object")
        or normalized.get("business_object_name")
        or _detect_business_object(context.request.text)
    )
    return normalized


def normalize_language_code(value: object, default: str = "") -> str:
    text = str(value or "").strip()
    if not text:
        return default
    alias_key = text.lower().replace("-", "_")
    return LANGUAGE_ALIASES.get(alias_key) or LANGUAGE_ALIASES.get(text) or text


def _detect_target_language(text: str) -> str:
    lowered = text.lower()
    for hint, code in LANGUAGE_HINTS:
        if hint in text or hint in lowered:
            return code
    return ""


def _is_batch_sync_request(payload: JsonDict, text: str) -> bool:
    if any(isinstance(payload.get(key), list) for key in TEMPLATE_LIST_KEYS):
        return True
    if payload.get("templateIds"):
        return True
    if payload.get("syncAll") or payload.get("batch") or payload.get("batchSync"):
        return True
    if _scope_payload(payload) and not _has_single_template_ref(payload):
        return True
    return any(word in text.lower() for word in ("同步", "全部", "所有", "批量", "all templates"))


def _has_single_template_ref(payload: JsonDict) -> bool:
    return bool(payload.get("templateId") or payload.get("id") or payload.get("sourceTemplate") or _looks_like_template_detail(payload))


def _scope_payload(payload: JsonDict) -> JsonDict:
    scope = dict(payload.get("scope") or {}) if isinstance(payload.get("scope"), dict) else {}
    for key in SCOPE_KEYS:
        if payload.get(key):
            scope[key] = payload[key]
    if scope.get("documentId") and not scope.get("msgDocumentId"):
        scope["msgDocumentId"] = scope["documentId"]
    if scope.get("msgDocumentId") and not scope.get("documentId"):
        scope["documentId"] = scope["msgDocumentId"]
    if scope.get("billNo") and not scope.get("documentId"):
        scope["documentId"] = scope["billNo"]
        scope["msgDocumentId"] = scope["billNo"]
    if scope.get("businessObjectCode") and not scope.get("documentId"):
        scope["documentId"] = scope["businessObjectCode"]
        scope["msgDocumentId"] = scope["businessObjectCode"]
    if scope.get("templateTypeId") and not scope.get("msgTemplateTypeId"):
        scope["msgTemplateTypeId"] = scope["templateTypeId"]
    if scope.get("typeId") and not scope.get("msgTemplateTypeId"):
        scope["msgTemplateTypeId"] = scope["typeId"]
    if (scope.get("documentId") or scope.get("msgDocumentId")) and not scope.get("groupCode"):
        scope["groupCode"] = "businessprocess"
    return scope


def _candidate_templates(payload: JsonDict) -> List[JsonDict]:
    if _looks_like_template_detail(payload):
        return [payload]
    for key in TEMPLATE_LIST_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict) or isinstance(item, str)]
    source = payload.get("sourceTemplate")
    if isinstance(source, list):
        return [item for item in source if isinstance(item, dict) or isinstance(item, str)]
    if isinstance(source, dict) or isinstance(source, str):
        return [source]
    if isinstance(payload.get("template"), dict) or isinstance(payload.get("template"), str):
        return [payload["template"]]
    ids = payload.get("templateIds") or payload.get("ids")
    if isinstance(ids, list):
        return [{"id": item} for item in ids]
    template_id = payload.get("templateId") or payload.get("id")
    if template_id:
        return [{"id": template_id}]
    return []


def _extract_template_candidates(response: JsonDict | List[JsonDict] | None) -> List[JsonDict]:
    data = _unwrap_platform_data(response)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if not isinstance(data, dict):
        return []
    templates_vo = data.get("templatesVOList")
    if isinstance(templates_vo, list):
        result: List[JsonDict] = []
        for item in templates_vo:
            if not isinstance(item, dict):
                continue
            current = item.get("currentOrgTemps")
            if isinstance(current, list):
                result.extend([entry for entry in current if isinstance(entry, dict)])
            elif _template_id(item) or item.get("templateCode"):
                result.append(item)
        return result
    for key in ("currentOrgTemps", "templates", "templateList", "records", "list"):
        value = data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _resolve_template_detail(context: AgentContext, template_ref: JsonDict | str, payload: JsonDict, scope: JsonDict) -> JsonDict:
    detail = _coerce_template_detail(template_ref, payload)
    if detail:
        return detail
    template_id = _template_id(template_ref)
    if not template_id:
        return {}
    response = context.platform.get_template_detail(template_id, scope, dry_run=context.request.dry_run)
    return _coerce_template_detail(_unwrap_platform_data(response), payload)


def _coerce_template_detail(raw: object, payload: JsonDict) -> JsonDict:
    data = _unwrap_platform_data(raw)
    if not isinstance(data, dict):
        return {}
    if isinstance(data.get("detail"), dict):
        data = data["detail"]
    if isinstance(data.get("templateDetail"), dict):
        data = data["templateDetail"]
    if data.get("messageTempVO") or data.get("messageTemplateContentVOList"):
        detail = copy.deepcopy(data)
        detail.setdefault("messageTempVO", {})
        detail.setdefault("messageTemplateContentVOList", [])
        return detail

    contents = data.get("contents") or data.get("contentList")
    if not contents and (data.get("content") or data.get("title") or data.get("templateName")):
        channel_type = data.get("channelType")
        channels = data.get("channels")
        if not channel_type and isinstance(channels, list) and channels:
            channel_type = channels[0]
        content = str(data.get("content") or "")
        contents = [
            {
                "id": data.get("contentId") or "",
                "channelType": channel_type or "mail",
                "language": data.get("language") or payload.get("sourceLanguage") or DEFAULT_SOURCE_LANGUAGE,
                "title": data.get("title") or data.get("templateName") or "",
                "content": content,
                "raw": data.get("raw") or content,
                "contentType": data.get("contentType") or ("html" if "<" in content else "text"),
            }
        ]
    if not isinstance(contents, list):
        return {}

    message_temp = {
        "id": data.get("id") or data.get("templateId") or "",
        "templateName": data.get("templateName") or data.get("title") or "",
        "templateCode": data.get("templateCode") or data.get("template_code") or "",
        "description": data.get("description") or "",
        "languageDict": data.get("languageDict") or {},
    }
    for key in SCOPE_KEYS:
        if data.get(key):
            message_temp[key] = data[key]
        elif payload.get(key):
            message_temp[key] = payload[key]
    return {"messageTempVO": message_temp, "messageTemplateContentVOList": copy.deepcopy(contents)}


def _translate_detail(context: AgentContext, detail: JsonDict, payload: JsonDict) -> JsonDict:
    target_language = payload["targetLanguage"]
    requested_source = payload["sourceLanguage"]
    message_temp = _message_temp_vo(detail)
    contents = _message_contents(detail)
    if not contents:
        return _error_result("template.content.required", "Template content is required for translation.")

    _trace(context, "\u68c0\u67e5\u901a\u9053\u914d\u7f6e")
    source_items = _source_content_items(contents, requested_source, target_language)
    issues: List[JsonDict] = []
    for skipped in source_items["skipped"]:
        issues.append(
            {
                "code": "template.language.exists",
                "message": f"\u6a21\u677f\u3010{_template_name(detail) or _template_code(detail) or _template_id(detail)}\u3011\u7684\u3010{skipped['channelType']}\u3011\u901a\u9053\u5df2\u5b58\u5728{target_language}\u7248\u672c\uff0c\u8df3\u8fc7\u540c\u6b65\u3002",
                "severity": "warning",
                "channelType": skipped["channelType"],
            }
        )
    if source_items["fallbackLanguage"] and source_items["fallbackLanguage"] != requested_source:
        issues.append(
            {
                "code": "template.source_language.fallback",
                "message": f"Source language {requested_source} was not found for at least one channel; used available configured language instead.",
                "severity": "warning",
            }
        )
    _trace(context, "\u68c0\u67e5\u76ee\u6807\u8bed\u79cd\u662f\u5426\u5b58\u5728")
    if not source_items["items"] and source_items["skipped"]:
        return {
            "ok": True,
            "status": "skipped",
            "summary": "target language already exists for all configured channels",
            "targetLanguage": target_language,
            "sourceLanguage": requested_source,
            "issues": issues,
        }
    if not source_items["items"]:
        return _error_result("template.source_language.missing", f"No source content found for {requested_source}.")

    _trace(context, "\u4fdd\u62a4\u53d8\u91cf\u548c\u5bcc\u6587\u672c")
    _trace(context, "\u7ffb\u8bd1\u6a21\u677f")
    translated_contents: List[JsonDict] = []
    for source_item, source_language in source_items["items"]:
        translated_item = context.llm.translate_template(source_item, target_language, source_language)
        translated_item["id"] = ""
        translated_item["language"] = target_language
        translated_item["status"] = "add"
        if "raw" in translated_item:
            translated_item["raw"] = str(translated_item.get("content") or translated_item.get("raw") or "")
        translated_item["variables"] = extract_variables(str(translated_item.get("content") or ""))
        translated_contents.append(translated_item)

    source_language = source_items["sourceLanguage"]
    translated_message_temp = _translate_message_temp(context, message_temp, source_language, target_language)
    platform_payload = {
        "messageTemplateContentVOList": contents + translated_contents,
        "messageTempVO": translated_message_temp,
    }
    save_preview = _save_preview(detail, translated_message_temp, translated_contents, target_language)
    return {
        "ok": True,
        "status": "translated",
        "summary": f"template translated to {target_language}",
        "templateId": _template_id(detail),
        "templateCode": _template_code(detail),
        "templateName": _template_name(detail),
        "sourceLanguage": source_language,
        "targetLanguage": target_language,
        "translatedContents": translated_contents,
        "savePreview": save_preview,
        "platformPayload": platform_payload,
        "issues": issues,
    }


def _message_temp_vo(detail: JsonDict) -> JsonDict:
    message_temp = detail.get("messageTempVO") if isinstance(detail.get("messageTempVO"), dict) else {}
    return copy.deepcopy(message_temp)


def _message_contents(detail: JsonDict) -> List[JsonDict]:
    contents = detail.get("messageTemplateContentVOList") or detail.get("contents") or []
    return copy.deepcopy(contents) if isinstance(contents, list) else []


def _source_content_items(contents: List[JsonDict], source_language: str, target_language: str) -> JsonDict:
    normalized_source = normalize_language_code(source_language, DEFAULT_SOURCE_LANGUAGE)
    by_channel: dict[str, list[JsonDict]] = {}
    for item in contents:
        channel = _channel_type(item)
        if channel:
            by_channel.setdefault(channel, []).append(item)

    selected: list[tuple[JsonDict, str]] = []
    skipped: list[JsonDict] = []
    fallback_language = ""
    for channel, channel_items in by_channel.items():
        if any(normalize_language_code(item.get("language")) == target_language for item in channel_items):
            skipped.append({"channelType": channel})
            continue
        source_item = next((item for item in channel_items if normalize_language_code(item.get("language")) == normalized_source), None)
        item_language = normalized_source
        if source_item is None:
            source_item = channel_items[0]
            item_language = normalize_language_code(source_item.get("language"), normalized_source)
            fallback_language = fallback_language or item_language
        selected.append((source_item, item_language))
    return {
        "items": selected,
        "skipped": skipped,
        "sourceLanguage": selected[0][1] if selected else normalized_source,
        "fallbackLanguage": fallback_language,
    }


def _translate_message_temp(context: AgentContext, message_temp: JsonDict, source_language: str, target_language: str) -> JsonDict:
    result = copy.deepcopy(message_temp)
    language_dict = copy.deepcopy(result.get("languageDict") or {})
    template_names = dict(language_dict.get("templateName") or {})
    descriptions = dict(language_dict.get("description") or {})
    source_name = _language_value(template_names, source_language) or str(result.get("templateName") or "")
    source_description = _language_value(descriptions, source_language) or str(result.get("description") or "")
    translated = context.llm.translate_template({"title": source_name, "content": source_description}, target_language, source_language)
    template_names[target_language] = {"id": "", "value": str(translated.get("title") or source_name)}
    descriptions[target_language] = {"id": "", "value": str(translated.get("content") or source_description)}
    language_dict["templateName"] = template_names
    language_dict["description"] = descriptions
    result["languageDict"] = language_dict
    if source_name and not result.get("templateName"):
        result["templateName"] = source_name
    if source_description and not result.get("description"):
        result["description"] = source_description
    return result


def _language_value(entries: JsonDict, language: str) -> str:
    normalized = normalize_language_code(language)
    candidates = [normalized, language]
    for candidate in candidates:
        value = entries.get(candidate) if isinstance(entries, dict) else None
        if isinstance(value, dict):
            if value.get("value"):
                return str(value["value"])
        elif value:
            return str(value)
    return ""


def _channel_type(item: JsonDict) -> str:
    return str(item.get("channelType") or item.get("channel") or item.get("type") or "").strip()


def _save_preview(detail: JsonDict, message_temp: JsonDict, contents: List[JsonDict], target_language: str) -> List[JsonDict]:
    template_names = {}
    language_dict = message_temp.get("languageDict") if isinstance(message_temp.get("languageDict"), dict) else {}
    if isinstance(language_dict.get("templateName"), dict):
        template_names = language_dict["templateName"]
    translated_name = _language_value(template_names, target_language) or _template_name(detail)
    return [
        {
            "templateId": _template_id(detail),
            "channel": _channel_type(item),
            "language": target_language,
            "templateName": translated_name,
            "templateTitle": str(item.get("title") or ""),
            "templateContent": str(item.get("content") or ""),
        }
        for item in contents
    ]


def _template_id(template: object) -> str:
    if not isinstance(template, dict):
        return str(template or "")
    message_temp = template.get("messageTempVO") if isinstance(template.get("messageTempVO"), dict) else {}
    return str(template.get("id") or template.get("templateId") or template.get("tempId") or message_temp.get("id") or "")


def _template_code(template: object) -> str:
    if not isinstance(template, dict):
        return ""
    message_temp = template.get("messageTempVO") if isinstance(template.get("messageTempVO"), dict) else {}
    return str(template.get("templateCode") or template.get("template_code") or message_temp.get("templateCode") or "")


def _template_name(template: object) -> str:
    if not isinstance(template, dict):
        return ""
    message_temp = template.get("messageTempVO") if isinstance(template.get("messageTempVO"), dict) else {}
    return str(template.get("templateName") or template.get("title") or message_temp.get("templateName") or "")


def _looks_like_template_detail(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    return bool(
        value.get("messageTempVO")
        or value.get("messageTemplateContentVOList")
        or value.get("contents")
        or value.get("content")
        or value.get("title")
    )


def _unwrap_platform_data(value: object) -> object:
    if isinstance(value, dict) and "data" in value and value.get("data") is not None:
        return value["data"]
    return value


def _extract_domain_tree(response: JsonDict) -> List[JsonDict]:
    data = response.get("tree") if isinstance(response, dict) else None
    if isinstance(data, list):
        return data
    data = _unwrap_platform_data(response)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        children = data.get("children")
        if isinstance(children, list):
            return [item for item in children if isinstance(item, dict)]
        tree = data.get("tree")
        if isinstance(tree, list):
            return [item for item in tree if isinstance(item, dict)]
        return [data]
    return []


def _needs_business_object_resolution(payload: JsonDict) -> bool:
    return bool(payload.get("businessObject") and not (payload.get("businessObjectCode") or payload.get("documentId")))


def _business_object_resolution(context: AgentContext, payload: JsonDict) -> JsonDict:
    cached = context.scratch.get("businessObjectResolution")
    if isinstance(cached, dict) and cached.get("ok"):
        return cached
    if not payload.get("businessObject"):
        return {}
    tree = context.scratch.get("domainTree")
    if not isinstance(tree, list):
        response = context.platform.get_domain_tree(payload, dry_run=context.request.dry_run)
        tree = _extract_domain_tree(response) or MOCK_DOMAIN_TREE
        context.scratch["domainTree"] = tree
        _trace(context, "\u8c03\u7528\u9886\u57df\u6811 MCP")
    result = BusinessObjectResolver(context.llm).resolve(str(payload.get("businessObject") or ""), tree)
    if result.get("ok"):
        context.scratch["businessObjectResolution"] = result
    return result


def _trace(context: AgentContext, message: str) -> None:
    lock = context.scratch.setdefault("_templateTraceLock", Lock())
    with lock:
        trace = context.scratch.setdefault("templateExecutionTrace", [])
        if message and message not in trace:
            trace.append(message)


def _sync_limits(payload: JsonDict) -> tuple[int, int]:
    batch_size = _coerce_positive_int(
        payload.get("batchSize")
        or payload.get("batch_size")
        or payload.get("templateBatchSize")
        or DEFAULT_TEMPLATE_SYNC_BATCH_SIZE,
        DEFAULT_TEMPLATE_SYNC_BATCH_SIZE,
    )
    max_concurrency = _coerce_positive_int(
        payload.get("maxConcurrency")
        or payload.get("max_concurrency")
        or payload.get("parallelism")
        or DEFAULT_TEMPLATE_SYNC_MAX_CONCURRENCY,
        DEFAULT_TEMPLATE_SYNC_MAX_CONCURRENCY,
    )
    return batch_size, min(max_concurrency, batch_size)


def _coerce_positive_int(value: object, default: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    return result if result > 0 else default


def _chunked(items: List[JsonDict], size: int) -> List[List[JsonDict]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _sync_template_entry(context: AgentContext, template_ref: JsonDict | str, payload: JsonDict, scope: JsonDict) -> JsonDict:
    try:
        detail = _resolve_template_detail(context, template_ref, payload, scope)
        template_id = _template_id(detail) or _template_id(template_ref)
        template_code = _template_code(detail) or _template_code(template_ref)
        template_name = _template_name(detail) or _template_name(template_ref)
        if not detail:
            return {
                "kind": "failed",
                "failed": {"templateId": template_id, "templateCode": template_code, "templateName": template_name, "reason": "detail missing"},
                "issues": [
                    {
                        "code": "template.detail.required",
                        "message": f"Template {template_code or template_id or template_name or '<unknown>'} detail is required.",
                        "severity": "error",
                    }
                ],
            }

        translated = _translate_detail(context, detail, payload)
        issues = list(translated.get("issues") or [])
        if translated.get("status") == "skipped":
            return {
                "kind": "skipped",
                "skipped": {
                    "templateId": template_id,
                    "templateCode": template_code,
                    "templateName": template_name,
                    "targetLanguage": payload["targetLanguage"],
                    "reason": "already exists",
                },
                "issues": issues,
            }
        if not translated.get("ok", True):
            return {
                "kind": "failed",
                "failed": {
                    "templateId": template_id,
                    "templateCode": template_code,
                    "templateName": template_name,
                    "reason": translated.get("summary") or "translation failed",
                },
                "issues": issues,
            }

        _trace(context, "\u4fdd\u5b58\u76ee\u6807\u8bed\u79cd\u6a21\u677f")
        save_response = context.platform.save_template(translated["platformPayload"], dry_run=context.request.dry_run)
        return {
            "kind": "translated",
            "translated": {
                "templateId": template_id,
                "templateCode": template_code,
                "templateName": template_name,
                "sourceLanguage": translated["sourceLanguage"],
                "targetLanguage": translated["targetLanguage"],
                "translatedContentCount": len(translated.get("translatedContents") or []),
                "savePreview": translated.get("savePreview") or [],
                "platformPayload": translated["platformPayload"],
                "saveResponse": save_response,
            },
            "issues": issues,
        }
    except Exception as exc:  # pragma: no cover - defensive guard for concurrent sync
        template_id = _template_id(template_ref)
        template_code = _template_code(template_ref)
        template_name = _template_name(template_ref)
        return {
            "kind": "failed",
            "failed": {
                "templateId": template_id,
                "templateCode": template_code,
                "templateName": template_name,
                "reason": str(exc),
            },
            "issues": [
                {
                    "code": "template.sync.failed",
                    "message": str(exc),
                    "severity": "error",
                }
            ],
        }


def _detect_business_object(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    patterns = (
        r"\u5c06(.+?)\u4e0b\u7684.*?\u6a21\u677f.*?\u540c\u6b65",
        r"(.+?)\u4e0b\u7684.*?\u6a21\u677f.*?\u540c\u6b65",
        r"\u540c\u6b65(.+?)\u4e0b\u7684.*?\u6a21\u677f",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return _clean_business_object_name(match.group(1))
    return ""


def _clean_business_object_name(value: str) -> str:
    value = str(value or "").strip()
    for prefix in ("\u5c06", "\u628a", "\u8bf7", "\u5e2e\u6211"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
    return value.strip()


def _error_result(code: str, message: str, extra: JsonDict | None = None) -> JsonDict:
    result: JsonDict = {
        "ok": False,
        "summary": message,
        "issues": [{"code": code, "message": message, "severity": "error"}],
    }
    if extra:
        result.update(extra)
    return result


def extract_variables(text: str) -> List[str]:
    variables = re.findall(r"\{\{\s*([a-zA-Z0-9_.-]+)\s*\}\}", text)
    variables.extend(re.findall(r"\$\{\s*([a-zA-Z0-9_.-]+)\s*\}", text))
    variables.extend(re.findall(r"#\{\s*([a-zA-Z0-9_.-]+)\s*\}", text))
    variables.extend(re.findall(r"\[(?:md|UIMD|OBJ)#([^\]]+)\]", text))
    return list(dict.fromkeys(variables))
