"""Template translation and localization sync agent."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Dict, List

from ..models import AgentResult, AgentStep, JsonDict, Tool
from ..react import AgentContext, ReActAgent


DEFAULT_SOURCE_LANGUAGE = "en_US"

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

LANGUAGE_HINTS = (
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
)

TEMPLATE_LIST_KEYS = ("templates", "templateList", "templateDetails", "sourceTemplates")


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
        }

    def plan(self, context: AgentContext) -> List[JsonDict]:
        payload = _normalize_request(context, _template_payload(context))
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
        scope = _scope_payload(payload)
        issues: List[JsonDict] = []
        translated_templates: List[JsonDict] = []
        skipped_templates: List[JsonDict] = []
        failed_templates: List[JsonDict] = []

        templates = _candidate_templates(payload)
        list_response: JsonDict | None = None
        if not templates:
            list_response = context.platform.find_templates(scope, dry_run=context.request.dry_run)
            templates = _extract_template_candidates(list_response)

        if not templates:
            return _error_result(
                "template.scope.empty",
                "No templates were found. Provide template.templates or configure platform access for the requested scope.",
                extra={"scope": scope, "templateListResponse": list_response or {}},
            )

        for template_ref in templates:
            detail = _resolve_template_detail(context, template_ref, payload, scope)
            template_id = _template_id(detail) or _template_id(template_ref)
            template_code = _template_code(detail) or _template_code(template_ref)
            template_name = _template_name(detail) or _template_name(template_ref)
            if not detail:
                failed = {"templateId": template_id, "templateCode": template_code, "templateName": template_name, "reason": "detail missing"}
                failed_templates.append(failed)
                issues.append(
                    {
                        "code": "template.detail.required",
                        "message": f"Template {template_code or template_id or template_name or '<unknown>'} detail is required.",
                        "severity": "error",
                    }
                )
                continue

            translated = _translate_detail(context, detail, payload)
            issues.extend(translated.get("issues") or [])
            if translated.get("status") == "skipped":
                skipped_templates.append(
                    {
                        "templateId": template_id,
                        "templateCode": template_code,
                        "templateName": template_name,
                        "targetLanguage": payload["targetLanguage"],
                        "reason": "already exists",
                    }
                )
                continue
            if not translated.get("ok", True):
                failed_templates.append(
                    {
                        "templateId": template_id,
                        "templateCode": template_code,
                        "templateName": template_name,
                        "reason": translated.get("summary") or "translation failed",
                    }
                )
                continue

            save_response = context.platform.save_template(translated["platformPayload"], dry_run=context.request.dry_run)
            translated_templates.append(
                {
                    "templateId": template_id,
                    "templateCode": template_code,
                    "templateName": template_name,
                    "sourceLanguage": translated["sourceLanguage"],
                    "targetLanguage": translated["targetLanguage"],
                    "translatedContentCount": len(translated.get("translatedContents") or []),
                    "platformPayload": translated["platformPayload"],
                    "saveResponse": save_response,
                }
            )

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

    if _target_language_exists(message_temp, contents, target_language):
        issue = {
            "code": "template.language.exists",
            "message": f"Template {_template_code(detail) or _template_name(detail) or _template_id(detail)} already has {target_language}; skipped.",
            "severity": "warning",
        }
        return {
            "ok": True,
            "status": "skipped",
            "summary": "target language already exists",
            "targetLanguage": target_language,
            "sourceLanguage": requested_source,
            "issues": [issue],
        }

    source_items, source_language = _source_content_items(contents, requested_source)
    issues: List[JsonDict] = []
    if source_language != requested_source:
        issues.append(
            {
                "code": "template.source_language.fallback",
                "message": f"Source language {requested_source} was not found; used {source_language} instead.",
                "severity": "warning",
            }
        )
    if not source_items:
        return _error_result("template.source_language.missing", f"No source content found for {requested_source}.")

    translated_contents: List[JsonDict] = []
    for source_item in source_items:
        translated_item = context.llm.translate_template(source_item, target_language, source_language)
        translated_item["id"] = ""
        translated_item["language"] = target_language
        translated_item["status"] = "add"
        if "raw" in translated_item:
            translated_item["raw"] = str(translated_item.get("content") or translated_item.get("raw") or "")
        translated_item["variables"] = extract_variables(str(translated_item.get("content") or ""))
        translated_contents.append(translated_item)

    translated_message_temp = _translate_message_temp(context, message_temp, source_language, target_language)
    platform_payload = {
        "messageTemplateContentVOList": contents + translated_contents,
        "messageTempVO": translated_message_temp,
    }
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
        "platformPayload": platform_payload,
        "issues": issues,
    }


def _message_temp_vo(detail: JsonDict) -> JsonDict:
    message_temp = detail.get("messageTempVO") if isinstance(detail.get("messageTempVO"), dict) else {}
    return copy.deepcopy(message_temp)


def _message_contents(detail: JsonDict) -> List[JsonDict]:
    contents = detail.get("messageTemplateContentVOList") or detail.get("contents") or []
    return copy.deepcopy(contents) if isinstance(contents, list) else []


def _target_language_exists(message_temp: JsonDict, contents: List[JsonDict], target_language: str) -> bool:
    for item in contents:
        if normalize_language_code(item.get("language")) == target_language:
            return True
    language_dict = message_temp.get("languageDict") if isinstance(message_temp.get("languageDict"), dict) else {}
    for key in ("templateName", "description"):
        entries = language_dict.get(key)
        if isinstance(entries, dict) and _language_value(entries, target_language):
            return True
    return False


def _source_content_items(contents: List[JsonDict], source_language: str) -> tuple[List[JsonDict], str]:
    normalized_source = normalize_language_code(source_language, DEFAULT_SOURCE_LANGUAGE)
    matches = [item for item in contents if normalize_language_code(item.get("language")) == normalized_source]
    if matches:
        return matches, normalized_source
    for item in contents:
        fallback = normalize_language_code(item.get("language"))
        if fallback:
            return [candidate for candidate in contents if normalize_language_code(candidate.get("language")) == fallback], fallback
    return contents, normalized_source


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
    variables.extend(re.findall(r"\[(?:md|UIMD|OBJ)#([^\]]+)\]", text))
    return list(dict.fromkeys(variables))
