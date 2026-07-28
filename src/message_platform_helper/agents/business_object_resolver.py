"""Business object matching against a domain tree."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Any

from ..llm import LLMClient
from ..models import JsonDict


@dataclass
class BusinessObjectResolver:
    """Resolve a business object name from the already-fetched domain tree."""

    llm: LLMClient | None = None

    def resolve(self, name: str, tree: list[JsonDict]) -> JsonDict:
        query = _normalize_name(name)
        nodes = _flatten_tree(tree)
        if not query:
            return _error("business_object.name.required", "Business object name is required.")
        if not nodes:
            return _error("business_object.tree.empty", "Domain tree is empty; cannot resolve business object.")

        exact = [node for node in nodes if _normalize_name(node["name"]) == query]
        if len(exact) == 1:
            return _resolved(exact[0], 1.0, "exact")
        if len(exact) > 1:
            return _ambiguous(exact, "Multiple business objects matched exactly.")

        scored = sorted(
            (
                {**node, "score": _semantic_score(query, _normalize_name(node["name"]))}
                for node in nodes
            ),
            key=lambda item: item["score"],
            reverse=True,
        )
        candidates = [item for item in scored if item["score"] >= 0.72][:5]
        if len(candidates) == 1:
            return _resolved(candidates[0], float(candidates[0]["score"]), "semantic")
        if len(candidates) > 1 and candidates[0]["score"] - candidates[1]["score"] <= 0.08:
            return _ambiguous(candidates, "Multiple candidate business objects require confirmation.")
        if candidates:
            return _resolved(candidates[0], float(candidates[0]["score"]), "semantic")

        llm_match = self._llm_match(name, scored[:10])
        if llm_match:
            matched = next((node for node in nodes if node["code"] == llm_match.get("businessObjectCode")), None)
            if matched:
                return _resolved(matched, float(llm_match.get("confidence") or 0.75), "llm")

        return _error(
            "business_object.not_found",
            f"Business object '{name}' was not found in the domain tree.",
            {"candidates": [_candidate(item) for item in scored[:5]]},
        )

    def _llm_match(self, name: str, candidates: list[JsonDict]) -> JsonDict:
        if self.llm is None or not candidates:
            return {}
        try:
            raw = self.llm.complete_json(
                "Resolve a business object by semantic meaning from candidate domain-tree nodes.",
                {
                    "name": name,
                    "candidates": [_candidate(item) for item in candidates],
                    "instruction": "Return businessObjectName, businessObjectCode, documentId, confidence, or ambiguousCandidates.",
                },
            )
        except Exception:
            return {}
        if isinstance(raw.get("ambiguousCandidates"), list):
            return {}
        return raw if raw.get("businessObjectCode") else {}


def _flatten_tree(tree: list[JsonDict]) -> list[JsonDict]:
    result: list[JsonDict] = []

    def visit(node: Any, path: list[str]) -> None:
        if not isinstance(node, dict):
            return
        name = str(node.get("name") or node.get("nodeName") or node.get("labelName") or node.get("title") or "").strip()
        code = str(node.get("code") or node.get("nodeCode") or node.get("labelCode") or node.get("id") or node.get("nodeId") or "").strip()
        document_id = str(
            node.get("documentId")
            or node.get("msgDocumentId")
            or node.get("billNo")
            or node.get("nodeCode")
            or node.get("code")
            or ""
        ).strip()
        children = node.get("children") if isinstance(node.get("children"), list) else []
        current_path = [*path, name] if name else path
        if name and code:
            result.append(
                {
                    "businessObjectName": name,
                    "name": name,
                    "businessObjectCode": code,
                    "code": code,
                    "documentId": document_id or code,
                    "path": current_path,
                    "childrenCount": len(children),
                }
            )
        for child in children:
            visit(child, current_path)

    for item in tree:
        visit(item, [])
    return result


def _semantic_score(query: str, candidate: str) -> float:
    if not query or not candidate:
        return 0.0
    if query in candidate or candidate in query:
        return 0.86
    q_terms = set(_terms(query))
    c_terms = set(_terms(candidate))
    overlap = len(q_terms & c_terms) / max(len(q_terms | c_terms), 1)
    ratio = difflib.SequenceMatcher(None, query, candidate).ratio()
    return max(overlap, ratio)


def _terms(value: str) -> list[str]:
    ascii_terms = re.findall(r"[a-zA-Z0-9]+", value.lower())
    cjk_chars = [char for char in value if "\u4e00" <= char <= "\u9fff"]
    return ascii_terms + cjk_chars


def _normalize_name(value: str) -> str:
    return re.sub(r"[\s_\-:：/\\]+", "", str(value or "").strip().lower())


def _resolved(node: JsonDict, score: float, strategy: str) -> JsonDict:
    return {
        "ok": True,
        "businessObjectName": node["businessObjectName"],
        "documentId": node.get("documentId") or node["businessObjectCode"],
        "businessObjectCode": node["businessObjectCode"],
        "matchStrategy": strategy,
        "score": score,
        "candidates": [_candidate(node)],
    }


def _ambiguous(candidates: list[JsonDict], message: str) -> JsonDict:
    return {
        "ok": False,
        "ambiguous": True,
        "summary": message,
        "candidates": [_candidate(item) for item in candidates[:5]],
        "issues": [{"code": "business_object.ambiguous", "message": message, "severity": "warning"}],
    }


def _candidate(node: JsonDict) -> JsonDict:
    return {
        "businessObjectName": node.get("businessObjectName") or node.get("name") or "",
        "documentId": node.get("documentId") or node.get("code") or "",
        "businessObjectCode": node.get("businessObjectCode") or node.get("code") or "",
        "path": node.get("path") or [],
        "score": node.get("score"),
    }


def _error(code: str, message: str, extra: JsonDict | None = None) -> JsonDict:
    result: JsonDict = {"ok": False, "summary": message, "issues": [{"code": code, "message": message, "severity": "error"}]}
    if extra:
        result.update(extra)
    return result
