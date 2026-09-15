"""Shared data models for the helper service."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Callable, Dict, List, Optional


JsonDict = Dict[str, Any]


def now_ts() -> float:
    return time.time()


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value


def deep_merge(base: JsonDict, patch: JsonDict) -> JsonDict:
    result = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@dataclass
class AssistantRequest:
    text: str
    session_id: str = "default"
    user_id: str = "anonymous"
    tenant_id: str = ""
    locale: str = "zh_CN"
    payload: JsonDict = field(default_factory=dict)
    dry_run: bool = True
    request_id: str = ""
    trace_id: str = ""


@dataclass
class UserProfile:
    user_id: str = "anonymous"
    tenant_id: str = ""
    locale: str = "zh_CN"
    preferred_channels: List[str] = field(default_factory=list)
    business_domains: List[str] = field(default_factory=list)
    traits: JsonDict = field(default_factory=dict)
    updated_at: float = field(default_factory=now_ts)


@dataclass
class ConversationMemory:
    session_id: str
    profile: UserProfile = field(default_factory=UserProfile)
    summary: str = ""
    facts: JsonDict = field(default_factory=dict)
    operation_history: List[JsonDict] = field(default_factory=list)
    recent_messages: List[JsonDict] = field(default_factory=list)
    updated_at: float = field(default_factory=now_ts)


@dataclass
class KnowledgeChunk:
    id: str
    title: str
    content: str
    source: str = "manual"
    tags: List[str] = field(default_factory=list)
    score: float = 0.0
    metadata: JsonDict = field(default_factory=dict)


@dataclass
class IntentResult:
    intent: str
    confidence: float
    request_type: str = ""
    domain: str = ""
    operation: str = ""
    rationale: str = ""
    labels: List[str] = field(default_factory=list)
    metadata: JsonDict = field(default_factory=dict)


@dataclass
class DecisionResult:
    intent: str
    request_type: str
    domain: str
    operation: str
    workflow: str
    need_rag: bool
    need_tools: bool
    selected_agents: List[str]
    selected_tools: List[str]
    confidence: float
    search_query: str = ""
    rationale: str = ""
    missing_slots: List[str] = field(default_factory=list)
    metadata: JsonDict = field(default_factory=dict)


@dataclass
class ReasoningDecision:
    need_retrieval: bool
    selected_agents: List[str]
    confidence: float
    rationale: str
    request_type: str = "action"
    domain: str = "implementation"
    operation: str = "execute"
    missing_slots: List[str] = field(default_factory=list)
    search_query: str = ""
    intent: str = ""
    workflow: str = ""
    need_tools: bool = False
    selected_tools: List[str] = field(default_factory=list)
    metadata: JsonDict = field(default_factory=dict)


@dataclass
class AgentStep:
    agent: str
    thought: str
    action: str
    observation: str
    status: str = "ok"
    data: JsonDict = field(default_factory=dict)


@dataclass
class AgentResult:
    agent: str
    ok: bool
    output: JsonDict = field(default_factory=dict)
    issues: List[JsonDict] = field(default_factory=list)
    next_actions: List[str] = field(default_factory=list)
    steps: List[AgentStep] = field(default_factory=list)


@dataclass
class HelperResponse:
    ok: bool
    session_id: str
    decision: ReasoningDecision
    memory: ConversationMemory
    retrieved: List[KnowledgeChunk] = field(default_factory=list)
    results: List[AgentResult] = field(default_factory=list)
    issues: List[JsonDict] = field(default_factory=list)
    next_actions: List[str] = field(default_factory=list)
    trace_id: str = ""
    request_id: str = ""
    conversation_id: str = ""
    metrics: JsonDict = field(default_factory=dict)


ToolHandler = Callable[[JsonDict], JsonDict]


@dataclass
class Tool:
    name: str
    description: str
    handler: ToolHandler
    requires_context: bool = False
    # JSON Schema for arguments, used when exporting the tool to LLM
    # function-calling protocols. Empty dict means no declared parameters.
    parameters: JsonDict = field(default_factory=dict)

    def run(self, payload: JsonDict, context: Any | None = None) -> JsonDict:
        if self.requires_context:
            return self.handler(payload, context)  # type: ignore[misc]
        return self.handler(payload)


@dataclass(frozen=True)
class AgentSpec:
    name: str
    description: str = ""
    capabilities: List[str] = field(default_factory=list)
    priority: int = 100
    enabled: bool = True


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str = ""
    category: str = "function"
    capabilities: List[str] = field(default_factory=list)
    priority: int = 100
    enabled: bool = True
    timeout_seconds: int = 30
    parameters: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class WorkflowStep:
    name: str
    agent: str
    tools: List[str] = field(default_factory=list)
    required: bool = True


@dataclass(frozen=True)
class WorkflowDefinition:
    name: str
    description: str = ""
    steps: List[WorkflowStep] = field(default_factory=list)
    enabled: bool = True


@dataclass(frozen=True)
class StreamingEvent:
    type: str
    data: JsonDict = field(default_factory=dict)
    trace_id: str = ""
    request_id: str = ""
    conversation_id: str = ""
    sequence: int = 0
    created_at: float = field(default_factory=now_ts)


@dataclass
class EmailChannelConfig:
    email: str
    config_name: str
    mail_host: str
    mail_port: int
    username: str
    password: str = ""
    smtp_ssl: bool = False
    smtp_tls: bool = True
    verify_user: str = ""
    default_status: int = 1
