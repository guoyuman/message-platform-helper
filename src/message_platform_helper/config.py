"""Runtime configuration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    data_dir_explicit: bool = False
    config_dir: Path | None = None
    redis_url: str = ""
    llm_provider: str = "offline_rule_based"
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "gpt-4.1-mini"
    llm_temperature: float = 0.2
    llm_response_format: str = "json_object"
    platform_base_url: str = ""
    business_agent_url: str = ""
    platform_headers: Dict[str, str] | None = None
    rag_database_url: str = ""


def load_settings() -> Settings:
    _load_dotenv()
    data_dir_raw = os.environ.get("MESSAGE_HELPER_DATA_DIR", "")
    data_dir = Path(data_dir_raw) if data_dir_raw else default_data_dir()
    config_dir_raw = os.environ.get("MESSAGE_HELPER_CONFIG_DIR", "")
    config_dir = Path(config_dir_raw) if config_dir_raw else default_config_dir()
    model_config = _load_default_model_config(config_dir)
    headers_raw = os.environ.get("MESSAGE_HELPER_PLATFORM_HEADERS_JSON", "")
    headers: Dict[str, str] = {}
    if headers_raw:
        parsed = json.loads(headers_raw)
        headers = {str(key): str(value) for key, value in parsed.items()}
    llm_base_url = (
        os.environ.get("MESSAGE_HELPER_LLM_BASE_URL")
        or _env_from_config(model_config, "base_url_env", "baseUrlEnv")
        or os.environ.get("OPENAI_BASE_URL")
        or _string_value(model_config, "base_url", "baseUrl", "api_base", "apiBase")
        or ""
    ).rstrip("/")
    llm_api_key = (
        os.environ.get("MESSAGE_HELPER_LLM_API_KEY")
        or _env_from_config(model_config, "api_key_env", "apiKeyEnv")
        or os.environ.get("OPENAI_API_KEY")
        or _string_value(model_config, "api_key", "apiKey", "key")
        or ""
    )
    llm_model = (
        os.environ.get("MESSAGE_HELPER_LLM_MODEL")
        or _env_from_config(model_config, "model_env", "modelEnv")
        or os.environ.get("OPENAI_MODEL")
        or _string_value(model_config, "model")
        or "gpt-4.1-mini"
    )
    llm_provider = os.environ.get("MESSAGE_HELPER_LLM_PROVIDER") or _string_value(model_config, "provider")
    if not llm_provider:
        llm_provider = "openai_compatible" if llm_base_url and llm_api_key else "offline_rule_based"
    return Settings(
        data_dir=data_dir,
        data_dir_explicit=bool(data_dir_raw),
        config_dir=config_dir,
        redis_url=os.environ.get("MESSAGE_HELPER_REDIS_URL", ""),
        llm_provider=llm_provider,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        llm_temperature=_float_value(os.environ.get("MESSAGE_HELPER_LLM_TEMPERATURE"), model_config.get("temperature"), default=0.2),
        llm_response_format=os.environ.get("MESSAGE_HELPER_LLM_RESPONSE_FORMAT")
        or _string_value(model_config, "response_format", "responseFormat")
        or "json_object",
        platform_base_url=os.environ.get("MESSAGE_HELPER_PLATFORM_BASE_URL", "").rstrip("/"),
        business_agent_url=os.environ.get("MESSAGE_HELPER_BUSINESS_AGENT_URL", "").rstrip("/"),
        platform_headers=headers,
        rag_database_url=os.environ.get("MESSAGE_HELPER_RAG_DATABASE_URL", ""),
    )


def default_data_dir() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "data"


def _load_dotenv(path: Path | None = None) -> None:
    env_path = path or Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = _dotenv_value(value)


def _dotenv_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


@dataclass(frozen=True)
class ConfigCatalog:
    config_dir: Path
    prompts: Dict[str, str]
    models: JsonDict
    policies: JsonDict
    workflows: JsonDict
    agents: JsonDict
    tools: JsonDict


def default_config_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "config"


def load_platform_config(config_dir: Path | None = None) -> ConfigCatalog:
    root = config_dir or default_config_dir()
    return ConfigCatalog(
        config_dir=root,
        prompts=_load_prompts(root / "prompts"),
        models=_load_json_like(root / "models.yaml"),
        policies=_load_json_like(root / "policies.yaml"),
        workflows=_load_json_like(root / "workflows.yaml"),
        agents=_load_json_like(root / "agents.yaml"),
        tools=_load_json_like(root / "tools.yaml"),
    )


def load_rag_config(config_dir: Path | None = None) -> JsonDict:
    """Load the JSON-compatible RAG config kept in ``config/rag.yaml``."""
    root = config_dir or default_config_dir()
    return _load_json_like(root / "rag.yaml")


def _load_prompts(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    prompts: Dict[str, str] = {}
    for item in sorted(path.glob("*.txt")):
        prompts[item.stem] = item.read_text(encoding="utf-8")
    return prompts


def _load_json_like(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    return dict(json.loads(text))


def _load_default_model_config(config_dir: Path) -> JsonDict:
    models = _load_json_like(config_dir / "models.yaml")
    default = models.get("default")
    return dict(default) if isinstance(default, dict) else {}


def _string_value(data: JsonDict, *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if value is not None:
            return str(value)
    return ""


def _env_from_config(data: JsonDict, *keys: str) -> str:
    env_name = _string_value(data, *keys)
    return os.environ.get(env_name, "") if env_name else ""


def _float_value(*values: Any, default: float) -> float:
    for value in values:
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return default
