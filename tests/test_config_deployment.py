from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from message_platform_helper.config import load_platform_config, load_settings
from message_platform_helper.llm import OpenAICompatibleLLMClient, RuleBasedLLMClient, build_llm, describe_llm
from message_platform_helper.workflow import build_workflow_registry_from_config


ROOT = Path(__file__).resolve().parents[1]


class ConfigDeploymentTests(unittest.TestCase):
    def test_default_platform_config_loads(self) -> None:
        catalog = load_platform_config(ROOT / "config")
        workflows = build_workflow_registry_from_config(catalog.workflows)

        self.assertIn("intent_classifier", catalog.prompts)
        self.assertEqual(catalog.models["default"]["provider"], "openai_compatible")
        self.assertTrue(catalog.policies["rag_request_types"]["query"])
        self.assertTrue(catalog.policies["rag_intents"]["knowledge_query"])
        self.assertIn("template_workflow", workflows.names())
        self.assertTrue(catalog.agents["agents"])
        self.assertTrue(catalog.tools["tools"])

    def test_deployment_assets_exist(self) -> None:
        self.assertTrue((ROOT / "Dockerfile").exists())
        self.assertTrue((ROOT / "deploy" / "kubernetes" / "message-platform-helper.yaml").exists())

    def test_settings_load_llm_from_model_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / "config"
            data_dir = root / "data"
            config_dir.mkdir()
            (config_dir / "models.yaml").write_text(
                """{
  "default": {
    "provider": "openai_compatible",
    "base_url": "https://llm.example.test/v1/",
    "api_key": "config-key",
    "model": "config-model",
    "temperature": 0.4
  }
}""",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"MESSAGE_HELPER_CONFIG_DIR": str(config_dir), "MESSAGE_HELPER_DATA_DIR": str(data_dir)}, clear=True):
                settings = load_settings()

        self.assertEqual(settings.llm_provider, "openai_compatible")
        self.assertEqual(settings.llm_base_url, "https://llm.example.test/v1")
        self.assertEqual(settings.llm_api_key, "config-key")
        self.assertEqual(settings.llm_model, "config-model")
        self.assertEqual(settings.llm_temperature, 0.4)
        self.assertIsInstance(build_llm(settings), OpenAICompatibleLLMClient)

    def test_env_llm_values_override_model_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / "config"
            data_dir = root / "data"
            config_dir.mkdir()
            (config_dir / "models.yaml").write_text(
                """{
  "default": {
    "provider": "openai_compatible",
    "base_url": "https://config.example.test/v1",
    "api_key": "config-key",
    "model": "config-model"
  }
}""",
                encoding="utf-8",
            )
            env = {
                "MESSAGE_HELPER_CONFIG_DIR": str(config_dir),
                "MESSAGE_HELPER_DATA_DIR": str(data_dir),
                "MESSAGE_HELPER_LLM_BASE_URL": "https://env.example.test/v1",
                "MESSAGE_HELPER_LLM_API_KEY": "env-key",
                "MESSAGE_HELPER_LLM_MODEL": "env-model",
            }

            with patch.dict(os.environ, env, clear=True):
                settings = load_settings()

        self.assertEqual(settings.llm_base_url, "https://env.example.test/v1")
        self.assertEqual(settings.llm_api_key, "env-key")
        self.assertEqual(settings.llm_model, "env-model")

    def test_openai_env_aliases_enable_llm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                "MESSAGE_HELPER_DATA_DIR": tmp,
                "OPENAI_BASE_URL": "https://openai-compatible.example.test/v1",
                "OPENAI_API_KEY": "alias-key",
                "OPENAI_MODEL": "alias-model",
            }
            with patch.dict(os.environ, env, clear=True):
                settings = load_settings()

        self.assertEqual(settings.llm_base_url, "https://openai-compatible.example.test/v1")
        self.assertEqual(settings.llm_api_key, "alias-key")
        self.assertEqual(settings.llm_model, "alias-model")
        self.assertIsInstance(build_llm(settings), OpenAICompatibleLLMClient)

    def test_llm_health_metadata_explains_offline_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings_env = {"MESSAGE_HELPER_DATA_DIR": tmp, "MESSAGE_HELPER_LLM_MODEL": "only-model"}
            with patch.dict(os.environ, settings_env, clear=True):
                settings = load_settings()
        llm = build_llm(settings)
        status = describe_llm(llm, settings)

        self.assertIsInstance(llm, RuleBasedLLMClient)
        self.assertEqual(status["mode"], "offline")
        self.assertEqual(status["model"], "only-model")
        self.assertFalse(status["baseUrlConfigured"])
        self.assertFalse(status["apiKeyConfigured"])


if __name__ == "__main__":
    unittest.main()
