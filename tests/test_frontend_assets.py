from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FrontendAssetTests(unittest.TestCase):
    def test_frontend_assets_exist(self) -> None:
        index = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        script = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
        react_entry = (ROOT / "web" / "src" / "main.jsx").read_text(encoding="utf-8")
        styles = (ROOT / "web" / "styles.css").read_text(encoding="utf-8")
        package_json = (ROOT / "web" / "package.json").read_text(encoding="utf-8")
        vite_config = (ROOT / "web" / "vite.config.js").read_text(encoding="utf-8")

        self.assertIn('<div id="root"></div>', index)
        self.assertIn("/src/main.jsx", script)
        self.assertIn("/api/chat", react_entry)
        self.assertIn("/api/memory/sessions", react_entry)
        self.assertIn("ThinkingMessage", react_entry)
        self.assertIn("thinkingStage", react_entry)
        self.assertNotIn("Knowledge Update", react_entry)
        self.assertNotIn("knowledgePath", react_entry)
        self.assertIn(".app-shell", styles)
        self.assertIn(".thinking-dots", styles)
        self.assertIn("vite", package_json)
        self.assertIn("127.0.0.1:8790", vite_config)


if __name__ == "__main__":
    unittest.main()
