from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FrontendAssetTests(unittest.TestCase):
    def test_frontend_assets_exist(self) -> None:
        index = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        script = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
        styles = (ROOT / "web" / "styles.css").read_text(encoding="utf-8")

        self.assertIn('<form id="chatForm"', index)
        self.assertIn("/api/chat", script)
        self.assertIn(".app-shell", styles)


if __name__ == "__main__":
    unittest.main()
