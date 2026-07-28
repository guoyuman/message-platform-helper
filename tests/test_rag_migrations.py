from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from alembic.config import Config
from message_platform_helper.rag.db import Base


ROOT = Path(__file__).resolve().parents[1]


class RagMigrationTests(unittest.TestCase):
    def test_alembic_config_points_to_migrations(self) -> None:
        config = Config(str(ROOT / "alembic.ini"))

        self.assertEqual(config.get_main_option("script_location"), "migrations")



if __name__ == "__main__":
    unittest.main()
