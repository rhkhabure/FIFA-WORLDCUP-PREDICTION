import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import get_env, load_env_file


class ConfigTests(unittest.TestCase):
    def test_load_env_file_parses_values_without_overwriting_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                "# comment\n"
                "export CONFIG_TEST_QUOTED=\"quoted value\"\n"
                "CONFIG_TEST_PLAIN=plain value\n"
                "CONFIG_TEST_EXISTING=file value\n",
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {"CONFIG_TEST_EXISTING": "environment value"},
                clear=False,
            ):
                os.environ.pop("CONFIG_TEST_QUOTED", None)
                os.environ.pop("CONFIG_TEST_PLAIN", None)
                load_env_file(env_path)

                self.assertEqual(get_env("CONFIG_TEST_QUOTED"), "quoted value")
                self.assertEqual(get_env("CONFIG_TEST_PLAIN"), "plain value")
                self.assertEqual(
                    get_env("CONFIG_TEST_EXISTING"), "environment value"
                )

    def test_get_env_stringifies_non_string_defaults(self):
        self.assertEqual(get_env("CONFIG_TEST_MISSING", Path("fallback")), "fallback")


if __name__ == "__main__":
    unittest.main()
