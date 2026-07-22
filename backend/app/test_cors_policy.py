import os
import unittest
from unittest.mock import patch

from backend.app.cors_policy import LOCAL_ORIGIN_REGEX, cors_options


class CorsPolicyTests(unittest.TestCase):
    def test_defaults_to_local_mirid_origins(self):
        with patch.dict(os.environ, {}, clear=True):
            options = cors_options()

        self.assertEqual(options["allow_origins"], [])
        self.assertEqual(options["allow_origin_regex"], LOCAL_ORIGIN_REGEX)
        self.assertTrue(options["allow_credentials"])

    def test_accepts_explicit_additional_origins(self):
        with patch.dict(
            os.environ,
            {"MIRID_CORS_ORIGINS": "https://example.test, http://192.168.1.8:3000"},
            clear=True,
        ):
            options = cors_options()

        self.assertEqual(
            options["allow_origins"],
            ["https://example.test", "http://192.168.1.8:3000"],
        )
        self.assertTrue(options["allow_credentials"])

    def test_wildcard_is_explicit_and_disables_credentials(self):
        with patch.dict(os.environ, {"MIRID_CORS_ORIGINS": "*"}, clear=True):
            options = cors_options()

        self.assertEqual(options["allow_origins"], ["*"])
        self.assertIsNone(options["allow_origin_regex"])
        self.assertFalse(options["allow_credentials"])


if __name__ == "__main__":
    unittest.main()
