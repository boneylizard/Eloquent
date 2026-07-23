from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from review_github_issue import (
    ReadOnlyCodebase,
    ReviewerError,
    choose_loaded_model,
    parse_issue_reference,
    redact_sensitive_text,
)


class IssueReferenceTests(unittest.TestCase):
    def test_parses_full_issue_url(self):
        reference = parse_issue_reference(
            "https://github.com/boneylizard/Eloquent/issues/4"
        )
        self.assertEqual(reference.owner, "boneylizard")
        self.assertEqual(reference.repository, "Eloquent")
        self.assertEqual(reference.number, 4)

    def test_parses_bare_issue_number_for_eloquent(self):
        reference = parse_issue_reference("3")
        self.assertEqual(reference.web_url, "https://github.com/boneylizard/Eloquent/issues/3")


class RedactionTests(unittest.TestCase):
    def test_redacts_common_credentials_and_windows_username(self):
        source = (
            "Authorization: Bearer secret-token\n"
            "api_key=abcdef123456\n"
            "C:\\Users\\Bernard\\AppData\\Local\\Mirid"
        )
        redacted = redact_sensitive_text(source)
        self.assertNotIn("secret-token", redacted)
        self.assertNotIn("abcdef123456", redacted)
        self.assertNotIn("Bernard", redacted)
        self.assertIn("C:\\Users\\<user>", redacted)


class ModelSelectionTests(unittest.TestCase):
    def setUp(self):
        self.response = {
            "models": [
                {
                    "type": "llm",
                    "key": "publisher/model",
                    "display_name": "Model",
                    "capabilities": {"trained_for_tool_use": True},
                    "loaded_instances": [
                        {
                            "id": "publisher/model@q4",
                            "config": {"context_length": 8192},
                        }
                    ],
                }
            ]
        }

    def test_selects_only_loaded_model(self):
        model_id, details = choose_loaded_model(self.response)
        self.assertEqual(model_id, "publisher/model@q4")
        self.assertTrue(details["trained_for_tool_use"])

    def test_requested_model_must_be_loaded(self):
        with self.assertRaises(ReviewerError):
            choose_loaded_model(self.response, "another/model")


class ReadOnlyCodebaseTests(unittest.TestCase):
    def test_searches_and_reads_source_but_blocks_private_paths(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "frontend" / "src" / "Example.jsx"
            source.parent.mkdir(parents=True)
            source.write_text("const message = 'socket failed';\n", encoding="utf-8")
            private = root / "personal" / "notes.txt"
            private.parent.mkdir()
            private.write_text("socket failed with private data\n", encoding="utf-8")
            env_file = root / ".env"
            env_file.write_text("API_KEY=secret\n", encoding="utf-8")

            codebase = ReadOnlyCodebase(root)
            search = codebase.search_code("socket failed")
            self.assertIn("frontend/src/Example.jsx", search)
            self.assertNotIn("personal/notes.txt", search)
            read = codebase.read_file("frontend/src/Example.jsx", 1, 1)
            self.assertIn("socket failed", read)
            with self.assertRaises(ReviewerError):
                codebase.read_file("personal/notes.txt")
            with self.assertRaises(ReviewerError):
                codebase.read_file(".env")


if __name__ == "__main__":
    unittest.main()
