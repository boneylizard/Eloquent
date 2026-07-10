import unittest

from backend.app.openai_compat import extract_openai_stream_delta_parts


class ExtractOpenAiStreamDeltaPartsTests(unittest.TestCase):
    def test_extracts_standard_delta_content(self):
        chunk = {"choices": [{"delta": {"content": "hello"}}]}
        content, reasoning = extract_openai_stream_delta_parts(chunk)
        self.assertEqual(content, "hello")
        self.assertEqual(reasoning, "")

    def test_extracts_reasoning_and_content(self):
        chunk = {"choices": [{"delta": {"reasoning": "think", "content": "answer"}}]}
        content, reasoning = extract_openai_stream_delta_parts(chunk)
        self.assertEqual(content, "answer")
        self.assertEqual(reasoning, "think")

    def test_extracts_message_content_when_no_delta(self):
        chunk = {"choices": [{"message": {"content": "final intro json"}}]}
        content, reasoning = extract_openai_stream_delta_parts(chunk)
        self.assertEqual(content, "final intro json")
        self.assertEqual(reasoning, "")

    def test_extracts_legacy_choice_text_when_no_delta(self):
        chunk = {"choices": [{"text": "{\"headline\":\"Hi\"}"}]}
        content, reasoning = extract_openai_stream_delta_parts(chunk)
        self.assertEqual(content, "{\"headline\":\"Hi\"}")
        self.assertEqual(reasoning, "")

    def test_empty_delta_has_no_exposed_text_or_reasoning(self):
        chunk = {"choices": [{"delta": {"role": "assistant"}}]}
        content, reasoning = extract_openai_stream_delta_parts(chunk)
        self.assertEqual(content, "")
        self.assertEqual(reasoning, "")

    def test_extracts_choice_level_reasoning_when_delta_empty(self):
        chunk = {"choices": [{"delta": {}, "reasoning": "hidden stream"}]}
        content, reasoning = extract_openai_stream_delta_parts(chunk)
        self.assertEqual(content, "")
        self.assertEqual(reasoning, "hidden stream")


if __name__ == "__main__":
    unittest.main()
