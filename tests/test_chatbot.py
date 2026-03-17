import unittest
from unittest.mock import patch, MagicMock
from src.chatbot import build_messages, ask


class TestBuildMessages(unittest.TestCase):

    def test_system_message_first(self):
        msgs = build_messages("What is RAG?", "some context")
        self.assertEqual(msgs[0]["role"], "system")

    def test_includes_question(self):
        msgs = build_messages("What is RAG?", "some context")
        last = msgs[-1]["content"]
        self.assertIn("What is RAG?", last)

    def test_includes_context(self):
        msgs = build_messages("question", "my context")
        last = msgs[-1]["content"]
        self.assertIn("my context", last)

    def test_history_appended(self):
        history = [{"user": "hi", "assistant": "hello"}]
        msgs = build_messages("follow up", "ctx", history)
        roles = [m["role"] for m in msgs]
        self.assertIn("user", roles)
        self.assertIn("assistant", roles)

    def test_empty_history(self):
        msgs = build_messages("q", "ctx", [])
        self.assertEqual(len(msgs), 2)

    def test_history_trimmed_to_max(self):
        history = [{"user": f"q{i}", "assistant": f"a{i}"} for i in range(20)]
        msgs = build_messages("final", "ctx", history)
        self.assertLessEqual(len(msgs), 25)


class TestAsk(unittest.TestCase):

    @patch("src.chatbot.search")
    @patch("src.chatbot.get_cached", return_value=None)
    def test_returns_no_results_message(self, mock_cache, mock_search):
        mock_search.return_value = []
        result = ask("anything", [], history=[])
        self.assertIn("couldn't find", result)

    @patch("src.chatbot.openai.ChatCompletion.create")
    @patch("src.chatbot.filter_results")
    @patch("src.chatbot.search")
    @patch("src.chatbot.get_cached", return_value=None)
    def test_returns_answer(self, mock_cache, mock_search, mock_filter, mock_openai):
        mock_search.return_value = [{"content": "info", "score": 0.9, "metadata": {}}]
        mock_filter.return_value = [{"content": "info", "score": 0.9, "metadata": {}}]
        mock_openai.return_value = {
            "choices": [{"message": {"content": "The answer is 42"}}],
            "usage": {"total_tokens": 100}
        }
        result = ask("What is the answer?", [], history=[])
        self.assertEqual(result, "The answer is 42")


if __name__ == "__main__":
    unittest.main()
