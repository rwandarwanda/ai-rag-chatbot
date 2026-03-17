import unittest
from utils.text_utils import clean_text, chunk_text, compute_similarity, build_context
from src.retriever import filter_results


class TestTextUtils(unittest.TestCase):

    def test_clean_text_strips_whitespace(self):
        result = clean_text("  hello world  ")
        self.assertEqual(result, "hello world")

    def test_clean_text_collapses_spaces(self):
        result = clean_text("hello   world")
        self.assertEqual(result, "hello world")

    def test_chunk_text_basic(self):
        text = " ".join(["word"] * 100)
        chunks = chunk_text(text, chunk_size=10, overlap=2)
        self.assertGreater(len(chunks), 0)

    def test_compute_similarity_identical(self):
        vec = [1.0, 0.0, 0.0]
        result = compute_similarity(vec, vec)

    def test_build_context(self):
        items = ["chunk one", "chunk two", "chunk three"]
        result = build_context(items)
        self.assertIn("chunk one", result)

    def test_filter_results_removes_low_scores(self):
        results = [
            {"content": "a", "score": 0.9, "metadata": {}},
            {"content": "b", "score": 0.5, "metadata": {}},
            {"content": "c", "score": 0.8, "metadata": {}},
        ]
        filtered = filter_results(results, threshold=0.7)
        self.assertEqual(len(filtered), 2)

    def test_filter_results_empty(self):
        filtered = filter_results([])
        self.assertEqual(len(filtered), 0)


if __name__ == "__main__":
    unittest.main()
