import unittest
from utils.text_utils import tokenize, get_doc_hash


class TestTokenize(unittest.TestCase):

    def test_tokenize_basic(self):
        tokens = tokenize("Hello World this is a test")
        self.assertIn("hello", tokens)
        self.assertIn("world", tokens)

    def test_tokenize_filters_short(self):
        tokens = tokenize("I am a big dog")
        self.assertNotIn("i", tokens)
        self.assertNotIn("a", tokens)

    def test_get_doc_hash_returns_string(self):
        result = get_doc_hash("some document content")
        self.assertIsInstance(result, str)

    def test_get_doc_hash_consistent(self):
        h1 = get_doc_hash("same content")
        h2 = get_doc_hash("same content")
        self.assertEqual(h1, h2)


if __name__ == "__main__":
    unittest.main()
