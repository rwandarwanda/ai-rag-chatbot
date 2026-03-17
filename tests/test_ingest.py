import unittest
import os
import tempfile
import json
from src.ingest import load_json_docs, process_documents, dedup_documents


class TestLoadJsonDocs(unittest.TestCase):

    def test_valid_json_list(self):
        data = [{"content": "hello world", "source": "test.txt"}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name
        result = load_json_docs(path)
        self.assertEqual(len(result), 1)
        os.unlink(path)

    def test_raises_on_non_list(self):
        data = {"content": "not a list"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name
        with self.assertRaises(ValueError):
            load_json_docs(path)
        os.unlink(path)

    def test_raises_on_missing_content(self):
        data = [{"source": "no content here"}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name
        with self.assertRaises(ValueError):
            load_json_docs(path)
        os.unlink(path)


class TestProcessDocuments(unittest.TestCase):

    def test_creates_chunks(self):
        docs = [{"content": " ".join(["word"] * 600), "filename": "test.txt"}]
        result = process_documents(docs)
        self.assertGreater(len(result), 1)

    def test_assigns_sequential_ids(self):
        docs = [{"content": "short doc", "filename": "a.txt"}]
        result = process_documents(docs)
        self.assertEqual(result[0]["id"], 0)

    def test_metadata_source(self):
        docs = [{"content": "some text", "filename": "myfile.txt"}]
        result = process_documents(docs)
        self.assertEqual(result[0]["metadata"]["source"], "myfile.txt")


class TestDedupDocuments(unittest.TestCase):

    def test_removes_duplicates(self):
        docs = [
            {"content": "same content", "filename": "a.txt"},
            {"content": "same content", "filename": "b.txt"},
            {"content": "different content", "filename": "c.txt"},
        ]
        result = dedup_documents(docs)
        self.assertEqual(len(result), 2)

    def test_empty_list(self):
        self.assertEqual(dedup_documents([]), [])


if __name__ == "__main__":
    unittest.main()
