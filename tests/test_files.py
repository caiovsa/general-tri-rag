"""File loading and corpus signature checks."""

import tempfile
import unittest
from pathlib import Path

from shared.hotpot_runner import corpus_signature, sha256_file
from shared.ingest import load_documents


class FileTest(unittest.TestCase):
    def test_sha256_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.txt"
            path.write_text("abc", encoding="utf-8")
            self.assertEqual(len(sha256_file(path)), 64)
            self.assertIsNone(sha256_file(Path(tmp) / "missing.txt"))

    def test_directory_signature_changes_with_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("a", encoding="utf-8")
            dataset = type("D", (), {"doc_list": None, "docs_dir": root, "file_pattern": "*.txt"})()
            before = corpus_signature(dataset)
            (root / "a.txt").write_text("abcd", encoding="utf-8")
            after = corpus_signature(dataset)
            self.assertNotEqual(before["sha256"], after["sha256"])
            self.assertEqual(after["files"], 1)

    def test_binary_pdf_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fake.pdf"
            path.write_bytes(b"%PDF-1.4\n" + bytes(range(256)) * 4)
            with self.assertRaises(ValueError):
                load_documents(path)

    def test_plain_text_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.txt"
            path.write_text("This is a normal text document.\n", encoding="utf-8")
            documents = load_documents(path)
            self.assertEqual(len(documents), 1)


if __name__ == "__main__":
    unittest.main()
