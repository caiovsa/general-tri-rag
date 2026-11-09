"""Invariants of the generated mini30 track (skipped before it is built)."""

import json
import unittest
from pathlib import Path

from scripts.metaqa_make_mini30 import OUT_CHECKSUM, OUT_DOC, OUT_EVAL, OUT_MANIFEST, verify

REPO = Path(__file__).resolve().parents[1]
METAQA = REPO / "MetaQA"


class Mini30Test(unittest.TestCase):
    def setUp(self):
        self.eval_path = METAQA / OUT_EVAL
        self.manifest_path = METAQA / OUT_MANIFEST
        if not self.eval_path.exists() or not self.manifest_path.exists():
            self.skipTest("mini30 not generated yet (run `make mini30`)")
        self.chosen = [json.loads(line) for line in self.eval_path.read_text(encoding="utf-8").splitlines()
                       if line.strip()]
        self.docs = [line.strip() for line in self.manifest_path.read_text(encoding="utf-8").splitlines()
                     if line.strip()]
        self.v1 = [json.loads(line) for line in (METAQA / "metaqa_eval_v1.jsonl")
                   .read_text(encoding="utf-8").splitlines() if line.strip()]
        self.full = [line.strip() for line in (METAQA / "corpus_manifest.txt")
                     .read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_counts_and_subsets(self):
        must_have = len({title for row in self.chosen for title in row["supporting_facts_titles"]})
        verify(self.chosen, self.docs, self.v1, self.full, must_have, 80)

    def test_checksum_file_exists(self):
        self.assertTrue((METAQA / OUT_CHECKSUM).exists())
        self.assertTrue(OUT_DOC.exists())


if __name__ == "__main__":
    unittest.main()
