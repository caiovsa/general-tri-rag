"""Graph ingest guards and Phase 3 fingerprint, offline."""

import tempfile
import unittest
from pathlib import Path

from phase2_graph_rag.ingest import CountingExtractionLLM
from shared.config import Dataset
from shared.hotpot_runner import run_fingerprint


class PolicySkipTest(unittest.TestCase):
    def test_policy_error_is_skipped(self):
        class PolicyError(Exception):
            code = "invalid_prompt"

        self.assertTrue(CountingExtractionLLM._is_policy_error(
            PolicyError("flagged as potentially violating our usage policy")))

    def test_other_errors_are_not_skipped(self):
        class LengthError(Exception):
            code = "context_length_exceeded"

        self.assertFalse(CountingExtractionLLM._is_policy_error(
            LengthError("maximum context length exceeded")))

    def test_empty_response(self):
        self.assertEqual(CountingExtractionLLM._empty_response().message.content, "")


class Phase3FingerprintTest(unittest.TestCase):
    def test_graph_database_and_ontology_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_file = root / "eval.jsonl"
            eval_file.write_text('{"question":"q","answer":"a"}\n', encoding="utf-8")
            docs = root / "docs"
            docs.mkdir()
            (docs / "a.txt").write_text("alpha", encoding="utf-8")
            dataset = Dataset(
                name="metaqa", eval_file=eval_file, docs_dir=docs, doc_list=None,
                results_file=root / "r.json", detailed_file=root / "d.json",
                eval_results_file=root / "e.json", eval_report_file=root / "r.txt",
                per_question_file=root / "pq.jsonl",
                qdrant_collection="c", neo4j_uri="neo4j://localhost",
                neo4j_database="phase2-db", chunk_size=256, chunk_overlap=32, top_k=20,
                tag_movies=True, prompt_style="shared", neo4j_ontology_database="phase3-db",
            )
            fingerprint = run_fingerprint(dataset, 3)
            self.assertEqual(fingerprint["graph_database"], "phase3-db")
            self.assertEqual(len(fingerprint["ontology_sha256"]), 64)
            self.assertNotEqual(fingerprint["graph_database"], dataset.neo4j_database)


if __name__ == "__main__":
    unittest.main()
