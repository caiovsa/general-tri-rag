"""Offline tests: no API calls, no running Qdrant/Neo4j required."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from shared.config import Dataset, dataset_config, phase3_database, settings


class DatasetConfigTest(unittest.TestCase):
    def setUp(self):
        self._saved = (settings.DATASET, settings.TOP_K,
                       settings.LOCAL_DOCS_DIR, settings.LOCAL_EVAL_FILE)
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        (root / "docs").mkdir()
        (root / "docs" / "a.txt").write_text("hello", encoding="utf-8")
        (root / "eval.jsonl").write_text('{"question":"q","answer":"a"}\n', encoding="utf-8")

    def tearDown(self):
        (settings.DATASET, settings.TOP_K,
         settings.LOCAL_DOCS_DIR, settings.LOCAL_EVAL_FILE) = self._saved
        self._tmp.cleanup()

    def _resolve(self, name):
        settings.DATASET = name
        settings.TOP_K = 0
        return dataset_config(phase=1)

    def test_tracks_have_distinct_stores(self):
        seen = {}
        for name in ("hotpot", "metaqa", "metaqa_mini", "metaqa_mini30"):
            dataset = self._resolve(name)
            seen[name] = (dataset.qdrant_collection, dataset.neo4j_database,
                          str(dataset.results_file.parent))
            self.assertNotEqual(dataset.qdrant_collection, "")
            self.assertNotEqual(dataset.neo4j_database, "")
        self.assertEqual(len({value[0] for value in seen.values()}), 4)
        self.assertEqual(len({value[1] for value in seen.values()}), 4)
        self.assertEqual(len({value[2] for value in seen.values()}), 4)

    def test_meta_protocol(self):
        for name in ("metaqa", "metaqa_mini", "metaqa_mini30"):
            dataset = self._resolve(name)
            self.assertEqual(dataset.top_k, 20)
            self.assertEqual((dataset.chunk_size, dataset.chunk_overlap), (256, 32))
            self.assertTrue(dataset.tag_movies)
            self.assertEqual(dataset.prompt_style, "shared")
            self.assertIsNotNone(dataset.doc_list)

    def test_phase3_database_is_separate(self):
        for name in ("hotpot", "metaqa", "metaqa_mini", "metaqa_mini30"):
            dataset = self._resolve(name)
            self.assertNotEqual(phase3_database(dataset), dataset.neo4j_database,
                                f"phase 3 would reuse {name}'s phase 2 database")

    def test_local_track(self):
        settings.LOCAL_DOCS_DIR = str(Path(self._tmp.name) / "docs")
        settings.LOCAL_EVAL_FILE = str(Path(self._tmp.name) / "eval.jsonl")
        dataset = self._resolve("local")
        self.assertEqual(dataset.name, "local")
        self.assertFalse(dataset.tag_movies)
        self.assertEqual(dataset.prompt_style, "shared")
        self.assertEqual(dataset.docs_dir, Path(settings.LOCAL_DOCS_DIR))

    def test_frozen_files_exist(self):
        for name in ("metaqa", "metaqa_mini"):
            dataset = self._resolve(name)
            self.assertTrue(dataset.eval_file.exists(), dataset.eval_file)
            self.assertTrue(dataset.doc_list.exists(), dataset.doc_list)


class FingerprintTest(unittest.TestCase):
    def _dataset(self, root: Path) -> Dataset:
        eval_file = root / "eval.jsonl"
        eval_file.write_text('{"question":"q","answer":"a"}\n', encoding="utf-8")
        docs = root / "docs"
        docs.mkdir()
        (docs / "a.txt").write_text("alpha", encoding="utf-8")
        return Dataset(
            name="unit", eval_file=eval_file, docs_dir=docs, doc_list=None,
            results_file=root / "results.json", detailed_file=root / "detailed.json",
            eval_results_file=root / "eval_results.json", eval_report_file=root / "report.txt",
            per_question_file=root / "per_question.jsonl",
            qdrant_collection="unit", neo4j_uri="neo4j://localhost", neo4j_database="unit",
            chunk_size=256, chunk_overlap=32, top_k=8,
        )

    def test_deterministic_and_content_sensitive(self):
        from shared.hotpot_runner import run_fingerprint
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self._dataset(root)
            first = run_fingerprint(dataset, 1)
            second = run_fingerprint(dataset, 1)
            self.assertEqual(first, second)
            (root / "docs" / "a.txt").write_text("beta", encoding="utf-8")
            self.assertNotEqual(first["corpus"]["sha256"],
                                run_fingerprint(dataset, 1)["corpus"]["sha256"])

    def test_resume_refuses_changed_config(self):
        from shared.hotpot_runner import _check_resume, run_fingerprint
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self._dataset(root)
            fingerprint = run_fingerprint(dataset, 1)
            (root / "run_meta.json").write_text(
                json.dumps({"fingerprint": fingerprint}), encoding="utf-8")
            _check_resume(root / "detailed.json", fingerprint, [{"question": "q", "answer": "a"}])
            changed = replace(dataset, top_k=20)
            with self.assertRaises(SystemExit):
                _check_resume(root / "detailed.json", run_fingerprint(changed, 1),
                              [{"question": "q", "answer": "a"}])

    def test_resume_refuses_legacy_run_without_meta(self):
        from shared.hotpot_runner import _check_resume, run_fingerprint
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self._dataset(root)
            with self.assertRaises(SystemExit):
                _check_resume(root / "detailed.json", run_fingerprint(dataset, 1),
                              [{"question": "q", "answer": "a"}])


if __name__ == "__main__":
    unittest.main()
