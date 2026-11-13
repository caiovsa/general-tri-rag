"""Phase 3b paths and fact parsing, offline."""

import unittest

from phase3b_ontology_rag.extraction import parse_facts
from scripts.compare_runs import mcnemar_exact
from shared.config import dataset_config, phase3_database, phase3b_database, phase3b_dataset, settings


class Phase3bConfigTest(unittest.TestCase):
    def setUp(self):
        self._saved = (settings.DATASET, settings.TOP_K)
        settings.DATASET = "metaqa_mini30"
        settings.TOP_K = 0

    def tearDown(self):
        settings.DATASET, settings.TOP_K = self._saved

    def test_paths_and_database_are_separate(self):
        base = dataset_config(phase=3)
        schema = phase3b_dataset(3)
        self.assertEqual(schema.eval_file, base.eval_file)
        self.assertEqual(schema.chunk_size, base.chunk_size)
        self.assertEqual(schema.top_k, base.top_k)
        self.assertEqual(schema.results_file.parent.name, "phase3b")
        self.assertEqual(schema.neo4j_ontology_database, phase3b_database("metaqa_mini30"))
        self.assertNotEqual(phase3_database(schema), phase3_database(base))
        self.assertNotEqual(phase3b_database("metaqa_mini30"), phase3_database(base))

    def test_database_names_distinct(self):
        names = {phase3b_database(track) for track in
                 ("metaqa", "metaqa_mini", "metaqa_mini30")}
        self.assertEqual(len(names), 3)


class FactParsingTest(unittest.TestCase):
    def test_valid_json_with_fences(self):
        raw = '''```json
        {"actors": ["A", "a", "Movie"], "directors": ["D"],
         "writers": [], "genres": ["Drama"], "languages": ["English"],
         "year": "1999"}
        ```'''
        facts, error = parse_facts(raw, movie="Movie")
        self.assertIsNone(error)
        self.assertEqual(facts["actors"], ["A"])
        self.assertEqual(facts["directors"], ["D"])
        self.assertEqual(facts["genres"], ["Drama"])
        self.assertEqual(facts["year"], "1999")

    def test_invalid_json(self):
        facts, error = parse_facts("not json", movie="Movie")
        self.assertEqual(facts, {})
        self.assertIsNotNone(error)

    def test_year_regex_and_missing_fields(self):
        facts, error = parse_facts('{"year": "released in 2004"}', movie="Movie")
        self.assertIsNone(error)
        self.assertEqual(facts["year"], "2004")
        self.assertEqual(facts["actors"], [])

    def test_duplicate_and_self_names_are_filtered(self):
        facts, error = parse_facts(
            '{"actors": ["Cher", "cher", "Cher"], "year": null}', movie="Cher")
        self.assertIsNone(error)
        self.assertEqual(facts["actors"], [])


class McNemarTest(unittest.TestCase):
    def test_known_table(self):
        self.assertAlmostEqual(mcnemar_exact(10, 2), 0.0386, places=4)

    def test_no_discordance(self):
        self.assertEqual(mcnemar_exact(0, 0), 1.0)


if __name__ == "__main__":
    unittest.main()
