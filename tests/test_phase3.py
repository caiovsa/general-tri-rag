"""Phase 3 schema + entity matching, offline."""

import unittest

from phase3_ontology_rag.retriever import MIN_NAME_LEN, _contains, match_entity
from phase3_ontology_rag.schema import CLASSES, PERSON_RELATIONS, RELATIONS, ontology_hash


class SchemaTest(unittest.TestCase):
    def test_hash_is_stable(self):
        self.assertEqual(ontology_hash(), ontology_hash())
        self.assertEqual(len(ontology_hash()), 64)

    def test_relations_cover_typed_edges(self):
        for relation in ("starred_actors", "directed_by", "written_by",
                         "has_genre", "in_language", "release_year"):
            self.assertIn(relation, RELATIONS)
            self.assertEqual(len(RELATIONS[relation]), 3)
        for label in CLASSES:
            self.assertIsInstance(label, str)
        self.assertEqual(set(PERSON_RELATIONS), {"starred_actors", "directed_by", "written_by"})


class EntityMatchTest(unittest.TestCase):
    def setUp(self):
        self.names = {
            "movies": ["Alien", "Alien Resurrection", "Kiss and Tell", "It"],
            "persons": ["Cher", "Sela Ward", "Shirley Temple"],
        }

    def test_longest_movie_wins(self):
        self.assertEqual(match_entity("who directed Alien Resurrection", self.names),
                         ("Alien Resurrection", "Movie"))

    def test_person_match(self):
        self.assertEqual(match_entity("what movies did Cher act in", self.names),
                         ("Cher", "Person"))

    def test_word_boundaries(self):
        self.assertFalse(_contains("this is a title about aliens", "Alien"))

    def test_short_names_are_ignored(self):
        self.assertLess(len("It"), MIN_NAME_LEN + 1)
        self.assertIsNone(match_entity("is it a movie", self.names))


if __name__ == "__main__":
    unittest.main()
