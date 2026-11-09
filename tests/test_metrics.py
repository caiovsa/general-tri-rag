"""Spot checks for the deterministic metric helpers."""

import unittest

from shared.benchmark import (
    contains_word,
    evidence_recall,
    exact_match,
    gold_recall,
    loose_normalize,
    normalize_answer,
    relaxed_em,
    token_f1,
)


class MetricTest(unittest.TestCase):
    def test_gold_recall(self):
        self.assertEqual(gold_recall("The Matrix, Reloaded", ["The Matrix", "Reloaded"]), 1.0)
        self.assertEqual(gold_recall("The Matrix", ["The Matrix", "Reloaded"]), 0.5)
        self.assertIsNone(gold_recall("x", []))
        self.assertIsNone(gold_recall("x", None))
        self.assertEqual(gold_recall("A", ["A"]), 1.0)  # degenerate article-only item

    def test_evidence_recall(self):
        self.assertEqual(evidence_recall(["[Movie: Alien]\ntext"], ["Alien"]), 1.0)
        self.assertEqual(evidence_recall(["no tag"], ["Alien"]), 0.0)
        self.assertIsNone(evidence_recall([], []))

    def test_em_and_f1(self):
        self.assertEqual(exact_match("Yes.", "yes"), 1)
        self.assertEqual(relaxed_em("Yes, American.", "American"), 1)
        self.assertEqual(token_f1("chief of protocol", "chief of protocol"), 1.0)
        self.assertEqual(token_f1("protocol", "chief of protocol"), 0.5)

    def test_normalization(self):
        self.assertEqual(normalize_answer("The  Matrix!"), "matrix")
        self.assertEqual(loose_normalize("A Test, ok"), "a test ok")
        self.assertTrue(contains_word("the matrix reloaded", "matrix"))
        self.assertFalse(contains_word("matrixes", "matrix"))


if __name__ == "__main__":
    unittest.main()
