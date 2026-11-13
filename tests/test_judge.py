"""Judge verdict + extracted-answer parsing, offline."""

import unittest

from shared.benchmark import extract_answer, parse_judge_verdict


class JudgeParserTest(unittest.TestCase):
    def test_incorrect_is_zero(self):
        self.assertEqual(parse_judge_verdict("INCORRECT")[0], 0)
        self.assertEqual(parse_judge_verdict("incorrect.")[0], 0)

    def test_correct_is_one(self):
        self.assertEqual(parse_judge_verdict("CORRECT")[0], 1)
        self.assertEqual(parse_judge_verdict("The answer is CORRECT.")[0], 1)

    def test_not_correct_is_zero(self):
        self.assertEqual(parse_judge_verdict("NOT CORRECT")[0], 0)

    def test_unparseable_is_none(self):
        self.assertIsNone(parse_judge_verdict("banana")[0])
        self.assertIsNone(parse_judge_verdict("")[0])


class ExtractedAnswerTest(unittest.TestCase):
    def test_yes_no_is_preserved(self):
        self.assertEqual(extract_answer("Yes. Both are American."), "Yes")
        self.assertEqual(extract_answer("No — different cities."), "No")
        self.assertEqual(extract_answer("yes, indeed"), "yes")

    def test_filler_is_stripped(self):
        self.assertEqual(extract_answer("Based on the provided context, Paris"), "Paris")
        self.assertEqual(extract_answer("The answer is 1999"), "1999")


if __name__ == "__main__":
    unittest.main()
