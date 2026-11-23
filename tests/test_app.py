import unittest
from unittest.mock import patch

import streamlit as st

with patch.object(st, "set_page_config"):
    from src.app import _validate_question_text


class AddQuestionValidationTests(unittest.TestCase):
    def test_strips_whitespace_and_accepts_question_mark(self) -> None:
        self.assertEqual(
            _validate_question_text("  What is a valid question?  "),
            "What is a valid question?",
        )
        self.assertEqual(_validate_question_text("यह एक अच्छा सवाल है?"), "यह एक अच्छा सवाल है?")

    def test_rejects_empty_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "Enter question text"):
            _validate_question_text(" \t ")

    def test_rejects_line_breaks(self) -> None:
        for value in ("What is the answer\n to this question?", "What is the answer? \n"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "line break"):
                _validate_question_text(value)

    def test_rejects_text_shorter_than_ten_characters(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 10 characters"):
            _validate_question_text("Short?")

    def test_rejects_text_without_question_mark(self) -> None:
        with self.assertRaisesRegex(ValueError, "question mark"):
            _validate_question_text("This is not a question")


if __name__ == "__main__":
    unittest.main()
