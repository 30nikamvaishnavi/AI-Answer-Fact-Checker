import unittest

import pandas as pd

from src.metrics import VERDICTS, calculate_summary, error_reviews


def _reviews() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"question_id": "q1", "language": "en", "verdict": "Correct", "reviewed_at": "2026-01-01"},
            {"question_id": "q2", "language": "en", "verdict": "Wrong", "reviewed_at": "2026-01-01"},
            {"question_id": "q3", "language": "hi", "verdict": "Outdated", "reviewed_at": "2026-01-01"},
            {"question_id": "q4", "language": "hi", "verdict": "Correct", "reviewed_at": "2026-01-01"},
        ]
    )


class SummaryTests(unittest.TestCase):
    def test_summary_has_verdict_shares_overall_and_by_language(self) -> None:
        summary = calculate_summary(_reviews())
        overall = summary[summary["scope"] == "Overall"].set_index("verdict")
        english = summary[summary["language"] == "en"].set_index("verdict")

        self.assertEqual(len(overall), len(VERDICTS))
        self.assertEqual(overall.loc["Correct", "count"], 2)
        self.assertEqual(overall.loc["Correct", "share_percent"], 50.0)
        self.assertEqual(english.loc["Wrong", "share_percent"], 50.0)
        self.assertEqual(english["share_percent"].sum(), 100.0)

    def test_summary_uses_latest_review_for_each_question(self) -> None:
        reviews = pd.concat(
            [
                _reviews(),
                pd.DataFrame(
                    [
                        {
                            "question_id": "q1",
                            "language": "en",
                            "verdict": "Wrong",
                            "reviewed_at": "2026-02-01",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

        summary = calculate_summary(reviews)
        overall = summary[summary["scope"] == "Overall"].set_index("verdict")

        self.assertEqual(overall["n_questions"].iloc[0], 4)
        self.assertEqual(overall.loc["Correct", "count"], 1)
        self.assertEqual(overall.loc["Wrong", "count"], 2)

    def test_problem_table_contains_only_latest_wrong_or_outdated(self) -> None:
        reviews = _reviews()
        reviews = pd.concat(
            [
                reviews,
                pd.DataFrame(
                    [
                        {
                            "question_id": "q2",
                            "language": "en",
                            "verdict": "Correct",
                            "reviewed_at": "2026-02-01",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

        problems = error_reviews(reviews)

        self.assertEqual(set(problems["question_id"]), {"q3"})

    def test_empty_reviews_return_empty_summary(self) -> None:
        summary = calculate_summary(pd.DataFrame(columns=["question_id", "language", "verdict"]))

        self.assertTrue(summary.empty)
        self.assertIn("share_percent", summary.columns)


if __name__ == "__main__":
    unittest.main()
