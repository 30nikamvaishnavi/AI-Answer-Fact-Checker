"""Verdict distribution and latest-review summary calculations."""

import pandas as pd


VERDICTS = ("Correct", "Partly Correct", "Outdated", "Wrong", "Cannot Verify")


def latest_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Keep the most recent saved review for each question."""
    if reviews.empty:
        return reviews.copy()
    ordered = reviews.copy()
    if "reviewed_at" in ordered.columns:
        ordered = ordered.sort_values("reviewed_at", kind="stable")
    return ordered.drop_duplicates("question_id", keep="last")


def calculate_summary(reviews: pd.DataFrame) -> pd.DataFrame:
    """Return verdict counts and percentage shares overall, by language, and by category."""
    columns = [
        "scope",
        "language",
        "category",
        "verdict",
        "count",
        "share_percent",
        "n_questions",
    ]
    current = latest_reviews(reviews)
    if current.empty:
        return pd.DataFrame(columns=columns)

    groups: list[tuple[str, str | None, str | None, pd.DataFrame]] = [
        ("Overall", None, None, current)
    ]
    groups.extend(
        (str(language), str(language), None, group)
        for language, group in current.groupby("language", dropna=False, sort=True)
    )
    if "category" in current.columns:
        groups.extend(
            (f"Category: {category}", None, str(category), group)
            for category, group in current.dropna(subset=["category"]).groupby(
                "category", sort=True
            )
        )
    rows = []
    for scope, language, category, group in groups:
        total = len(group)
        counts = group["verdict"].value_counts()
        for verdict in VERDICTS:
            count = int(counts.get(verdict, 0))
            rows.append(
                {
                    "scope": scope,
                    "language": language,
                    "category": category,
                    "verdict": verdict,
                    "count": count,
                    "share_percent": count * 100.0 / total,
                    "n_questions": total,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _model_caught_false_premise(answer: object) -> bool:
    """Heuristically detect explicit rejection of a false premise in an answer."""
    text = " ".join(str(answer).casefold().split())
    rejection_phrases = (
        "false premise",
        "premise is false",
        "incorrect premise",
        "the premise is incorrect",
        "incorrectly assumes",
        "there is no evidence",
        "no evidence that",
        "no such",
        "does not exist",
        "doesn't exist",
        "never existed",
        "it is a myth",
        "is a myth",
        "this is not true",
        "there was no",
        "यह धारणा गलत है",
        "गलत धारणा",
        "यह सच नहीं है",
        "ऐसा नहीं है",
        "कोई प्रमाण नहीं",
        "कोई सबूत नहीं",
        "अस्तित्व नहीं",
    )
    return any(phrase in text for phrase in rejection_phrases)


def false_premise_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Return latest false-premise reviews with a heuristic model-caught indicator."""
    current = latest_reviews(reviews)
    if current.empty or "category" not in current.columns:
        result = current.iloc[0:0].copy()
    else:
        result = current[current["category"] == "false_premise"].copy()

    answers = result["model_answer"] if "model_answer" in result.columns else pd.Series(index=result.index, dtype=object)
    result["model_caught_false_premise"] = answers.map(_model_caught_false_premise).astype(bool)
    return result


def error_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Return the latest reviews labeled Wrong or Outdated."""
    current = latest_reviews(reviews)
    return current[current["verdict"].isin(("Wrong", "Outdated"))].copy()
