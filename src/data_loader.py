"""CSV loading, validation, and review persistence helpers."""

from pathlib import Path

import pandas as pd


QUESTION_COLUMNS = ("question_id", "language", "category", "question")
ANSWER_COLUMNS = ("question_id", "model_answer")
REVIEW_COLUMNS = (
    "question_id",
    "language",
    "category",
    "question",
    "model_answer",
    "verdict",
    "source_url",
    "note",
    "reviewer",
    "reviewed_at",
)


def _validate_columns(frame: pd.DataFrame, required: tuple[str, ...], source: str) -> pd.DataFrame:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")
    return frame


def load_questions(path: str | Path) -> pd.DataFrame:
    questions = _validate_columns(pd.read_csv(path), QUESTION_COLUMNS, str(path))
    if questions["question_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate question_id values.")
    invalid_languages = sorted(set(questions["language"].dropna()) - {"en", "hi"})
    if invalid_languages:
        raise ValueError(f"{path} contains unsupported languages: {', '.join(invalid_languages)}")
    return questions


def load_answers(path: str | Path) -> pd.DataFrame:
    answers = _validate_columns(pd.read_csv(path), ANSWER_COLUMNS, str(path))
    return answers.drop_duplicates("question_id", keep="last")


def validate_reviews(frame: pd.DataFrame, source: str = "Reviews") -> pd.DataFrame:
    return _validate_columns(frame, REVIEW_COLUMNS, source)


def load_reviews(path: str | Path) -> pd.DataFrame:
    destination = Path(path)
    if not destination.exists():
        return pd.DataFrame(columns=REVIEW_COLUMNS)
    return validate_reviews(pd.read_csv(destination), str(destination))


def append_review(row: dict, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    new_review = validate_reviews(pd.DataFrame([row]), "New review")
    if destination.exists():
        existing = load_reviews(destination)
        new_review = pd.concat([existing, new_review], ignore_index=True)
    new_review.to_csv(destination, index=False)
