"""Streamlit interface for reviewing AI-generated factual answers."""

import csv
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import streamlit as st

try:
    from .data_loader import append_review, load_answers, load_questions, load_reviews
    from .metrics import (
        VERDICTS,
        calculate_summary,
        error_reviews,
        false_premise_reviews,
        latest_reviews,
    )
except ImportError:
    from data_loader import append_review, load_answers, load_questions, load_reviews
    from metrics import VERDICTS, calculate_summary, error_reviews, false_premise_reviews, latest_reviews


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = PROJECT_ROOT / "data" / "questions.csv"
ANSWERS_PATH = PROJECT_ROOT / "data" / "answers.csv"
REVIEWS_PATH = PROJECT_ROOT / "reviews" / "reviews.csv"
VERDICT_OPTIONS = ("— Select a verdict —", *VERDICTS)

st.set_page_config(page_title="AI Answer Fact Checker", page_icon="✅", layout="wide")


def _load_review_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    try:
        questions = load_questions(QUESTIONS_PATH)
        answers = (
            load_answers(ANSWERS_PATH)
            if ANSWERS_PATH.exists()
            else pd.DataFrame(columns=("question_id", "model_answer"))
        )
        reviews = load_reviews(REVIEWS_PATH)
        return questions, answers, reviews
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.error(f"Could not load project data: {error}")
        return None


def _valid_source_url(value: str) -> bool:
    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _validate_question_text(value: str) -> str:
    if not value.strip():
        raise ValueError("Enter question text before adding it.")
    if any(line_break in value for line_break in ("\r", "\n", "\u2028", "\u2029")):
        raise ValueError("Question text must not contain a line break.")
    question_text = value.strip()
    if len(question_text) < 10:
        raise ValueError("Question text must be at least 10 characters long.")
    if not question_text.endswith(("?", "？")):
        raise ValueError("End the question with a question mark (?).")
    return question_text


def _add_question_page(questions: pd.DataFrame) -> None:
    st.title("Add a question")
    language = st.selectbox("Language", ("en", "hi"))
    categories = sorted(questions["category"].dropna().astype(str).unique().tolist())
    category_choice = st.selectbox(
        "Category",
        [*categories, None],
        format_func=lambda category: category if category is not None else "Add a new category...",
    )
    category = (
        st.text_input("New category")
        if category_choice is None
        else category_choice
    )
    question_text = st.text_area("Question")

    if not st.button("Add question", type="primary"):
        return
    try:
        question_text = _validate_question_text(question_text)
    except ValueError as error:
        st.error(str(error))
        return
    if questions["question"].eq(question_text).any():
        st.error("That exact question already exists.")
        return
    if not category or not category.strip():
        st.error("Choose or enter a category before adding the question.")
        return

    question_numbers = [
        int(question_id[1:])
        for question_id in questions["question_id"].astype(str)
        if question_id.startswith("q") and question_id[1:].isdigit()
    ]
    next_question_id = f"q{max(question_numbers, default=0) + 1}"
    new_question = {
        "question_id": next_question_id,
        "language": language,
        "category": category.strip(),
        "question": question_text,
    }

    try:
        if QUESTIONS_PATH.stat().st_size:
            with QUESTIONS_PATH.open("rb") as existing_file:
                existing_file.seek(-1, 2)
                if existing_file.read(1) not in (b"\n", b"\r"):
                    with QUESTIONS_PATH.open("a", encoding="utf-8", newline="") as output:
                        output.write("\n")
        with QUESTIONS_PATH.open("a", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=questions.columns)
            writer.writerow(new_question)
    except OSError as error:
        st.error(f"Could not add the question: {error}")
        return

    st.success(
        f"Added {next_question_id}. Run `python -m src.generate_answers` to generate its answer; "
        "that command skips questions that already have answers."
    )


def _review_page(questions: pd.DataFrame, answers: pd.DataFrame, reviews: pd.DataFrame) -> None:
    st.title("Review model answers")
    if answers.empty:
        st.info("Answers have not been generated yet. Run `python -m src.generate_answers` first.")
        st.subheader("Questions")
        st.dataframe(
            questions[["question_id", "language", "category", "question"]],
            hide_index=True,
            width="stretch",
            column_config={
                "question": st.column_config.TextColumn("question", width="large"),
            },
        )
        return

    answer_by_id = dict(
        zip(answers["question_id"].astype(str), answers["model_answer"].astype(str))
    )
    questions_with_answers = questions[
        questions["question_id"].astype(str).isin(set(answers["question_id"].astype(str)))
    ].copy()
    if questions_with_answers.empty:
        st.warning("No generated answers match the questions. Run the answer-generation script.")
        return

    latest = latest_reviews(reviews)
    reviewed_ids = set(latest["question_id"].astype(str))
    all_question_ids = questions_with_answers["question_id"].astype(str).tolist()
    only_unreviewed = st.sidebar.checkbox("Only unreviewed questions")
    candidate_ids = (
        [question_id for question_id in all_question_ids if question_id not in reviewed_ids]
        if only_unreviewed
        else all_question_ids
    )
    if not candidate_ids:
        st.success("All questions with answers have been reviewed.")
        return

    current_id = st.session_state.get("review_question_id")
    if current_id not in candidate_ids:
        current_id = candidate_ids[0]
    candidate_positions = [all_question_ids.index(question_id) + 1 for question_id in candidate_ids]
    jump_position = st.session_state.get("review_question_number")
    if jump_position in candidate_positions:
        current_id = all_question_ids[jump_position - 1]
    current_position = all_question_ids.index(current_id) + 1

    previous_column, next_column, jump_column = st.columns([1, 1, 3])
    with previous_column:
        if st.button("Previous", disabled=candidate_ids.index(current_id) == 0):
            current_id = candidate_ids[candidate_ids.index(current_id) - 1]
    with next_column:
        if st.button("Next", disabled=candidate_ids.index(current_id) == len(candidate_ids) - 1):
            current_id = candidate_ids[candidate_ids.index(current_id) + 1]

    current_position = all_question_ids.index(current_id) + 1
    st.session_state["review_question_number"] = current_position
    with jump_column:
        selected_position = st.selectbox(
            "Jump to question number",
            options=candidate_positions,
            index=candidate_positions.index(current_position),
            key="review_question_number",
        )
    current_id = all_question_ids[selected_position - 1]
    st.session_state["review_question_id"] = current_id
    question = questions_with_answers.iloc[selected_position - 1]
    question_id = str(question["question_id"])
    model_answer = str(answer_by_id[question_id])

    st.caption(
        f"Question {selected_position} of {len(all_question_ids)} - "
        f"{len(reviewed_ids.intersection(all_question_ids))} reviewed"
    )
    st.markdown(f"**Language:** {question['language']}  ·  **Category:** {question['category']}")
    if str(question["category"]) == "time_sensitive":
        st.info("Check this on the day and write today's date in the notes.")
    question_column, answer_column = st.columns(2)
    with question_column:
        st.subheader("Question")
        st.markdown(str(question["question"]))
    with answer_column:
        st.subheader("Model answer")
        st.markdown(model_answer)

    saved_review = latest[latest["question_id"].astype(str) == question_id]
    previous_review = saved_review.iloc[-1] if not saved_review.empty else None
    previous_verdict = (
        str(previous_review["verdict"])
        if previous_review is not None and str(previous_review["verdict"]) in VERDICTS
        else VERDICT_OPTIONS[0]
    )
    verdict_index = VERDICT_OPTIONS.index(previous_verdict)

    with st.form(f"review_{question_id}"):
        verdict = st.selectbox("Verdict", VERDICT_OPTIONS, index=verdict_index)
        reviewer = st.text_input(
            "Reviewer name (optional)",
            value=str(previous_review["reviewer"])
            if previous_review is not None
            else "",
        )
        source_url = st.text_input(
            "Source URL",
            value=str(previous_review["source_url"])
            if previous_review is not None
            else "",
        )
        note = st.text_area(
            "Notes",
            value=str(previous_review["note"]) if previous_review is not None else "",
            max_chars=500,
        )
        submitted = st.form_submit_button("Save review", type="primary")

    if submitted:
        if verdict == VERDICT_OPTIONS[0]:
            st.error("Choose a verdict before saving.")
            return
        if verdict != "Cannot Verify" and not _valid_source_url(source_url):
            st.error("Provide a valid http:// or https:// source URL for this verdict.")
            return
        if source_url.strip() and not _valid_source_url(source_url):
            st.error("The source URL must start with http:// or https://.")
            return
        review = {
            "question_id": question_id,
            "language": question["language"],
            "category": question["category"],
            "question": question["question"],
            "model_answer": model_answer,
            "verdict": verdict,
            "source_url": source_url.strip(),
            "note": note.strip(),
            "reviewer": reviewer.strip(),
            "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        try:
            append_review(review, REVIEWS_PATH)
        except (OSError, ValueError, pd.errors.ParserError) as error:
            st.error(f"Could not save review: {error}")
            return
        st.success(f"Review saved to {REVIEWS_PATH.relative_to(PROJECT_ROOT)}.")


def _summary_page(reviews: pd.DataFrame) -> None:
    st.title("Review summary")
    summary = calculate_summary(reviews)
    if summary.empty:
        st.info("No reviews have been saved yet.")
        return

    overall = summary[summary["scope"] == "Overall"].set_index("verdict")
    correct_share = overall.loc["Correct", "share_percent"]
    st.metric("Questions reviewed", int(overall["n_questions"].iloc[0]))
    st.metric("Share rated Correct", f"{correct_share:.1f}%")

    st.subheader("Verdict share overall")
    overall_chart = overall[["share_percent"]].rename(columns={"share_percent": "Share (%)"})
    st.bar_chart(overall_chart)

    st.subheader("Verdict share by language")
    by_language = summary[summary["language"].notna()]
    language_chart = by_language.pivot(index="language", columns="verdict", values="share_percent")
    language_chart = language_chart.reindex(columns=VERDICTS, fill_value=0)
    st.bar_chart(language_chart)

    st.subheader("Verdict share by category")
    by_category = summary[summary["category"].notna()]
    category_chart = by_category.pivot(index="category", columns="verdict", values="share_percent")
    category_chart = category_chart.reindex(columns=VERDICTS, fill_value=0)
    st.bar_chart(category_chart)

    st.dataframe(
        summary.rename(
            columns={
                "scope": "Scope",
                "language": "Language",
                "category": "Category",
                "verdict": "Verdict",
                "count": "Count",
                "share_percent": "Share (%)",
                "n_questions": "Questions in scope",
            }
        ),
        hide_index=True,
        width="stretch",
    )

    st.subheader("Wrong and Outdated answers")
    problems = error_reviews(reviews)
    if problems.empty:
        st.info("There are no latest reviews labeled Wrong or Outdated.")
    else:
        columns = ["question_id", "language", "category", "question", "model_answer", "verdict", "note", "source_url"]
        st.dataframe(problems[columns], hide_index=True, width="stretch")

    st.subheader("False-premise questions")
    false_premises = false_premise_reviews(reviews)
    if false_premises.empty:
        st.info("There are no latest reviews for false_premise questions.")
    else:
        st.caption(
            "The caught indicator is a text heuristic for explicit rejection of the premise."
        )
        columns = [
            "question_id",
            "language",
            "question",
            "model_answer",
            "model_caught_false_premise",
        ]
        st.dataframe(
            false_premises[columns].rename(
                columns={"model_caught_false_premise": "Model caught false premise (heuristic)"}
            ),
            hide_index=True,
            width="stretch",
        )

    st.download_button(
        "Export summary as CSV",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name="answer_fact_checker_summary.csv",
        mime="text/csv",
    )


def main() -> None:
    loaded = _load_review_data()
    if loaded is None:
        return
    questions, answers, reviews = loaded
    page = st.sidebar.radio("Page", ("Review answers", "Summary", "Rubric", "Add question"))
    if page == "Review answers":
        _review_page(questions, answers, reviews)
    elif page == "Summary":
        _summary_page(reviews)
    elif page == "Add question":
        _add_question_page(questions)
    else:
        st.title("Review rubric")
        st.markdown(
            """
- **Correct:** Material factual claims are accurate and supported by a reliable source.
- **Partly Correct:** Useful accurate information is present, but there is a meaningful omission, imprecision, or error.
- **Outdated:** A material claim may once have been correct but is no longer current.
- **Wrong:** One or more central factual claims are false or materially misleading.
- **Cannot Verify:** Available reliable sources are insufficient to confirm the claims; a source URL is not required for this verdict.
"""
        )
        st.caption("A source URL is required for every verdict except Cannot Verify.")


if __name__ == "__main__":
    main()
