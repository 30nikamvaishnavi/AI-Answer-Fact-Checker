"""Generate missing question answers using the Gemini API."""

import argparse
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError

try:
    from .data_loader import load_answers, load_questions
except ImportError:
    from data_loader import load_answers, load_questions


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = PROJECT_ROOT / "data" / "questions.csv"
ANSWERS_PATH = PROJECT_ROOT / "data" / "answers.csv"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
API_KEY_PLACEHOLDER = "your_gemini_api_key_here"
RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}


def _load_api_key(report_status: bool = True) -> str:
    dotenv_path = PROJECT_ROOT / ".env"
    dotenv_found = dotenv_path.is_file()
    if report_status:
        print(f".env file found: {'yes' if dotenv_found else 'no'}")
        print(f".env path: {dotenv_path.resolve()}")
    if dotenv_found:
        load_dotenv(dotenv_path)

    api_key = os.getenv("GOOGLE_API_KEY")
    configured = bool(api_key and api_key.strip() and api_key.strip() != API_KEY_PLACEHOLDER)
    if report_status:
        print(f"GOOGLE_API_KEY set to a non-placeholder value: {'yes' if configured else 'no'}")
    if not configured:
        raise RuntimeError(
            "GOOGLE_API_KEY is missing or still the placeholder. "
            "Edit .env and set GOOGLE_API_KEY before generating answers."
        )
    return api_key


def _print_available_models(client: genai.Client) -> None:
    try:
        for model in client.models.list():
            if "generateContent" not in (getattr(model, "supported_actions", None) or []):
                continue
            name = getattr(model, "name", None)
            if name:
                print(name.removeprefix("models/"))
    except APIError as error:
        raise RuntimeError(
            "Could not list models. Check API-key access and network connectivity."
        ) from error


def _is_retryable(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(error, "code", None)
    try:
        if int(status) in RETRYABLE_CODES:
            return True
    except (TypeError, ValueError):
        pass
    error_name = type(error).__name__.lower()
    return any(token in error_name for token in ("ratelimit", "resourceexhausted", "serviceunavailable", "deadlineexceeded", "internalservererror", "temporarilyunavailable"))


def _is_model_access_error(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(error, "code", None)
    try:
        if int(status) in {403, 404}:
            return True
    except (TypeError, ValueError):
        pass
    error_name = type(error).__name__.lower()
    message = str(error).lower()
    return any(token in error_name for token in ("notfound", "permissiondenied", "forbidden")) or any(
        token in message for token in ("not found", "permission", "does not exist")
    )


def generate_with_retries(
    client: genai.Client,
    model_name: str,
    question: str,
    language: str,
    max_retries: int,
    base_delay: float,
) -> str:
    language_name = "Hindi" if language == "hi" else "English"
    prompt = (
        f"Answer this factual question in {language_name}. Be concise and factual. "
        "If you are uncertain, say so instead of inventing details.\n\n"
        f"Question: {question}"
    )
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            answer = getattr(response, "text", None)
            if not answer or not answer.strip():
                raise ValueError("The Gemini API returned an empty answer.")
            return answer.strip()
        except Exception as error:
            if attempt >= max_retries or not _is_retryable(error):
                raise
            delay = base_delay * (2**attempt) + random.uniform(0, base_delay)
            print(
                f"Temporary API error ({error}); retrying in {delay:.1f}s "
                f"({attempt + 1}/{max_retries}).",
                file=sys.stderr,
            )
            time.sleep(delay)
    raise RuntimeError("Answer generation stopped without a response.")


def _save_answer(question_id: str, answer: str, path: Path) -> None:
    row = pd.DataFrame([{"question_id": question_id, "model_answer": answer}])
    if path.exists():
        existing = load_answers(path)
        row = pd.concat([existing, row], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    row.to_csv(path, index=False)


def generate_answers(
    questions_path: Path = QUESTIONS_PATH,
    answers_path: Path = ANSWERS_PATH,
    model_name: str | None = None,
    max_retries: int = 5,
    base_delay: float = 1.0,
    request_delay: float = 1.0,
) -> list[str]:
    api_key = _load_api_key()
    model_name = model_name or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL

    questions = load_questions(questions_path)
    existing = load_answers(answers_path) if answers_path.exists() else pd.DataFrame(columns=("question_id", "model_answer"))
    answered_ids = set(existing["question_id"].astype(str))
    pending = questions[~questions["question_id"].astype(str).isin(answered_ids)]
    if pending.empty:
        print("All questions already have answers; nothing to generate.")
        return []

    client = genai.Client(api_key=api_key)
    failures: list[str] = []
    for index, row in enumerate(pending.itertuples(index=False)):
        question_id = str(row.question_id)
        try:
            answer = generate_with_retries(
                client,
                model_name,
                str(row.question),
                str(row.language),
                max_retries,
                base_delay,
            )
            _save_answer(question_id, answer, answers_path)
            print(f"Saved answer for {question_id}.")
        except Exception as error:
            if _is_model_access_error(error):
                raise RuntimeError(
                    f"Gemini could not use model '{model_name}'. Check GEMINI_MODEL and "
                    "confirm your Google API key has permission to use this model. "
                    "Run `python -m src.generate_answers --list-models` to see available models."
                ) from error
            failures.append(question_id)
            print(f"Failed to generate answer for {question_id}: {error}", file=sys.stderr)
        if index < len(pending) - 1 and request_delay > 0:
            time.sleep(request_delay)

    if failures:
        raise RuntimeError(f"Answer generation failed for question IDs: {', '.join(failures)}")
    return [str(question_id) for question_id in pending["question_id"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="Gemini model name; overrides GEMINI_MODEL.")
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List generation models available to the configured API key.",
    )
    parser.add_argument("--max-retries", type=int, default=5, help="Retries after transient errors.")
    parser.add_argument("--base-delay", type=float, default=1.0, help="Base retry backoff in seconds.")
    parser.add_argument("--request-delay", type=float, default=1.0, help="Delay between API requests in seconds.")
    args = parser.parse_args()
    if args.max_retries < 0 or args.base_delay < 0 or args.request_delay < 0:
        parser.error("Retry count and delays must not be negative.")
    if args.list_models:
        try:
            api_key = _load_api_key(report_status=False)
            _print_available_models(genai.Client(api_key=api_key))
        except (OSError, ValueError, RuntimeError) as error:
            parser.exit(1, f"Error: {error}\n")
        return
    try:
        generate_answers(
            model_name=args.model,
            max_retries=args.max_retries,
            base_delay=args.base_delay,
            request_delay=args.request_delay,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
