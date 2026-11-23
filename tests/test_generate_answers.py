import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
from google.genai.errors import APIError

from src import generate_answers as generator


class GenerateAnswersTests(unittest.TestCase):
    def test_placeholder_key_reports_dotenv_status_and_stops_before_api(self) -> None:
        output = io.StringIO()

        with (
            patch.dict("os.environ", {"GOOGLE_API_KEY": generator.API_KEY_PLACEHOLDER}),
            patch.object(generator.genai, "Client") as create_client,
            redirect_stdout(output),
        ):
            with self.assertRaisesRegex(RuntimeError, r"Edit \.env"):
                generator.generate_answers()

        status = output.getvalue()
        self.assertIn(".env file found: yes", status)
        self.assertIn(f".env path: {(generator.PROJECT_ROOT / '.env').resolve()}", status)
        self.assertIn("GOOGLE_API_KEY set to a non-placeholder value: no", status)
        create_client.assert_not_called()

    def test_retries_rate_limit_then_returns_generated_answer(self) -> None:
        api_error = APIError(429, {"message": "rate limited"})
        models = SimpleNamespace(
            generate_content=Mock(side_effect=[api_error, SimpleNamespace(text="Answer")])
        )
        client = SimpleNamespace(models=models)

        with (
            patch.object(generator.time, "sleep"),
            patch.object(generator.random, "uniform", return_value=0),
        ):
            answer = generator.generate_with_retries(
                client,
                "gemini-test",
                "Sample question",
                "en",
                max_retries=1,
                base_delay=0,
            )

        self.assertEqual(answer, "Answer")
        self.assertEqual(models.generate_content.call_count, 2)
        self.assertEqual(
            models.generate_content.call_args.kwargs["model"],
            "gemini-test",
        )

    def test_skips_answered_questions_and_saves_only_missing_answer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions_path = root / "questions.csv"
            answers_path = root / "answers.csv"
            questions = pd.DataFrame(
                [
                    {
                        "question_id": "q1",
                        "language": "en",
                        "category": "science",
                        "question": "Q1",
                    },
                    {
                        "question_id": "q2",
                        "language": "hi",
                        "category": "science",
                        "question": "Q2",
                    },
                ]
            )
            questions.to_csv(questions_path, index=False)
            pd.DataFrame([{"question_id": "q1", "model_answer": "Existing"}]).to_csv(
                answers_path, index=False
            )
            generate_content = Mock(return_value=SimpleNamespace(text="New answer"))
            client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))

            with (
                patch.dict(
                    "os.environ",
                    {"GOOGLE_API_KEY": "test-key", "GEMINI_MODEL": "gemini-configured"},
                ),
                patch.object(generator.genai, "Client", return_value=client) as create_client,
                patch.object(generator.time, "sleep"),
            ):
                generated = generator.generate_answers(
                    questions_path=questions_path,
                    answers_path=answers_path,
                    max_retries=0,
                    request_delay=0,
                )

            saved = pd.read_csv(answers_path).set_index("question_id")["model_answer"]

        self.assertEqual(generated, ["q2"])
        self.assertEqual(create_client.call_count, 1)
        self.assertEqual(generate_content.call_count, 1)
        self.assertEqual(generate_content.call_args.kwargs["model"], "gemini-configured")
        self.assertEqual(saved.to_dict(), {"q1": "Existing", "q2": "New answer"})

    def test_model_permission_error_explains_how_to_check_configuration(self) -> None:
        class PermissionDenied(Exception):
            status_code = 403

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions_path = root / "questions.csv"
            pd.DataFrame(
                [{"question_id": "q1", "language": "en", "category": "science", "question": "Q1"}]
            ).to_csv(questions_path, index=False)
            client = SimpleNamespace(
                models=SimpleNamespace(generate_content=Mock(side_effect=PermissionDenied()))
            )

            with (
                patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
                patch.object(generator.genai, "Client", return_value=client),
            ):
                with self.assertRaisesRegex(RuntimeError, "Check GEMINI_MODEL.*permission.*--list-models"):
                    generator.generate_answers(
                        questions_path=questions_path,
                        answers_path=root / "answers.csv",
                        max_retries=0,
                        request_delay=0,
                    )

    def test_list_models_prints_only_generation_model_names(self) -> None:
        client = SimpleNamespace(
            models=SimpleNamespace(
                list=Mock(
                    return_value=[
                        SimpleNamespace(
                            name="models/gemini-3.5-flash-lite",
                            supported_actions=["generateContent"],
                        ),
                        SimpleNamespace(
                            name="models/embedding-example",
                            supported_actions=["embedContent"],
                        ),
                    ]
                )
            )
        )
        output = io.StringIO()

        with redirect_stdout(output):
            generator._print_available_models(client)

        self.assertEqual(output.getvalue(), "gemini-3.5-flash-lite\n")


if __name__ == "__main__":
    unittest.main()
