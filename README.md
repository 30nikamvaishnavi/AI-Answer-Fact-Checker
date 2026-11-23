# AI-Answer-Fact-Checker

A human-in-the-loop tool for checking the factual accuracy of Gemini answers to English and Hindi questions. The app uses Streamlit and pandas; the answer-generation script uses Google's current `google-genai` SDK. A Google API key is required to generate answers. No results are generated or evaluated until you run the project.

## Setup

1. Install Python 3.10 or newer.
2. From this project directory, create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env`, set `GOOGLE_API_KEY` to your Gemini API key, and optionally set `GEMINI_MODEL`. Keep `.env` private; it is ignored by git.

## How to run

Generate answers for questions not already present in `data/answers.csv`:

```powershell
.\.venv\Scripts\python.exe -m src.generate_answers
```

The model name is set with `GEMINI_MODEL` in `.env`; the default is `gemini-3.5-flash-lite`. Older model names may not be available to new API keys. Run `python -m src.generate_answers --list-models` to list generation models available to your key. You can also override the model with `--model`, and configure retry count and request delay with `--max-retries` and `--request-delay`. API rate limits and transient errors are retried with exponential backoff. Questions already present in `data/answers.csv` are skipped, and each successful answer is saved incrementally. If the model is not found or access is denied, check `GEMINI_MODEL` and run the model-list command above.

Start the review app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/app.py
```

Select a question to review its generated answer. Reviews are appended to `reviews/reviews.csv`; the latest review for each question is used in summary calculations. The Summary page shows verdict shares and the Wrong and Outdated answers, and provides a CSV download of the summary.

Run the unit tests from this directory:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Review rubric

- **Correct:** The answer's material factual claims are accurate and supported by a reliable source.
- **Partly Correct:** The answer contains useful accurate information but also has a meaningful omission, imprecision, or error.
- **Outdated:** The answer may once have been correct, but a material claim is no longer current.
- **Wrong:** One or more central factual claims are false or materially misleading.
- **Cannot Verify:** The reviewer cannot confirm the answer's claims using available reliable sources. Add a short note explaining the verification limitation where useful. A source URL is optional for this verdict; all other verdicts require one.

Use the source URL for the evidence consulted and the note to describe the error or verification limitation, rather than restating the full answer.

## Data and method

The author wrote the 50 questions: 25 in English and 25 in Hindi, covering factual, time-sensitive, and false-premise categories. Answers come from the Gemini API. The author checks each answer against named reliable sources and records the review.

## Results

Complete these placeholders after generating and reviewing answers. The sample questions are not evaluated results.

- Questions reviewed: [N questions]
- Share rated Correct: [% correct]
- Common error types: [common error types]
