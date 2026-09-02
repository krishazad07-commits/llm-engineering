"""
eval_generation.py — measures the generation half of the pipeline.

For each question in golden_qa.jsonl:
  1. Retrieve top-K chunks
  2. Generate an answer
  3. Check whether the model abstained (output contains INSUFFICIENT_CONTEXT)
  4. Compare against ground truth (answerable=True/False from the golden set)

Reports:
  - Correct abstentions on unanswerables (should be high)
  - Correct attempts on answerables (should be high)
  - The two failure modes: hallucinations & over-refusals

Writes per-question results to eval_generation_results.jsonl for later inspection.
"""

import json
import os
import time
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from generate_answer import RetrievedChunk, generate_answer
from google import genai
from pgvector.psycopg import register_vector
from search_documents import retrieve

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

GOLDEN_SET_PATH = Path("golden_qa.jsonl")
RESULTS_PATH = Path("eval_generation_results.jsonl")

TOP_K = 5
ABSTENTION_TOKEN = "INSUFFICIENT_CONTEXT"


# ---------------------------------------------------------------------------
# Reused from sanity_generate.py — same shape mapping
# ---------------------------------------------------------------------------
def to_retrieved_chunks(raw_rows: list[tuple]) -> list[RetrievedChunk]:
    """Map retriever tuples to RetrievedChunk dataclasses."""
    chunks = []

    for row in raw_rows:
        _doc_id, source, page, chunk_id, content, distance = row

        chunks.append(
            RetrievedChunk(
                chunk_id=str(chunk_id),
                source_doc=source,
                page=page,
                content=content,
                similarity=1 - distance,
            )
        )

    return chunks


# ---------------------------------------------------------------------------
# The one real check: did the model abstain?
# ---------------------------------------------------------------------------
def did_abstain(answer: str) -> bool:
    """
    Return True if the model chose to abstain.

    Rule: substring match on ABSTENTION_TOKEN, case-sensitive.
    - "INSUFFICIENT_CONTEXT" → True (clean abstention)
    - "INSUFFICIENT_CONTEXT. The doc says nothing about X" → True
    - "insufficient context" → False
    - "The answer is not clear" → False
    """
    return ABSTENTION_TOKEN in answer


# ---------------------------------------------------------------------------
# Load the golden set
# ---------------------------------------------------------------------------
def load_golden_set(path: Path) -> list[dict]:
    """Load all questions from the JSONL file into a list of dicts."""
    with open(path, encoding="utf-8") as f:   # ← add "as f"
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set in .env")

    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set in .env")

    client = genai.Client(api_key=GOOGLE_API_KEY)

    questions = load_golden_set(GOLDEN_SET_PATH)

    print(f"Loaded {len(questions)} questions from {GOLDEN_SET_PATH}")

    # Counters we'll fill in as we go.
    # Naming convention:
    #   n_*        = totals in each class
    #   correct_*  = the model did the right thing

    n_answerable = 0
    n_unanswerable = 0

    correct_attempts = 0
    correct_abstentions = 0

    hallucinations = []
    over_refusals = []

    per_question_results = []

    # Open results file once so each question can be written immediately.
    with (
        open(RESULTS_PATH, "w", encoding="utf-8") as results_file,
        psycopg.connect(DATABASE_URL) as conn,
    ):
        register_vector(conn)
    # ... rest of loop unchanged

        for i, q in enumerate(questions, start=1):
            question_text = q["question"]
            is_answerable = q["answerable"]

            print(f"[{i}/{len(questions)}] {question_text[:70]}...")

            # Retrieve top-K chunks for this question.
            raw = retrieve(
                client,
                conn,
                question_text,
                k=TOP_K,
            )

            chunks = to_retrieved_chunks(raw)

            # Generate an answer using the retrieved chunks.
            answer = generate_answer(
                client,
                question_text,
                chunks,
            )

            abstained = did_abstain(answer)

            # Classify this result into one of four buckets.
            if is_answerable:
                n_answerable += 1

                if abstained:
                    over_refusals.append(q)
                else:
                    correct_attempts += 1

            else:
                n_unanswerable += 1

                if abstained:
                    correct_abstentions += 1
                else:
                    hallucinations.append(q)

            # Save per-question result for later inspection.
            per_question_results.append(
                {
                    "id": q["id"],
                    "question": question_text,
                    "answerable": is_answerable,
                    "abstained": abstained,
                    "answer": answer,
                }
            )

            # Write each result immediately instead of waiting until the end.
            results_file.write(
                json.dumps(
                    per_question_results[-1],
                    ensure_ascii=False,
                )
                + "\n"
            )
            results_file.flush()

            # Give the API a short pause between questions.
            time.sleep(4)

    print(
        f"\nWrote {len(per_question_results)} results "
        f"to {RESULTS_PATH}"
    )

    # ---------- Summary ----------
    abstention_rate = (
        correct_abstentions / n_unanswerable
        if n_unanswerable
        else 0.0
    )

    attempt_rate = (
        correct_attempts / n_answerable
        if n_answerable
        else 0.0
    )

    print("\n" + "=" * 60)
    print("GENERATION EVAL RESULTS")
    print("=" * 60)

    print(f"Unanswerable questions: {n_unanswerable}")
    print(
        f"  Correct abstentions:  "
        f"{correct_abstentions} ({abstention_rate:.1%})"
    )
    print(
        f"  Hallucinations:       "
        f"{len(hallucinations)}"
    )

    print()

    print(f"Answerable questions:   {n_answerable}")
    print(
        f"  Correct attempts:     "
        f"{correct_attempts} ({attempt_rate:.1%})"
    )
    print(
        f"  Over-refusals:        "
        f"{len(over_refusals)}"
    )

    # Print failure IDs so we can inspect them later.
    if hallucinations:
        print(
            f"\nHallucination IDs: "
            f"{[q['id'] for q in hallucinations]}"
        )

    if over_refusals:
        print(
            f"Over-refusal IDs: "
            f"{[q['id'] for q in over_refusals]}"
        )

    # Close the results file after all summary prints.
    results_file.close()


if __name__ == "__main__":
    main()