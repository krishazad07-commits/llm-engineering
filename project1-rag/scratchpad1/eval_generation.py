"""
eval_generation.py — measures the generation half of the pipeline.

BATCH-OPTIMIZED VERSION:
  - Embeds all queries upfront in one API call (10× faster, no per-min rate limits)
  - Loop only does DB search + generation per question
"""

import json
import os
import time
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from generate_answer import RetrievedChunk, generate_answer
from google import genai
from groq import Groq
from pgvector.psycopg import register_vector
from search_documents import embed_query_batch, search

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOLDEN_SET_PATH = Path("golden_qa.jsonl")
RESULTS_PATH = Path("eval_generation_results.jsonl")

TOP_K = 5
ABSTENTION_TOKEN = "INSUFFICIENT_CONTEXT"


# ---------------------------------------------------------------------------
# Same as before — no changes
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


def did_abstain(answer: str) -> bool:
    return ABSTENTION_TOKEN in answer


def load_golden_set(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------------------
# Main loop — now with upfront batch embedding
# ---------------------------------------------------------------------------
def main() -> None:
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set in .env")
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set in .env")
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set in .env")
    groq_client = Groq(api_key=GROQ_API_KEY)
    client = genai.Client(api_key=GOOGLE_API_KEY)
    questions = load_golden_set(GOLDEN_SET_PATH)
    print(f"Loaded {len(questions)} questions from {GOLDEN_SET_PATH}")

    # -----------------------------------------------------------------------
    # Step 1: Extract just the question strings from the golden set.
    # -----------------------------------------------------------------------
    question_texts = [q["question"] for q in questions]

    # -----------------------------------------------------------------------
    # Step 2: Embed all questions in ONE API call.
    # -----------------------------------------------------------------------
    print(f"Batch-embedding {len(question_texts)} queries in one API call...")
    query_vectors = embed_query_batch(client, question_texts)
    print(f"Got {len(query_vectors)} vectors. Starting eval loop.\n")

    # -----------------------------------------------------------------------
    # Counters (same as before)
    # -----------------------------------------------------------------------
    n_answerable = 0
    n_unanswerable = 0
    correct_attempts = 0
    correct_abstentions = 0
    hallucinations = []
    over_refusals = []
    per_question_results = []

    # -----------------------------------------------------------------------
    # Step 3: Loop. Now uses pre-computed vectors + calls search() directly.
    # -----------------------------------------------------------------------
    with (
        open(RESULTS_PATH, "w", encoding="utf-8") as results_file,
        psycopg.connect(DATABASE_URL) as conn,
    ):
        register_vector(conn)

        for i, (q, query_vector) in enumerate(zip(questions, query_vectors), start=1):
            question_text = q["question"]
            is_answerable = q["answerable"]

            print(f"[{i}/{len(questions)}] {question_text[:70]}...")

            # Search using the pre-computed embedding.
            raw = search(conn, query_vector, TOP_K)

            chunks = to_retrieved_chunks(raw)

            # Generate answer using retrieved chunks.
            answer = generate_answer(groq_client, question_text, chunks)

            abstained = did_abstain(answer)

            # Same 4-quadrant classification as before.
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

            per_question_results.append(
                {
                    "id": q["id"],
                    "question": question_text,
                    "answerable": is_answerable,
                    "abstained": abstained,
                    "answer": answer,
                }
            )

            # Incremental write — same crash-safe pattern from Day 19.
            results_file.write(
                json.dumps(
                    per_question_results[-1],
                    ensure_ascii=False,
                )
                + "\n"
            )
            results_file.flush()

            # Protect Gemini generation rate limit.
            time.sleep(0.5)

    print(f"\nWrote {len(per_question_results)} results to {RESULTS_PATH}")

    # ---------- Summary ----------
    abstention_rate = correct_abstentions / n_unanswerable if n_unanswerable else 0.0
    attempt_rate = correct_attempts / n_answerable if n_answerable else 0.0

    print("\n" + "=" * 60)
    print("GENERATION EVAL RESULTS")
    print("=" * 60)
    print(f"Unanswerable questions: {n_unanswerable}")
    print(f"  Correct abstentions:  {correct_abstentions} ({abstention_rate:.1%})")
    print(f"  Hallucinations:       {len(hallucinations)}")
    print()
    print(f"Answerable questions:   {n_answerable}")
    print(f"  Correct attempts:     {correct_attempts} ({attempt_rate:.1%})")
    print(f"  Over-refusals:        {len(over_refusals)}")

    if hallucinations:
        print(f"\nHallucination IDs: {[q['id'] for q in hallucinations]}")

    if over_refusals:
        print(f"Over-refusal IDs: {[q['id'] for q in over_refusals]}")


if __name__ == "__main__":
    main()
