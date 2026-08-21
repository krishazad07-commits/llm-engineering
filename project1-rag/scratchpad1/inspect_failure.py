"""
inspect_failure.py — dump retrieval + labeling for a single golden question.

Usage: uv run scratchpad1/inspect_failure.py q_026
"""

import json
import os
import sys

import psycopg
from dotenv import load_dotenv
from eval_retrieval import GOLDEN_PATH, TOP_K, is_relevant
from google import genai
from pgvector.psycopg import register_vector
from search_documents import retrieve

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")


def find_question(qid: str) -> dict:
    """Load the golden set and return the question with matching id."""
    with GOLDEN_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("id") == qid:
                return item
    raise ValueError(f"Question {qid} not found in {GOLDEN_PATH}")


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: uv run scratchpad1/inspect_failure.py <qid>")

    qid = sys.argv[1]

    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set in .env")
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set in .env")

    question = find_question(qid)

    print(f"Question: {question['question']}")
    print(f"Hint: {question['relevant_chunk_hint']}")
    print(f"Category: {question['category']}")
    print()

    client = genai.Client(api_key=GOOGLE_API_KEY)

    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)

        retrieved = retrieve(
            client,
            conn,
            question["question"],
            TOP_K,
        )

    for rank, (
        doc_id,
        source,
        page,
        chunk_id,
        content,
        distance,
    ) in enumerate(retrieved, start=1):

        label = is_relevant(
            content,
            question["relevant_chunk_hint"],
        )

        match = "[MATCH]" if label else "[no match]"

        print(
            f"#{rank} {match} "
            f"page={page} "
            f"distance={distance:.4f}"
        )
        print(f"    {content[:250]}...")
        print()


if __name__ == "__main__":
    main()