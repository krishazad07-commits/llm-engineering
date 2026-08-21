"""
eval_retrieval.py — measure baseline retrieval quality on the golden set.

Usage: uv run scratchpad1/eval_retrieval.py

Reads:  golden_qa.jsonl (loads only answerable questions, skips 8 unanswerables)
Writes: eval_results.jsonl (per-question metrics)
        prints an aggregate report to stdout

Metrics computed: Recall@1, Recall@5, Recall@10, Precision@5, MRR, NDCG@10
"""
import re
import json
import os
from pathlib import Path



import psycopg
from dotenv import load_dotenv
from google import genai
from pgvector.psycopg import register_vector
from search_documents import retrieve

load_dotenv()


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")


GOLDEN_PATH = Path(__file__).parent/"golden_qa.jsonl"

RESULTS_PATH = Path(__file__).parent/"eval_results.jsonl"

TOP_K = 10

def load_scoreable_questions(path: Path) -> list[dict]:
    """Load golden QA entries that count toward retrieval metrics.

    Excludes unanswerable questions (no relevant chunk exists) and
    table_dependent questions (hint format doesn't support substring matching).
    Both are tracked separately in the final report.
    """
    questions = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)

            if item.get("answerable") is not True:
                continue
            if item.get("category") == "table_dependent":
                continue

            questions.append(item)

    return questions


# Matches "(Page 5)" or "(page 12)" or "(p. 5)" or "(p.5)"
PAGE_MARKER = re.compile(r"\((?:page|p\.?)\s*\d+\)", re.IGNORECASE)


def is_relevant(retrieved_text: str, hint: str) -> bool:
    """Return True if any cleaned segment of the hint appears in the retrieved text."""
    # Remove page markers like "(Page 5)"
    cleaned = PAGE_MARKER.sub("", hint)

    # Some hints have multiple quoted segments joined by ";"
    segments = [seg.strip().strip('"').strip("'") for seg in cleaned.split(";")]
    segments = [seg for seg in segments if seg]

    haystack = retrieved_text.lower()
    for segment in segments:
        if segment.lower() in haystack:
            return True

    return False


def label_ranking(retrieved_chunks: list[tuple], hint: str) -> list[bool]:
    """Label each retrieved chunk as relevant or not, preserving ranking order."""
    return [
        is_relevant(chunk[4], hint)
        for chunk in retrieved_chunks
    ]