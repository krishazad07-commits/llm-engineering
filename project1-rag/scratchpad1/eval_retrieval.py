"""
eval_retrieval.py — measure baseline retrieval quality on the golden set.

Usage: uv run scratchpad1/eval_retrieval.py

Reads:  golden_qa.jsonl (loads only answerable questions, skips 8 unanswerables)
Writes: eval_results.jsonl (per-question metrics)
        prints an aggregate report to stdout

Metrics computed: Recall@1, Recall@5, Recall@10, Precision@5, MRR, NDCG@10
"""
import json
import math
import os
import re
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

def recall_at_k(labels: list[bool], k: int) -> float:
    """Return 1.0 if any of the top-k labels is True, else 0.0.

    Binary per-question recall — averaged across the golden set to produce
    the aggregate Recall@k metric.
    """
    top_k = labels[:k]
    return 1.0 if any(top_k) else 0.0

def precision_at_k(labels: list[bool], k: int) -> float:
    """Return the fraction of top-k labels that are True."""
    top_k = labels[:k]
    relevant_count = sum(top_k)
    return relevant_count / float(k)

def reciprocal_rank(labels: list[bool]) -> float:
    """Return 1 / (rank of first True), or 0.0 if no True is found.

    Rank is 1-indexed (rank 1 = position 0 in the list).
    """
    for index, label in enumerate(labels):
        if label:
            return 1 / (index + 1)

    return 0.0

def ndcg_at_k(labels: list[bool], k: int) -> float:
    """Return NDCG@k for a binary single-relevance ranking.

    In our simplified case (max 1 True per question), NDCG@k = 1 / log2(rank + 1)
    where rank is the 1-indexed position of the first True in labels[:k], or
    0.0 if no True appears.
    """
    top_k = labels[:k]

    for index, label in enumerate(top_k):
        if label:
            return 1 / math.log2((index + 1) + 1)

    return 0.0

def report(results: list[dict]) -> None:
    """Print aggregate + per-category metrics to stdout."""
    if not results:
        print("No results to report.")
        return

    metric_keys = [
        "recall@1",
        "recall@5",
        "recall@10",
        "precision@5",
        "mrr",
        "ndcg@10",
    ]

    def avg(items: list[dict], key: str) -> float:
        return sum(item[key] for item in items) / len(items)

    # TODO 1: overall metrics
    print("\n=== Overall Retrieval Metrics ===")
    print(f"Questions: {len(results)}")

    for key in metric_keys:
        print(f"{key}: {avg(results, key):.3f}")

    # TODO 2: group results by category
    by_category = {}

    for item in results:
        category = item["category"]
        by_category.setdefault(category, []).append(item)

    # TODO 3: per-category metrics
    print("\n=== By Category ===")

    for category in sorted(by_category):
        category_results = by_category[category]

        print(f"\n[{category}] n={len(category_results)}")

        for key in metric_keys:
            print(f"{key}: {avg(category_results, key):.3f}")

    # TODO 4: exclusions
    print(
        "\nExcluded: 8 unanswerable, 5 table_dependent "
        "(see LOG for reasons)"
    )
def main():
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set in .env")
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set in .env")

    client = genai.Client(api_key=GOOGLE_API_KEY)
    questions = load_scoreable_questions(GOLDEN_PATH)
    print(
        f"Loaded {len(questions)} scoreable questions "
        f"(excluded: 8 unanswerable, 5 table_dependent)\n"
    )

    results = []

    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)

        for i, q in enumerate(questions, start=1):
            # TODO 1: retrieve top-k chunks
            retrieved = retrieve(
                client,
                conn,
                q["question"],
                TOP_K,
            )

            # TODO 2: label the retrieved chunks
            labels = label_ranking(
                retrieved,
                q["relevant_chunk_hint"],
            )

            # TODO 3: calculate metrics
            result = {
                "id": q["id"],
                "category": q["category"],
                "recall@1": recall_at_k(labels, 1),
                "recall@5": recall_at_k(labels, 5),
                "recall@10": recall_at_k(labels, 10),
                "precision@5": precision_at_k(labels, 5),
                "mrr": reciprocal_rank(labels),
                "ndcg@10": ndcg_at_k(labels, 10),
            }

            results.append(result)

            # TODO 4: print progress
            print(
                f"[{i}/{len(questions)}] "
                f"{q['id']} "
                f"category={q['category']} "
                f"recall@10={result['recall@10']:.3f}"
            )

    # TODO 5: write results as JSONL
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    # TODO 6: report comes next
    report(results)


if __name__ == "__main__":
    main()