"""
eval_retrieval.py — measure baseline retrieval quality on the golden set.

Usage: uv run scratchpad1/eval_retrieval.py

Reads:  golden_qa.jsonl (loads only answerable questions, skips 8 unanswerables)

Writes: eval_results.jsonl (per-question metrics)
        prints an aggregate report to stdout

Metrics computed: Recall@1, Recall@5, Recall@10, Precision@5, MRR, NDCG@10
                 Chunk Coverage@5, Chunk Coverage@10
"""
import argparse
import json
import math
import os
import re
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from google import genai
from hybrid_search import build_bm25_index, retrieve_hybrid
from pgvector.psycopg import register_vector
from rerank_search import retrieve_reranked
from search_documents import retrieve
from search_documents_contextual import retrieve as retrieve_contextual

load_dotenv()


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")


GOLDEN_PATH = Path(__file__).parent / "golden_qa.jsonl"
RESULTS_PATH = Path(__file__).parent / "eval_results.jsonl"
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
PAGE_MARKER = re.compile(
    r"\((?:page|p\.?)\s*\d+\)",
    re.IGNORECASE,
)


def is_relevant(chunk: tuple, record: dict) -> bool:
    """Return True if the chunk is relevant to the question.

    Prefers ID set-membership when relevant_chunk_ids is present on the record.
    Falls back to hint-substring matching for records that lack the field
    (backward-compat with older golden set schema).
    """
    chunk_id = chunk[0]
    relevant_ids = record.get("relevant_chunk_ids")

    # Use annotated chunk IDs when the field exists.
    if relevant_ids is not None:
        return chunk_id in set(relevant_ids)

    # Fall back to the old hint-substring logic.
    hint = record.get("relevant_chunk_hint", "")

    cleaned = PAGE_MARKER.sub("", hint)

    segments = [
        seg.strip().strip('"').strip("'")
        for seg in cleaned.split(";")
    ]

    segments = [seg for seg in segments if seg]

    haystack = chunk[4].lower()

    for segment in segments:
        if segment.lower() in haystack:
            return True

    return False


def chunk_coverage_at_k(
    retrieved_chunks: list[tuple],
    record: dict,
    k: int,
) -> float:
    """Return fraction of relevant_chunk_ids that appear in the top-k retrieved.

    Diagnostic metric that distinguishes 1-of-3 from 3-of-3 multi-hop retrievals.
    Returns 0.0 when relevant_chunk_ids is missing or empty.
    """
    relevant_ids = record.get("relevant_chunk_ids")

    # Missing or empty annotation → undefined coverage.
    if not relevant_ids:
        return 0.0

    # Get IDs from the top-k retrieved chunks.
    retrieved_ids = {chunk[0] for chunk in retrieved_chunks[:k]}

    # Calculate fraction of relevant IDs retrieved.
    relevant_ids_set = set(relevant_ids)

    return len(relevant_ids_set & retrieved_ids) / len(relevant_ids_set)


def label_ranking(
    retrieved_chunks: list[tuple],
    record: dict,
) -> list[bool]:
    """Label each retrieved chunk as relevant or not, preserving ranking order."""
    return [
        is_relevant(chunk, record)
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
        "chunk_coverage@5",
        "chunk_coverage@10",
    ]

    def avg(items: list[dict], key: str) -> float:
        return sum(item[key] for item in items) / len(items)

    # Overall metrics
    print("\n=== Overall Retrieval Metrics ===")

    print(f"Questions: {len(results)}")

    for key in metric_keys:
        print(f"{key}: {avg(results, key):.3f}")

    # Group results by category
    by_category = {}

    for item in results:
        category = item["category"]
        by_category.setdefault(category, []).append(item)

    # Per-category metrics
    print("\n=== By Category ===")

    for category in sorted(by_category):
        category_results = by_category[category]

        print(f"\n[{category}] n={len(category_results)}")

        for key in metric_keys:
            print(f"{key}: {avg(category_results, key):.3f}")

    # Exclusions
    print(
        "\nExcluded: 8 unanswerable, 5 table_dependent "
        "(see LOG for reasons)"
    )


def main():
    # Parse CLI args
    parser = argparse.ArgumentParser(
        description="Run retrieval evaluation on the golden set.",
    )
    parser.add_argument(
        "--retriever",
        choices=["vector", "hybrid", "reranked", "contextual"],
        default="vector",
        help="Which retriever to evaluate.",
        
    )
    args = parser.parse_args()

    print(f"\n=== Running eval with retriever: {args.retriever} ===\n")

    # Write results to a separate file for each retriever
    results_path = Path(__file__).parent / (
        f"eval_results_{args.retriever}.jsonl"
    )

    # Check required environment variables
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

        # TODO: only build BM25 when args.retriever == "hybrid"
        bm25, chunks = build_bm25_index(conn)

        for i, q in enumerate(questions, start=1):

            # Retrieve top-k chunks using the selected retriever
            if args.retriever == "vector":
                retrieved = retrieve(
                    client,
                    conn,
                    q["question"],
                    TOP_K,
                )

            elif args.retriever == "hybrid":
                retrieved = retrieve_hybrid(
                    client,
                    conn,
                    bm25,
                    chunks,
                    q["question"],
                    TOP_K,
                )

            elif args.retriever == "reranked":
                retrieved = retrieve_reranked(
                    client,
                    conn,
                    q["question"],
                    k=TOP_K,
                    wide_k=20,
                )
            elif args.retriever == "contextual":
                retrieved = retrieve_contextual(
                    client,
                    conn,
                    q["question"],
                    TOP_K,
                )

            # Label the retrieved chunks
            labels = label_ranking(
                retrieved,
                q,
            )

            # Calculate metrics
            result = {
                "id": q["id"],
                "category": q["category"],
                "recall@1": recall_at_k(labels, 1),
                "recall@5": recall_at_k(labels, 5),
                "recall@10": recall_at_k(labels, 10),
                "precision@5": precision_at_k(labels, 5),
                "mrr": reciprocal_rank(labels),
                "ndcg@10": ndcg_at_k(labels, 10),
                "chunk_coverage@5": chunk_coverage_at_k(
                    retrieved,
                    q,
                    5,
                ),
                "chunk_coverage@10": chunk_coverage_at_k(
                    retrieved,
                    q,
                    10,
                ),
            }

            results.append(result)

            # Print progress
            print(
                f"[{i}/{len(questions)}] "
                f"{q['id']} "
                f"category={q['category']} "
                f"recall@10={result['recall@10']:.3f}"
            )

    # Write results as JSONL
    with results_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    # Report
    print(f"\n\n### Results for retriever: {args.retriever} ###")
    report(results)


if __name__ == "__main__":
    main()