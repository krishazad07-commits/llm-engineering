"""
Consolidate per-question retrieval eval results into an aggregate + per-category table.
Reads eval_results_{retriever}.jsonl files, computes means, prints markdown tables.
No LLM calls — pure aggregation over already-computed metrics.
"""

import json
from pathlib import Path
from statistics import mean

RETRIEVERS = ["vector", "reranked", "contextual"]
METRICS = ["recall@1", "recall@5", "recall@10", "precision@5", "mrr",
           "ndcg@10", "chunk_coverage@5", "chunk_coverage@10"]


def load_rows(retriever: str) -> list[dict]:
    path = Path(f"eval_results_{retriever}.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def aggregate(rows: list[dict]) -> dict[str, float]:
    return {m: mean(r[m] for r in rows) for m in METRICS}


def by_category(rows: list[dict]) -> dict[str, dict[str, float]]:
    cats = sorted({r["category"] for r in rows})
    return {
        c: aggregate([r for r in rows if r["category"] == c])
        for c in cats
    }


def print_markdown_table(title: str, header: list[str], rows: list[list[str]]) -> None:
    print(f"\n**{title}**\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")


def fmt(x: float) -> str:
    return f"{x:.3f}"


def main() -> None:
    data = {r: load_rows(r) for r in RETRIEVERS}

    # Overall
    header = ["Retriever"] + METRICS
    rows = [[r] + [fmt(aggregate(data[r])[m]) for m in METRICS] for r in RETRIEVERS]
    print_markdown_table("Overall (n=37)", header, rows)

    # Per-category, per-retriever
    all_cats = sorted({r["category"] for r in data["vector"]})
    for cat in all_cats:
        cat_rows = []
        n = None
        for retr in RETRIEVERS:
            filtered = [r for r in data[retr] if r["category"] == cat]
            n = len(filtered)
            agg = aggregate(filtered)
            cat_rows.append([retr] + [fmt(agg[m]) for m in METRICS])
        print_markdown_table(f"{cat} (n={n})", header, cat_rows)


if __name__ == "__main__":
    main()