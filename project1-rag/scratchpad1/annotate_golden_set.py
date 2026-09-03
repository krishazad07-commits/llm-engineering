# scratchpad1/annotate_golden_set.py
"""Interactive annotation: for each scoreable question, show top-K vector
candidates and prompt for the DB primary keys that actually contain the answer.
Writes after each; safe to Ctrl-C and resume.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from google import genai

# Reuse the retrieval setup from your existing script.
from search_documents import retrieve

load_dotenv()
DATABASE_URL = os.environ["DATABASE_URL"]
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
GOLDEN_PATH = Path(__file__).parent / "golden_qa.jsonl"

CANDIDATE_K = 15
PREVIEW_CHARS = 400


def load_golden(path: Path) -> list[dict]:
    """Read JSONL into memory."""
    records: list[dict] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def save_golden(path: Path, records: list[dict]) -> None:
    """Atomically rewrite the JSONL file."""
    tmp_path = path.with_suffix(".jsonl.tmp")

    with tmp_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    tmp_path.replace(path)


def is_scoreable(record: dict) -> bool:
    """Only answerable, non-table questions need chunk annotation."""
    return (
        record.get("answerable") is True and record.get("category") != "table_dependent"
    )


def needs_annotation(record: dict) -> bool:
    """Missing key means not yet annotated."""
    return "relevant_chunk_ids" not in record


def display_candidates(chunks: list[tuple]) -> None:
    """Display retrieved candidates.

    retrieve() returns:
        (id, source_doc, page, chunk_id, content, distance)
    """
    print("\nTop candidates:\n")

    for rank, chunk in enumerate(chunks, start=1):
        chunk_id = chunk[0]  # DB primary key
        page = chunk[2]
        text = chunk[4]
        distance = chunk[5]

        preview = text.replace("\n", " ")[:PREVIEW_CHARS]

        print(f"[{rank:2}] id={chunk_id:<4} page={page:<2} dist={distance:.3f}")
        print(f"     {preview}")
        print()


def parse_input(raw: str) -> tuple[str, list[int] | None]:
    """Return (action, ids).

    action ∈ {"ids", "none", "skip", "quit"}
    """
    if raw in {"q", "quit"}:
        return ("quit", None)

    if raw in {"s", "skip"}:
        return ("skip", None)

    if raw in {"", "none"}:
        return ("none", [])

    ids = [int(part.strip()) for part in raw.split(",")]

    return ("ids", ids)


def main() -> None:
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set.")

    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set.")

    records = load_golden(GOLDEN_PATH)

    scoreable = [r for r in records if is_scoreable(r)]
    todo = [r for r in scoreable if needs_annotation(r)]

    print(f"{len(todo)} of {len(scoreable)} scoreable questions need annotation.\n")

    client = genai.Client(api_key=GOOGLE_API_KEY)

    with psycopg.connect(DATABASE_URL) as conn:
        for record in todo:
            print("=" * 90)
            print(f"ID       : {record['id']}")
            print(f"Category : {record['category']}")
            print("\nQUESTION")
            print(record["question"])

            print("\nHINT")
            print(record.get("relevant_chunk_hint") or "(none)")

            print("\nEXPECTED ANSWER")
            print(record["expected_answer"])

            candidates = retrieve(
                client,
                conn,
                record["question"],
                k=CANDIDATE_K,
            )

            display_candidates(candidates)

            while True:
                raw = input("ids (comma-sep) / none / skip / quit: ").strip().lower()

                try:
                    action, ids = parse_input(raw)
                    break
                except ValueError:
                    print("Couldn't parse input. Example: 42,57,91")
                    continue

            if action == "quit":
                print("\nStopping. Progress already saved for previous questions.")
                return

            if action == "skip":
                print("Skipped.\n")
                continue

            # ids is [] for "none", otherwise list[int]
            record["relevant_chunk_ids"] = ids
            save_golden(GOLDEN_PATH, records)

            print(f"✓ Saved relevant_chunk_ids = {ids}\n")

    print("All done.")


if __name__ == "__main__":
    main()
