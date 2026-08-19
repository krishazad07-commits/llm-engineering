"""
review_golden_set.py — hand-review golden set candidates against the source PDF.

Usage: uv run scratchpad1/review_golden_set.py

Reads:  golden_qa_draft.jsonl
Writes: golden_qa.jsonl       (accepted + edited)
        rejections.jsonl      (rejected with reasons)

Both output files are append-only JSONL for crash safety and resume.
Already-reviewed candidates (by id) are skipped on restart.
"""

import json
from pathlib import Path

# TODO: three Path constants
# DRAFT_PATH points to golden_qa_draft.jsonl (input)
# GOLDEN_PATH points to golden_qa.jsonl (accepted output)
# REJECT_PATH points to rejections.jsonl (rejected output)
# All three live in the same directory as this script.

DRAFT_PATH = Path(__file__).parent / "golden_qa_draft.jsonl"
GOLDEN_PATH = Path(__file__).parent / "golden_qa.jsonl"
REJECT_PATH = Path(__file__).parent / "rejections.jsonl"

def load_reviewed_ids(golden_path: Path, reject_path: Path) -> set[str]:
    """Return the set of candidate IDs already decided (accepted or rejected)."""
    reviewed: set[str] = set()

    for path in (golden_path, reject_path):
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        reviewed.add(record["id"])

    return reviewed
def display_candidate(candidate: dict) -> None:
    """Print candidate fields in a readable format."""

    print("\n" + "=" * 80)
    print(f"ID: {candidate.get('id', 'N/A')}")
    print("=" * 80)

    print("\nQUESTION:")
    print(candidate.get("question", ""))

    print("\nEXPECTED ANSWER:")
    print(candidate.get("expected_answer", ""))

    print("\nRELEVANT CHUNK HINT:")
    print(candidate.get("relevant_chunk_hint", ""))

    print("\nMETADATA:")
    print(f"  Category   : {candidate.get('category', '')}")
    print(f"  Difficulty : {candidate.get('difficulty', '')}")
    print(f"  Answerable : {candidate.get('answerable', '')}")

    print("\n" + "-" * 80)   

def get_decision() -> str:
    """Prompt for accept/edit/reject/skip/quit. Returns one of: a, e, r, s, q."""
    valid = {"a", "e", "r", "s", "q"}
    prompt = "\n[a]ccept  [e]dit  [r]eject  [s]kip  [q]uit > "

    while True:
        # TODO 1: read one line of input, using `prompt` as the message
        choice = input(prompt)

        # TODO 2: normalize it — strip whitespace and lowercase it
        choice = choice.strip().lower()

        # TODO 3: if it's in the valid set, return it
        if choice in valid:
            return choice

        # TODO 4: otherwise, print an error message and let the loop retry
        print("Invalid choice. Please enter a, e, r, s, or q.")

def get_rejection_reason() -> str:
    """Prompt for a one-line rejection reason."""
    return input("Reason for rejection: ").strip()

def load_drafts(path: Path) -> list[dict]:
    """Load all candidates from the draft JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def edit_candidate(candidate: dict) -> dict:
    """Let the user edit any field. Press enter on a field to keep its current value."""
    editable_fields = ["question", "expected_answer", "relevant_chunk_hint", "category", "difficulty"]

    for field in editable_fields:
        current = candidate.get(field, "")
        new_value = input(f"{field} [{current}]: ").strip()
        if new_value:
            candidate[field] = new_value

    return candidate

def main():
    drafts = load_drafts(DRAFT_PATH)
    already_reviewed = load_reviewed_ids(GOLDEN_PATH, REJECT_PATH)
    remaining = [c for c in drafts if c["id"] not in already_reviewed]

    print(f"{len(already_reviewed)} already reviewed. {len(remaining)} remaining.\n")

    for i, candidate in enumerate(remaining, start=1):
        print(f"[{i}/{len(remaining)}]")
        display_candidate(candidate)
        decision = get_decision()

        # Quit the review session
        if decision == "q":
            print("\nQuitting review session.")
            break

        # Skip this candidate without saving anything
        if decision == "s":
            continue

        # Accept the candidate
        if decision == "a":
            with open(GOLDEN_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(candidate) + "\n")

        # Edit the candidate, then accept it
        if decision == "e":
            candidate = edit_candidate(candidate)
            with open(GOLDEN_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(candidate) + "\n")

        # Reject the candidate
        if decision == "r":
            reason = get_rejection_reason()
            candidate["rejection_reason"] = reason

            with open(REJECT_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(candidate) + "\n")

    print("\nReview session complete.")


if __name__ == "__main__":
    main()