"""
Eval harness for Project 2 agent.

Reads golden_tasks.jsonl, runs each task through run_agent, scores
the answer against the expected field, prints a report.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from unittest.mock import patch

# Make src/ importable
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from agent import run_agent  # noqa: E402


TASKS_PATH = Path(__file__).parent / "golden_tasks.jsonl"


# ---------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------

def score_task(expected: dict, answer: str) -> tuple[bool, str]:
    """
    Returns (passed, reason).

    expected shape (one of):
      {"must_contain": ["a", "b"]}            → ALL of these must appear
      {"must_contain_any_of": [["a"], ["b"]]} → ANY one group must fully match
      {"starts_with": "INSUFFICIENT_CONTEXT"} → answer must start with this
    """
    if answer is None:
        return False, "no answer returned"

    lower = answer.lower()

    if "starts_with" in expected:
        prefix = expected["starts_with"]
        if answer.strip().startswith(prefix):
            return True, "refusal matched"
        return False, f"expected prefix '{prefix}', got '{answer[:60]}...'"

    if "must_contain" in expected:
        missing = [s for s in expected["must_contain"] if s.lower() not in lower]
        if not missing:
            return True, "all required strings found"
        return False, f"missing: {missing}"

    if "must_contain_any_of" in expected:
        for group in expected["must_contain_any_of"]:
            if all(s.lower() in lower for s in group):
                return True, f"matched group: {group}"
        return False, f"no group matched. Groups: {expected['must_contain_any_of']}"

    return False, "unknown expected shape"


# ---------------------------------------------------------------
# Runner
# ---------------------------------------------------------------

def run_one(task: dict, auto_reject_refunds: bool = True) -> dict:
    """Run a single task and return result + score."""
    needs_refund_stub = task["category"] == "refund_flow"

    try:
        if needs_refund_stub and auto_reject_refunds:
            # Patch input() so the HITL gate auto-answers 'n'
            with patch("builtins.input", return_value="n"):
                result = run_agent(task["question"], verbose=False)
        else:
            result = run_agent(task["question"], verbose=False)
    except Exception as e:
        return {
            **task,
            "passed": False,
            "reason": f"EXCEPTION: {type(e).__name__}: {e}",
            "answer": None,
            "steps": None,
            "main_input_tokens": None,
            "main_output_tokens": None,
        }

    passed, reason = score_task(task["expected"], result["answer"])
    return {
        **task,
        "passed": passed,
        "reason": reason,
        "answer": result["answer"],
        "steps": result["steps"],
        "main_input_tokens": result["main_input_tokens"],
        "main_output_tokens": result["main_output_tokens"],
    }


def main():
    tasks = [json.loads(line) for line in TASKS_PATH.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(tasks)} tasks\n")

    results = []
    for i, task in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] {task['id']} ({task['category']}) ... ", end="", flush=True)
        r = run_one(task)
        results.append(r)
        print("PASS" if r["passed"] else "FAIL")

    # ---- Report ----
    print("\n" + "=" * 70)
    print("REPORT")
    print("=" * 70)

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    print(f"\nOverall: {passed}/{total} passed ({passed/total*100:.0f}%)")

    # Per category
    by_cat = defaultdict(lambda: {"passed": 0, "total": 0})
    for r in results:
        by_cat[r["category"]]["total"] += 1
        if r["passed"]:
            by_cat[r["category"]]["passed"] += 1
    print("\nBy category:")
    for cat, s in sorted(by_cat.items()):
        print(f"  {cat:20} {s['passed']}/{s['total']}")

    # Cost / step averages (only on runs that completed)
    completed = [r for r in results if r["steps"] is not None]
    if completed:
        avg_steps = sum(r["steps"] for r in completed) / len(completed)
        avg_in = sum(r["main_input_tokens"] for r in completed) / len(completed)
        avg_out = sum(r["main_output_tokens"] for r in completed) / len(completed)
        print(f"\nAverages (over {len(completed)} completed runs):")
        print(f"  steps:              {avg_steps:.1f}")
        print(f"  main input tokens:  {avg_in:,.0f}")
        print(f"  main output tokens: {avg_out:,.0f}")

    # Failures
    failures = [r for r in results if not r["passed"]]
    if failures:
        print(f"\n--- {len(failures)} FAILURES ---")
        for r in failures:
            print(f"\n[{r['id']}] {r['category']}")
            print(f"  Q: {r['question']}")
            print(f"  Reason: {r['reason']}")
            if r["answer"]:
                ans = r["answer"][:300].replace("\n", " ")
                print(f"  Answer: {ans}{'...' if len(r['answer']) > 300 else ''}")

    # Save detailed JSON for later analysis / log
    out_path = Path(__file__).parent / "eval_results.jsonl"
    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()