import json
from pathlib import Path

path = Path(__file__).parent / "golden_qa.jsonl"

empty = 0
non_empty = 0
missing = 0

with path.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        record = json.loads(line)
        rel = record.get("relevant_chunk_ids")
        if rel is None:
            missing += 1
        elif rel == []:
            empty += 1
        else:
            non_empty += 1

print(f"empty ([]): {empty}")
print(f"non-empty:  {non_empty}")
print(f"missing:    {missing}")