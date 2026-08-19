"""
build_golden_set.py — draft candidate questions for the golden dataset.

Usage: uv run build_golden_set.py
Output: golden_qa_draft.jsonl (one JSON object per line)

This is a DRAFT generator. Every candidate must be hand-reviewed
against the PDF before promotion to golden_qa.jsonl.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEN_MODEL = "gemini-3.5-flash-lite"

LETTER_PATH = Path(__file__).parent.parent / "data" / "berkshire_2023.pdf"
OUTPUT_PATH = Path(__file__).parent / "golden_qa_draft.jsonl"

# Category plan — locked
CATEGORIES = {
    "extractive": 15,
    "inferential": 8,
    "multi_hop": 8,
    "partial": 6,
    "table_dependent": 5,
    "unanswerable": 8,
}


# --- Category prompts ---

EXTRACTIVE_PROMPT = """You are helping build an evaluation dataset for a RAG system over Warren Buffett's 2023 Berkshire Hathaway shareholder letter.

Generate exactly 15 EXTRACTIVE questions about this letter.

An EXTRACTIVE question is one whose answer is a specific fact stated directly in the letter — a number, a name, a date, a quote. The answer should be:
- Directly stated in the text (not requiring inference or calculation)
- Verifiable by pointing to a single sentence or short passage
- Unambiguous (there is exactly one correct answer)

For each question, return a JSON object with these fields:
- "question": the question text
- "expected_answer": a full sentence answer, not just a keyword
- "relevant_chunk_hint": a short quote (10-30 words) from the letter that contains the answer. Include the page number if you can identify it.
- "category": always "extractive"
- "difficulty": "easy", "medium", or "hard"
- "answerable": always true

Return your response as a JSON ARRAY of exactly 15 such objects. No wrapping object, no preamble, no markdown fences — just the array."""

UNANSWERABLE_PROMPT = """You are helping build an evaluation dataset for a RAG system over Warren Buffett's 2023 Berkshire Hathaway shareholder letter.

Generate exactly 8 UNANSWERABLE questions about this letter.

An UNANSWERABLE question is one that:
- has no answer anywhere in the letter
- the question is one a real reader of the letter might plausibly ask — for example, about a topic Buffett touches on but doesn't elaborate on, or about a related entity the letter mentions but doesn't cover.
- should not be topics entirely unrelated to Berkshire, business, or investing — the question must be plausible in context

For each question, return a JSON object with these fields:
- "question": the question text
- "expected_answer": always the literal string "INSUFFICIENT_CONTEXT"
- "relevant_chunk_hint": always the JSON null value (not the string "null")
- "category": always "unanswerable"
- "difficulty": "easy", "medium", or "hard"
- "answerable": always false

Before finalizing each question, verify the letter genuinely does not answer it. Do not include questions the letter partially or fully addresses.

Return your response as a JSON ARRAY of exactly 8 such objects. No wrapping object, no preamble, no markdown fences — just the array."""

INFERENTIAL_PROMPT = """You are helping build an evaluation dataset for a RAG system over Warren Buffett's 2023 Berkshire Hathaway shareholder letter.

Generate exactly 8 INFERENTIAL questions about this letter.
An INFERENTIAL question is one whose answer is not stated directly in the letter, but can be inferred from the text. The answer should be:
- Answer requires one small reasoning step on stated facts — for example, a calculation (like computing year-over-year growth from two stated numbers), a comparison ('which is larger, X or Y'), or a simple synthesis of two nearby facts. The raw facts must be in the letter; the answer itself is not.
- Verifiable by pointing to a single sentence or short passage that supports the inference
- Unambiguous (there is exactly one correct answer)

for each question, return a JSON object with these fields:
- "question": the question text
- "expected_answer": a full sentence answer, not just a keyword
- "relevant_chunk_hint": a short quote (10-30 words) from the letter that supports the inference. Include the page number if you can identify it.
- "category": always "inferential"
- "difficulty": "easy", "medium", or "hard"
- "answerable": always true

Return your response as a JSON ARRAY of exactly 8 such objects. No wrapping object, no preamble, no markdown fences — just the array."""

MULTI_HOP_PROMPT = """You are helping build an evaluation dataset for a RAG system over Warren Buffett's 2023 Berkshire Hathaway shareholder letter.

Generate exactly 8 MULTI-HOP questions about this letter.
A MULTI-HOP question is one whose answer requires reasoning across multiple parts of the letter. The answer should be:
- Answer requires reasoning across multiple parts of the letter — for example, synthesizing two or more facts from different sections,
- Verifiable by pointing to multiple sentences or short passages that support the answer
- Unambiguous (there is exactly one correct answer)

for each question, return a JSON object with this fields:
- "question": the question text    
- "expected_answer": a full sentence answer, not just a keyword
- "relevant_chunk_hint": "TWO OR MORE short quotes (10-30 words each) from DIFFERENT sections of the letter, with page numbers"
- "category": always "multi_hop"
- "difficulty": "easy", "medium", or "hard"
- "answerable": always true

Return your response as a JSON ARRAY of exactly 8 such objects. No wrapping object, no preamble, no markdown fences — just the array."""

PARTIAL_PROMPT = """You are helping build an evaluation dataset for a RAG system over Warren Buffett's 2023 Berkshire Hathaway shareholder letter.

Generate exactly 6 PARTIAL questions about this letter.

A PARTIAL question is one where the letter answers one part of the question but does not provide the complete information needed to answer every part. The question should be:
- Partially answerable from the letter: the letter clearly provides some of the requested information.
- Partially unanswerable: at least one important part of the question is not stated anywhere in the letter.
- A plausible question that a real reader might ask, not a contrived or unrelated question.
- Designed so that the correct response should state what the letter DOES say and clearly identify what information is missing, rather than simply returning INSUFFICIENT_CONTEXT.
- Unambiguous about which part is supported by the letter and which part is missing.

For example, if the letter says Buffett and Munger first met in 1959 when Munger was 35, but does not explain how they met, a good partial question would be:
"When did Buffett and Munger first meet, how old was Munger at the time, and how did they meet?"
The expected answer should state the first two facts and explicitly say that the letter does not describe how they met.

For each question, return a JSON object with these fields:
- "question": the question text
- "expected_answer": a full sentence answer that states the information supported by the letter AND explicitly identifies the missing or unanswered part
- "relevant_chunk_hint": a short quote (10-30 words) from the letter containing the supported information. If the question requires multiple supported facts, include the relevant quote(s). Include the page number if you can identify it.
- "category": always "partial"
- "difficulty": "easy", "medium", or "hard"
- "answerable": always true

Before finalizing each question, verify that the letter genuinely answers one important part of the question but does not answer the remaining part. Do not include questions where the entire answer is present, or questions where none of the answer is present.

Return your response as a JSON ARRAY of exactly 6 such objects. No wrapping object, no preamble, no markdown fences — just the array."""

TABLE_DEPENDENT_PROMPT = """You are helping build an evaluation dataset for a RAG system over Warren Buffett's 2023 Berkshire Hathaway shareholder letter.

Generate exactly 5 TABLE-DEPENDENT questions about this letter.

A TABLE-DEPENDENT question is one whose answer requires reading a specific cell or value from one of the tables in the letter, rather than relying only on surrounding prose. The question should be:
- Answerable by identifying a specific value in a table in the letter.
- Dependent on the exact row and column location of the value.
- Based on one of the letter's two important tables: the operating earnings table on page 10, or the year-by-year performance table on pages 16-17.
- Unambiguous, with exactly one correct cell or table value.
- Written as a realistic question a reader might ask about the information in the table.

For each question, return a JSON object with these fields:
- "question": the question text
- "expected_answer": the exact value from the relevant table, such as "15.8%" or "$5,428M"
- "relevant_chunk_hint": identify the table and the exact location of the answer, for example: "Operating earnings table, page 10, row: Insurance-underwriting, column: 2023"
- "category": always "table_dependent"
- "difficulty": "easy", "medium", or "hard"
- "answerable": always true

The answer must come directly from the table cell. Do not require calculations, inference, or information from unrelated prose. Prefer questions that test whether the system correctly preserves table structure, row labels, column labels, and exact cell values.

Before finalizing each question, verify that the expected answer is an exact value shown in the specified table and that the table location is correct.

Return your response as a JSON ARRAY of exactly 5 such objects. No wrapping object, no preamble, no markdown fences — just the array."""
# --- Generator function ---


def generate_candidates_for_category(
    client: genai.Client,
    category: str,
    count: int,
    letter_part: types.Part,
    prompt_for_category: str,
) -> list[dict]:
    """
    Ask Gemini to draft `count` candidate questions for `category`.
    Returns a list of dicts matching the golden set schema
    (minus the id field — assigned later).
    """
    user_contents = [
        letter_part,
        prompt_for_category,
    ]

    response = client.models.generate_content(
        model=GEN_MODEL,
        contents=user_contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.7,
        ),
    )

    candidates = json.loads(response.text)
    return candidates


def main():
    # Step 1: env var guard
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set in .env")

    # Step 2: build the Gemini client
    client = genai.Client(api_key=GOOGLE_API_KEY)

    # Step 3: load the PDF as a Gemini Part (once, reused across all categories)
    letter_bytes = LETTER_PATH.read_bytes()
    letter_part = types.Part.from_bytes(
        data=letter_bytes,
        mime_type="application/pdf",
    )

    # Step 4: map category name -> prompt string (so the loop can look up the right prompt)
    prompts_by_category = {
        "extractive": EXTRACTIVE_PROMPT,
        "unanswerable": UNANSWERABLE_PROMPT,
        "inferential": INFERENTIAL_PROMPT,
        "multi_hop": MULTI_HOP_PROMPT,
        "partial": PARTIAL_PROMPT,
        "table_dependent": TABLE_DEPENDENT_PROMPT,
    }

    # Step 5: loop through categories, generate candidates, collect them all
    all_candidates = []
    for category, count in CATEGORIES.items():
        prompt = prompts_by_category[category]
        print(f"Generating {count} {category} questions...")

        candidates = generate_candidates_for_category(
            client,
            category,
            count,
            letter_part,
            prompt,
        )

        print(f"  → got {len(candidates)} candidates")
        all_candidates.extend(candidates)

    # Step 6: assign unique IDs and write to JSONL
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for i, candidate in enumerate(all_candidates, start=1):
            candidate["id"] = f"q_{i:03d}"  # e.g. q_001, q_042
            f.write(json.dumps(candidate) + "\n")

    print(f"\nWrote {len(all_candidates)} candidates to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
