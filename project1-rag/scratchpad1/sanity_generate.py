"""
sanity_generate.py — one-off smoke test for the generation pipeline.

Runs one answerable and one unanswerable question through
retrieve → generate and prints the results for eyeballing.
"""

import os

import psycopg
from dotenv import load_dotenv
from generate_answer import RetrievedChunk, generate_answer
from google import genai
from pgvector.psycopg import register_vector
from search_documents import retrieve

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

client = genai.Client(api_key=GOOGLE_API_KEY)

QUESTIONS = [
    ("ANSWERABLE",   "On what date did Berkshire's Japanese purchases begin?"),
    ("UNANSWERABLE", "What specific university did Ajit Jain attend for his undergraduate education in India?"),
]


def to_retrieved_chunks(raw_rows: list[tuple]) -> list[RetrievedChunk]:
    """Map retriever tuples to RetrievedChunk dataclasses.

    Retriever returns: (doc_id, source, page, chunk_id, content, distance)
    Dataclass wants:   (chunk_id, source_doc, page, content, similarity)

    Notes:
    - Drop `doc_id` (plumbing key, not needed for generation).
    - Convert cosine distance → similarity via (1 - distance).
    """
    chunks = []
    for row in raw_rows:
        _doc_id, source, page, chunk_id, content, distance = row
        chunks.append(RetrievedChunk(
            chunk_id=str(chunk_id),
            source_doc=source,
            page=page,
            content=content,
            similarity=1 - distance,
        ))
    return chunks


with psycopg.connect(DATABASE_URL) as conn:
    register_vector(conn)

    for label, question in QUESTIONS:
        print("=" * 70)
        print(f"[{label}] {question}")
        print("=" * 70)

        raw = retrieve(client, conn, question, k=5)
        chunks = to_retrieved_chunks(raw)

        print(f"\nRetrieved {len(chunks)} chunks. Top chunk id: "
              f"{chunks[0].source_doc}:p{chunks[0].page}:c{chunks[0].chunk_id}\n")

        answer = generate_answer(client, question, chunks)

        print("ANSWER:")
        print(answer)
        print()