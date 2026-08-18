"""
rag.py — full RAG pipeline: question in, grounded answer out.
"""

import os

import psycopg
from dotenv import load_dotenv
from generate_answer import RetrievedChunk, generate_answer
from google import genai
from pgvector.psycopg import register_vector
from search_documents import embed_query, search

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
GEN_MODEL = "gemini-3.6-flash"


def rows_to_chunks(rows) -> list[RetrievedChunk]:
    """Convert search() tuples into RetrievedChunk dataclass instances."""
    return [
        RetrievedChunk(
            chunk_id=f"{source}_p{page}_c{chunk_idx}",
            source_doc=source,
            page=page,
            content=content,
            similarity=1 - distance,
        )
        for (doc_id, source, page, chunk_idx, content, distance) in rows
    ]


def ask(client, conn, question: str, k: int = 5) -> str:
    query_vector = embed_query(client, question)
    rows = search(conn, query_vector, k)
    chunks = rows_to_chunks(rows)

    # DEBUG — print what we retrieved
    print(f"\n--- Retrieved {len(chunks)} chunks for: {question!r} ---")
    for i, c in enumerate(chunks, 1):
        print(f"  #{i} sim={c.similarity:.4f}  [{c.chunk_id}]")
        print(f"      {c.content[:120]}...")
    print("--- End retrieval ---\n")

    answer = generate_answer(client, question, chunks)
    return answer


def main():
    # Step 1: env var guards (copy from search_documents.py main)
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set in .env")
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set in .env")

    # Step 2: build the Gemini client
    client = genai.Client(api_key=GOOGLE_API_KEY)

    # Step 3: define the question you're asking
    question = "List the exact float figures from the insurance operations table."
    print(f"question: {question!r}\n")
    # Step 4: open DB connection, register pgvector type, call ask()
    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)
        answer = ask(client, conn, question, k=5)

    # Step 5: pretty-print the result
    print(f"Q: {question}\n")
    print(f"A: {answer}")


if __name__ == "__main__":
    main()
