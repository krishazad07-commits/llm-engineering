"""Throwaway debug script — compare vector vs BM25 rankings on one query."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import psycopg
from dotenv import load_dotenv
from google import genai
from hybrid_search import build_bm25_index, search_bm25
from pgvector.psycopg import register_vector
from search_documents import retrieve

load_dotenv()

QUERY = "In what year did Ajit Jain join Berkshire?"


def main():
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    with psycopg.connect(os.getenv("DATABASE_URL")) as conn:
        register_vector(conn)
        bm25, chunks = build_bm25_index(conn)

        vector_results = retrieve(client, conn, QUERY, 10)
        bm25_results = search_bm25(QUERY, bm25, chunks, 10)

    print(f"Query: {QUERY}\n")

    print("=== VECTOR ===")
    for rank, r in enumerate(vector_results, 1):
        print(f"{rank}. id={r[0]} page={r[2]} {r[4][:80]}")

    print("\n=== BM25 ===")
    for rank, r in enumerate(bm25_results, 1):
        print(f"{rank}. id={r[0]} page={r[2]} {r[4][:80]}")


if __name__ == "__main__":
    main()