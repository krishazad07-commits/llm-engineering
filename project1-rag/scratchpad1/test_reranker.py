"""Quick sanity check: does reranker reorder in a way that looks sensible?"""

import os

import psycopg
from dotenv import load_dotenv
from google import genai
from pgvector.psycopg import register_vector
from rerank_search import retrieve_reranked
from search_documents import retrieve

load_dotenv()

QUERY = "In what year did U.S. oil production fall to five million BOEPD?"  # pick one multi_hop question from golden_qa.jsonl


def show(label, results):
    print(f"\n=== {label} ===")
    for i, chunk in enumerate(results[:5], 1):
        chunk_id, _source, page, _chunk_idx, content, _distance = chunk
        preview = content[:80].replace("\n", " ")
        print(f"  {i}. id={chunk_id} p{page} | {preview}...")


client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
with psycopg.connect(os.getenv("DATABASE_URL")) as conn:
    register_vector(conn)
    vec = retrieve(client, conn, QUERY, k=10)
    rer = retrieve_reranked(client, conn, QUERY, k=10, wide_k=20)

show("VECTOR top 5", vec)
show("RERANKED top 5", rer)
