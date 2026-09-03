"""
search_documents_contextual.py — vector retrieval against the situated
(context + content) embeddings in documents_contextual.

Same signature as search_documents.retrieve() so it's a drop-in swap
in the eval harness.
"""

import psycopg
from google import genai
from search_documents import embed_query  # reuse — query side is unchanged


def search(conn: psycopg.Connection, query_vector: list[float], k: int):
    """Return top-k rows from documents_contextual ranked by cosine distance."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, source_doc, page, chunk_id, content,
                   embedding <=> %s::vector AS distance
            FROM documents_contextual
            WHERE embedding IS NOT NULL
            ORDER BY distance
            LIMIT %s
            """,
            (query_vector, k),
        )
        return cur.fetchall()


def retrieve(
    client: genai.Client,
    conn: psycopg.Connection,
    query: str,
    k: int = 10,
) -> list[tuple]:
    """Embed a query and return top-k chunks from documents_contextual."""
    query_vector = embed_query(client, query)
    return search(conn, query_vector, k)
