import os

import psycopg
from dotenv import load_dotenv
from gemini_generate import retry_on_transient
from google import genai
from google.genai import types
from pgvector.psycopg import register_vector

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
TOP_K = 3


@retry_on_transient
def embed_query_batch(
    client: genai.Client,
    queries: list[str],
) -> list[list[float]]:
    """
    Embed a list of user queries in ONE API call with RETRIEVAL_QUERY task type.
    Each embedding is MRL-truncated to EMBED_DIM.

    Returns a list of vectors in the SAME ORDER as the input queries.
    """
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=queries,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )

    vectors = [embedding.values[:EMBED_DIM] for embedding in result.embeddings]

    assert all(len(v) == EMBED_DIM for v in vectors), (
        f"Embedding dim mismatch on batch of {len(queries)}"
    )

    return vectors


def embed_query(client: genai.Client, query_text: str) -> list[float]:
    """
    Convenience wrapper: embed a single query.

    Thin wrapper over embed_query_batch so single-query and batch-query
    callers share the same underlying logic and retry behavior.
    """
    return embed_query_batch(client, [query_text])[0]


def search(conn: psycopg.Connection, query_vector: list[float], k: int):
    """Return top-k documents ranked by cosine distance to query_vector."""
    with conn.cursor() as cur:
        cur.execute(
            """
            select id, source_doc, page, chunk_id, content,
                   embedding <=> %s::vector as distance
            from documents
            order by distance
            limit %s
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
    """Embed a query and return top-k chunks from the DB. Caller owns client + conn lifecycle."""
    # TODO 1: embed the query using embed_query()
    query_vector = embed_query(client, query)

    # TODO 2: call search() with the vector and k
    results = search(conn, query_vector, k)
    # TODO 3: return the results
    return results


def main():
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set in .env")
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set in .env")

    client = genai.Client(api_key=GOOGLE_API_KEY)

    query = "What did Warren Buffett say about Charlie Munger?"
    print(f"Query: {query!r}\n")

    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)
        results = retrieve(client, conn, query, TOP_K)

    if not results:
        print("No documents found. Is the documents table populated?")
        return

    for rank, (doc_id, source, page, chunk_id, content, distance) in enumerate(
        results, start=1
    ):
        similarity = 1 - distance
        source_label = (
            f"{source} p.{page} chunk {chunk_id}" if source else f"id={doc_id}"
        )
        print(
            f"#{rank}  {source_label}  distance={distance:.4f}  similarity={similarity:.4f}"
        )
        print(f"     {content[:250]}...\n")


if __name__ == "__main__":
    main()
