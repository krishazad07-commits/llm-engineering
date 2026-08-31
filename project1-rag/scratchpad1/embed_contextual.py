"""
embed_contextual.py — embed (context + content) for every row in
documents_contextual and write the vector to the embedding column.

Resume-safe: skips rows that already have an embedding.
Uses Gemini's batch embedding API for speed.

Usage: uv run scratchpad1/embed_contextual.py
"""

import os

import psycopg
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pgvector.psycopg import register_vector

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
BATCH_SIZE = 20


def load_pending_rows(conn: psycopg.Connection) -> list[tuple]:
    """Load rows from documents_contextual that don't yet have an embedding.

    Returns list of (id, context, content).
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, context, content
            FROM documents_contextual
            WHERE embedding IS NULL
            ORDER BY id
            """
        )
        return cur.fetchall()


def build_situated_text(context: str, content: str) -> str:
    """Prepend context to content — Decision 1A."""
    return context + "\n\n" + content


def embed_batch(
    client: genai.Client,
    texts: list[str],
) -> list[list[float]]:
    """Embed a batch of texts with MRL truncation to 768 dims.

    Returns list of embedding vectors, one per input text, in order.
    """
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=EMBED_DIM,
        ),
    )

    return [emb.values for emb in response.embeddings]


def update_embedding(
    conn: psycopg.Connection,
    row_id: int,
    embedding: list[float],
) -> None:
    """Write the embedding to the row with matching id."""
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE documents_contextual
            SET embedding = %s
            WHERE id = %s
            """,
            (embedding, row_id),
        )
    conn.commit()


def main():
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set in .env")
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set in .env")

    client = genai.Client(api_key=GOOGLE_API_KEY)

    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)

        rows = load_pending_rows(conn)
        print(f"Loaded {len(rows)} rows pending embedding")

        # Process in batches
                # Process in batches
        for batch_start in range(0, len(rows), BATCH_SIZE):
            batch = rows[batch_start:batch_start + BATCH_SIZE]

            # Build situated texts for this batch
            texts = [
                build_situated_text(context, content)
                for _row_id, context, content in batch
            ]

            print(
                f"  Embedding batch "
                f"{batch_start // BATCH_SIZE + 1} ({len(batch)} rows)..."
            )

            embeddings = embed_batch(client, texts)

            # Write each embedding back to its row
            for (row_id, _ctx, _cnt), emb in zip(
                batch,
                embeddings,
                strict=True,
            ):
                update_embedding(conn, row_id, emb)

            print(
                f"  ✓ Batch done. "
                f"{batch_start + len(batch)}/{len(rows)}"
            )


    print("\n✓ All embeddings written.")


if __name__ == "__main__":
    main()