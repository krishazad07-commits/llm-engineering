import asyncio
import os
import time
from pathlib import Path

import psycopg
from chunker import chunk_text
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pdf_ingest import extract_pages
from pgvector.psycopg import register_vector

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
BATCH_SIZE = 10
MAX_CONCURRENT_BATCHES = 3

PDF_PATH = Path(
    "C:\\Users\\Krish\\Documents\\code\\llm-engineering\\project1-rag\\data\\berkshire_2023.pdf"
)
SOURCE_DOC = "berkshire_2023"


async def embed_batch(
    client: genai.Client,
    texts: list[str],
    semaphore: asyncio.Semaphore,
    batch_idx: int,
) -> list[list[float]]:
    """Embed one batch of texts, capped by semaphore."""
    async with semaphore:
        print(f"  [batch {batch_idx}] embedding {len(texts)} chunks...")
        result = await client.aio.models.embed_content(
            model=EMBED_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        vectors = [e.values[:EMBED_DIM] for e in result.embeddings]
        for v in vectors:
            assert len(v) == EMBED_DIM, f"dim mismatch: got {len(v)}"
        print(f"  [batch {batch_idx}] done, got {len(vectors)} vectors")
        return vectors


async def embed_all_chunks(chunks: list[dict]) -> list[dict]:
    """Embed all chunks concurrently in batches. Returns chunks with 'embedding' added."""
    client = genai.Client(api_key=GOOGLE_API_KEY)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_BATCHES)

    # split chunks into batches
    batches = [chunks[i : i + BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)]
    print(
        f"Split {len(chunks)} chunks into {len(batches)} batches of up to {BATCH_SIZE}"
    )

    # fire all batches concurrently, bounded by semaphore
    tasks = [
        embed_batch(client, [c["text"] for c in batch], semaphore, idx)
        for idx, batch in enumerate(batches)
    ]
    batch_results = await asyncio.gather(*tasks, return_exceptions=True)

    # flatten and attach embeddings back to chunks
    enriched = []
    for batch, batch_vectors in zip(batches, batch_results):
        if isinstance(batch_vectors, Exception):
            print(
                f"  [warn] batch failed: {batch_vectors!r} — skipping {len(batch)} chunks"
            )
            continue
        for chunk, vector in zip(batch, batch_vectors):
            enriched.append({**chunk, "embedding": vector})

    return enriched


def insert_chunks(chunks: list[dict], source_doc: str):
    """Insert embedded chunks into Supabase, idempotent via ON CONFLICT."""
    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            rows = [
                (source_doc, c["page"], c["chunk_id"], c["text"], c["embedding"])
                for c in chunks
            ]
            cur.executemany(
                """
                insert into documents (source_doc, page, chunk_id, content, embedding)
                values (%s, %s, %s, %s, %s)
                on conflict (source_doc, page, chunk_id) do nothing
                """,
                rows,
            )
        conn.commit()
    print(f"Inserted {len(chunks)} rows (duplicates silently skipped).")


async def main():
    if not GOOGLE_API_KEY or not DATABASE_URL:
        raise RuntimeError("GOOGLE_API_KEY or DATABASE_URL not set in .env")
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found at {PDF_PATH}.")

    # 1. parse + chunk
    print(f"Parsing {PDF_PATH.name}...")
    pages = extract_pages(PDF_PATH)
    all_chunks = []
    for page in pages:
        page_chunks = chunk_text(page["text"], max_size=1000, overlap=150)
        for chunk_idx, text in enumerate(page_chunks):
            all_chunks.append(
                {
                    "page": page["page"],
                    "chunk_id": chunk_idx,
                    "text": text,
                }
            )
    print(f"Produced {len(all_chunks)} chunks from {len(pages)} pages.\n")

    # 2. embed (concurrently)
    print("Embedding chunks concurrently...")
    t_start = time.perf_counter()
    enriched = await embed_all_chunks(all_chunks)
    t_embed = time.perf_counter() - t_start
    print(f"\nEmbedded {len(enriched)}/{len(all_chunks)} chunks in {t_embed:.2f}s")
    print(f"  → {t_embed / len(enriched) * 1000:.0f}ms per chunk (wall clock)\n")

    # 3. insert into Supabase
    print("Inserting into Supabase...")
    t_start = time.perf_counter()
    insert_chunks(enriched, SOURCE_DOC)
    t_insert = time.perf_counter() - t_start
    print(f"Inserted in {t_insert:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
