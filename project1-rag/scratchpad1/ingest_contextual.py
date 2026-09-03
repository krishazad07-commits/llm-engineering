"""
ingest_contextual.py — generate context for every chunk, insert into
documents_contextual. Leaves `embedding` NULL; embedding is a separate step.

Resume-safe: skips chunks already present in documents_contextual.
Retries transient Gemini failures via gemini_generate.py.

Usage: uv run scratchpad1/ingest_contextual.py
"""

import os
import time

import psycopg
from dotenv import load_dotenv
from gemini_generate import generate_context
from google import genai

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

# Rate-limit spacing: 15 RPM = 1 call per 4s. Add margin.
SLEEP_BETWEEN_CALLS = 4.5


PROMPT_TEMPLATE = """<document>
CHUNK BEFORE:
{before}

CURRENT CHUNK:
{current}

CHUNK AFTER:
{after}
</document>

<chunk_to_situate>
{current}
</chunk_to_situate>

Write a 1-2 sentence context explaining what this chunk covers and how it fits into the surrounding document, so that a search system can better find this chunk. Keep it under 50 words. Output only the context — no preamble or explanation."""


def load_all_chunks(conn: psycopg.Connection) -> list[tuple]:
    """Load every chunk from documents, ordered by (source_doc, chunk_id).

    Returns list of tuples: (id, source_doc, page, chunk_id, content).
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, source_doc, page, chunk_id, content
            FROM documents
            ORDER BY source_doc, page, chunk_id
            """
        )
        return cur.fetchall()


def load_done_chunk_ids(conn: psycopg.Connection) -> set[int]:
    """Return the set of `id` values already in documents_contextual.

    Used for resume-safety — we skip chunks we've already processed.
    We match on `id` because documents.id is what gets copied over.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id
            FROM documents_contextual
            """
        )
        return {row[0] for row in cur.fetchall()}


def build_prompt(
    before: tuple | None,
    current: tuple,
    after: tuple | None,
) -> str:
    """Assemble the situating prompt for one chunk.

    `before` / `after` may be None at the document edges.
    Chunk tuples are (id, source_doc, page, chunk_id, content).
    """
    before_text = before[4] if before is not None else "[No previous chunk]"
    current_text = current[4]
    after_text = after[4] if after is not None else "[No next chunk]"

    return PROMPT_TEMPLATE.format(
        before=before_text,
        current=current_text,
        after=after_text,
    )


def insert_contextual(
    conn: psycopg.Connection,
    original_id: int,
    source_doc: str,
    page: int,
    chunk_id: int,
    content: str,
    context: str,
) -> None:
    """Insert one row into documents_contextual. Embedding left NULL."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents_contextual
                (id, source_doc, page, chunk_id, content, context)
            VALUES
                (%s, %s, %s, %s, %s, %s)
            """,
            (
                original_id,
                source_doc,
                page,
                chunk_id,
                content,
                context,
            ),
        )

    conn.commit()


def main():
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set in .env")
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set in .env")

    client = genai.Client(api_key=GOOGLE_API_KEY)

    with psycopg.connect(DATABASE_URL) as conn:
        all_chunks = load_all_chunks(conn)
        done_ids = load_done_chunk_ids(conn)

        print(
            f"Loaded {len(all_chunks)} chunks, "
            f"{len(done_ids)} already done, "
            f"{len(all_chunks) - len(done_ids)} to process"
        )

        for i, chunk in enumerate(all_chunks):
            chunk_id_pk = chunk[0]

            # Skip if already done (resume-safety).
            if chunk_id_pk in done_ids:
                print(f"[{i + 1}/{len(all_chunks)}] id={chunk_id_pk} — skip (done)")
                continue

            # Find neighbors by list position (Q1 approach A).
            before = all_chunks[i - 1] if i > 0 else None
            after = all_chunks[i + 1] if i < len(all_chunks) - 1 else None

            prompt = build_prompt(
                before=before,
                current=chunk,
                after=after,
            )

            context = generate_context(
                client=client,
                prompt=prompt,
                model="gemini-3.5-flash-lite",
            )

            insert_contextual(
                conn,
                original_id=chunk_id_pk,
                source_doc=chunk[1],
                page=chunk[2],
                chunk_id=chunk[3],
                content=chunk[4],
                context=context,
            )

            print(
                f"[{i + 1}/{len(all_chunks)}] id={chunk_id_pk} "
                f"page={chunk[2]} — done "
                f"(context {len(context)} chars)"
            )

            # Rate-limit spacing.
            # Rate-limit spacing.
            time.sleep(SLEEP_BETWEEN_CALLS)

    print("\n✓ All chunks processed.")


if __name__ == "__main__":
    main()
