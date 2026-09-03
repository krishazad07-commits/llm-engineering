"""Print the full content of one chunk by DB id."""

import os
import sys

import psycopg
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.environ["DATABASE_URL"]


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: uv run show_chunk.py <id>")
        sys.exit(1)

    chunk_id = int(sys.argv[1])

    with psycopg.connect(DATABASE_URL) as conn, conn.cursor() as cur:
        cur.execute(
            "select id, page, content from documents where id = %s",
            (chunk_id,),
        )
        row = cur.fetchone()

    if row is None:
        print(f"no chunk with id={chunk_id}")
        return

    print(f"id={row[0]}  page={row[1]}")
    print("-" * 80)
    print(row[2])


if __name__ == "__main__":
    main()
