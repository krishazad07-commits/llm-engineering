import os
from dotenv import load_dotenv
import psycopg
from pgvector.psycopg import register_vector

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def main():
    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)

        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            result = cur.fetchone()
            print("Connected to:", result[0])

if __name__ == "__main__":
    main()