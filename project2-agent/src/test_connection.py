import os
from pathlib import Path
from dotenv import load_dotenv
import psycopg

env_path = Path(r"C:\Users\Krish\Documents\code\llm-engineering\.env").resolve()
print(f"Looking for .env at: {env_path}")
print(f".env exists there?  {env_path.exists()}")

loaded = load_dotenv(dotenv_path=env_path)
print(f"load_dotenv returned: {loaded}")

print("SUPABASE_* keys loaded:", [k for k in os.environ if k.startswith("SUPABASE")])

db_url = os.getenv("SUPABASE_DB_URL_P2")
print(f"SUPABASE_DB_URL_P2 raw value: {repr(db_url)}")

if db_url is None:
    raise SystemExit("Stopping — variable didn't load. See prints above.")

with psycopg.connect(db_url) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM customers;")
        print(f"Customer count: {cur.fetchone()[0]}")