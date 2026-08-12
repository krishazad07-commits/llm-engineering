import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import psycopg
from pgvector.psycopg import register_vector

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

EMBED_DIM = 768

documents = [
    "A team of criminals plans an elaborate diamond heist from a high-security vault.",
    "A courtroom drama where a young lawyer defends an innocent man accused of murder.",
    "Astronauts stranded on Mars must survive using scientific ingenuity and limited supplies.",
    "A world-class chef opens a small restaurant and rediscovers their love for cooking.",
    "How photosynthesis converts sunlight into chemical energy in plant cells.",
]

def embed_documents(client, texts):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    return [e.values[:EMBED_DIM] for e in result.embeddings]

def main():
    client = genai.Client(api_key=GOOGLE_API_KEY)
    
    print(f"Embedding {len(documents)} documents...")
    embeddings = embed_documents(client, documents)
    print(f"Got {len(embeddings)} embeddings, each with {len(embeddings[0])} dimensions.")
    
    rows = list(zip(documents, embeddings))
    
    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
                rows,
            )
        conn.commit()
    
    print(f"Inserted {len(rows)} rows into documents table.")

if __name__ == "__main__":
    main()