import os

from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

prompts = [
    "hi",
    "Explain retrieval-augmented generation in one sentence.",
    "Explain retrieval-augmented generation in one sentence. " * 20,
]

for p in prompts:
    result = client.models.count_tokens(
        model="gemini-3.5-flash",
        contents=p,
    )
    print(f"chars={len(p):5d}  tokens={result.total_tokens}  preview={p[:50]!r}")