import os

from dotenv import load_dotenv
from gemini_generate import generate_context
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

prompt = "Say hello in exactly 5 words."
result = generate_context(client, prompt)
print(f"Result: {result!r}")