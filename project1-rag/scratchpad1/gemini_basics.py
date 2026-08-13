import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents="In one sentence, what is retrieval-augmented generation?",
)

print(response.text)

stream = client.models.generate_content_stream(
    model="gemini-3.5-flash",
    contents="Explain vector databases in about 300 words.",
)

for i, chunk in enumerate(stream):
    print(f"--- chunk {i} ---")
    print(repr(chunk.text))