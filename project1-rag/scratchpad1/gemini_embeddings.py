import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import math
load_dotenv()
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

result = client.models.embed_content(
    model="gemini-embedding-001",
    contents="A group of thieves plans an elaborate bank heist.",
)

embedding = result.embeddings[0]
print(type(embedding))
print(len(embedding.values))
print(embedding.values[:10])

text = "A group of thieves plans an elaborate bank heist."

doc_embedding = client.models.embed_content(
    model="gemini-embedding-001",
    contents=text,
    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
).embeddings[0]

query_embedding = client.models.embed_content(
    model="gemini-embedding-001",
    contents=text,
    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
).embeddings[0]

print(doc_embedding.values[:5])
print(query_embedding.values[:5])
print(doc_embedding.values[:5] == query_embedding.values[:5])

def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(y * y for y in b))
    return dot_product / (magnitude_a * magnitude_b)

similarity = cosine_similarity(doc_embedding.values, query_embedding.values)
print(f"same sentence, doc vs query embedding: {similarity:.4f}")

unrelated = client.models.embed_content(
    model="gemini-embedding-001",
    contents="Photosynthesis converts sunlight into chemical energy in plants.",
    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
).embeddings[0]

similarity_unrelated = cosine_similarity(doc_embedding.values, unrelated.values)
print(f"heist sentence vs photosynthesis sentence: {similarity_unrelated:.4f}")

documents = [
    "A group of thieves plans an elaborate bank heist.",
    "Astronauts discover a mysterious signal from deep space.",
    "A jury deliberates the fate of a man accused of murder.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "A chef prepares a five-course tasting menu in a busy kitchen.",
]

doc_embeddings = [
    client.models.embed_content(
        model="gemini-embedding-001",
        contents=doc,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    ).embeddings[0]
    for doc in documents
]

query = "Tell me about a crime and robbery movie."
query_embedding = client.models.embed_content(
    model="gemini-embedding-001",
    contents=query,
    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
).embeddings[0]

scored = [
    (doc, cosine_similarity(query_embedding.values, doc_emb.values))
    for doc, doc_emb in zip(documents, doc_embeddings)
]

scored.sort(key=lambda pair: pair[1], reverse=True)

print(f"Query: {query}\n")
for doc, score in scored:
    print(f"{score:.4f}  {doc}")