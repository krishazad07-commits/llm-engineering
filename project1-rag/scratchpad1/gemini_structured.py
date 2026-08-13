import asyncio
import os
from datetime import UTC, datetime

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field, ValidationError

load_dotenv()
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

class MovieRecommendation(BaseModel):
    title: str = Field(min_length=1, max_length=100)
    year: int = Field(ge=1888, le=datetime.now(tz=UTC).year + 1)
    one_line_reason: str = Field(min_length=1, max_length=500)

async def recommend_movie(topic: str) -> MovieRecommendation | None:
    prompt = f"Recommend one great movie about: {topic}."
    
    # Pre-flight: count tokens before we send. Cheap habit that saves you later.
    token_count = await client.aio.models.count_tokens(
        model="gemini-3.5-flash",
        contents=prompt,
    )
    print(f"[{topic}] prompt tokens: {token_count.total_tokens}")

    try:
        response = await client.aio.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=MovieRecommendation,
            ),
        )
        return response.parsed
    except errors.APIError as e:
        print(f"[{topic}] API error: {e}")
        return None
    except ValidationError as e:
        print(f"[{topic}] validation error: {e}")
        return None

async def main() -> None:
    topics = [
        "a heist movie",
        "a slow sci-fi",
        "a courtroom drama",
    ]

    results = await asyncio.gather(
        *(recommend_movie(t) for t in topics),
        return_exceptions=True,  
    )

    print("\n--- results ---")
    for topic, result in zip(topics, results):
        if isinstance(result, Exception):
            print(f"[{topic}] crashed: {type(result).__name__}: {result}")
        elif result is None:
            print(f"[{topic}] no recommendation (see error above)")
        else:
            print(f"[{topic}] {result.title} ({result.year}) — {result.one_line_reason}")


if __name__ == "__main__":
    asyncio.run(main())

