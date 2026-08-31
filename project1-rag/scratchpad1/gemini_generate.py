"""
gemini_generate.py — text generation helper with retry-on-transient-failure.

Retries only on rate limits (429) and server overload (503).
Does NOT retry on 400 (bad request) or auth errors — those aren't transient.
"""

import time

from google import genai
from google.genai import errors

# Which HTTP codes are worth retrying.
RETRYABLE_CODES = {429, 503}
MAX_ATTEMPTS = 3


def generate_context(
    client: genai.Client,
    prompt: str,
    model: str = "gemini-3.5-flash-lite",
) -> str:
    """Generate text for a prompt with retry on transient errors.

    Raises the final exception if all attempts fail.
    """
    last_error: Exception | None = None

    for attempt in range(MAX_ATTEMPTS):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )
            return response.text.strip()

        except (errors.ClientError, errors.ServerError) as e:
            # Only retry transient failures.
            # For non-retryable errors such as 400 or auth failures,
            # immediately propagate the exception.
            if e.code not in RETRYABLE_CODES:
                raise

            last_error = e

            # Exponential backoff: 5s, 15s, 45s
            wait = 5 * (3 ** attempt)
            print(
                f"  [retry] attempt {attempt + 1} failed "
                f"(code {e.code}): waiting {wait}s"
            )
            time.sleep(wait)

    # All attempts failed — raise the last error we saw
    assert last_error is not None
    raise last_error