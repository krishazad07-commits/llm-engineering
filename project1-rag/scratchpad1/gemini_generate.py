"""
gemini_generate.py — generic retry wrapper for Gemini API calls.

Provides @retry_on_transient decorator that retries any function on
429 (rate limit) and 503 (server overload). Non-retryable errors
(400, auth) propagate immediately.

Also exports generate_context() as a convenience for text generation.
"""

import functools
import time
from collections.abc import Callable

from google import genai
from google.genai import errors

RETRYABLE_CODES = {429, 503}
MAX_ATTEMPTS = 3


def retry_on_transient[T](fn: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator: retry the wrapped function on transient Gemini errors.

    Retries on 429/503, raises immediately on other errors.
    Exponential backoff: 5s, 15s, 45s.

    Usage:
        @retry_on_transient
        def my_api_call(...):
            return client.models.something(...)
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs) -> T:
        last_error: Exception | None = None

        for attempt in range(MAX_ATTEMPTS):
            try:
                # Call the wrapped function with all arguments.
                return fn(*args, **kwargs)

            except (errors.ClientError, errors.ServerError) as e:
                # Only retry transient errors.
                # Other errors such as 400 or authentication errors
                # should immediately propagate to the caller.
                if e.code not in RETRYABLE_CODES:
                    raise

                last_error = e

                # Exponential backoff: 5s, 15s, 45s.
                wait = 5 * (3 ** attempt)

                print(
                    f"  [retry] {fn.__name__} attempt {attempt + 1} failed "
                    f"(code {e.code}): waiting {wait}s"
                )

                time.sleep(wait)

        # All attempts exhausted — raise the last error we saw.
        assert last_error is not None
        raise last_error

    return wrapper


# ---------------------------------------------------------------------------
# Convenience function for text generation — thin wrapper using the decorator
# ---------------------------------------------------------------------------
@retry_on_transient
def generate_context(
    client: genai.Client,
    prompt: str,
    model: str = "gemini-3.5-flash-lite",
    system_instruction: str | None = None,
) -> str:
    """Generate text for a prompt. Automatically retries on transient errors."""

    config = None

    if system_instruction is not None:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0,
        )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )

    return response.text.strip()