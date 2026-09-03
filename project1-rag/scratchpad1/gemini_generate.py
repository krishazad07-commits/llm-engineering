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
from groq import APIError as GroqAPIError
from groq import Groq
from groq import RateLimitError as GroqRateLimitError

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

            except (
                errors.ClientError,
                errors.ServerError,
                GroqAPIError,
                GroqRateLimitError,
            ) as e:
                # Normalize error code across providers.
                # Gemini errors use .code; Groq errors use .status_code.
                code = getattr(e, "code", None) or getattr(e, "status_code", None)

                # Only retry transient errors.
                # Non-retryable errors (400, auth) propagate immediately.
                if code not in RETRYABLE_CODES:
                    raise

                last_error = e

                # Exponential backoff: 5s, 15s, 45s.
                wait = 5 * (3**attempt)

                print(
                    f"  [retry] {fn.__name__} attempt {attempt + 1} failed "
                    f"(code {code}): waiting {wait}s"
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


# ---------------------------------------------------------------------------
# Groq generation — provider swap for text generation
# ---------------------------------------------------------------------------

GROQ_MODEL = "openai/gpt-oss-120b"


@retry_on_transient
def generate_with_groq(
    client: Groq,
    prompt: str,
    system_instruction: str | None = None,
    model: str = GROQ_MODEL,
) -> str:
    """
    Generate text via Groq/Llama. Provider-agnostic retry via @retry_on_transient.

    Same interface shape as generate_context so callers can swap providers
    by changing one function call, not their whole flow.
    """

    # Build messages list.
    messages = []

    if system_instruction is not None:
        messages.append(
            {
                "role": "system",
                "content": system_instruction,
            }
        )

    messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    # Call Groq chat completion.
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )

    # Extract generated text.
    return response.choices[0].message.content.strip()
