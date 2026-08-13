import os

from dotenv import load_dotenv
from google import genai
from google.genai import errors

load_dotenv()
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

def ask_gemini(prompt: str, model: str = "gemini-3.5-flash") -> str | None:
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        return response.text
    except errors.ClientError as e:
        print(f"[client error] your request was bad: {e.code} — {e.message}")
        return None
    except errors.ServerError as e:
        print(f"[server error] Gemini's fault, already retried: {e.code} — {e.message}")
        return None
    except errors.APIError as e:
        print(f"[api error] something else went wrong: {e}")
        return None

# Test 1: the broken call from before
print("--- Test 1: bad model name ---")
result = ask_gemini("hello", model="gemini-does-not-exist")
print("returned:", result)

# Test 2: normal call, should work
print("\n--- Test 2: normal call ---")
result = ask_gemini("Say 'hi' in exactly one word.")
print("returned:", result)