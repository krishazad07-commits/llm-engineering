import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

MODEL = "gemini-3.5-flash"
MAX_STEPS = 5


# ---------- The actual tool implementations (fake data) ----------


def get_weather(city: str) -> dict:
    fake_data = {
        "Ahmedabad": {"temp_c": 34, "condition": "sunny"},
        "Mumbai": {"temp_c": 30, "condition": "humid"},
        "Delhi": {"temp_c": 28, "condition": "hazy"},
    }
    return fake_data.get(city, {"error": f"No data for {city}"})


def get_time(timezone: str) -> str:
    fake_times = {
        "IST": "14:30",
        "UTC": "09:00",
        "PST": "01:00",
    }
    return fake_times.get(timezone, f"Unknown timezone: {timezone}")


def add(a: float, b: float) -> float:
    return a + b


# ---------- Tool registry: name → callable ----------

TOOL_REGISTRY = {
    "get_weather": get_weather,
    "get_time": get_time,
    "add": add,
}


# ---------- Tool schemas the model sees ----------

TOOLS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="get_weather",
                description="Get the current weather for a given city. Returns temperature in Celsius and general conditions.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "city": types.Schema(
                            type=types.Type.STRING,
                            description="Name of the city, e.g. 'Ahmedabad'",
                        ),
                    },
                    required=["city"],
                ),
            ),
            types.FunctionDeclaration(
                name="get_time",
                description="Get the current time in a given timezone. Supported: IST, UTC, PST.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "timezone": types.Schema(
                            type=types.Type.STRING,
                            description="Timezone code, e.g. 'IST'",
                        ),
                    },
                    required=["timezone"],
                ),
            ),
            types.FunctionDeclaration(
                name="add",
                description="Add two numbers together and return the sum.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "a": types.Schema(
                            type=types.Type.NUMBER, description="First number"
                        ),
                        "b": types.Schema(
                            type=types.Type.NUMBER, description="Second number"
                        ),
                    },
                    required=["a", "b"],
                ),
            ),
        ]
    )
]


# ---------- Dispatch: run a tool by name ----------


def execute_tool(name: str, args: dict):
    if name not in TOOL_REGISTRY:
        return {"error": f"Unknown tool: {name}"}
    try:
        return TOOL_REGISTRY[name](**args)
    except Exception as e:  # noqa: BLE001 — intentional at tool-dispatch boundary
        return {"error": f"Tool {name} failed: {type(e).__name__}: {e}"}


# ---------- The agent loop ----------


def run_agent(client: genai.Client, user_query: str) -> str:
    history: list[types.Content] = [
        types.Content(role="user", parts=[types.Part.from_text(text=user_query)])
    ]

    for step in range(1, MAX_STEPS + 1):
        print(f"\n--- Step {step} ---")

        response = client.models.generate_content(
            model=MODEL,
            contents=history,
            config=types.GenerateContentConfig(tools=TOOLS),
        )

        model_content = response.candidates[0].content
        history.append(model_content)

        function_calls = [
            p.function_call for p in model_content.parts if p.function_call
        ]

        if not function_calls:
            final_text = "".join(p.text for p in model_content.parts if p.text)
            print(f"Model finished. Final answer:\n{final_text}")
            return final_text

        print(f"Model requested {len(function_calls)} tool call(s):")
        tool_response_parts = []
        for call in function_calls:
            print(f"  → {call.name}({dict(call.args)})")
            result = execute_tool(call.name, dict(call.args))
            print(f"    result: {result}")
            tool_response_parts.append(
                types.Part.from_function_response(
                    name=call.name, response={"result": result}
                )
            )

        history.append(types.Content(role="user", parts=tool_response_parts))

    return f"[MAX_STEPS={MAX_STEPS} reached without final answer]"


def main():
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set in .env")

    client = genai.Client(api_key=GOOGLE_API_KEY)

    query = "What's the weather in Ahmedabad and Mumbai, and what's 47 + 89?"
    print(f"Query: {query!r}")

    answer = run_agent(client, query)
    print(f"\n=== FINAL ANSWER ===\n{answer}")


if __name__ == "__main__":
    main()
