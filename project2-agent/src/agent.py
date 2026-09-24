"""
Agent loop — the Day 1 hand-rolled version.

Takes a user question, sends it to Groq with our three tool schemas,
runs any tools Groq asks for, feeds results back, repeats until Groq
returns a final text answer or MAX_STEPS is hit.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from tools import find_customer, get_invoice, list_customer_invoices
from tool_schemas import ALL_TOOLS

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "openai/gpt-oss-120b"
MAX_STEPS = 10

# Map tool names (as the LLM sees them) → the actual Python functions
TOOL_REGISTRY = {
    "find_customer": find_customer,
    "get_invoice": get_invoice,
    "list_customer_invoices": list_customer_invoices,
}

SYSTEM_PROMPT = (
    "You are a helpful assistant for a company's operations team. "
    "You can look up customer information and invoices using the tools provided. "
    "Answer the user's question by calling tools as needed, then producing a clear final answer. "
    "If a tool returns an error, read the error message and try a different approach. "
    "When you have enough information to answer, stop calling tools and just answer."
)


def run_tool(name: str, arguments: dict) -> str:
    """
    Execute a tool by name with the given arguments. Return the result as a string.

    Errors are caught and returned as strings so the LLM can see them and self-correct.
    """
    try:
        # TODO 1: look up the function in TOOL_REGISTRY by name
        fn = TOOL_REGISTRY.get(name)

        # TODO 2: if the tool name isn't in the registry, return an error string
        # (LLMs occasionally hallucinate tool names)
        if fn is None:
            return f"Error: unknown tool '{name}'"

        # TODO 3: call the function by unpacking the arguments dict.
        # Example: {"query": "GrubMatch"} → fn(query="GrubMatch")
        result = fn(**arguments)

        # TODO 4: serialize the result to a JSON string so we can send it back as tool content
        return json.dumps(result)

    except Exception as e:
        # Return the error as a string, don't raise. The LLM needs to see it.
        return f"Error: {type(e).__name__}: {e}"


def run_agent(question: str, verbose: bool = True) -> str:
    """
    Run the agent loop for a single question. Return the final text answer.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for step in range(MAX_STEPS):
        if verbose:
            print(f"\n--- Step {step + 1} ---")

        # TODO 5: call Groq. Pass model, messages, and tools=ALL_TOOLS.
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=ALL_TOOLS,
        )
        msg = response.choices[0].message

        # Append the assistant's message to the history.
        # Groq returns a Pydantic-like object; convert to dict for consistency.
        messages.append(msg.model_dump(exclude_none=True))

        # TODO 6: check if the LLM wants to call tools.
        # If msg.tool_calls is None or empty, we're done — return msg.content.
        if not msg.tool_calls:
            if verbose:
                print(f"Final answer: {msg.content}")
            return msg.content

        # TODO 7: for each tool call, run it and append the result as a "tool" message.
        # Each tool result MUST include tool_call_id matching the one the LLM sent.
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            if verbose:
                print(f"  Tool call: {name}({arguments})")

            result = run_tool(name, arguments)

            if verbose:
                # Truncate very long results in the log
                display = result if len(result) < 200 else result[:200] + "..."
                print(f"  → {display}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    # If we fell out of the loop, we hit MAX_STEPS
    return f"[MAX_STEPS ({MAX_STEPS}) reached without a final answer]"


if __name__ == "__main__":
    # The test task — same one you predicted 3 steps for
    question = "What is GrubMatch Foods' most recent invoice?"
    answer = run_agent(question)
    print(f"\n=== FINAL ===\n{answer}")
