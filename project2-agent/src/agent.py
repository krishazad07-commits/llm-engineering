"""
Agent loop — the Day 1 hand-rolled version.

Takes a user question, sends it to Groq with our six tool schemas,
runs any tools Groq asks for, feeds results back, repeats until Groq
returns a final text answer or MAX_STEPS is hit.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from tools import (
    find_customer,
    get_invoice,
    list_customer_invoices,
    get_customer,
    search_invoices,
    list_customer_tickets,
    issue_refund,
    execute_refund,   # ← add
    reject_refund,    # ← add
)
from tool_schemas import ALL_TOOLS

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "openai/gpt-oss-120b"
MAX_STEPS = 10
DESTRUCTIVE_TOOLS = {"issue_refund"}
# Map tool names (as the LLM sees them) → the actual Python functions
TOOL_REGISTRY = {
    "find_customer": find_customer,
    "get_invoice": get_invoice,
    "list_customer_invoices": list_customer_invoices,
    "get_customer": get_customer,
    "search_invoices": search_invoices,
    "list_customer_tickets": list_customer_tickets,
    "issue_refund": issue_refund,
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
        fn = TOOL_REGISTRY.get(name)

        # LLMs occasionally hallucinate tool names
        if fn is None:
            return f"Error: unknown tool '{name}'"

        # Call the function by unpacking the arguments dict.
        # Example: {"query": "GrubMatch"} → fn(query="GrubMatch")
        result = fn(**arguments)

        # Serialize the result to a JSON string so we can send it back as tool content
        return json.dumps(result)

    except Exception as e:
        # Return the error as a string, don't raise. The LLM needs to see it.
        return f"Error: {type(e).__name__}: {e}"

def handle_destructive_confirmation(
    name: str,
    arguments: dict,
    proposal_result: str,
) -> str:
    """
    Show the proposed destructive action to the user, wait for y/n,
    and either execute or reject via the appropriate helper.

    Returns the final tool result string to hand back to the model —
    reflecting execution, rejection, or a no-op if the proposal was
    already handled.
    """

    # TODO 1: parse proposal_result as JSON
    # If it's not JSON (i.e. the tool errored during proposal),
    # return it unchanged — nothing to confirm.

    try:
        proposal_dict = json.loads(proposal_result)
    except (json.JSONDecodeError, TypeError):
        return proposal_result

    # TODO 2: if the proposal came back as 'already_executed',
    # return proposal_result unchanged. No gate needed.

    if proposal_dict.get("status") == "already_executed":
        return proposal_result

    # TODO 3: print a confirmation prompt showing:
    # - tool name
    # - arguments (invoice_id, amount, reason)
    # - refund_id from the proposal

    print("\n" + "=" * 60)
    print("DESTRUCTIVE ACTION — CONFIRMATION REQUIRED")
    print("=" * 60)
    print(f"Tool:      {name}")
    print(f"Invoice:   {arguments.get('invoice_id')}")
    print(f"Amount:    {arguments.get('amount')}")
    print(f"Reason:    {arguments.get('reason')}")
    print(f"Refund ID: {proposal_dict.get('refund_id')}")
    print("=" * 60)

    user_input = input("Approve? [y/N]: ").strip().lower()
    print(f"→ {'APPROVED' if user_input == 'y' else 'REJECTED'}\n")

    # TODO 4: dispatch based on name and user_input

    if name == "issue_refund":
        if user_input == "y":
            final = execute_refund(proposal_dict["refund_id"])
        else:
            final = reject_refund(proposal_dict["refund_id"])
        return json.dumps(final)

    # Unknown destructive tool — shouldn't happen if DESTRUCTIVE_TOOLS
    # and this dispatch stay in sync. Fail loudly rather than silently
    # returning None.
    raise ValueError(
        f"Tool '{name}' is in DESTRUCTIVE_TOOLS but has no confirmation "
        f"handler. Add a branch to handle_destructive_confirmation."
    )

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

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=ALL_TOOLS,
        )
        msg = response.choices[0].message

        # Append the assistant's message to the history.
        # Groq returns a Pydantic-like object; convert to dict for consistency.
        messages.append(msg.model_dump(exclude_none=True))

        # If msg.tool_calls is None or empty, we're done — return msg.content.
        if not msg.tool_calls:
            if verbose:
                print(f"Final answer: {msg.content}")
            return msg.content

        # For each tool call, run it and append the result as a "tool" message.
        # Each tool result MUST include tool_call_id matching the one the LLM sent.
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            if verbose:
                print(f"  Tool call: {name}({arguments})")

            result = run_tool(name, arguments)

            # HITL gate for destructive tools
            if name in DESTRUCTIVE_TOOLS:
                result = handle_destructive_confirmation(name, arguments, result)

            if verbose:
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
    question = (
        "Issue a refund for KrishTech Solutions' most recent invoice. "
        "Customer says they were double-charged."
    )
    answer = run_agent(question)
    print(f"\n=== FINAL ===\n{answer}")