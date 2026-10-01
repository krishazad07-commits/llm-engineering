"""
Isolated research subagent.

The subagent investigates a customer using read-only tools and returns
a short summary to the main agent. Raw tool results stay inside this
isolated loop.
"""

import json

from groq import Groq

from tools import (
    find_customer,
    get_invoice,
    list_customer_invoices,
    get_customer,
    search_invoices,
    list_customer_tickets,
)
from tool_schemas import ALL_TOOLS


# Same model/client setup as agent.py
client = Groq()
MODEL = "openai/gpt-oss-120b"

LAST_RUN_STATS: list[dict] = []
# ---- Which tools the subagent is allowed to use ----
# Read-only only. issue_refund / execute_refund / reject_refund
# are intentionally NOT available to the subagent.
SUBAGENT_TOOL_NAMES = {
    "find_customer",
    "get_invoice",
    "list_customer_invoices",
    "get_customer",
    "search_invoices",
    "list_customer_tickets",
}


# Map only approved tools to their Python functions.
SUBAGENT_REGISTRY = {
    "find_customer": find_customer,
    "get_invoice": get_invoice,
    "list_customer_invoices": list_customer_invoices,
    "get_customer": get_customer,
    "search_invoices": search_invoices,
    "list_customer_tickets": list_customer_tickets,
}


# ALL_TOOLS contains the schemas the main agent sees.
# Filter it so the subagent only receives read-only tools.
SUBAGENT_SCHEMAS = [
    schema
    for schema in ALL_TOOLS
    if schema["function"]["name"] in SUBAGENT_TOOL_NAMES
]


SUBAGENT_SYSTEM = """You are a research subagent.

Your job: investigate one customer and write a short written report for the
main agent. You have read-only access to customer, invoice, and ticket data.

Rules:
- Use the tools to gather what you need. Don't guess.
- When you have enough, STOP calling tools and write a 1–2 paragraph summary.
- The summary is the ONLY thing the main agent will see. Make it count.
- No bullet lists longer than 5 items.
- If something looks off (missing data, contradictions), say so in the summary.
"""


def subagent_loop(
    customer_id: str,
    question: str,
    max_steps: int = 10,
) -> dict:
    """
    Runs an isolated exploration loop.

    Returns:
      {
        "summary": str,
        "steps_used": int,
        "input_tokens": int,
        "output_tokens": int,
      }
    """

    messages = [
        {
            "role": "user",
            "content": (
                f"Customer ID: {customer_id}\n"
                f"Question: {question}"
            ),
        }
    ]

    total_in = 0
    total_out = 0

    for step in range(max_steps):

        # Send the entire isolated subagent conversation.
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SUBAGENT_SYSTEM,
                },
                *messages,
            ],
            tools=SUBAGENT_SCHEMAS,
        )

        # Track token usage across ALL internal subagent turns.
        if response.usage:
            total_in += response.usage.prompt_tokens or 0
            total_out += response.usage.completion_tokens or 0

        msg = response.choices[0].message

        # Store the assistant's message so the next subagent turn
        # can see its previous tool calls / reasoning output.
        messages.append(
            msg.model_dump(exclude_none=True)
        )

        # No tool calls means the subagent has produced its final report.
        if not msg.tool_calls:
            LAST_RUN_STATS.append({
                "steps": step + 1,
                "input_tokens": total_in,
                "output_tokens": total_out,
            })
            return {
                "summary": msg.content or "",
                "steps_used": step + 1,
                "input_tokens": total_in,
                "output_tokens": total_out,
            }

        # Execute every tool requested by the subagent.
        for tool_call in msg.tool_calls:

            name = tool_call.function.name

            # Safety check — the model must not escape the read-only set.
            if name not in SUBAGENT_REGISTRY:
                result = {
                    "error": f"Tool '{name}' is not available to the subagent."
                }

            else:
                try:
                    arguments = json.loads(
                        tool_call.function.arguments
                    )

                    fn = SUBAGENT_REGISTRY[name]

                    result = fn(**arguments)

                except json.JSONDecodeError as e:
                    result = {
                        "error": f"Invalid JSON arguments: {e}"
                    }

                except Exception as e:
                    result = {
                        "error": (
                            f"Error running {name}: "
                            f"{type(e).__name__}: {e}"
                        )
                    }

            # Tool messages must contain string content.
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, default=str),
                }
            )
        # Max steps reached without a final summary.
    LAST_RUN_STATS.append({
        "steps": max_steps,
        "input_tokens": total_in,
        "output_tokens": total_out,
    })
    return {
        "summary": (
            "Subagent reached max_steps without producing "
            "a final summary."
        ),
        "steps_used": max_steps,
        "input_tokens": total_in,
        "output_tokens": total_out,
    }