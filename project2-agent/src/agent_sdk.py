"""
Agent — Claude Agent SDK version.

Same three tools as the hand-rolled agent.py (find_customer,
list_customer_invoices, issue_refund), same HITL gate on issue_refund,
different framework.

Uses:
- Claude Agent SDK (claude_agent_sdk)
- claude-haiku-4-5 (via Claude Code subprocess)
- In-process MCP server for tools
- PreToolUse hook for the HITL gate (see NOTE below)

NOTE on the gate: this file previously used can_use_tool. That callback
is silently shadowed for MCP tools that appear in allowed_tools — a
CanUseToolShadowedWarning is emitted but the tool still executes.
Rewrote to use a PreToolUse hook + HookMatcher, which the SDK invokes
deterministically before the tool runs. The can_use_tool function
below is kept commented out as an artifact of what didn't work.
"""

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from claude_agent_sdk import (
    query,
    tool,
    create_sdk_mcp_server,
    ClaudeAgentOptions,
    PermissionResultAllow,
    PermissionResultDeny,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    HookMatcher,
)

from tools import (
    find_customer as _find_customer,
    list_customer_invoices as _list_customer_invoices,
    issue_refund as _issue_refund,
    execute_refund,
    reject_refund,
)

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

DESTRUCTIVE_TOOLS = {"issue_refund"}


# ============================================================
# Tool wrappers
# ============================================================

@tool(
    "find_customer",
    (
        "Searches for customers by name or email using a case-insensitive "
        "substring match. Use this tool when you need to find a customer "
        "but do not already know their customer ID. Returns a list of "
        "matching customers."
    ),
    {"query": str},
)
async def find_customer(args: dict) -> dict:
    try:
        result = _find_customer(args["query"])
        return {
            "content": [{"type": "text", "text": json.dumps(result)}]
        }
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Error: {e}"}],
            "is_error": True,
        }


@tool(
    "list_customer_invoices",
    (
        "Retrieves all invoices belonging to a customer using their "
        "customer ID. Use when you already have a customer ID and need "
        "their invoice history. Do not use to search by name."
    ),
    {"customer_id": int},
)
async def list_customer_invoices(args: dict) -> dict:
    try:
        result = _list_customer_invoices(args["customer_id"])
        return {
            "content": [{"type": "text", "text": json.dumps(result)}]
        }
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Error: {e}"}],
            "is_error": True,
        }


@tool(
    "issue_refund",
    (
        "Proposes a refund on a specific invoice. THIS ACTION REQUIRES "
        "HUMAN APPROVAL — calling this tool does not immediately issue "
        "the refund; the system will confirm with the operator before "
        "executing. Use when the user has explicitly asked to issue a "
        "refund and you have the invoice ID and refund amount."
    ),
    {"invoice_id": int, "amount": float, "reason": str},
)
async def issue_refund(args: dict) -> dict:
    try:
        result = _issue_refund(
            args["invoice_id"],
            args["amount"],
            args["reason"],
        )
        return {
            "content": [{"type": "text", "text": json.dumps(result)}]
        }
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Error: {e}"}],
            "is_error": True,
        }


# ============================================================
# HITL gate — DEPRECATED: kept as an artifact of what didn't work.
# Silently shadowed for MCP tools listed in allowed_tools.
# Real gate is refund_gate_hook + HookMatcher below.
# ============================================================

# async def can_use_tool(tool_name: str, tool_input: dict, context):
#     if tool_name not in DESTRUCTIVE_TOOLS:
#         return PermissionResultAllow()
#
#     print("\n" + "=" * 60)
#     print("DESTRUCTIVE ACTION — CONFIRMATION REQUIRED")
#     print("=" * 60)
#     print(f"Tool:      {tool_name}")
#     print(f"Invoice:   {tool_input.get('invoice_id')}")
#     print(f"Amount:    {tool_input.get('amount')}")
#     print(f"Reason:    {tool_input.get('reason')}")
#     print("=" * 60)
#
#     user_input = input("Approve? [y/N]: ").strip().lower()
#     print(f"→ {'APPROVED' if user_input == 'y' else 'REJECTED'}\n")
#
#     if user_input == "y":
#         return PermissionResultAllow()
#
#     return PermissionResultDeny(message="User rejected the refund request.")


# ============================================================
# PreToolUse hook — the deterministic HITL gate
# ============================================================

async def refund_gate_hook(
    input_data: dict,   # PreToolUseHookInput, a dict subclass
    tool_use_id: str | None,
    context,            # HookContext
) -> dict:
    """
    Fires before any tool matched by our HookMatcher runs.

    Returns:
        {} to allow, or a block-decision dict to deny.
    """
    tool_name = input_data.get("tool_name", "")
    tool_input = input_data.get("tool_input", {})

    # Safety net: this hook is registered only for issue_refund via
    # HookMatcher, but check the name anyway in case matcher is broad.
    if "issue_refund" not in tool_name:
        return {}

    print("\n" + "=" * 60)
    print("DESTRUCTIVE ACTION — CONFIRMATION REQUIRED (via PreToolUse hook)")
    print("=" * 60)
    print(f"Tool:      {tool_name}")
    print(f"Invoice:   {tool_input.get('invoice_id')}")
    print(f"Amount:    {tool_input.get('amount')}")
    print(f"Reason:    {tool_input.get('reason')}")
    print("=" * 60)

    user_input = input("Approve? [y/N]: ").strip().lower()
    print(f"→ {'APPROVED' if user_input == 'y' else 'REJECTED'}\n")

    if user_input == "y":
        return {}   # allow

    # Deny — standard hook JSON output shape
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": "User rejected the refund request.",
        }
    }


# ============================================================
# Runner
# ============================================================

async def run_agent(question: str):
    """
    Run the SDK agent for one question. Print streamed messages.
    """

    server = create_sdk_mcp_server(
        name="project2-tools",
        version="1.0.0",
        tools=[
            find_customer,
            list_customer_invoices,
            issue_refund,
        ],
    )

    options = ClaudeAgentOptions(
        model="claude-haiku-4-5",
        mcp_servers={"project2": server},
        allowed_tools=[
            "mcp__project2__find_customer",
            "mcp__project2__list_customer_invoices",
            "mcp__project2__issue_refund",   # gated by the hook, not the allow-list
        ],
        hooks={
            "PreToolUse": [
                HookMatcher(
                    matcher="mcp__project2__issue_refund",
                    hooks=[refund_gate_hook],
                )
            ]
        },
        system_prompt=(
            "You are a helpful assistant for a company's operations team. "
            "Use the provided tools to look up customer information and "
            "invoices, then answer clearly."
        ),
        max_turns=10,
    )

    step = 0

    async for message in query(prompt=question, options=options):
        if isinstance(message, AssistantMessage):
            has_content = any(
                isinstance(b, (TextBlock, ToolUseBlock))
                for b in message.content
            )
            if has_content:
                step += 1
                print(f"\n--- Step {step} ---")
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"Text: {block.text}")
                    elif isinstance(block, ToolUseBlock):
                        print(f"Tool: {block.name}({block.input})")

        elif isinstance(message, ResultMessage):
            print("\n=== FINAL ===")
            print(f"Stop reason: {message.stop_reason}")
            print(f"Turns: {message.num_turns}")
            print(f"Duration: {message.duration_ms}ms")
            print(f"Cost: ${message.total_cost_usd:.4f}")
            print(f"Tokens: in={message.usage.get('input_tokens')} "
                  f"out={message.usage.get('output_tokens')} "
                  f"cache_read={message.usage.get('cache_read_input_tokens')} "
                  f"cache_create={message.usage.get('cache_creation_input_tokens')}")
            if message.permission_denials:
                print(f"Permission denials: {message.permission_denials}")


if __name__ == "__main__":
    question = (
        "Issue a refund for GrubMatch Foods' most recent invoice. "
        "Customer says the goods were damaged in transit."
    )
    asyncio.run(run_agent(question))