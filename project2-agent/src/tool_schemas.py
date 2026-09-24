"""
Tool schemas — how the LLM sees our three drawers.

Each schema tells the LLM:
- The tool's name (matches the Python function name exactly)
- What the tool does, when to use it, when NOT to use it
- What arguments it takes
- What it returns (mentioned inside description, since JSON Schema
  can't formally describe return types)

These are passed as `tools=[...]` in the Groq API call.
"""

FIND_CUSTOMER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "find_customer",
        "description": (
            "Searches for customers by name or email using a "
            "case-insensitive substring match. Use this tool when you "
            "need to find a customer but do not already know their "
            "customer ID. Do not use this tool when you already have "
            "a customer ID and need their invoices. It returns a list "
            "of matching customers, which may contain multiple matches "
            "or may be an empty list if no customers match."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A non-empty search string containing part or all "
                        "of a customer's name or email. Examples include "
                        "'GrubMatch', 'krishtech', or 'grubmatch.foods@'."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}


GET_INVOICE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_invoice",
        "description": (
            "Retrieves one specific invoice using its invoice ID. Use "
            "this tool only when you already have the invoice ID. Do "
            "not use this tool to search for invoices by customer name, "
            "email, or other text. It returns the complete invoice "
            "details, including the customer ID, customer name, amount, "
            "status, due date, and issued date."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "invoice_id": {
                    "type": "integer",
                    "description": (
                        "The unique ID of the invoice to retrieve. "
                        "It must be a positive integer and should only "
                        "be provided when the invoice ID is already known."
                    ),
                },
            },
            "required": ["invoice_id"],
        },
    },
}


LIST_CUSTOMER_INVOICES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_customer_invoices",
        "description": (
            "Retrieves all invoices belonging to a customer using their "
            "customer ID. Use this tool when you already have a customer "
            "ID and need one or more of that customer's invoices, such as "
            "when answering questions about their most recent invoice. "
            "Do not use this tool to search for a customer by name or "
            "email; use find_customer for that. The invoices are returned "
            "newest first, and an existing customer with no invoices "
            "returns an empty list."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "integer",
                    "description": (
                        "The unique ID of the customer whose invoices "
                        "should be retrieved. It must be a positive integer."
                    ),
                },
            },
            "required": ["customer_id"],
        },
    },
}


# All schemas together, ready to pass to Groq
ALL_TOOLS = [
    FIND_CUSTOMER_SCHEMA,
    GET_INVOICE_SCHEMA,
    LIST_CUSTOMER_INVOICES_SCHEMA,
]
