"""
Tool schemas — how the LLM sees our six tools.

Each schema tells the LLM:
- The tool's name (matches the Python function name exactly)
- What the tool does, when to use it, when NOT to use it
- What arguments it takes
- What it returns (mentioned inside description, since JSON Schema
  can't formally describe return types)

These are passed as tools=[...] in the Groq API call.

Day 27 note: get_customer and search_invoices were deliberately
degraded during the tool-selection experiment. All descriptions are
restored here. See LOG.md Day 27 for the degraded versions and
findings.
"""


# ============================================================
# CUSTOMER TOOL 1
# ============================================================

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


# ============================================================
# INVOICE TOOL 1
# ============================================================

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


# ============================================================
# INVOICE TOOL 2
# ============================================================

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


# ============================================================
# CUSTOMER TOOL 2
# ============================================================

GET_CUSTOMER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_customer",
        "description": (
            "Retrieves one specific customer using their customer ID. "
            "Use this tool only when you already have the customer ID "
            "and need the full customer record. Do not use this tool "
            "to search by name or email; use find_customer for that. "
            "It returns the complete customer record, including id, "
            "name, email, and created_at."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "integer",
                    "description": (
                        "The unique ID of the customer to retrieve. "
                        "It must be a positive integer and should only "
                        "be provided when the customer ID is already known."
                    ),
                },
            },
            "required": ["customer_id"],
        },
    },
}


# ============================================================
# INVOICE SEARCH
# ============================================================

SEARCH_INVOICES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_invoices",
        "description": (
            "Searches invoices across all customers by status or by an "
            "overdue-before date. Use this tool when you need to find "
            "invoices matching a global filter — for example, every "
            "unpaid invoice, or every invoice overdue before a given "
            "date — without knowing which customer they belong to. "
            "Do not use this tool when you already have a customer ID "
            "and only need that customer's invoices; use "
            "list_customer_invoices for that. Returns up to 20 "
            "matching invoices."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "status_or_date_filter": {
                    "type": "string",
                    "description": (
                        "Either an invoice status — one of 'paid', "
                        "'unpaid', or 'overdue' — or an ISO date string "
                        "such as '2025-01-01'. A date returns all unpaid "
                        "invoices whose due date is before that date."
                    ),
                },
            },
            "required": ["status_or_date_filter"],
        },
    },
}


# ============================================================
# CUSTOMER TICKETS
# ============================================================

LIST_CUSTOMER_TICKETS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_customer_tickets",
        "description": (
            "Lists all support tickets belonging to a customer using their "
            "customer ID. Use this tool when you need to find or review a "
            "customer's support tickets. Do not use this tool for invoice "
            "history; use list_customer_invoices for that. It returns a "
            "list of tickets ordered newest first, and an existing customer "
            "with no tickets returns an empty list."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "integer",
                    "description": (
                        "The unique ID of the customer whose support tickets "
                        "should be retrieved. It must be a positive integer."
                    ),
                },
            },
            "required": ["customer_id"],
        },
    },
}


# ============================================================
# ALL TOOLS
# ============================================================

ALL_TOOLS = [
    FIND_CUSTOMER_SCHEMA,
    GET_INVOICE_SCHEMA,
    LIST_CUSTOMER_INVOICES_SCHEMA,
    GET_CUSTOMER_SCHEMA,
    SEARCH_INVOICES_SCHEMA,
    LIST_CUSTOMER_TICKETS_SCHEMA,
]