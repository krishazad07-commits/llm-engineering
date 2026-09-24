"""
Drawer 1: find_customer
Searches the customers table by name or email (case-insensitive substring).
Returns a list of matching customer dicts. Empty list if no matches.
"""

import os
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")
DB_URL = os.environ["SUPABASE_DB_URL_P2"]


def find_customer(query: str) -> list[dict[str, Any]]:
    """
    Search customers by name OR email, case-insensitive substring match.

    Args:
        query: search string (e.g. "GrubMatch", "krishtech", "grubmatch.foods@")

    Returns:
        List of matching customer dicts, each with keys: id, name, email.
        Empty list if no matches.
    """

    # Input validation: prevent empty/whitespace queries
    # from becoming "%%" and matching every customer.
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

    # Remove accidental leading/trailing spaces
    query = query.strip()

    sql = """
        SELECT id, name, email
        FROM customers
        WHERE name ILIKE %s
           OR email ILIKE %s
        ORDER BY id
    """

    pattern = f"%{query}%"

    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (pattern, pattern))
            rows = cur.fetchall()

    return [
        {
            "id": row[0],
            "name": row[1],
            "email": row[2],
        }
        for row in rows
    ]
class InvoiceNotFoundError(Exception):
    """Raised when get_invoice is called with an ID that doesn't exist."""
    pass


def get_invoice(invoice_id: int) -> dict[str, Any]:
    """
    Fetch a single invoice by ID, including the customer's name.

    Args:
        invoice_id: primary key from the invoices table

    Returns:
        Dict with keys: id, customer_id, customer_name, amount, status,
        due_date, issued_at

    Raises:
        InvoiceNotFoundError: if no invoice with this ID exists
        ValueError: if invoice_id is not a positive integer
    """

    # TODO 1: input validation
    if not isinstance(invoice_id, int) or isinstance(invoice_id, bool):
        raise ValueError("invoice_id must be a positive integer")

    if invoice_id <= 0:
        raise ValueError("invoice_id must be a positive integer")

    # TODO 2: SQL with JOIN
    sql = """
        SELECT
            invoices.id,
            invoices.customer_id,
            customers.name AS customer_name,
            invoices.amount,
            invoices.status,
            invoices.due_date,
            invoices.issued_at
        FROM invoices
        JOIN customers
            ON customers.id = invoices.customer_id
        WHERE invoices.id = %s
    """

    # TODO 3: execute and fetch one row
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (invoice_id,))
            row = cur.fetchone()

    # TODO 4: handle missing invoice
    if row is None:
        raise InvoiceNotFoundError(
            f"Invoice with ID {invoice_id} was not found"
        )

    # TODO 5: convert PostgreSQL types to JSON-friendly values
    return {
        "id": row[0],
        "customer_id": row[1],
        "customer_name": row[2],
        "amount": float(row[3]),
        "status": row[4],
        "due_date": row[5].isoformat(),
        "issued_at": row[6].isoformat(),
    }

class CustomerNotFoundError(Exception):
    """Raised when list_customer_invoices is called with a customer_id that doesn't exist."""
    pass


def list_customer_invoices(customer_id: int) -> list[dict[str, Any]]:
    """
    List all invoices for a given customer, newest first.
    """

    # TODO 1: input validation
    if not isinstance(customer_id, int) or isinstance(customer_id, bool) or customer_id <= 0:
        raise ValueError("customer_id must be a positive integer")

    # TODO 2 & 3: check customer exists, then fetch invoices
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:

            # Check whether customer exists
            cur.execute(
                "SELECT 1 FROM customers WHERE id = %s LIMIT 1",
                (customer_id,)
            )

            if cur.fetchone() is None:
                raise CustomerNotFoundError(
                    f"Customer with id {customer_id} does not exist"
                )

            # Fetch customer's invoices, newest first
            cur.execute(
                """
                SELECT id, amount, status, due_date, issued_at
                FROM invoices
                WHERE customer_id = %s
                ORDER BY issued_at DESC
                """,
                (customer_id,)
            )

            rows = cur.fetchall()

    # TODO 4: convert rows to dictionaries
    return [
        {
            "id": row[0],
            "amount": float(row[1]),
            "status": row[2],
            "due_date": row[3].isoformat() if row[3] else None,
            "issued_at": row[4].isoformat() if row[4] else None,
        }
        for row in rows
    ]

if __name__ == "__main__":
    # Manual smoke tests
    # Run:
    #   uv run src/tools.py

    print("Test 1 — exact-ish name:",
          find_customer("GrubMatch Foods"))

    print("Test 2 — partial name (should return 2):",
          find_customer("GrubMatch"))

    print("Test 3 — email substring:",
          find_customer("krishtech"))

    print("Test 4 — no match (should return []):",
          find_customer("NonexistentCorp"))

    print("Test 5 — empty string (should raise ValueError):")
    try:
        find_customer("")
        print("  FAIL: did not raise")
    except ValueError as e:
        print(f"  PASS: raised ValueError: {e}")

    print("Test 6 — whitespace only (should also raise):")
    try:
        find_customer("   ")
        print("  FAIL: did not raise")
    except ValueError as e:
        print(f"  PASS: raised ValueError: {e}")

    print("Test 7 — padded query (should still match):")
    print(f"  {find_customer('  GrubMatch  ')}")
    print("\n--- Drawer 2: get_invoice ---")

    print("Test 8 — existing invoice (should return dict with customer_name):")
    print(f"  {get_invoice(16)}")   # this is GrubMatch's most recent, our ground truth

    print("Test 9 — nonexistent invoice (should raise InvoiceNotFoundError):")
    try:
        get_invoice(999999)
        print("  FAIL: did not raise")
    except InvoiceNotFoundError as e:
        print(f"  PASS: raised: {e}")

    print("Test 10 — invalid input type (should raise ValueError):")
    try:
        get_invoice("not a number")   # LLM might pass a string
        print("  FAIL: did not raise")
    except ValueError as e:
        print(f"  PASS: raised: {e}")

    print("Test 11 — zero or negative (should raise ValueError):")
    try:
        get_invoice(-1)
        print("  FAIL: did not raise")
    except ValueError as e:
        print(f"  PASS: raised: {e}")
        print("\n--- Drawer 3: list_customer_invoices ---")

    print("Test 12 — customer with many invoices (GrubMatch, id 3, should return 16):")
    invoices = list_customer_invoices(3)
    print(f"  Got {len(invoices)} invoices")
    print(f"  Most recent: {invoices[0]}")   # should be invoice id 16, ₹2450, overdue

    print("Test 13 — customer with zero invoices (ZeroBill, id 20, should return []):")
    print(f"  {list_customer_invoices(20)}")

    print("Test 14 — nonexistent customer (should raise CustomerNotFoundError):")
    try:
        list_customer_invoices(999999)
        print("  FAIL: did not raise")
    except CustomerNotFoundError as e:
        print(f"  PASS: raised: {e}")

    print("Test 15 — invalid input (should raise ValueError):")
    try:
        list_customer_invoices("not an int")
        print("  FAIL: did not raise")
    except ValueError as e:
        print(f"  PASS: raised: {e}")