"""
Six tools for the Project 2 agent.

Drawer 1: find_customer         — search customers by name or email
Drawer 2: get_invoice           — fetch one invoice by ID
Drawer 3: list_customer_invoices — all invoices for a customer
Drawer 4: get_customer          — fetch one customer by ID
Drawer 5: search_invoices       — global invoice search by status or date
Drawer 6: list_customer_tickets — all support tickets for a customer
"""

import os
from pathlib import Path
from typing import Any
from datetime import datetime
from decimal import Decimal
import psycopg
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")
DB_URL = os.environ["SUPABASE_DB_URL_P2"]


# ============================================================
# Custom exceptions
# ============================================================

class InvoiceNotFoundError(Exception):
    """Raised when an invoice_id doesn't exist."""
    pass


class CustomerNotFoundError(Exception):
    """Raised when a customer_id doesn't exist."""
    pass


# ============================================================
# Drawer 1: find_customer
# ============================================================

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
        {"id": row[0], "name": row[1], "email": row[2]}
        for row in rows
    ]


# ============================================================
# Drawer 2: get_invoice
# ============================================================

def get_invoice(invoice_id: int) -> dict[str, Any]:
    """
    Fetch a single invoice by ID, including the customer's name.

    Raises:
        InvoiceNotFoundError: if no invoice with this ID exists
        ValueError: if invoice_id is not a positive integer
    """

    if not isinstance(invoice_id, int) or isinstance(invoice_id, bool):
        raise ValueError("invoice_id must be a positive integer")

    if invoice_id <= 0:
        raise ValueError("invoice_id must be a positive integer")

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

    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (invoice_id,))
            row = cur.fetchone()

    if row is None:
        raise InvoiceNotFoundError(
            f"Invoice with ID {invoice_id} was not found"
        )

    return {
        "id": row[0],
        "customer_id": row[1],
        "customer_name": row[2],
        "amount": float(row[3]),
        "status": row[4],
        "due_date": row[5].isoformat(),
        "issued_at": row[6].isoformat(),
    }


# ============================================================
# Drawer 3: list_customer_invoices
# ============================================================

def list_customer_invoices(customer_id: int) -> list[dict[str, Any]]:
    """
    List all invoices for a given customer, newest first.

    Raises:
        CustomerNotFoundError: if the customer does not exist
        ValueError: if customer_id is not a positive integer
    """

    if (
        not isinstance(customer_id, int)
        or isinstance(customer_id, bool)
        or customer_id <= 0
    ):
        raise ValueError("customer_id must be a positive integer")

    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:

            cur.execute(
                "SELECT 1 FROM customers WHERE id = %s LIMIT 1",
                (customer_id,),
            )

            if cur.fetchone() is None:
                raise CustomerNotFoundError(
                    f"Customer with id {customer_id} does not exist"
                )

            cur.execute(
                """
                SELECT id, amount, status, due_date, issued_at
                FROM invoices
                WHERE customer_id = %s
                ORDER BY issued_at DESC
                """,
                (customer_id,),
            )

            rows = cur.fetchall()

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


# ============================================================
# Drawer 4: get_customer
# ============================================================

def get_customer(customer_id: int) -> dict[str, Any]:
    """
    Fetch a single customer by ID.

    Raises:
        CustomerNotFoundError: if the customer does not exist
        ValueError: if customer_id is not a positive integer
    """

    if (
        not isinstance(customer_id, int)
        or isinstance(customer_id, bool)
        or customer_id <= 0
    ):
        raise ValueError("customer_id must be a positive integer")

    sql = """
        SELECT *
        FROM customers
        WHERE id = %s
    """

    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (customer_id,))
            row = cur.fetchone()

            if row is None:
                raise CustomerNotFoundError(
                    f"Customer with id {customer_id} does not exist"
                )

            columns = [desc.name for desc in cur.description]

    result = {}
    for column, value in zip(columns, row):
        if hasattr(value, "isoformat"):
            value = value.isoformat()
        result[column] = value

    return result


# ============================================================
# Drawer 5: search_invoices
# ============================================================

def search_invoices(status_or_date_filter: str) -> list[dict[str, Any]]:
    """
    Search invoices by status or overdue-before date.

    Accepts:
        "paid" | "unpaid" | "overdue"
        ISO date such as "2025-01-01" — returns unpaid invoices due before that date

    Returns:
        Up to 20 matching invoices.
    """

    if not isinstance(status_or_date_filter, str):
        raise ValueError("status_or_date_filter must be a string")

    status_or_date_filter = status_or_date_filter.strip().lower()

    if not status_or_date_filter:
        raise ValueError("status_or_date_filter must be a non-empty string")

    status_values = {"paid", "unpaid", "overdue"}

    if status_or_date_filter in status_values:
        sql = """
            SELECT *
            FROM invoices
            WHERE status = %s
            LIMIT 20
        """
        params = (status_or_date_filter,)

    else:
        try:
            filter_date = datetime.fromisoformat(status_or_date_filter).date()
        except ValueError:
            raise ValueError(
                "Filter must be a valid invoice status "
                "or an ISO date such as '2025-01-01'"
            )

        sql = """
            SELECT *
            FROM invoices
            WHERE due_date < %s
              AND status != 'paid'
            LIMIT 20
        """
        params = (filter_date,)

    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            columns = [desc.name for desc in cur.description]

    results = []
    for row in rows:
        item = {}
        for column, value in zip(columns, row):
            if isinstance(value, Decimal):
                value = float(value)
            elif hasattr(value, "isoformat"):
                value = value.isoformat()
            item[column] = value
        results.append(item)

    return results


# ============================================================
# Drawer 6: list_customer_tickets
# ============================================================

def list_customer_tickets(customer_id: int) -> list[dict[str, Any]]:
    """
    List all support tickets belonging to a customer, newest first.

    An existing customer with no tickets returns an empty list.

    Raises:
        CustomerNotFoundError: if the customer does not exist
        ValueError: if customer_id is not a positive integer
    """

    if (
        not isinstance(customer_id, int)
        or isinstance(customer_id, bool)
        or customer_id <= 0
    ):
        raise ValueError("customer_id must be a positive integer")

    sql = """
        SELECT *
        FROM support_tickets
        WHERE customer_id = %s
        ORDER BY created_at DESC
    """

    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM customers WHERE id = %s LIMIT 1",
                (customer_id,),
            )

            if cur.fetchone() is None:
                raise CustomerNotFoundError(
                    f"Customer with id {customer_id} does not exist"
                )

            cur.execute(sql, (customer_id,))
            rows = cur.fetchall()
            columns = [desc.name for desc in cur.description]

    results = []
    for row in rows:
        item = {}
        for column, value in zip(columns, row):
            if isinstance(value, Decimal):
                value = float(value)
            elif hasattr(value, "isoformat"):
                value = value.isoformat()
            item[column] = value
        results.append(item)

    return results


# ============================================================
# Manual smoke tests
# ============================================================

if __name__ == "__main__":
    # Run: uv run src/tools.py

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
    print(f"  {get_invoice(16)}")

    print("Test 9 — nonexistent invoice (should raise InvoiceNotFoundError):")
    try:
        get_invoice(999999)
        print("  FAIL: did not raise")
    except InvoiceNotFoundError as e:
        print(f"  PASS: raised: {e}")

    print("Test 10 — invalid input type (should raise ValueError):")
    try:
        get_invoice("not a number")
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
    print(f"  Most recent: {invoices[0]}")

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

    print("\n--- New Tools: Phase A ---")

    print("Test 16 — get_customer(1):")
    try:
        print(f"  {get_customer(1)}")
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")

    print("\nTest 17 — search_invoices('paid'):")
    try:
        invoices = search_invoices("paid")
        print(f"  Got {len(invoices)} invoices")
        print(f"  First result: {invoices[0] if invoices else '[]'}")
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")

    print("\nTest 18 — search_invoices('overdue'):")
    try:
        invoices = search_invoices("overdue")
        print(f"  Got {len(invoices)} invoices")
        print(f"  First result: {invoices[0] if invoices else '[]'}")
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")

    print("\nTest 19 — search_invoices('2025-01-01'):")
    try:
        invoices = search_invoices("2025-01-01")
        print(f"  Got {len(invoices)} invoices")
        print(f"  First result: {invoices[0] if invoices else '[]'}")
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")

    print("\nTest 20 — list_customer_tickets(1):")
    try:
        tickets = list_customer_tickets(1)
        print(f"  Got {len(tickets)} tickets")
        print(f"  First result: {tickets[0] if tickets else '[]'}")
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")