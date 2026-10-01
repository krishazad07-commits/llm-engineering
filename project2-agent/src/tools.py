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
# Drawer 7: issue_refund (DESTRUCTIVE — requires HITL approval)
# ============================================================

# ============================================================
# Drawer 7: issue_refund (DESTRUCTIVE — requires HITL approval)
# ============================================================

def issue_refund(
    invoice_id: int,
    amount: float,
    reason: str,
) -> dict[str, Any]:
    """
    Propose a refund on an invoice.

    Idempotency behaviour:
    - First call with (invoice_id, amount): creates a 'proposed' row
      in refund_requests, returns {status: 'proposed', refund_id: ...}
    - The agent loop's HITL gate then asks the user to approve.
    - On approval, the loop calls execute_refund(refund_id) which
      marks the row 'executed' and updates the invoice.
    - On rejection, the loop calls reject_refund(refund_id) which
      marks the row 'rejected'.
    - Re-calling issue_refund with the same (invoice_id, amount) while
      a proposal is still pending returns the same refund_id — no
      duplicate row created.
    - Re-calling after execution returns {status: 'already_executed'}
      so the model knows not to re-propose.

    Args:
        invoice_id: which invoice to refund
        amount: refund amount in the invoice's currency
        reason: why the refund is being issued

    Returns:
        Dict with keys: status, refund_id, invoice_id, amount, reason
        status is one of: 'proposed', 'already_executed'

    Raises:
        ValueError: if inputs are invalid
        InvoiceNotFoundError: if invoice_id does not exist
    """

    # TODO 1: input validation

    # invoice_id must be a positive int; bool excluded
    if isinstance(invoice_id, bool) or not isinstance(invoice_id, int):
        raise ValueError("invoice_id must be a positive integer")

    if invoice_id <= 0:
        raise ValueError("invoice_id must be a positive integer")

    # amount must be int or float; bool excluded
    if isinstance(amount, bool) or not isinstance(amount, (int, float)):
        raise ValueError("amount must be a positive number")

    if amount <= 0:
        raise ValueError("amount must be greater than 0")

    # Normalise: always float, rounded to 2 decimal places.
    # Guards against LLM math imprecision and int/float inconsistency.
    amount = round(float(amount), 2)

    # reason must be a non-empty string
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")

    reason = reason.strip()

    # TODO 2 + 3 + 4: verify invoice, check for existing proposal, insert if fresh

    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:

            # Verify invoice exists
            cur.execute(
                "SELECT 1 FROM invoices WHERE id = %s",
                (invoice_id,),
            )

            if cur.fetchone() is None:
                raise InvoiceNotFoundError(
                    f"Invoice {invoice_id} not found"
                )

            # Check for an existing active proposal for this
            # (invoice_id, amount) pair.
            # Note: 'approved' is a transient state used inside the
            # agent loop between gate approval and execute_refund;
            # it should not be visible to callers here.
            cur.execute(
                """
                SELECT id, status
                FROM refund_requests
                WHERE invoice_id = %s
                  AND amount = %s
                  AND status IN ('proposed', 'executed')
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (invoice_id, amount),
            )

            existing = cur.fetchone()

            if existing:
                refund_id, status = existing

                if status == "proposed":
                    return {
                        "status": "proposed",
                        "refund_id": str(refund_id),
                        "invoice_id": invoice_id,
                        "amount": amount,
                        "reason": reason,
                    }

                if status == "executed":
                    return {
                        "status": "already_executed",
                        "refund_id": str(refund_id),
                        "invoice_id": invoice_id,
                        "amount": amount,
                        "reason": reason,
                    }

            # Fresh proposal — insert. Wrap in try/except to catch
            # the race where another turn INSERTed the same pair
            # between our SELECT and INSERT. The partial unique
            # index will raise UniqueViolation; we roll back, read
            # the winner, and return it as if idempotent.
            try:
                cur.execute(
                    """
                    INSERT INTO refund_requests
                        (invoice_id, amount, reason)
                    VALUES (%s, %s, %s)
                    RETURNING id
                    """,
                    (invoice_id, amount, reason),
                )
                refund_id = cur.fetchone()[0]

                return {
                    "status": "proposed",
                    "refund_id": str(refund_id),
                    "invoice_id": invoice_id,
                    "amount": amount,
                    "reason": reason,
                }

            except psycopg.errors.UniqueViolation:
                # Someone won the race. Roll back our INSERT, read
                # the winning row, and return it.
                conn.rollback()

                cur.execute(
                    """
                    SELECT id FROM refund_requests
                    WHERE invoice_id = %s
                      AND amount = %s
                      AND status IN ('proposed', 'executed')
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (invoice_id, amount),
                )
                refund_id = cur.fetchone()[0]

                return {
                    "status": "proposed",
                    "refund_id": str(refund_id),
                    "invoice_id": invoice_id,
                    "amount": amount,
                    "reason": reason,
                }

def execute_refund(refund_id: str) -> dict[str, Any]:
    """
    Mark a proposed refund as executed and update the invoice.

    Called by the agent loop after the user approves at the gate.
    Idempotent: calling on an already-executed row returns the
    existing state without changes.

    Raises:
        ValueError: if refund_id is not a valid UUID string or the
        row is in an invalid state ('rejected', not found)
    """

    # TODO 1: validate refund_id looks like a UUID string

    if not isinstance(refund_id, str) or not refund_id.strip():
        raise ValueError("refund_id must be a non-empty UUID string")

    refund_id = refund_id.strip()

    if len(refund_id) != 36:
        raise ValueError("refund_id must be a valid UUID string")

    # TODO 2: transaction

    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:

            # a) SELECT the refund row FOR UPDATE
            cur.execute(
                """
                SELECT id, invoice_id, amount, reason, status, executed_at
                FROM refund_requests
                WHERE id = %s
                FOR UPDATE
                """,
                (refund_id,),
            )

            row = cur.fetchone()

            if row is None:
                raise ValueError(
                    f"Refund request {refund_id} not found"
                )

            (
                db_refund_id,
                invoice_id,
                amount,
                reason,
                status,
                executed_at,
            ) = row

            # b) Already executed → idempotent no-op
            if status == "executed":
                return {
                    "status": "already_executed",
                    "refund_id": str(db_refund_id),
                    "invoice_id": invoice_id,
                    "amount": float(amount),
                    "reason": reason,
                    "executed_at": (
                        executed_at.isoformat()
                        if executed_at
                        else None
                    ),
                }

            # c) Rejected → cannot execute
            if status == "rejected":
                raise ValueError(
                    f"Refund {refund_id} has already been rejected"
                )

            # Only proposed refunds can be executed
            if status != "proposed":
                raise ValueError(
                    f"Refund {refund_id} is in invalid state: {status}"
                )

            # d) Mark refund as executed
            cur.execute(
                """
                UPDATE refund_requests
                SET status = 'executed',
                    executed_at = NOW()
                WHERE id = %s
                """,
                (refund_id,),
            )

            # Update the invoice status
            cur.execute(
                """
                UPDATE invoices
                SET status = 'refunded'
                WHERE id = %s
                """,
                (invoice_id,),
            )

            # e) Return final state
            cur.execute(
                """
                SELECT id, invoice_id, amount, reason, status, executed_at
                FROM refund_requests
                WHERE id = %s
                """,
                (refund_id,),
            )

            final_row = cur.fetchone()

            (
                db_refund_id,
                invoice_id,
                amount,
                reason,
                status,
                executed_at,
            ) = final_row

            return {
                "status": status,
                "refund_id": str(db_refund_id),
                "invoice_id": invoice_id,
                "amount": float(amount),
                "reason": reason,
                "executed_at": (
                    executed_at.isoformat()
                    if executed_at
                    else None
                ),
            }


def reject_refund(refund_id: str) -> dict[str, Any]:
    """
    Mark a proposed refund as rejected.

    Called by the agent loop after the user rejects at the gate.
    Idempotent: calling on an already-rejected row is a no-op.
    """

    # TODO 1: validate refund_id

    if not isinstance(refund_id, str) or not refund_id.strip():
        raise ValueError("refund_id must be a non-empty UUID string")

    refund_id = refund_id.strip()

    if len(refund_id) != 36:
        raise ValueError("refund_id must be a valid UUID string")

    # TODO 2: reject proposed refund

    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:

            cur.execute(
                """
                UPDATE refund_requests
                SET status = 'rejected'
                WHERE id = %s
                  AND status = 'proposed'
                """,
                (refund_id,),
            )

            # Select current state.
            # If already rejected, UPDATE changes nothing,
            # but SELECT still returns the existing state.

            cur.execute(
                """
                SELECT id, invoice_id, amount, reason, status
                FROM refund_requests
                WHERE id = %s
                """,
                (refund_id,),
            )

            row = cur.fetchone()

            if row is None:
                raise ValueError(
                    f"Refund request {refund_id} not found"
                )

            (
                db_refund_id,
                invoice_id,
                amount,
                reason,
                status,
            ) = row

            return {
                "status": status,
                "refund_id": str(db_refund_id),
                "invoice_id": invoice_id,
                "amount": float(amount),
                "reason": reason,
            }
def explore_customer_profile(
    customer_id: int,
    question: str,
) -> dict[str, Any]:
    """
    Run an isolated read-only research subagent for a customer.

    Only the final summary crosses the subagent boundary.
    Token counts and raw tool results stay inside the subagent.
    """
    from subagent import subagent_loop

    result = subagent_loop(
        customer_id=str(customer_id),
        question=question,
    )

    return {
        "summary": result["summary"],
    }
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

    print("\nTest 21 — issue_refund fresh proposal:")
try:
    result = issue_refund(1, 100.0, "smoke test")
    print(f"  {result}")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")

print("\nTest 22 — issue_refund duplicate (should return same refund_id):")
try:
    result = issue_refund(1, 100.0, "smoke test again")
    print(f"  {result}")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {e}")

print("\nTest 23 — issue_refund invalid amount:")
try:
    issue_refund(1, -5, "negative")
    print("  FAIL: did not raise")
except ValueError as e:
    print(f"  PASS: raised: {e}")