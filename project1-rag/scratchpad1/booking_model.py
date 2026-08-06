from pydantic import BaseModel, Field, ValidationError
from typing import Literal
from datetime import date


class Booking(BaseModel):
    model_config = {"extra": "forbid"}

    id: str
    customer_name: str = Field(min_length=1, max_length=100)
    booking_date: date
    check_in_date: date
    check_out_date: date
    guests: int = Field(gt=0, le=20)
    total_price: float = Field(gt=0)
    status: Literal["pending", "confirmed", "cancelled"]


# ─── Test 1: valid booking ───
print("--- Test 1: Valid booking ---")
booking1 = Booking(
    id="kjsafg",
    customer_name="Krish",
    booking_date=date(2026, 4, 24),
    check_in_date=date(2026, 5, 1),
    check_out_date=date(2026, 5, 5),
    guests=19,
    total_price=600000,
    status="confirmed"
)
print(booking1.model_dump_json(indent=2))


# ─── Test 2: multiple field constraint violations ───
print("\n--- Test 2: Invalid — 0 guests and negative price ---")
try:
    Booking(
        id="kjsafg",
        customer_name="Krish",
        booking_date=date(2026, 4, 24),
        check_in_date=date(2026, 5, 1),
        check_out_date=date(2026, 5, 5),
        guests=0,
        total_price=-600000,
        status="confirmed"
    )
except ValidationError as e:
    print(f"Caught {len(e.errors())} errors:")
    for err in e.errors():
        print(f"  - {err['loc'][0]}: {err['msg']}")


# ─── Test 3: status not in Literal ───
print("\n--- Test 3: Invalid — status not in allowed list ---")
try:
    Booking(
        id="abc123",
        customer_name="Krish",
        booking_date=date(2026, 4, 24),
        check_in_date=date(2026, 5, 1),
        check_out_date=date(2026, 5, 5),
        guests=2,
        total_price=5000,
        status="draft"
    )
except ValidationError as e:
    print(e)


# ─── Test 4: extra field caught by forbid ───
print("\n--- Test 4: Invalid — extra field ---")
try:
    Booking(
        id="abc123",
        customer_name="Krish",
        booking_date=date(2026, 4, 24),
        check_in_date=date(2026, 5, 1),
        check_out_date=date(2026, 5, 5),
        guests=2,
        total_price=5000,
        status="confirmed",
        location="Ahmedabad"
    )
except ValidationError as e:
    print(e)