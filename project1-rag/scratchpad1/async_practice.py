import asyncio
import time


async def customer_data(name: str, delay: float, should_fail: bool = False) -> str:
    print(f"order preparing {name}...")
    await asyncio.sleep(delay)  # simulates a network call taking `delay` seconds
    if should_fail:
        raise ValueError(f"{name} failed!")
    print(f"order prepared {name}")
    return f"{name} result"


async def customer_sequential() -> None:
    start = time.time()
    c1 = await customer_data("customer A", 3)
    c2 = await customer_data("customer B", 2.5)
    c3 = await customer_data("customer C", 2)
    elapsed = time.time() - start
    print(f"\nSequential took {elapsed:.2f} seconds")
    print([c1, c2, c3])


async def customer_concurrent() -> None:
    start = time.time()
    c1, c2, c3 = await asyncio.gather(
        customer_data("customer A", 3, should_fail=False),
        customer_data("customer B", 2.5, should_fail=True),   # this one fails
        customer_data("customer C", 2, should_fail=False),
        return_exceptions=True,
    )
    elapsed = time.time() - start
    print(f"\nConcurrent (with return_exceptions=True) took {elapsed:.2f} seconds")
    for r in [c1, c2, c3]:
        if isinstance(r, Exception):
            print(f"Error: {r}")
        else:
            print(f"Success: {r}")


async def customer_concurrent_no_protection() -> None:
    start = time.time()
    try:
        results = await asyncio.gather(
            customer_data("customer A", 3, should_fail=False),
            customer_data("customer B", 2.5, should_fail=True),   # this one fails
            customer_data("customer C", 2, should_fail=False),
            # no return_exceptions=True this time
        )
        print(results)
    except ValueError as e:
        elapsed = time.time() - start
        print(f"\nConcurrent (WITHOUT return_exceptions) crashed after {elapsed:.2f} seconds")
        print(f"Whole gather() call failed: {e}")
        print("Note: we lost A and C's results even though they succeeded (or were about to).")


asyncio.run(customer_sequential())
print("\n" + "=" * 40 + "\n")
asyncio.run(customer_concurrent())
print("\n" + "=" * 40 + "\n")
asyncio.run(customer_concurrent_no_protection())