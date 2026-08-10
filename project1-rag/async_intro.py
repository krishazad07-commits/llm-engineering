import asyncio

async def greet()->str:
    return"Herllo, World!"

async def main():
    result= await greet()
    print(result)

asyncio.run(main())

async def slow_task() -> str:
    print("Task started")
    return "done"

async def main() -> None:
    result = await slow_task()  # NOW it runs
    print(result)

asyncio.run(main())


import time

async def fetch_data(name: str, delay: float) -> str:
    print(f"Starting {name}...")
    await asyncio.sleep(delay)  # simulates a network call taking `delay` seconds
    print(f"Finished {name}")
    return f"{name} result"


async def run_sequential() -> None:
    start = time.time()
    r1 = await fetch_data("Task A", 2)
    r2 = await fetch_data("Task B", 2)
    r3 = await fetch_data("Task C", 2)
    elapsed = time.time() - start
    print(f"\nSequential took {elapsed:.2f} seconds")
    print([r1, r2, r3])


async def run_concurrent() -> None:
    start = time.time()
    r1, r2, r3 = await asyncio.gather(
        fetch_data("Task A", 2),
        fetch_data("Task B", 2),
        fetch_data("Task C", 2),
    )
    elapsed = time.time() - start
    print(f"\nConcurrent took {elapsed:.2f} seconds")
    print([r1, r2, r3])


asyncio.run(run_sequential())
print("\n" + "="*40 + "\n")
asyncio.run(run_concurrent())
