import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List

# TODO: re-export common transforms for easy importing in pipelines


async def apply_transform_with_concurrency(
    transform: Callable[[Any], Any],
    inputs: List[dict],
    max_concurrency: int,
) -> List[Any]:
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        return list(await asyncio.gather(
            *[loop.run_in_executor(executor, transform, inp) for inp in inputs]
        ))
