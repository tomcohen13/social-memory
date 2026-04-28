import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial, update_wrapper
from typing import Any, Callable, List

# TODO: re-export common transforms for easy importing in pipelines


class Transform:
    """A named, tqdm-trackable callable. Accepts bound kwargs like functools.partial."""

    def __init__(self, fn: Callable, name: str = None, **kwargs):
        self._fn = partial(fn, **kwargs) if kwargs else fn
        if isinstance(self._fn, partial):
            update_wrapper(self._fn, fn)
        self.__name__ = name or fn.__name__

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class TransformList:

    def __init__(self, transforms: list[Callable]):
        self.transforms = transforms
        self.transform_names = set(t.__name__ for t in transforms)

    def __contains__(self, transform: str) -> bool:
        if not isinstance(transform, str):
            raise ValueError("Transform must be a callable or a string representing the transform name.")
        return transform in self.transform_names

    def __iter__(self):
        return iter(self.transforms)

    def draw_pipeline(self) -> str:
        return " -> ".join(t.__name__ for t in self.transforms)

    def add(self, new_transform: Callable) -> None:
        if new_transform.__name__ in self.transform_names:
            raise ValueError(f"Transform {new_transform.__name__} already exists in the list.")
        self.transforms.append(new_transform)
        self.transform_names.add(new_transform.__name__)

    def add_after(self, new_transform: Callable, after_transform: str) -> None:
        if after_transform not in self:
            raise ValueError(f"Transform {after_transform} not found in the list. Got {self.transform_names}")
        index = next(i for i, t in enumerate(self.transforms) if t.__name__ == after_transform)
        self.transforms.insert(index + 1, new_transform)
        self.transform_names.add(new_transform.__name__)

    def add_before(self, new_transform: Callable, before_transform: str) -> None:
        if before_transform not in self:
            raise ValueError(f"Transform {before_transform} not found in the list.")
        index = next(i for i, t in enumerate(self.transforms) if t.__name__ == before_transform)
        self.transforms.insert(index, new_transform)
        self.transform_names.add(new_transform.__name__)


async def apply_transform_with_concurrency(
    transform: Callable[[Any], Any],
    inputs: List[dict],
    max_concurrency: int,
) -> List[Any]:
    from tqdm.asyncio import tqdm
    loop = asyncio.get_running_loop()
    desc = getattr(transform, "__name__", str(transform))
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        return list(await tqdm.gather(
            *[loop.run_in_executor(executor, transform, inp) for inp in inputs],
            desc=desc,
        ))
