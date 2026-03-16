from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")


def progress_iter(iterable: Iterable[T], enabled: bool, desc: str) -> Iterator[T]:
    if not enabled:
        yield from iterable
        return

    try:
        from tqdm import tqdm

        yield from tqdm(iterable, desc=desc)
        return
    except Exception:
        pass

    total = len(iterable) if hasattr(iterable, "__len__") else None
    for idx, item in enumerate(iterable, start=1):
        if total is None:
            print(f"[{desc}] step {idx}")
        else:
            print(f"[{desc}] {idx}/{total}")
        yield item

