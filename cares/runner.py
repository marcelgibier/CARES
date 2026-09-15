from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, TypeVar

from .jsonio import load_json, save_json
from .log import get_logger

log = get_logger(__name__)

T = TypeVar("T")


class Checkpoint:
    def __init__(self, path: str | Path, *, save_every: int = 50) -> None:
        self.path = Path(path)
        self.save_every = save_every
        self.data: dict[str, Any] = load_json(self.path, default={}) or {}
        self._lock = asyncio.Lock()
        self._since_save = 0

    def __len__(self) -> int:
        return len(self.data)

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def pending(
        self,
        items: Iterable[T],
        *,
        key: Callable[[T], str],
        is_valid: Callable[[Any], bool] | None = None,
    ) -> list[T]:
        todo = []
        for item in items:
            cached = self.data.get(key(item))
            if cached is not None and (is_valid is None or is_valid(cached)):
                continue
            todo.append(item)
        return todo

    async def record(self, key: str, value: Any) -> None:
        async with self._lock:
            self.data[key] = value
            self._since_save += 1
            if self._since_save >= self.save_every:
                self._save()

    def drop(self, is_valid: Callable[[Any], bool]) -> int:
        before = len(self.data)
        self.data = {k: v for k, v in self.data.items() if is_valid(v)}
        removed = before - len(self.data)
        if removed:
            self._save()
        return removed

    def flush(self) -> None:
        self._save()

    def _save(self) -> None:
        save_json(self.path, self.data)
        self._since_save = 0


def _progress(coros: Sequence[Awaitable[Any]], desc: str):
    try:
        from tqdm.asyncio import tqdm_asyncio
    except ImportError:  # pragma: no cover
        return asyncio.gather(*coros)
    return tqdm_asyncio.gather(*coros, desc=desc)


async def run_pool(
    items: Sequence[T],
    worker: Callable[[T], Awaitable[Any]],
    *,
    concurrency: int,
    desc: str = "",
) -> list[Any]:
    if not items:
        return []

    sem = asyncio.Semaphore(max(1, concurrency))

    async def guarded(item: T) -> Any:
        async with sem:
            try:
                return await worker(item)
            except Exception as exc:  # noqa: BLE001
                log.error("Unhandled failure on %.200r: %s", item, exc)
                return None

    return await _progress([guarded(item) for item in items], desc)


def run_async(coro: Awaitable[Any]) -> Any:
    try:
        return asyncio.run(coro)
    except RuntimeError as exc:
        if "asyncio.run() cannot be called from a running event loop" in str(exc):
            raise RuntimeError(
                "An asyncio loop is already running: await the coroutine directly."
            ) from exc
        raise
