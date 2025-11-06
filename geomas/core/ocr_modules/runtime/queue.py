"""Asynchronous work queue with retry helpers."""

from __future__ import annotations

import asyncio
import logging
import random
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Generic,
    Optional,
    Tuple,
    TypeVar,
)
from fpdf.errors import FPDFUnicodeEncodingException

from geomas.core.ocr_modules.io.hashing import short_hash


def _item_path(item: Any) -> Path | None:
    if isinstance(item, Path):
        return item
    for attr in ("src", "path", "normalized", "preproc"):
        candidate = getattr(item, attr, None)
        if isinstance(candidate, Path):
            return candidate
    return None


def _item_identifier(item: Any) -> str:
    path = _item_path(item)
    if path is None:
        return f"<{type(item).__name__}>"
    try:
        return short_hash(path)
    except OSError:
        return f"<{type(item).__name__}>"

T = TypeVar("T")


class WorkQueue(asyncio.Queue[Optional[T]], Generic[T]):
    """Queue that can spawn worker tasks with retry/backoff logic."""

    def __init__(
        self,
        *,
        failure_q: asyncio.Queue[tuple[T, Exception]] | None = None,
        maxsize: int = 0,
    ) -> None:
        """Initialize queue with optional ``failure_q`` for failed items."""
        super().__init__(maxsize=maxsize)
        self._failure_q = failure_q

    async def _execute(
        self,
        worker: Callable[[T], Awaitable[None]],
        item: T,
        *,
        retries: int,
        backoff_ms: Tuple[int, int],
    ) -> Exception | None:
        """Run ``worker`` with retry/backoff and return the final error.

        All exceptions listed in the handler below trigger retry/backoff
        behavior before the failure is surfaced to callers.
        """

        for attempt in range(retries):
            try:
                await worker(item)
                return None
            except (
                OSError,
                RuntimeError,
                NotImplementedError,
                TimeoutError,
                FPDFUnicodeEncodingException,
            ) as exc:
                attempt_number = attempt + 1
                if attempt_number == retries:
                    logging.exception(
                        "Worker failed on attempt %s for item %s; retries exhausted",
                        attempt_number,
                        _item_identifier(item),
                    )
                    return exc
                logging.exception("Worker failed on attempt %s", attempt_number)
                delay = random.uniform(*backoff_ms) / 1000
                await asyncio.sleep(delay)

        return None  # pragma: no cover - loop always returns

    async def _queue_iter(self) -> AsyncIterator[T]:
        """Yield items until sentinel ``None`` is received."""
        while (item := await self.get()) is not None:
            yield item
        self.task_done()

    async def _run(
        self,
        worker: Callable[[T], Awaitable[None]],
        *,
        retries: int,
        backoff_ms: Tuple[int, int],
    ) -> None:
        """Consume items from the queue, applying retry/backoff."""

        async for item in self._queue_iter():
            if err := await self._execute(
                worker, item, retries=retries, backoff_ms=backoff_ms
            ):
                if self._failure_q is not None:
                    await self._failure_q.put((item, err))
                self.task_done()
                continue
            self.task_done()

    def start_workers(
        self,
        worker: Callable[[T], Awaitable[None]],
        *,
        concurrency: int,
        retries: int,
        backoff_ms: Tuple[int, int],
        create_task: (
            Callable[[Coroutine[Any, Any, None]], asyncio.Task[None]] | None
        ) = None,
    ) -> list[asyncio.Task[None]]:
        """Start ``concurrency`` worker tasks processing items from the queue."""

        create = create_task or asyncio.create_task
        return [
            create(self._run(worker, retries=retries, backoff_ms=backoff_ms))
            for _ in range(concurrency)
        ]


__all__ = ["WorkQueue"]
