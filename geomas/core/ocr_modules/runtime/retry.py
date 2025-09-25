"""Retry helpers with deterministic backoff."""

from __future__ import annotations

import logging
import random
import time
from typing import Callable, Tuple, TypeVar

T = TypeVar("T")


def retry_with_backoff(
    func: Callable[[], T],
    *,
    retries: int,
    backoff_ms: Tuple[int, int],
    logger: logging.Logger | None = None,
    retry_condition: Callable[[Exception], bool] | None = None,
) -> T:
    """Execute ``func`` with retry and jittered backoff.

    Parameters
    ----------
    func:
        Callable to execute.
    retries:
        Number of attempts before giving up. A value of ``1`` disables
        retrying.
    backoff_ms:
        ``(min_ms, max_ms)`` tuple controlling the random backoff interval.
    logger:
        Optional logger used to emit retry warnings.
    retry_condition:
        Optional predicate deciding whether ``exc`` is retryable. When omitted
        all exceptions are considered retryable.
    """

    for attempt in range(1, retries + 1):
        try:
            return func()
        except Exception as exc:
            if retry_condition and not retry_condition(exc):
                raise
            if attempt == retries:
                raise
            delay = random.uniform(*backoff_ms) / 1000
            if logger is not None:
                logger.warning("Operation failed (%s); retrying in %.1f s", exc, delay)
            time.sleep(delay)
    raise RuntimeError("unreachable")


__all__ = ["retry_with_backoff"]
