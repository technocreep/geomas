"""Batch utilities for OCR."""

from __future__ import annotations

from typing import Iterable, Sequence, TypeVar

T = TypeVar("T")


def iter_batches(seq: Sequence[T], size: int) -> Iterable[Sequence[T]]:
    """Yield consecutive slices of ``seq`` of length ``size``.

    Parameters
    ----------
    seq : Sequence[T]
        Sequence to partition into batches.
    size : int
        Maximum number of elements per batch.

    Returns
    -------
    Iterable[Sequence[T]]
        Generator yielding non-overlapping slices of the input sequence.
    """

    for i in range(0, len(seq), size):
        yield seq[i : i + size]


__all__ = ["iter_batches"]
