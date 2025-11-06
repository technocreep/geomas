"""Simple in-process metrics counters."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict

_counts: Counter[str] = Counter()
_latency_totals: Dict[str, float] = defaultdict(float)
_latency_counts: Counter[str] = Counter()


def increment(name: str, value: int = 1) -> None:
    """Increment counter *name* by *value*."""
    _counts[name] += value


def observe_latency(name: str, value: float) -> None:
    """Record latency *value* (in seconds) for metric *name*."""
    _latency_totals[name] += value
    _latency_counts[name] += 1


def get_metrics() -> Dict[str, float]:
    """Return a snapshot of current counters and latency aggregates."""
    data: Dict[str, float] = dict(_counts)
    for name, total in _latency_totals.items():
        data[f"{name}_sum"] = total
        data[f"{name}_count"] = float(_latency_counts[name])
    return data


__all__ = ["increment", "observe_latency", "get_metrics"]
