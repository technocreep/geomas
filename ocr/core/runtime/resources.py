"""Runtime resource definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Resources:
    """Resources available to the orchestrator.

    Parameters
    ----------
    max_workers:
        Baseline number of OCR tasks that may run concurrently across all
        devices.
    max_gpu_utilization:
        Fraction of ``max_workers`` permitted on each GPU.  The orchestrator
        computes the effective per-device limit as ``max_workers / len(devices)
        * max_gpu_utilization`` (with a minimum of one).
    devices:
        Optional list of GPU device IDs.  When provided, OCR work is distributed
        round-robin across these devices and guarded by per-device semaphores to
        enforce the above limit.
    """

    max_workers: int = 1
    max_gpu_utilization: float = 1.0
    devices: List[int] | None = None


__all__ = ["Resources"]
