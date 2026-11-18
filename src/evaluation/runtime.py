"""
Runtime Performance and Execution Measurement Utilities.

Academic Purpose:
Algorithm execution time is a critical practical evaluation metric.
Restoration algorithms must balance reconstruction quality (PSNR/SSIM) against
computational complexity and latency on CPU hardware.
"""

import time
from typing import Any, Callable, Dict, Tuple
import numpy as np


class ExecutionTimer:
    """Context manager for high-resolution performance timing using time.perf_counter()."""

    def __init__(self):
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.elapsed_seconds: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed_seconds = self.end_time - self.start_time
        self.elapsed_ms = self.elapsed_seconds * 1000.0


def time_function(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """
    Execute a function and measure its elapsed execution time in milliseconds.

    Parameters
    ----------
    func : callable
        Function to execute.
    *args, **kwargs
        Arguments to pass to func.

    Returns
    -------
    tuple of (result, elapsed_ms)
        - result: output of the function.
        - elapsed_ms: runtime in milliseconds.
    """
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000.0
    return result, elapsed_ms
