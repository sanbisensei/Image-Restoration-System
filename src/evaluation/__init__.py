"""
Evaluation and Benchmarking package:
- PSNR & MSE metrics
- SSIM (custom and reference)
- High-precision execution timer
- Structured benchmarking utilities
"""

from src.evaluation.psnr import compute_mse, compute_psnr
from src.evaluation.ssim import compute_custom_ssim, compute_reference_ssim, compute_ssim
from src.evaluation.runtime import ExecutionTimer, time_function
from src.evaluation.benchmarking import evaluate_restoration, results_to_dataframe

__all__ = [
    "compute_mse",
    "compute_psnr",
    "compute_custom_ssim",
    "compute_reference_ssim",
    "compute_ssim",
    "ExecutionTimer",
    "time_function",
    "evaluate_restoration",
    "results_to_dataframe",
]
