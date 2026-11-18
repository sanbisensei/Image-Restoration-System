"""
Benchmarking and Systematic Experiment Logging Utilities.

Combines PSNR, SSIM, and Runtime metrics into standardized diagnostic records.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.evaluation.psnr import compute_psnr, compute_mse
from src.evaluation.ssim import compute_custom_ssim, compute_reference_ssim
from src.utils.validation import validate_image


def evaluate_restoration(
    image_original: np.ndarray,
    image_degraded: np.ndarray,
    image_restored: np.ndarray,
    method_name: str,
    runtime_ms: float = 0.0,
    parameters: Optional[Dict[str, Any]] = None,
    verify_with_reference_ssim: bool = False,
) -> Dict[str, Any]:
    """
    Perform a complete quantitative evaluation of a restoration result against clean ground truth.

    Parameters
    ----------
    image_original : np.ndarray
        Clean reference ground-truth image.
    image_degraded : np.ndarray
        Degraded image prior to restoration.
    image_restored : np.ndarray
        Restored output image.
    method_name : str
        Name of the restoration algorithm.
    runtime_ms : float, default=0.0
        Execution latency in milliseconds.
    parameters : dict, optional
        Configuration parameters used during restoration.
    verify_with_reference_ssim : bool, default=False
        Whether to also record scikit-image reference SSIM for cross-validation.

    Returns
    -------
    dict
        Evaluation report dictionary containing:
        - method: Algorithm name
        - psnr_degraded: PSNR of degraded input (dB)
        - psnr_restored: PSNR of restored output (dB)
        - delta_psnr: Improvement in PSNR (dB)
        - ssim_degraded: SSIM of degraded input
        - ssim_restored: SSIM of restored output
        - delta_ssim: Improvement in SSIM
        - runtime_ms: Processing time in ms
        - parameters: Dictionary of filter settings
    """
    validate_image(image_original)
    validate_image(image_degraded)
    validate_image(image_restored)

    # 1. PSNR Metrics
    psnr_deg = compute_psnr(image_original, image_degraded)
    psnr_rest = compute_psnr(image_original, image_restored)
    delta_psnr = psnr_rest - psnr_deg

    # 2. SSIM Metrics (Custom implementation)
    ssim_deg = compute_custom_ssim(image_original, image_degraded)
    ssim_rest = compute_custom_ssim(image_original, image_restored)
    delta_ssim = ssim_rest - ssim_deg

    report = {
        "method": method_name,
        "psnr_degraded": round(psnr_deg, 2) if np.isfinite(psnr_deg) else "inf",
        "psnr_restored": round(psnr_rest, 2) if np.isfinite(psnr_rest) else "inf",
        "delta_psnr": round(delta_psnr, 2) if np.isfinite(delta_psnr) else "inf",
        "ssim_degraded": round(ssim_deg, 4),
        "ssim_restored": round(ssim_rest, 4),
        "delta_ssim": round(delta_ssim, 4),
        "runtime_ms": round(runtime_ms, 2),
        "parameters": parameters or {},
    }

    if verify_with_reference_ssim:
        ref_ssim_deg = compute_reference_ssim(image_original, image_degraded)
        ref_ssim_rest = compute_reference_ssim(image_original, image_restored)
        report["ref_ssim_degraded"] = round(ref_ssim_deg, 4)
        report["ref_ssim_restored"] = round(ref_ssim_rest, 4)
        report["ssim_abs_discrepancy"] = round(abs(ssim_rest - ref_ssim_rest), 6)

    return report


def results_to_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert a list of evaluation result dictionaries into a styled pandas DataFrame."""
    return pd.DataFrame(results)
