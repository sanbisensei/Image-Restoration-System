"""
Unit tests for Evaluation Metrics:
- PSNR & MSE
- SSIM (Custom implementation & Scikit-image reference comparison)
- Runtime measurement
- Benchmarking report generation
"""

import numpy as np
import pytest

from src.evaluation.psnr import compute_mse, compute_psnr
from src.evaluation.ssim import compute_custom_ssim, compute_reference_ssim, compute_ssim
from src.evaluation.runtime import ExecutionTimer, time_function
from src.evaluation.benchmarking import evaluate_restoration


# ---------------------------------------------------------------------
# MSE & PSNR Tests
# ---------------------------------------------------------------------

def test_mse_identical_images():
    img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
    assert compute_mse(img, img) == 0.0


def test_mse_known_difference():
    im1 = np.zeros((10, 10), dtype=np.uint8)
    im2 = np.full((10, 10), 10, dtype=np.uint8)
    # Difference is 10 everywhere -> MSE = 10^2 = 100.0
    assert compute_mse(im1, im2) == 100.0


def test_psnr_identical_images_is_infinity():
    # Mathematical identity: MSE = 0 -> PSNR = +inf
    img_u8 = np.ones((20, 20), dtype=np.uint8) * 150
    assert compute_psnr(img_u8, img_u8) == float("inf")

    img_flt = np.ones((20, 20), dtype=np.float64) * 0.6
    assert compute_psnr(img_flt, img_flt) == float("inf")


def test_psnr_known_values_uint8():
    im1 = np.full((10, 10), 100, dtype=np.uint8)
    im2 = np.full((10, 10), 101, dtype=np.uint8)
    # MSE = 1.0. For uint8 with range 255: PSNR = 20 * log10(255 / 1) = 48.1308 dB
    psnr = compute_psnr(im1, im2)
    expected = 20.0 * np.log10(255.0)
    assert np.isclose(psnr, expected, atol=1e-3)


def test_psnr_known_values_float():
    im1 = np.zeros((10, 10), dtype=np.float64)
    im2 = np.full((10, 10), 0.1, dtype=np.float64)
    # MSE = 0.01. For float [0, 1]: PSNR = 10 * log10(1.0 / 0.01) = 10 * log10(100) = 20.0 dB
    psnr = compute_psnr(im1, im2)
    assert np.isclose(psnr, 20.0, atol=1e-3)


def test_psnr_shape_mismatch_raises():
    im1 = np.zeros((10, 10), dtype=np.uint8)
    im2 = np.zeros((10, 12), dtype=np.uint8)
    with pytest.raises(ValueError, match="identical shapes"):
        compute_psnr(im1, im2)


# ---------------------------------------------------------------------
# SSIM Tests
# ---------------------------------------------------------------------

def test_custom_ssim_identical_images():
    img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
    ssim_val = compute_custom_ssim(img, img)
    assert np.isclose(ssim_val, 1.0, atol=1e-4)


def test_custom_ssim_decreases_with_noise():
    clean = np.ones((50, 50), dtype=np.float64) * 0.5
    clean[15:35, 15:35] = 0.9

    rng = np.random.default_rng(42)
    mild_noise = np.clip(clean + rng.normal(0, 0.05, clean.shape), 0.0, 1.0)
    heavy_noise = np.clip(clean + rng.normal(0, 0.20, clean.shape), 0.0, 1.0)

    ssim_mild = compute_custom_ssim(clean, mild_noise)
    ssim_heavy = compute_custom_ssim(clean, heavy_noise)

    assert 1.0 > ssim_mild > ssim_heavy > 0.0


def test_custom_ssim_cross_validation_with_skimage():
    # Rigorous academic validation: Compare our custom SSIM against scikit-image gold standard
    img1 = np.random.randint(50, 200, (64, 64), dtype=np.uint8)
    img2 = np.clip(img1.astype(int) + np.random.randint(-15, 15, (64, 64)), 0, 255).astype(np.uint8)

    custom_score = compute_custom_ssim(img1, img2)
    reference_score = compute_reference_ssim(img1, img2)

    # Both implementations should agree closely (within 0.02, accounting for minor boundary padding style)
    assert np.isclose(custom_score, reference_score, atol=0.02), (
        f"Custom SSIM ({custom_score:.4f}) deviates from reference ({reference_score:.4f})"
    )


# ---------------------------------------------------------------------
# Runtime and Benchmarking Tests
# ---------------------------------------------------------------------

def test_runtime_timer():
    with ExecutionTimer() as timer:
        # Simple computation
        arr = np.sum(np.ones((200, 200)))

    assert timer.elapsed_ms > 0.0
    assert timer.elapsed_seconds > 0.0


def test_evaluate_restoration_report():
    orig = np.ones((30, 30), dtype=np.uint8) * 100
    deg = orig.copy()
    deg[10:20, 10:20] = 50

    rest = orig.copy()
    rest[10:20, 10:20] = 90  # closer to original

    report = evaluate_restoration(
        image_original=orig,
        image_degraded=deg,
        image_restored=rest,
        method_name="TestFilter",
        runtime_ms=12.5,
        verify_with_reference_ssim=True,
    )

    assert report["method"] == "TestFilter"
    assert report["runtime_ms"] == 12.5
    assert report["psnr_restored"] > report["psnr_degraded"]
    assert report["delta_psnr"] > 0
    assert report["ssim_restored"] > report["ssim_degraded"]
    assert "ref_ssim_restored" in report
