"""
Phase 5 Quantitative Evaluation Metrics Demonstration Script.

Measures and compares:
- Peak Signal-to-Noise Ratio (PSNR in dB)
- Structural Similarity Index (SSIM) — Custom & Scikit-image reference
- Actual CPU execution time (ms)
across multiple degradation and restoration pairs.

Outputs are formatted as a pandas table and saved to outputs/phase5_benchmark_sample.csv.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from src.image_io import load_image
from src.degradation.gaussian_noise import add_gaussian_noise
from src.degradation.salt_pepper import add_salt_and_pepper_noise
from src.degradation.motion_blur import create_motion_blur_kernel, apply_motion_blur
from src.spatial.mean_filter import apply_mean_filter
from src.spatial.gaussian_filter import apply_gaussian_filter
from src.spatial.median_filter import apply_median_filter
from src.frequency.wiener_filter import apply_wiener_filter
from src.evaluation.runtime import time_function
from src.evaluation.benchmarking import evaluate_restoration, results_to_dataframe


def main():
    print("=" * 75)
    print("CSE 438 — Image Restoration System: Phase 5 Quantitative Evaluation")
    print("=" * 75)

    data_dir = Path("data/original")
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    img_path = data_dir / "geometric_shapes.png"
    clean_img = load_image(img_path, as_gray=True)
    print(f"\nLoaded reference image: {img_path.name} (dimensions {clean_img.shape})")

    eval_records = []

    # -------------------------------------------------------------
    # Case 1: Gaussian Noise (sigma=0.08) restored with Spatial Filters
    # -------------------------------------------------------------
    print("\n[Case 1] Benchmarking Gaussian Noise (σ=0.08)...")
    noisy_gauss = add_gaussian_noise(clean_img, sigma=0.08, seed=42)

    # 1A. Mean Filter 3x3
    rest_mean, t_mean = time_function(apply_mean_filter, noisy_gauss, kernel_size=3)
    rec_mean = evaluate_restoration(
        clean_img, noisy_gauss, rest_mean,
        method_name="Mean Filter 3×3",
        runtime_ms=t_mean,
        parameters={"kernel_size": 3},
        verify_with_reference_ssim=True,
    )
    eval_records.append(rec_mean)

    # 1B. Gaussian Filter (K=5, sigma=1.0)
    rest_gauss, t_gauss = time_function(apply_gaussian_filter, noisy_gauss, kernel_size=5, sigma=1.0)
    rec_gauss = evaluate_restoration(
        clean_img, noisy_gauss, rest_gauss,
        method_name="Gaussian Filter (K=5, σ=1.0)",
        runtime_ms=t_gauss,
        parameters={"kernel_size": 5, "sigma": 1.0},
        verify_with_reference_ssim=True,
    )
    eval_records.append(rec_gauss)

    # -------------------------------------------------------------
    # Case 2: Salt-and-Pepper Noise (density=0.10) restored with Median Filter
    # -------------------------------------------------------------
    print("\n[Case 2] Benchmarking Salt-and-Pepper Noise (density=10%)...")
    noisy_sp = add_salt_and_pepper_noise(clean_img, density=0.10, seed=42)

    # 2A. Median Filter 3x3
    rest_median, t_median = time_function(apply_median_filter, noisy_sp, kernel_size=3)
    rec_median = evaluate_restoration(
        clean_img, noisy_sp, rest_median,
        method_name="Median Filter 3×3",
        runtime_ms=t_median,
        parameters={"kernel_size": 3},
        verify_with_reference_ssim=True,
    )
    eval_records.append(rec_median)

    # -------------------------------------------------------------
    # Case 3: Motion Blur + Noise restored with Wiener Filter
    # -------------------------------------------------------------
    print("\n[Case 3] Benchmarking Motion Blur + Noise restored with Wiener Filter...")
    psf = create_motion_blur_kernel(length=9, angle=30.0)
    blurred = apply_motion_blur(clean_img, length=9, angle=30.0)
    noisy_blurred = add_gaussian_noise(blurred, sigma=0.01, seed=42)

    rest_wiener, t_wiener = time_function(apply_wiener_filter, noisy_blurred, psf, K=0.01)
    rec_wiener = evaluate_restoration(
        clean_img, noisy_blurred, rest_wiener,
        method_name="Wiener Filter (K=0.01)",
        runtime_ms=t_wiener,
        parameters={"K": 0.01, "motion_length": 9, "angle": 30.0},
        verify_with_reference_ssim=True,
    )
    eval_records.append(rec_wiener)

    # -------------------------------------------------------------
    # Format and Output Results Table
    # -------------------------------------------------------------
    df = results_to_dataframe(eval_records)
    print("\n" + "=" * 75)
    print("QUANTITATIVE BENCHMARK EVALUATION TABLE:")
    print("=" * 75)

    display_cols = [
        "method", "psnr_degraded", "psnr_restored", "delta_psnr",
        "ssim_degraded", "ssim_restored", "delta_ssim", "runtime_ms"
    ]
    print(df[display_cols].to_string(index=False))

    csv_path = out_dir / "phase5_benchmark_sample.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved benchmark metrics to: {csv_path}")

    print("\n" + "=" * 75)
    print("Phase 5 demonstration completed successfully!")
    print("=" * 75)


if __name__ == "__main__":
    main()
