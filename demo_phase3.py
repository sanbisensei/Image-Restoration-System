"""
Phase 3 Spatial Filters Demonstration Script.

Demonstrates:
1. Mean (Box) Filter (3x3, 5x5, 7x7) on Gaussian noise.
2. Gaussian Smoothing Filter (sigma=0.8, 1.5, 2.5) on Gaussian noise.
3. Decisive Comparison on Salt-and-Pepper Impulse Noise:
   - Mean Filter (blurs impulses into gray smudge artifacts)
   - Gaussian Filter (blurs impulses into halos)
   - Median Filter (completely eliminates impulse spikes, perfectly preserving sharp edges)

Outputs are saved to data/restored/ and outputs/phase3_spatial_filtering.png.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src.image_io import load_image, save_image
from src.degradation.gaussian_noise import add_gaussian_noise
from src.degradation.salt_pepper import add_salt_and_pepper_noise
from src.spatial.mean_filter import apply_mean_filter
from src.spatial.gaussian_filter import apply_gaussian_filter
from src.spatial.median_filter import apply_median_filter


def main():
    print("=" * 65)
    print("CSE 438 — Image Restoration System: Phase 3 Spatial Filters")
    print("=" * 65)

    data_dir = Path("data/original")
    rest_dir = Path("data/restored")
    out_dir = Path("outputs")

    rest_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load clean image
    img_path = data_dir / "geometric_shapes.png"
    if not img_path.exists():
        img_path = data_dir / "checkerboard.png"

    clean_img = load_image(img_path, as_gray=True)
    print(f"\nLoaded clean image: {img_path.name}")

    # 2. Synthesize test degradations
    # Gaussian noise (sigma=0.08)
    noisy_gauss = add_gaussian_noise(clean_img, sigma=0.08, seed=42)
    # Salt-and-pepper noise (density=0.10)
    noisy_sp = add_salt_and_pepper_noise(clean_img, density=0.10, seed=42)

    # -------------------------------------------------------------
    # Experiment A: Mean Filter Progression on Gaussian Noise
    # -------------------------------------------------------------
    print("\n[A] Testing Mean Filter (3x3, 5x5, 7x7) on Gaussian Noise...")
    mean_3 = apply_mean_filter(noisy_gauss, kernel_size=3)
    mean_5 = apply_mean_filter(noisy_gauss, kernel_size=5)
    mean_7 = apply_mean_filter(noisy_gauss, kernel_size=7)

    # -------------------------------------------------------------
    # Experiment B: Gaussian Filter Progression on Gaussian Noise
    # -------------------------------------------------------------
    print("\n[B] Testing Gaussian Filter (K=5,7,9) on Gaussian Noise...")
    gauss_mild = apply_gaussian_filter(noisy_gauss, kernel_size=5, sigma=1.0)
    gauss_mod = apply_gaussian_filter(noisy_gauss, kernel_size=7, sigma=1.8)
    gauss_strong = apply_gaussian_filter(noisy_gauss, kernel_size=9, sigma=2.8)

    # -------------------------------------------------------------
    # Experiment C: Impulse Noise Comparison (Mean vs Gauss vs Median)
    # -------------------------------------------------------------
    print("\n[C] Comparing Filters on Salt-and-Pepper Impulse Noise...")
    sp_mean_3 = apply_mean_filter(noisy_sp, kernel_size=3)
    sp_gauss_3 = apply_gaussian_filter(noisy_sp, kernel_size=5, sigma=1.0)
    sp_median_3 = apply_median_filter(noisy_sp, kernel_size=3)
    sp_median_5 = apply_median_filter(noisy_sp, kernel_size=5)

    # Save restored samples
    save_image(rest_dir / "restored_gauss_noise_mean3.png", mean_3)
    save_image(rest_dir / "restored_gauss_noise_gauss5.png", gauss_mild)
    save_image(rest_dir / "restored_salt_pepper_median3.png", sp_median_3)
    print("    Saved restored images to data/restored/")

    # -------------------------------------------------------------
    # Generate Visual Comparison Figure
    # -------------------------------------------------------------
    fig, axes = plt.subplots(3, 4, figsize=(14, 11))

    # Row 1: Mean Filter on Gaussian Noise
    row1 = [
        (clean_img, "1. Clean Reference"),
        (noisy_gauss, "2. Noisy (Gaussian σ=0.08)"),
        (mean_3, "3. Mean Filter 3×3"),
        (mean_7, "4. Mean Filter 7×7 (Heavy Blur)"),
    ]
    for ax, (img, title) in zip(axes[0], row1):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=10, pad=6)
        ax.axis("off")

    # Row 2: Gaussian Filter on Gaussian Noise
    row2 = [
        (clean_img, "1. Clean Reference"),
        (noisy_gauss, "2. Noisy (Gaussian σ=0.08)"),
        (gauss_mild, "3. Gaussian Filter (σ=1.0)"),
        (gauss_strong, "4. Gaussian Filter (σ=2.8)"),
    ]
    for ax, (img, title) in zip(axes[1], row2):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=10, pad=6)
        ax.axis("off")

    # Row 3: Impulse Noise Showdown (Mean vs Gaussian vs Median)
    row3 = [
        (noisy_sp, "1. Salt & Pepper (10% noise)"),
        (sp_mean_3, "2. Mean 3×3 (Smears Spikes)"),
        (sp_gauss_3, "3. Gaussian (Halos Spikes)"),
        (sp_median_3, "4. Median 3×3 (Spikes Removed!)"),
    ]
    for ax, (img, title) in zip(axes[2], row3):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=10, pad=6)
        ax.axis("off")

    plt.suptitle("CSE 438 DIP — Phase 3 Spatial Filtering Performance & Comparisons", fontsize=13, y=0.99)
    plt.tight_layout()

    out_fig = out_dir / "phase3_spatial_filtering.png"
    plt.savefig(out_fig, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"\nGenerated comparison figure: {out_fig}")

    print("\n" + "=" * 65)
    print("Phase 3 demonstration completed successfully!")
    print("=" * 65)


if __name__ == "__main__":
    main()
