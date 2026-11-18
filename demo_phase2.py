"""
Phase 2 Degradation Simulator Demonstration Script.

Applies all 4 classical degradation models to clean benchmark images:
1. Gaussian Noise (Low, Medium, High)
2. Salt-and-Pepper Impulse Noise (Low, Medium, High)
3. Gaussian Optical Blur (Mild, Moderate, Severe)
4. Directional Motion Blur (Horizontal, Diagonal, Vertical)

Saves degraded output images to data/degraded/ and generates a comprehensive
visual comparison grid in outputs/phase2_degradation_grid.png.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src.image_io import load_image, save_image
from src.degradation.gaussian_noise import add_gaussian_noise
from src.degradation.salt_pepper import add_salt_and_pepper_noise
from src.degradation.gaussian_blur import apply_gaussian_blur, create_gaussian_kernel
from src.degradation.motion_blur import apply_motion_blur, create_motion_blur_kernel


def main():
    print("=" * 65)
    print("CSE 438 — Image Restoration System: Phase 2 Degradation Simulator")
    print("=" * 65)

    data_dir = Path("data/original")
    deg_dir = Path("data/degraded")
    out_dir = Path("outputs")

    deg_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load clean benchmark image
    img_path = data_dir / "geometric_shapes.png"
    if not img_path.exists():
        img_path = data_dir / "checkerboard.png"

    clean_img = load_image(img_path, as_gray=True)
    print(f"\nLoaded clean image: {img_path.name} (shape {clean_img.shape})")

    # -------------------------------------------------------------
    # 1. Gaussian Noise (Additive Thermal/Sensor Noise)
    # -------------------------------------------------------------
    print("\n[1] Simulating Gaussian Noise...")
    g_low = add_gaussian_noise(clean_img, sigma=0.03, seed=42)
    g_med = add_gaussian_noise(clean_img, sigma=0.08, seed=42)
    g_high = add_gaussian_noise(clean_img, sigma=0.18, seed=42)
    save_image(deg_dir / "geometric_gaussian_noise_med.png", g_med)
    print("    Saved medium Gaussian noise image.")

    # -------------------------------------------------------------
    # 2. Salt-and-Pepper Noise (Bipolar Impulse Noise)
    # -------------------------------------------------------------
    print("\n[2] Simulating Salt-and-Pepper Noise...")
    sp_low = add_salt_and_pepper_noise(clean_img, density=0.02, seed=42)
    sp_med = add_salt_and_pepper_noise(clean_img, density=0.08, seed=42)
    sp_high = add_salt_and_pepper_noise(clean_img, density=0.20, seed=42)
    save_image(deg_dir / "geometric_salt_pepper_med.png", sp_med)
    print("    Saved medium Salt-and-Pepper noise image.")

    # -------------------------------------------------------------
    # 3. Gaussian Blur (Atmospheric / Defocus Blur)
    # -------------------------------------------------------------
    print("\n[3] Simulating Gaussian Optical Blur (via custom 2D convolution)...")
    gb_mild = apply_gaussian_blur(clean_img, kernel_size=5, sigma=1.0)
    gb_mod = apply_gaussian_blur(clean_img, kernel_size=9, sigma=2.0)
    gb_sev = apply_gaussian_blur(clean_img, kernel_size=15, sigma=3.5)
    save_image(deg_dir / "geometric_gaussian_blur_mod.png", gb_mod)
    print("    Saved moderate Gaussian blur image.")

    # -------------------------------------------------------------
    # 4. Motion Blur (Directional Camera Translation)
    # -------------------------------------------------------------
    print("\n[4] Simulating Directional Motion Blur (via custom 2D convolution)...")
    mb_horiz = apply_motion_blur(clean_img, length=11, angle=0.0)      # 0 deg: horizontal
    mb_diag = apply_motion_blur(clean_img, length=15, angle=45.0)      # 45 deg: diagonal
    mb_vert = apply_motion_blur(clean_img, length=15, angle=90.0)      # 90 deg: vertical
    save_image(deg_dir / "geometric_motion_blur_diag.png", mb_diag)
    print("    Saved diagonal motion blur image.")

    # -------------------------------------------------------------
    # Generate 4x4 Visual Comparison Figure
    # -------------------------------------------------------------
    grid_images = [
        # Row 1: Gaussian Noise
        (clean_img, "Original (Clean)"),
        (g_low, "Gaussian Noise (σ=0.03)"),
        (g_med, "Gaussian Noise (σ=0.08)"),
        (g_high, "Gaussian Noise (σ=0.18)"),

        # Row 2: Salt & Pepper Noise
        (clean_img, "Original (Clean)"),
        (sp_low, "Salt & Pepper (d=0.02)"),
        (sp_med, "Salt & Pepper (d=0.08)"),
        (sp_high, "Salt & Pepper (d=0.20)"),

        # Row 3: Gaussian Blur
        (clean_img, "Original (Clean)"),
        (gb_mild, "Gaussian Blur (K=5, σ=1.0)"),
        (gb_mod, "Gaussian Blur (K=9, σ=2.0)"),
        (gb_sev, "Gaussian Blur (K=15, σ=3.5)"),

        # Row 4: Motion Blur
        (clean_img, "Original (Clean)"),
        (mb_horiz, "Motion (L=11, 0° Horiz)"),
        (mb_diag, "Motion (L=15, 45° Diag)"),
        (mb_vert, "Motion (L=15, 90° Vert)"),
    ]

    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    for ax, (img, title) in zip(axes.flatten(), grid_images):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=10, pad=6)
        ax.axis("off")

    plt.suptitle("CSE 438 DIP — Classical Degradation Models Comparison Grid", fontsize=14, y=0.99)
    plt.tight_layout()

    out_grid_path = out_dir / "phase2_degradation_grid.png"
    plt.savefig(out_grid_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"\nGenerated comprehensive visual comparison grid: {out_grid_path}")

    print("\n" + "=" * 65)
    print("Phase 2 demonstration completed successfully!")
    print("=" * 65)


if __name__ == "__main__":
    main()
