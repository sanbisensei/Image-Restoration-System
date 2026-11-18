"""
Phase 4 Frequency Domain Restoration Demonstration Script.

Demonstrates:
1. 2D Discrete Fourier Transform (DFT) & Log Magnitude Spectrum Visualization.
2. Frequency Response (OTF) of Gaussian and Motion Blur PSFs:
   - Shows characteristic zero-crossing null bands of motion blur.
3. Deblurring Comparison under Noise:
   - Direct Inverse Filter (proves catastrophic noise amplification)
   - Thresholded Inverse Filter (truncated stabilization)
   - Wiener Filter (K=0.001, K=0.01, K=0.10) (demonstrates graceful noise-blur trade-off)

Outputs saved to data/restored/ and outputs/phase4_frequency_restoration.png.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src.image_io import load_image, save_image
from src.degradation.gaussian_noise import add_gaussian_noise
from src.degradation.gaussian_blur import create_gaussian_kernel, apply_gaussian_blur
from src.degradation.motion_blur import create_motion_blur_kernel, apply_motion_blur
from src.frequency.dft_visualization import compute_dft2, compute_magnitude_spectrum
from src.frequency.psf import psf_to_otf
from src.frequency.inverse_filter import apply_inverse_filter
from src.frequency.wiener_filter import apply_wiener_filter


def main():
    print("=" * 70)
    print("CSE 438 — Image Restoration System: Phase 4 Frequency Domain")
    print("=" * 70)

    data_dir = Path("data/original")
    rest_dir = Path("data/restored")
    out_dir = Path("outputs")

    rest_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load clean image
    img_path = data_dir / "checkerboard.png"
    clean_img = load_image(img_path, as_gray=True)
    print(f"\nLoaded clean image: {img_path.name}")

    # 2. Compute 2D DFT and Centered Magnitude Spectrum
    _, f_shift = compute_dft2(clean_img)
    clean_spec = compute_magnitude_spectrum(f_shift, log_scale=True)
    print("Computed 2D DFT & Log-compressed Magnitude Spectrum.")

    # 3. Simulate Degradation: Motion Blur + Small Gaussian Noise
    psf_motion = create_motion_blur_kernel(length=11, angle=45.0)
    blurred_motion = apply_motion_blur(clean_img, length=11, angle=45.0)
    noisy_blurred = add_gaussian_noise(blurred_motion, sigma=0.01, seed=42)

    # Compute OTF of the motion blur PSF for frequency visualization
    otf_motion = psf_to_otf(psf_motion, clean_img.shape)
    otf_shift = np.fft.fftshift(otf_motion)
    otf_spec = compute_magnitude_spectrum(otf_shift, log_scale=True)

    # 4. Apply Frequency Restoration
    print("\n[A] Applying Direct Inverse Filtering (demonstrating noise explosion)...")
    inv_direct = apply_inverse_filter(noisy_blurred, psf_motion, threshold=None)

    print("[B] Applying Thresholded Inverse Filtering (threshold=0.05)...")
    inv_thresh = apply_inverse_filter(noisy_blurred, psf_motion, threshold=0.05)

    print("[C] Applying Wiener Filtering across K parameter values...")
    wiener_low_k = apply_wiener_filter(noisy_blurred, psf_motion, K=0.001)
    wiener_opt_k = apply_wiener_filter(noisy_blurred, psf_motion, K=0.01)
    wiener_high_k = apply_wiener_filter(noisy_blurred, psf_motion, K=0.10)

    # Save restored images
    save_image(rest_dir / "motion_inv_direct.png", inv_direct)
    save_image(rest_dir / "motion_inv_thresholded.png", inv_thresh)
    save_image(rest_dir / "motion_wiener_opt.png", wiener_opt_k)
    print("Saved restored images to data/restored/")

    # 5. Visual Grid Comparison
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))

    # Row 1: Degradation & Spectra
    axes[0, 0].imshow(clean_img, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("1. Clean Original", fontsize=10)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(clean_spec, cmap="inferno", vmin=0.0, vmax=1.0)
    axes[0, 1].set_title("2. DFT Spectrum |F(u,v)|", fontsize=10)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(otf_spec, cmap="inferno", vmin=0.0, vmax=1.0)
    axes[0, 2].set_title("3. Motion OTF |H(u,v)| (Nulls)", fontsize=10)
    axes[0, 2].axis("off")

    axes[0, 3].imshow(noisy_blurred, cmap="gray", vmin=0, vmax=255)
    axes[0, 3].set_title("4. Degraded (Motion + Noise)", fontsize=10)
    axes[0, 3].axis("off")

    # Row 2: Frequency Restorations
    axes[1, 0].imshow(inv_direct, cmap="gray", vmin=0, vmax=255)
    axes[1, 0].set_title("5. Direct Inverse (Noise Blizzard)", fontsize=10)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(inv_thresh, cmap="gray", vmin=0, vmax=255)
    axes[1, 1].set_title("6. Thresholded Inverse", fontsize=10)
    axes[1, 1].axis("off")

    axes[1, 2].imshow(wiener_opt_k, cmap="gray", vmin=0, vmax=255)
    axes[1, 2].set_title("7. Wiener (Optimal K=0.01)", fontsize=10)
    axes[1, 2].axis("off")

    axes[1, 3].imshow(wiener_high_k, cmap="gray", vmin=0, vmax=255)
    axes[1, 3].set_title("8. Wiener (High K=0.10, Smooth)", fontsize=10)
    axes[1, 3].axis("off")

    plt.suptitle("CSE 438 DIP — Phase 4 Frequency Domain Restoration: Inverse vs Wiener Filtering", fontsize=13, y=0.98)
    plt.tight_layout()

    out_fig = out_dir / "phase4_frequency_restoration.png"
    plt.savefig(out_fig, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"\nGenerated comparison figure: {out_fig}")

    print("\n" + "=" * 70)
    print("Phase 4 demonstration completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
