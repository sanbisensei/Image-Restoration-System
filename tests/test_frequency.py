"""
Unit tests for Frequency Domain Restoration:
- DFT & Magnitude Spectrum
- PSF to OTF conversion
- Inverse Filtering
- Wiener Filtering
"""

import numpy as np
import pytest

from src.frequency.dft_visualization import (
    compute_dft2,
    compute_magnitude_spectrum,
    compute_phase_spectrum,
)
from src.frequency.psf import psf_to_otf, otf_to_psf
from src.frequency.inverse_filter import apply_inverse_filter
from src.frequency.wiener_filter import apply_wiener_filter
from src.degradation.gaussian_blur import create_gaussian_kernel, apply_gaussian_blur
from src.degradation.motion_blur import create_motion_blur_kernel, apply_motion_blur
from src.degradation.gaussian_noise import add_gaussian_noise


# ---------------------------------------------------------------------
# DFT & Spectrum Tests
# ---------------------------------------------------------------------

def test_dft_parsevals_theorem():
    # Spatial domain energy must equal frequency domain energy
    img = np.random.uniform(0.0, 1.0, (32, 32))
    f_raw, _ = compute_dft2(img)

    spatial_energy = np.sum(img ** 2)
    freq_energy = np.sum(np.abs(f_raw) ** 2) / (img.shape[0] * img.shape[1])

    assert np.isclose(spatial_energy, freq_energy)


def test_dft_dc_component_flat_image():
    # For a flat image, all energy is at DC (center of shifted spectrum)
    flat = np.ones((64, 64), dtype=np.float64) * 0.5
    _, f_shift = compute_dft2(flat)

    mag = np.abs(f_shift)
    cy, cx = flat.shape[0] // 2, flat.shape[1] // 2

    # Center is peak
    assert mag[cy, cx] == np.max(mag)
    # Non-DC components are approximately zero
    mag[cy, cx] = 0.0
    assert np.allclose(mag, 0.0, atol=1e-8)


def test_magnitude_spectrum_normalized():
    img = np.random.uniform(0.0, 1.0, (40, 40))
    _, f_shift = compute_dft2(img)
    mag_spec = compute_magnitude_spectrum(f_shift, log_scale=True)

    assert mag_spec.shape == img.shape
    assert 0.0 <= np.min(mag_spec)
    assert np.max(mag_spec) <= 1.0


# ---------------------------------------------------------------------
# PSF and OTF Tests
# ---------------------------------------------------------------------

def test_psf_impulse_otf_is_ones():
    # 3x3 impulse PSF centered at (1, 1)
    impulse_psf = np.zeros((3, 3), dtype=np.float64)
    impulse_psf[1, 1] = 1.0

    target_shape = (32, 32)
    otf = psf_to_otf(impulse_psf, target_shape)

    # An impulse at origin in space has a flat frequency response of all 1.0s
    assert np.allclose(otf, 1.0 + 0j)


def test_psf_dc_value_equals_kernel_sum():
    # The DC value H(0, 0) of the unshifted OTF is the sum of spatial kernel weights
    psf = create_gaussian_kernel(kernel_size=7, sigma=1.5)
    otf = psf_to_otf(psf, (64, 64))

    assert np.isclose(otf[0, 0].real, np.sum(psf))
    assert np.isclose(otf[0, 0].imag, 0.0)


def test_psf_otf_roundtrip():
    original_psf = create_gaussian_kernel(kernel_size=5, sigma=1.2)
    otf = psf_to_otf(original_psf, (64, 64))
    recovered_psf = otf_to_psf(otf, original_psf.shape)

    assert np.allclose(original_psf, recovered_psf, atol=1e-7)


# ---------------------------------------------------------------------
# Inverse Filter Tests
# ---------------------------------------------------------------------

def test_inverse_filter_noise_free_recovery():
    # In the absence of noise, inverse filtering must restore the original image
    # Use synthetic smooth pattern to avoid discrete boundary ringing
    y, x = np.mgrid[0:64, 0:64]
    clean = 0.5 + 0.4 * np.sin(2 * np.pi * x / 16.0)

    psf = create_gaussian_kernel(kernel_size=5, sigma=1.0)
    blurred = apply_gaussian_blur(clean, kernel_size=5, sigma=1.0, mode="reflect")

    restored = apply_inverse_filter(blurred, psf, threshold=0.01)

    # Restored image should closely match the clean image in the interior
    # (avoiding slight boundary border differences from circular FFT vs reflect padding)
    interior_clean = clean[8:56, 8:56]
    interior_restored = restored[8:56, 8:56]
    max_error = np.max(np.abs(interior_clean - interior_restored))

    assert max_error < 0.08, f"Expected accurate noise-free restoration, got max error {max_error}"


def test_inverse_filter_noise_amplification_demonstration():
    # Demonstrating the fundamental academic failure mode of inverse filtering
    clean = np.ones((64, 64), dtype=np.float64) * 0.5
    clean[20:44, 20:44] = 0.8

    psf = create_motion_blur_kernel(length=9, angle=0.0)
    blurred = apply_motion_blur(clean, length=9, angle=0.0)

    # Add very small Gaussian noise (sigma=0.01)
    noisy_blurred = add_gaussian_noise(blurred, sigma=0.01, seed=42)

    # Direct unconstrained inverse filter vs thresholded inverse filter
    unconstrained = apply_inverse_filter(noisy_blurred, psf, threshold=None)
    thresholded = apply_inverse_filter(noisy_blurred, psf, threshold=0.05)

    # Unconstrained inverse filter noise explodes, causing extreme variance
    var_unconstrained = np.var(unconstrained)
    var_thresholded = np.var(thresholded)

    # Thresholded filter should heavily curb the explosive variance
    assert var_thresholded < var_unconstrained


# ---------------------------------------------------------------------
# Wiener Filter Tests
# ---------------------------------------------------------------------

def test_wiener_filter_noise_free_recovery():
    y, x = np.mgrid[0:64, 0:64]
    clean = 0.5 + 0.4 * np.sin(2 * np.pi * x / 16.0)

    psf = create_gaussian_kernel(kernel_size=5, sigma=1.0)
    blurred = apply_gaussian_blur(clean, kernel_size=5, sigma=1.0)

    # Very small K (approaching inverse filter)
    restored = apply_wiener_filter(blurred, psf, K=1e-4)

    interior_clean = clean[8:56, 8:56]
    interior_restored = restored[8:56, 8:56]
    max_error = np.max(np.abs(interior_clean - interior_restored))

    assert max_error < 0.08


def test_wiener_vs_inverse_on_noisy_image():
    # Key comparison: On an image with BOTH blur AND noise,
    # Wiener filtering decisively outperforms direct inverse filtering!
    clean = np.ones((64, 64), dtype=np.float64) * 0.5
    clean[16:48, 16:48] = 0.9

    psf = create_gaussian_kernel(kernel_size=7, sigma=1.5)
    blurred = apply_gaussian_blur(clean, kernel_size=7, sigma=1.5)
    noisy_blurred = add_gaussian_noise(blurred, sigma=0.02, seed=123)

    inv_restored = apply_inverse_filter(noisy_blurred, psf, threshold=None)
    wiener_restored = apply_wiener_filter(noisy_blurred, psf, K=0.02)

    # Mean squared error against ground truth clean image
    mse_inverse = np.mean((clean - inv_restored) ** 2)
    mse_wiener = np.mean((clean - wiener_restored) ** 2)

    assert mse_wiener < mse_inverse, "Wiener filter MSE must be significantly lower than unconstrained inverse filter."


def test_wiener_filter_rgb():
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[8:24, 8:24, 0] = 200
    rgb[8:24, 8:24, 1] = 100
    rgb[8:24, 8:24, 2] = 50

    psf = create_gaussian_kernel(kernel_size=5, sigma=1.0)
    blurred = apply_gaussian_blur(rgb, kernel_size=5, sigma=1.0)
    restored = apply_wiener_filter(blurred, psf, K=0.01)

    assert restored.shape == (32, 32, 3)
    assert restored.dtype == np.uint8
