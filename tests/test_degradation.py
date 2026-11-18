"""
Unit tests for degradation simulator models:
- Gaussian Noise
- Salt-and-Pepper Noise
- Gaussian Blur
- Motion Blur
"""

import numpy as np
import pytest
from src.degradation.gaussian_noise import add_gaussian_noise
from src.degradation.salt_pepper import add_salt_and_pepper_noise
from src.degradation.gaussian_blur import create_gaussian_kernel, apply_gaussian_blur
from src.degradation.motion_blur import create_motion_blur_kernel, apply_motion_blur


# ---------------------------------------------------------
# Gaussian Noise Tests
# ---------------------------------------------------------

def test_gaussian_noise_reproducibility():
    img = np.ones((50, 50), dtype=np.uint8) * 128
    noisy1 = add_gaussian_noise(img, sigma=0.05, seed=42)
    noisy2 = add_gaussian_noise(img, sigma=0.05, seed=42)
    noisy_diff_seed = add_gaussian_noise(img, sigma=0.05, seed=99)

    assert np.array_equal(noisy1, noisy2), "Identical seed must yield identical noise."
    assert not np.array_equal(noisy1, noisy_diff_seed), "Different seeds should produce different noise."


def test_gaussian_noise_zero_sigma():
    img = np.array([[50, 100], [150, 200]], dtype=np.uint8)
    noisy = add_gaussian_noise(img, sigma=0.0, seed=123)
    assert np.array_equal(img, noisy)


def test_gaussian_noise_clipping_bounds():
    # High noise on extremes (0 and 255)
    img_u8 = np.zeros((40, 40), dtype=np.uint8)
    noisy_u8 = add_gaussian_noise(img_u8, sigma=0.5, seed=7)
    assert noisy_u8.min() >= 0
    assert noisy_u8.max() <= 255
    assert noisy_u8.dtype == np.uint8

    img_flt = np.ones((40, 40), dtype=np.float64)
    noisy_flt = add_gaussian_noise(img_flt, sigma=0.5, seed=7)
    assert noisy_flt.min() >= 0.0
    assert noisy_flt.max() <= 1.0
    assert noisy_flt.dtype == np.float64


def test_gaussian_noise_statistics():
    # Large flat image to verify empirical variance matches sigma^2
    target_sigma = 0.08
    img = np.ones((300, 300), dtype=np.float64) * 0.5
    noisy = add_gaussian_noise(img, sigma=target_sigma, seed=12345)

    noise_residual = noisy - img
    measured_mean = np.mean(noise_residual)
    measured_std = np.std(noise_residual)

    # Within 3 standard errors of the sample mean & std
    assert np.isclose(measured_mean, 0.0, atol=0.01)
    assert np.isclose(measured_std, target_sigma, atol=0.005)


# ---------------------------------------------------------
# Salt-and-Pepper Noise Tests
# ---------------------------------------------------------

def test_salt_pepper_reproducibility():
    img = np.ones((50, 50), dtype=np.uint8) * 128
    sp1 = add_salt_and_pepper_noise(img, density=0.1, seed=42)
    sp2 = add_salt_and_pepper_noise(img, density=0.1, seed=42)
    assert np.array_equal(sp1, sp2)


def test_salt_pepper_extreme_values():
    # Gray image (value 128)
    img = np.ones((100, 100), dtype=np.uint8) * 128
    sp = add_salt_and_pepper_noise(img, density=0.2, salt_ratio=0.5, seed=10)

    # Unique pixel values should only be: 128 (uncorrupted), 0 (pepper), 255 (salt)
    unique_vals = set(np.unique(sp))
    assert unique_vals.issubset({0, 128, 255})

    # Density verification: approximately 20% corrupted (within binomial confidence)
    corrupted_ratio = np.mean(sp != 128)
    assert np.isclose(corrupted_ratio, 0.20, atol=0.03)


def test_salt_pepper_zero_density():
    img = np.arange(25, dtype=np.uint8).reshape((5, 5))
    sp = add_salt_and_pepper_noise(img, density=0.0)
    assert np.array_equal(img, sp)


# ---------------------------------------------------------
# Gaussian Blur Tests
# ---------------------------------------------------------

def test_gaussian_kernel_properties():
    k = create_gaussian_kernel(kernel_size=7, sigma=1.5)
    assert k.shape == (7, 7)
    # Energy conservation
    assert np.isclose(np.sum(k), 1.0)
    # Radial symmetry: k == k.T and k == flipped(k)
    assert np.allclose(k, k.T)
    assert np.allclose(k, np.flip(k))
    # Center must be peak
    assert k[3, 3] == np.max(k)


def test_gaussian_blur_flat_image():
    # Blurring a uniform flat surface should not change pixel values
    flat = np.ones((30, 30), dtype=np.uint8) * 150
    blurred = apply_gaussian_blur(flat, kernel_size=5, sigma=1.0)
    assert np.allclose(flat, blurred)


def test_gaussian_blur_smooths_edge():
    # Step edge from 0 to 255
    edge = np.zeros((20, 20), dtype=np.float64)
    edge[:, 10:] = 1.0

    blurred = apply_gaussian_blur(edge, kernel_size=5, sigma=1.2)
    # The sharp step at column 10 should now have intermediate transition values
    assert 0.0 < blurred[10, 9] < 1.0
    assert 0.0 < blurred[10, 10] < 1.0


# ---------------------------------------------------------
# Motion Blur Tests
# ---------------------------------------------------------

def test_motion_blur_kernel_properties():
    k_horiz = create_motion_blur_kernel(length=7, angle=0.0)
    assert np.isclose(np.sum(k_horiz), 1.0)

    # For pure horizontal motion (angle 0°), all energy should be in the center row
    center = k_horiz.shape[0] // 2
    assert np.isclose(np.sum(k_horiz[center, :]), 1.0)

    # For pure vertical motion (angle 90°), all energy in center column
    k_vert = create_motion_blur_kernel(length=7, angle=90.0)
    assert np.isclose(np.sum(k_vert), 1.0)
    assert np.isclose(np.sum(k_vert[:, center]), 1.0)


def test_motion_blur_flat_image():
    flat = np.ones((25, 25), dtype=np.uint8) * 180
    blurred = apply_motion_blur(flat, length=9, angle=45.0)
    assert np.allclose(flat, blurred)


def test_motion_blur_direction():
    # Vertical line image: blur horizontally (angle 0°) should spread line horizontally
    line_img = np.zeros((31, 31), dtype=np.float64)
    line_img[:, 15] = 1.0

    blurred = apply_motion_blur(line_img, length=7, angle=0.0)
    # Energy should have spread to neighbors on the same row
    center_row = blurred[15, :]
    assert center_row[14] > 0.0
    assert center_row[16] > 0.0
