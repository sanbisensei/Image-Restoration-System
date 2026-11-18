"""
Unit tests for Spatial Domain Filters:
- Mean Filter (3x3, 5x5, 7x7)
- Gaussian Smoothing Filter
- Median Filter
"""

import numpy as np
import pytest
from src.spatial.mean_filter import create_mean_kernel, apply_mean_filter
from src.spatial.gaussian_filter import apply_gaussian_filter
from src.spatial.median_filter import apply_median_filter, apply_median_filter_grayscale


# ---------------------------------------------------------------------
# Mean Filter Tests
# ---------------------------------------------------------------------

@pytest.mark.parametrize("ksize", [3, 5, 7])
def test_mean_kernel_properties(ksize):
    k = create_mean_kernel(ksize)
    assert k.shape == (ksize, ksize)
    assert np.isclose(np.sum(k), 1.0)
    assert np.allclose(k, 1.0 / (ksize * ksize))


def test_mean_filter_known_matrix():
    # 3x3 image with single impulse of 100 in sea of 10s
    img = np.full((3, 3), 10, dtype=np.uint8)
    img[1, 1] = 100

    # Under 3x3 mean filter, center becomes (8*10 + 100) / 9 = 180 / 9 = 20
    out = apply_mean_filter(img, kernel_size=3, mode="constant")
    assert out[1, 1] == 20


def test_mean_filter_noise_suppression():
    # Random zero-mean Gaussian noise on flat background
    rng = np.random.default_rng(42)
    clean = np.ones((100, 100), dtype=np.float64) * 0.5
    noisy = clean + rng.normal(0, 0.1, clean.shape)

    filtered_3x3 = apply_mean_filter(noisy, kernel_size=3)
    filtered_5x5 = apply_mean_filter(noisy, kernel_size=5)

    var_noisy = np.var(noisy)
    var_3x3 = np.var(filtered_3x3)
    var_5x5 = np.var(filtered_5x5)

    # 3x3 filter should reduce variance by ~9x; 5x5 by ~25x
    assert var_3x3 < var_noisy / 5.0
    assert var_5x5 < var_3x3


# ---------------------------------------------------------------------
# Gaussian Filter Tests
# ---------------------------------------------------------------------

def test_gaussian_filter_flat_image():
    flat = np.full((30, 30), 120, dtype=np.uint8)
    smoothed = apply_gaussian_filter(flat, kernel_size=5, sigma=1.0)
    assert np.allclose(flat, smoothed)


def test_gaussian_filter_smoothing_progression():
    # Step edge from 0.0 to 1.0
    edge = np.zeros((30, 30), dtype=np.float64)
    edge[:, 15:] = 1.0

    smooth_small_sigma = apply_gaussian_filter(edge, kernel_size=7, sigma=0.8)
    smooth_large_sigma = apply_gaussian_filter(edge, kernel_size=7, sigma=2.5)

    # Higher sigma spreads the boundary further from column 15
    # Value at column 12 (3 pixels left of edge) should be higher with larger sigma
    assert smooth_large_sigma[15, 12] > smooth_small_sigma[15, 12]


def test_gaussian_filter_rgb():
    rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    rgb[:, :, 0] = 50
    rgb[:, :, 1] = 100
    rgb[:, :, 2] = 200

    out = apply_gaussian_filter(rgb, kernel_size=5, sigma=1.0)
    assert out.shape == (20, 20, 3)
    assert np.allclose(out[:, :, 0], 50)
    assert np.allclose(out[:, :, 1], 100)
    assert np.allclose(out[:, :, 2], 200)


# ---------------------------------------------------------------------
# Median Filter Tests
# ---------------------------------------------------------------------

def test_median_filter_known_matrix():
    # 3x3 matrix:
    # 10,  20, 30
    # 40, 100, 60
    # 70,  80, 90
    # Sorted 9 elements: 10, 20, 30, 40, 60, 70, 80, 90, 100
    # Median is 60!
    mat = np.array([
        [10, 20, 30],
        [40, 100, 60],
        [70, 80, 90]
    ], dtype=np.uint8)

    filtered = apply_median_filter_grayscale(mat, kernel_size=3, mode="constant")
    assert filtered[1, 1] == 60


def test_median_filter_impulse_noise_elimination():
    # Black image with isolated white salt pixel
    black_img = np.zeros((20, 20), dtype=np.uint8)
    black_img[10, 10] = 255

    med = apply_median_filter(black_img, kernel_size=3)
    # The salt pixel must be completely removed by the median filter
    assert med[10, 10] == 0
    assert np.all(med == 0)

    # White image with isolated black pepper pixel
    white_img = np.full((20, 20), 255, dtype=np.uint8)
    white_img[10, 10] = 0

    med_white = apply_median_filter(white_img, kernel_size=3)
    # The pepper pixel must be completely removed
    assert med_white[10, 10] == 255
    assert np.all(med_white == 255)


def test_median_filter_step_edge_preservation():
    # Vertical step edge: columns 0..9 are 0, columns 10..19 are 200
    edge = np.zeros((20, 20), dtype=np.uint8)
    edge[:, 10:] = 200

    med_filtered = apply_median_filter(edge, kernel_size=3, mode="reflect")

    # The step edge remains strictly preserved: no blurry intermediate values!
    unique_values = np.unique(med_filtered)
    assert set(unique_values).issubset({0, 200}), f"Expected sharp edge with only {0, 200}, got {unique_values}"

    # In contrast, mean filter produces intermediate blurry values:
    mean_filtered = apply_mean_filter(edge, kernel_size=3, mode="reflect")
    assert len(np.unique(mean_filtered)) > 2


@pytest.mark.parametrize("ksize", [3, 5, 7])
def test_median_filter_various_kernel_sizes(ksize):
    img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
    out = apply_median_filter(img, kernel_size=ksize)
    assert out.shape == img.shape
    assert out.dtype == img.dtype


def test_median_filter_rejects_even_kernel():
    img = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError, match="odd positive integer"):
        apply_median_filter(img, kernel_size=4)
