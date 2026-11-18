"""
Unit tests for custom 2D convolution engine.
"""

import numpy as np
import pytest
from src.spatial.convolution import convolve2d, convolve2d_grayscale


def test_convolution_identity_kernel():
    # 5x5 test image
    img = np.arange(25, dtype=np.float64).reshape((5, 5))
    # 3x3 identity kernel
    identity = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0]
    ])

    out = convolve2d_grayscale(img, identity, mode="reflect")
    assert np.allclose(img, out)


def test_convolution_known_small_matrix():
    # 3x3 simple image
    img = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])

    # 3x3 smoothing kernel
    box_kernel = np.ones((3, 3), dtype=np.float64) / 9.0

    # With mode='constant' (padded with zeros):
    # Center pixel (row 1, col 1): sum of all 9 elements = 45 / 9 = 5.0
    out_constant = convolve2d_grayscale(img, box_kernel, mode="constant")
    assert np.isclose(out_constant[1, 1], 5.0)

    # Top-left pixel with constant zero-padding:
    # Neighbors of (0, 0): (0,0)=1, (0,1)=2, (1,0)=4, (1,1)=5, rest 0
    # Sum = 1 + 2 + 4 + 5 = 12 -> 12 / 9 = 1.33333
    assert np.isclose(out_constant[0, 0], 12.0 / 9.0)


def test_convolution_kernel_flip_mathematical_correctness():
    # Asymmetric 3x3 kernel
    # In true convolution, kernel is flipped horizontally & vertically:
    # (f * h)(x, y) = sum f(x - s, y - t) * h(s, t)
    kernel = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])

    # Impulse image: 1.0 at center, 0 elsewhere
    impulse = np.zeros((5, 5), dtype=np.float64)
    impulse[2, 2] = 1.0

    # The convolution of an impulse with a filter should yield the flipped filter centered at the impulse
    conv_out = convolve2d_grayscale(impulse, kernel, mode="constant", flip_kernel=True)
    corr_out = convolve2d_grayscale(impulse, kernel, mode="constant", flip_kernel=False)

    # Extracted 3x3 response at center
    resp_conv = conv_out[1:4, 1:4]
    resp_corr = corr_out[1:4, 1:4]

    # Fundamental DSP theorem: The convolution of an impulse with a filter yields
    # the Point Spread Function h itself: delta * h = h.
    # Conversely, correlation without kernel flipping yields the flipped kernel: delta ★ h = h(-x, -y).
    assert np.allclose(resp_conv, kernel)
    assert np.allclose(resp_corr, np.flip(kernel))


def test_convolution_rgb_multi_channel():
    img_rgb = np.zeros((10, 10, 3), dtype=np.float64)
    img_rgb[:, :, 0] = 1.0
    img_rgb[:, :, 1] = 2.0
    img_rgb[:, :, 2] = 3.0

    box = np.ones((3, 3), dtype=np.float64) / 9.0
    out = convolve2d(img_rgb, box, mode="reflect")

    assert out.shape == (10, 10, 3)
    assert np.allclose(out[:, :, 0], 1.0)
    assert np.allclose(out[:, :, 1], 2.0)
    assert np.allclose(out[:, :, 2], 3.0)


def test_convolution_rejects_even_kernel():
    img = np.zeros((10, 10), dtype=np.float64)
    even_kernel = np.ones((4, 4))
    with pytest.raises(ValueError, match="Kernel dimensions must be odd"):
        convolve2d_grayscale(img, even_kernel)
