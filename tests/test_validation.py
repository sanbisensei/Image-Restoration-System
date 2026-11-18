"""
Unit tests for image validation utilities.
"""

import numpy as np
import pytest
from src.utils.validation import (
    validate_image,
    is_valid_grayscale,
    is_valid_rgb,
    get_image_info,
)


def test_validate_image_valid_grayscale():
    img = np.zeros((100, 100), dtype=np.uint8)
    # Should not raise
    validate_image(img)
    validate_image(img, require_2d=True)
    validate_image(img, require_uint8=True)
    assert is_valid_grayscale(img)
    assert not is_valid_rgb(img)


def test_validate_image_valid_rgb():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    validate_image(img, allow_color=True)
    validate_image(img, require_uint8=True)
    assert is_valid_rgb(img)
    assert not is_valid_grayscale(img)


def test_validate_image_float_normalized():
    img = np.random.uniform(0.0, 1.0, (50, 50)).astype(np.float64)
    validate_image(img, require_float=True)
    info = get_image_info(img)
    assert info["is_float"] is True
    assert info["is_grayscale"] is True
    assert 0.0 <= info["min"] <= 1.0
    assert 0.0 <= info["max"] <= 1.0


def test_validate_image_rejects_non_array():
    with pytest.raises(TypeError, match="Expected image to be np.ndarray"):
        validate_image([1, 2, 3])


def test_validate_image_rejects_empty():
    with pytest.raises(ValueError, match="Image array is empty"):
        validate_image(np.array([]))


def test_validate_image_rejects_nan_inf():
    img_nan = np.zeros((10, 10))
    img_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite values"):
        validate_image(img_nan)

    img_inf = np.zeros((10, 10))
    img_inf[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite values"):
        validate_image(img_inf)


def test_validate_image_rejects_wrong_dimensions():
    # 1D array
    with pytest.raises(ValueError, match="Expected 2D .* or 3D"):
        validate_image(np.zeros(10))

    # 4D array
    with pytest.raises(ValueError, match="Expected 2D .* or 3D"):
        validate_image(np.zeros((10, 10, 3, 2)))

    # 3D with 4 channels
    with pytest.raises(ValueError, match="Color images must have 3 channels"):
        validate_image(np.zeros((10, 10, 4)))


def test_validate_image_require_2d_flag():
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="requires a 2D grayscale image"):
        validate_image(rgb, require_2d=True)


def test_validate_image_float_range_violation():
    img_overflow = np.array([[0.0, 1.5], [0.2, 0.8]], dtype=np.float64)
    with pytest.raises(ValueError, match="must be in range \\[0.0, 1.0\\]"):
        validate_image(img_overflow, require_float=True)


def test_get_image_info():
    img = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    info = get_image_info(img)
    assert info["shape"] == (2, 2)
    assert info["dtype"] == "uint8"
    assert info["min"] == 10.0
    assert info["max"] == 40.0
    assert info["mean"] == 25.0
    assert info["is_grayscale"] is True
