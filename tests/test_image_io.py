"""
Unit tests for Image I/O and helper transformations.
"""

from pathlib import Path
import numpy as np
import pytest

from src.image_io import load_image, save_image, display_image, display_comparison
from src.utils.image_helpers import (
    to_float,
    to_uint8,
    rgb_to_gray,
    gray_to_rgb,
    ensure_grayscale,
)


@pytest.fixture
def original_data_dir():
    return Path(__file__).resolve().parent.parent / "data" / "original"


def test_to_float_and_to_uint8_conversion():
    # uint8 array
    u8 = np.array([[0, 128, 255]], dtype=np.uint8)
    flt = to_float(u8)

    assert flt.dtype == np.float64
    assert np.isclose(flt[0, 0], 0.0)
    assert np.isclose(flt[0, 1], 128 / 255.0)
    assert np.isclose(flt[0, 2], 1.0)

    # Convert back to uint8
    u8_back = to_uint8(flt)
    assert u8_back.dtype == np.uint8
    assert np.array_equal(u8, u8_back)


def test_to_uint8_clipping():
    # Out-of-bounds float values
    flt = np.array([[-0.5, 0.5, 1.5]], dtype=np.float64)
    u8 = to_uint8(flt, clip=True)
    assert u8[0, 0] == 0
    assert u8[0, 1] == 128
    assert u8[0, 2] == 255


def test_rgb_to_gray():
    # Pure red, green, blue pixels
    rgb = np.zeros((1, 3, 3), dtype=np.uint8)
    rgb[0, 0] = [255, 0, 0]  # Red
    rgb[0, 1] = [0, 255, 0]  # Green
    rgb[0, 2] = [0, 0, 255]  # Blue

    gray = rgb_to_gray(rgb)
    assert gray.shape == (1, 3)
    assert gray.dtype == np.uint8

    # Weights: 0.2989*R, 0.5870*G, 0.1140*B
    assert gray[0, 0] == round(0.2989 * 255)  # 76
    assert gray[0, 1] == round(0.5870 * 255)  # 150
    assert gray[0, 2] == round(0.1140 * 255)  # 29


def test_gray_to_rgb():
    gray = np.array([[50, 150]], dtype=np.uint8)
    rgb = gray_to_rgb(gray)
    assert rgb.shape == (1, 2, 3)
    assert np.array_equal(rgb[:, :, 0], gray)
    assert np.array_equal(rgb[:, :, 1], gray)
    assert np.array_equal(rgb[:, :, 2], gray)


def test_ensure_grayscale():
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    gray = ensure_grayscale(rgb)
    assert gray.ndim == 2
    assert gray.shape == (10, 10)

    # Calling on already grayscale returns 2D unchanged
    gray2 = ensure_grayscale(gray)
    assert gray2.ndim == 2
    assert np.array_equal(gray, gray2)


def test_load_image_png_grayscale(original_data_dir):
    img_path = original_data_dir / "checkerboard.png"
    assert img_path.exists(), "Checkerboard test image must exist."

    # Load as grayscale
    img_gray = load_image(img_path, as_gray=True)
    assert img_gray.ndim == 2
    assert img_gray.shape == (256, 256)
    assert img_gray.dtype == np.uint8


def test_load_image_rgb_channel_order(original_data_dir):
    img_path = original_data_dir / "color_test.jpg"
    assert img_path.exists(), "Color test image must exist."

    # Load RGB image
    img_rgb = load_image(img_path, as_gray=False)
    assert img_rgb.ndim == 3
    assert img_rgb.shape == (256, 256, 3)

    # Top-left quadrant of color_test is predominantly Red (R > 180, G < 100, B < 100)
    top_left_pixel = img_rgb[50, 50]
    assert top_left_pixel[0] > 180, f"Expected red channel to be high, got {top_left_pixel[0]}"
    assert top_left_pixel[2] < 100, f"Expected blue channel to be low, got {top_left_pixel[2]}"


def test_load_image_normalized_float(original_data_dir):
    img_path = original_data_dir / "checkerboard.png"
    img_flt = load_image(img_path, as_gray=True, to_normalized_float=True)
    assert img_flt.dtype == np.float64
    assert np.min(img_flt) >= 0.0
    assert np.max(img_flt) <= 1.0


def test_load_non_existent_file():
    with pytest.raises(FileNotFoundError):
        load_image("non_existent_image_12345.png")


def test_save_and_reload_image(tmp_path):
    # Create synthetic test array
    test_img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    save_path = tmp_path / "test_out.png"

    # Save
    saved_path = save_image(save_path, test_img)
    assert saved_path.exists()

    # Reload
    reloaded = load_image(save_path, as_gray=True)
    assert np.array_equal(test_img, reloaded)


def test_save_float_image(tmp_path):
    # Floating-point normalized image
    flt_img = np.linspace(0.0, 1.0, 10000).reshape((100, 100))
    save_path = tmp_path / "float_out.png"

    save_image(save_path, flt_img)
    assert save_path.exists()

    # Reloaded should be uint8 in [0, 255]
    reloaded = load_image(save_path, as_gray=True)
    assert reloaded.dtype == np.uint8
    assert reloaded.shape == (100, 100)
    assert reloaded.min() == 0
    assert reloaded.max() == 255


def test_display_image_and_comparison(tmp_path):
    img1 = np.zeros((32, 32), dtype=np.uint8)
    img2 = np.ones((32, 32), dtype=np.uint8) * 255

    # Test single display
    out_single = tmp_path / "single.png"
    fig1 = display_image(img1, title="Test Single", show=False, save_path=out_single)
    assert out_single.exists()
    assert fig1 is not None

    # Test grid display
    out_grid = tmp_path / "grid.png"
    fig2 = display_comparison([img1, img2], titles=["Black", "White"], cols=2, show=False, save_path=out_grid)
    assert out_grid.exists()
    assert fig2 is not None
