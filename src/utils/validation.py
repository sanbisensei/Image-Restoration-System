"""
Validation utilities for image arrays and processing parameters.

This module provides validation checks to guarantee that image arrays
passed to classical DIP algorithms conform to strict representations:
    1. Grayscale: 2D NumPy array of shape (H, W).
    2. Color (RGB): 3D NumPy array of shape (H, W, 3).
    3. Value Range:
       - uint8: integers in [0, 255]
       - float: floating-point values normalized to [0.0, 1.0]

Academic Principle:
Silently mixing floating-point normalized representations with integer [0, 255]
ranges causes severe numerical errors (e.g. integer overflow, black outputs).
Explicit validation prevents these defects early.
"""

from typing import Dict, Any, Tuple
import numpy as np


def validate_image(
    image: np.ndarray,
    allow_color: bool = True,
    require_2d: bool = False,
    require_float: bool = False,
    require_uint8: bool = False,
) -> None:
    """
    Validate that an input object is a well-formed NumPy image array.

    Parameters
    ----------
    image : np.ndarray
        Image array to validate.
    allow_color : bool, default=True
        Whether 3D 3-channel RGB images are accepted.
    require_2d : bool, default=False
        If True, only 2D grayscale images are accepted.
    require_float : bool, default=False
        If True, requires floating-point dtype in range [0.0, 1.0].
    require_uint8 : bool, default=False
        If True, requires np.uint8 dtype in range [0, 255].

    Raises
    ------
    TypeError
        If image is not a numpy ndarray.
    ValueError
        If dimensions, shapes, or value ranges are invalid.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected image to be np.ndarray, got {type(image).__name__}")

    if image.size == 0:
        raise ValueError("Image array is empty (size is 0).")

    if not np.all(np.isfinite(image)):
        raise ValueError("Image contains non-finite values (NaN or Inf).")

    # Shape validation
    ndim = image.ndim
    if require_2d:
        if ndim != 2:
            raise ValueError(f"Algorithm requires a 2D grayscale image (H, W), but received {ndim}D array with shape {image.shape}.")
    else:
        if ndim == 2:
            pass  # Valid 2D grayscale
        elif ndim == 3:
            if not allow_color:
                raise ValueError(f"Color images are not allowed here; expected 2D grayscale, got shape {image.shape}.")
            if image.shape[2] != 3:
                raise ValueError(f"Color images must have 3 channels (RGB); got shape {image.shape}.")
        else:
            raise ValueError(f"Expected 2D (grayscale) or 3D (RGB) image, got {ndim}D array with shape {image.shape}.")

    # Type & Range validation
    is_float_type = np.issubdtype(image.dtype, np.floating)
    is_uint8_type = image.dtype == np.uint8

    if require_float:
        if not is_float_type:
            raise ValueError(f"Algorithm requires floating-point image in [0.0, 1.0], but got dtype {image.dtype}.")
        min_val, max_val = float(np.min(image)), float(np.max(image))
        if min_val < -1e-6 or max_val > 1.0 + 1e-6:
            raise ValueError(f"Floating-point image must be in range [0.0, 1.0], but min={min_val:.4f}, max={max_val:.4f}.")

    if require_uint8:
        if not is_uint8_type:
            raise ValueError(f"Algorithm requires uint8 image in [0, 255], but got dtype {image.dtype}.")


def is_valid_grayscale(image: np.ndarray) -> bool:
    """Return True if image is a valid 2D array."""
    return isinstance(image, np.ndarray) and image.ndim == 2 and image.size > 0


def is_valid_rgb(image: np.ndarray) -> bool:
    """Return True if image is a valid 3D array with 3 color channels."""
    return isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 3 and image.size > 0


def get_image_info(image: np.ndarray) -> Dict[str, Any]:
    """
    Inspect an image array and return diagnostic statistics.

    Returns
    -------
    dict with keys:
        - shape: tuple of dimensions
        - dtype: data type string
        - min: minimum value
        - max: maximum value
        - mean: mean pixel value
        - std: standard deviation
        - is_float: bool
        - is_grayscale: bool
    """
    validate_image(image)
    return {
        "shape": image.shape,
        "dtype": str(image.dtype),
        "min": float(np.min(image)),
        "max": float(np.max(image)),
        "mean": float(np.mean(image)),
        "std": float(np.std(image)),
        "is_float": bool(np.issubdtype(image.dtype, np.floating)),
        "is_grayscale": bool(image.ndim == 2),
    }
