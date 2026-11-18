"""
Image conversion and transformation helpers.

Provides explicit, safe conversions between:
- uint8 [0, 255] and float64 [0.0, 1.0]
- RGB (H, W, 3) and Grayscale (H, W)

Academic Note:
Standard Luminance Weights (ITU-R BT.601-7):
    Y = 0.2989 * R + 0.5870 * G + 0.1140 * B
This models the human visual system's varying sensitivity to red, green, and blue light.
"""

from typing import Optional
import numpy as np
from src.utils.validation import validate_image, is_valid_grayscale, is_valid_rgb


def to_float(image: np.ndarray) -> np.ndarray:
    """
    Convert an image array to floating-point representation normalized to [0.0, 1.0].

    If already a floating-point array:
        - If values appear to be in [0.0, 255.0] (max > 2.0), normalizes by 255.0.
        - Otherwise returns float64 clipped to [0.0, 1.0].
    If uint8:
        - Divides by 255.0.

    Parameters
    ----------
    image : np.ndarray
        Input image.

    Returns
    -------
    np.ndarray
        float64 image in range [0.0, 1.0].
    """
    validate_image(image)

    if np.issubdtype(image.dtype, np.floating):
        img_float = image.astype(np.float64)
        max_val = np.max(img_float) if img_float.size > 0 else 0.0
        if max_val > 2.0:
            # Unnormalized float in [0, 255]
            return np.clip(img_float / 255.0, 0.0, 1.0)
        else:
            # Normalized float with potential minor overshoot
            return np.clip(img_float, 0.0, 1.0)

    if image.dtype == np.uint8:
        return (image.astype(np.float64)) / 255.0

    # For other integer types (e.g. uint16, int32)
    img_float = image.astype(np.float64)
    min_val, max_val = np.min(img_float), np.max(img_float)
    if max_val > min_val:
        return (img_float - min_val) / (max_val - min_val)
    return np.zeros_like(img_float, dtype=np.float64)


def to_uint8(
    image: np.ndarray,
    from_normalized: Optional[bool] = True,
    clip: bool = True,
) -> np.ndarray:
    """
    Convert an image array to uint8 representation in range [0, 255].

    Parameters
    ----------
    image : np.ndarray
        Input image.
    from_normalized : bool or None, default=True
        Whether the input float image is normalized to [0.0, 1.0].
        If None, automatically inferred based on maximum pixel value.
    clip : bool, default=True
        Whether to clamp values to prevent underflow/overflow before conversion.

    Returns
    -------
    np.ndarray
        uint8 array with values in [0, 255].
    """
    validate_image(image)

    if image.dtype == np.uint8:
        return image.copy()

    if np.issubdtype(image.dtype, np.floating):
        img_copy = image.copy().astype(np.float64)
        max_val = np.max(img_copy) if img_copy.size > 0 else 0.0

        # Auto-detect if from_normalized is None
        is_norm = from_normalized if from_normalized is not None else (max_val <= 2.0)

        if is_norm:
            # Float in [0.0, 1.0] (allowing for noise/filter overshoot)
            if clip:
                img_copy = np.clip(img_copy, 0.0, 1.0)
            return np.round(img_copy * 255.0).astype(np.uint8)
        else:
            # Float already on [0.0, 255.0] scale
            if clip:
                img_copy = np.clip(img_copy, 0.0, 255.0)
            return np.round(img_copy).astype(np.uint8)

    # General fallback for integer types
    if clip:
        img_clipped = np.clip(image, 0, 255)
    else:
        img_clipped = image
    return np.round(img_clipped).astype(np.uint8)


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image (H, W, 3) to 2D grayscale (H, W) using standard ITU-R BT.601 weights.

    Formula:
        Y = 0.2989 * R + 0.5870 * G + 0.1140 * B

    Parameters
    ----------
    image : np.ndarray
        RGB image of shape (H, W, 3).

    Returns
    -------
    np.ndarray
        2D grayscale array of shape (H, W), preserving input dtype.
    """
    validate_image(image, allow_color=True)

    if image.ndim == 2:
        return image.copy()

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image.shape}")

    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    # Compute luminance
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    if image.dtype == np.uint8:
        return np.round(np.clip(gray, 0, 255)).astype(np.uint8)
    else:
        return gray.astype(image.dtype)


def gray_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert a 2D grayscale image (H, W) into an RGB image (H, W, 3) by replicating channels.

    Parameters
    ----------
    image : np.ndarray
        2D grayscale array of shape (H, W).

    Returns
    -------
    np.ndarray
        RGB image of shape (H, W, 3).
    """
    validate_image(image, require_2d=True)
    return np.stack([image, image, image], axis=-1)


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Ensure the image is a 2D grayscale array.
    If 3D RGB, converts to grayscale; if 2D, returns a copy.
    """
    validate_image(image, allow_color=True)
    if image.ndim == 3:
        return rgb_to_gray(image)
    return image.copy()
