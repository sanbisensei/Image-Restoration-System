"""
Mean (Box / Average) Spatial Filter.

Academic Implementation Rule:
Implemented from scratch using NumPy mathematical operations and our custom
2D convolution engine. Does NOT call cv2.blur, cv2.boxFilter, or scipy filters.

Mathematical Formulation:
For a rectangular/square neighborhood S of size K x K centered at (x, y):
    f_restored(x, y) = (1 / (K * K)) * sum_{(s, t) in S} g(s, t)

All kernel coefficients are identical:
    h(s, t) = 1 / (K^2)

Why it Suppresses Noise:
For additive zero-mean random noise with variance sigma^2:
    Var(mean) = (1 / K^4) * sum_{i=1}^{K^2} Var(eta_i) = sigma^2 / (K^2)
Thus, a 3x3 mean filter reduces noise variance by a factor of 9, and a 5x5
filter reduces it by a factor of 25.

Why it Destroys Edges (Limitation):
Image edges are high-frequency spatial transitions (sharp gradients).
Because the mean filter is an unweighted low-pass filter, averaging pixels
across an edge blurs the boundary into a ramp, reducing image sharpness
and fine texture detail.
"""

from typing import Tuple
import numpy as np
from src.spatial.convolution import convolve2d
from src.utils.validation import validate_image
from src.utils.image_helpers import to_float, to_uint8


def create_mean_kernel(kernel_size: int = 3) -> np.ndarray:
    """
    Generate a normalized 2D mean (box) filter kernel.

    Parameters
    ----------
    kernel_size : int, default=3
        Side length of square kernel. Must be an odd positive integer.

    Returns
    -------
    np.ndarray
        2D float64 kernel of shape (kernel_size, kernel_size) where each element is 1 / (kernel_size^2).
    """
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"Kernel size must be an odd positive integer (3, 5, 7, ...); got {kernel_size}")

    num_pixels = kernel_size * kernel_size
    kernel = np.full((kernel_size, kernel_size), 1.0 / num_pixels, dtype=np.float64)
    return kernel


def apply_mean_filter(
    image: np.ndarray,
    kernel_size: int = 3,
    mode: str = "reflect",
) -> np.ndarray:
    """
    Apply spatial mean filtering to an image using custom 2D convolution.

    Parameters
    ----------
    image : np.ndarray
        Input image (uint8 or float64, grayscale or RGB).
    kernel_size : int, default=3
        Neighborhood window size (3 for 3x3, 5 for 5x5, 7 for 7x7).
    mode : str, default='reflect'
        Boundary padding mode ('reflect', 'edge', 'constant').

    Returns
    -------
    np.ndarray
        Filtered image matching input dtype and shape.
    """
    validate_image(image)

    was_uint8 = (image.dtype == np.uint8)
    img_float = to_float(image)

    kernel = create_mean_kernel(kernel_size=kernel_size)
    filtered_float = convolve2d(img_float, kernel, mode=mode, flip_kernel=False)
    filtered_float = np.clip(filtered_float, 0.0, 1.0)

    if was_uint8:
        return to_uint8(filtered_float)
    return filtered_float
