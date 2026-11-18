"""
Gaussian Spatial Smoothing Filter and Mathematical Kernel Generation.

Academic Implementation Rule:
Implemented from scratch using NumPy mathematical operations and our custom
2D convolution engine. Does NOT call cv2.GaussianBlur or scipy.ndimage.gaussian_filter.

Difference Between Mean and Gaussian Filtering:
1. Uniform vs Weighted:
   - Mean filter gives equal weight (1 / K^2) to all neighbors regardless of distance.
   - Gaussian filter weights neighbors inversely proportional to their squared distance
     from the center, following the bell curve exp(-(x^2 + y^2) / (2 * sigma^2)).
2. Edge Preservation:
   - Gaussian smoothing degrades edges more gracefully because distant pixels contribute
     substantially less than immediately adjacent pixels.
3. Frequency Response:
   - The Fourier transform of a Gaussian is another Gaussian (no side lobes / ringing).
   - The Fourier transform of a box/mean filter is a 2D sinc function, which produces
     noticeable high-frequency ripples (ringing artifacts) in spatial domain.
"""

from typing import Optional, Tuple
import numpy as np

from src.spatial.convolution import convolve2d
from src.utils.validation import validate_image
from src.utils.image_helpers import to_float, to_uint8


def create_gaussian_kernel(kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Generate a 2D Gaussian blur kernel mathematically from scratch.

    Parameters
    ----------
    kernel_size : int, default=5
        Side length of square kernel. Must be an odd positive integer (e.g. 3, 5, 7, 9).
    sigma : float, default=1.0
        Standard deviation of Gaussian distribution (spread of the blur).

    Returns
    -------
    np.ndarray
        2D float64 kernel of shape (kernel_size, kernel_size) normalized so sum(kernel) == 1.0.
    """
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"Kernel size must be an odd positive integer; got {kernel_size}")
    if sigma <= 0.0:
        raise ValueError(f"Gaussian sigma must be strictly positive (> 0); got {sigma}")

    radius = kernel_size // 2
    # Create coordinate grid centered at (0, 0)
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]

    # Gaussian 2D formula: exp( -(x^2 + y^2) / (2 * sigma^2) ) / (2 * pi * sigma^2)
    exponent = -(x**2 + y**2) / (2.0 * (sigma**2))
    kernel = np.exp(exponent) / (2.0 * np.pi * (sigma**2))

    # Normalize kernel to guarantee energy conservation (sum of weights = 1.0)
    kernel_sum = np.sum(kernel)
    if kernel_sum > 0:
        kernel = kernel / kernel_sum

    return kernel.astype(np.float64)


def apply_gaussian_filter(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.0,
    mode: str = "reflect",
) -> np.ndarray:
    """
    Apply Gaussian spatial smoothing filter using our custom convolution engine.

    Parameters
    ----------
    image : np.ndarray
        Input image (uint8 or float64, grayscale or RGB).
    kernel_size : int, default=5
        Side length of Gaussian kernel (odd integer >= 3).
    sigma : float, default=1.0
        Gaussian standard deviation. Higher values produce stronger smoothing.
    mode : str, default='reflect'
        Boundary padding mode ('reflect', 'edge', 'constant').

    Returns
    -------
    np.ndarray
        Smoothed image matching input dtype and shape.
    """
    validate_image(image)

    was_uint8 = (image.dtype == np.uint8)
    img_float = to_float(image)

    kernel = create_gaussian_kernel(kernel_size=kernel_size, sigma=sigma)
    filtered_float = convolve2d(img_float, kernel, mode=mode, flip_kernel=False)
    filtered_float = np.clip(filtered_float, 0.0, 1.0)

    if was_uint8:
        return to_uint8(filtered_float)
    return filtered_float
