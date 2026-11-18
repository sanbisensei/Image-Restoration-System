"""
Median Non-Linear Spatial Filter.

Academic Implementation Rule:
Implemented from scratch using NumPy order-statistic calculations on local
neighborhoods. Does NOT call cv2.medianBlur or scipy.signal.medfilt2d.

Mathematical Background:
Unlike mean or Gaussian filters, the median filter is a non-linear filter and
CANNOT be represented as a spatial convolution kernel.
For an odd-sized neighborhood S of size K x K centered at (x, y):
1. Extract the K^2 neighboring pixel values.
2. Sort them in ascending order:
       v_{(1)} <= v_{(2)} <= ... <= v_{(m)} <= ... <= v_{(K^2)}
3. The restored pixel is assigned the median value at index m = (K^2 // 2):
       f_restored(x, y) = median(S) = v_{(m)}

Why Median Filtering is Uniquely Effective for Salt-and-Pepper Noise:
1. Impulse Immunity:
   Salt (white, 255) and pepper (black, 0) pixels are statistical outliers that
   occupy extreme ranks in the sorted list (the lowest ranks for 0, the highest ranks for 255).
   As long as corrupted pixels constitute less than 50% of the local window,
   the median value is guaranteed to be an uncorrupted neighborhood pixel.
2. Edge Preservation:
   Unlike linear averaging, which averages across edges and smears boundaries,
   a monotonic step edge across the neighborhood maintains its step transition
   because the median of a monotonic sequence preserves the step position.
3. No Artificial Grays:
   A mean filter turns sharp black/white noise into blurry gray splotches.
   The median filter selects an actual existing neighboring value, preserving crispness.
"""

from typing import Union
import numpy as np
from src.utils.validation import validate_image
from src.utils.image_helpers import to_float, to_uint8


def apply_median_filter_grayscale(
    image: np.ndarray,
    kernel_size: int = 3,
    mode: str = "reflect",
) -> np.ndarray:
    """
    Apply median filtering to a 2D grayscale image from scratch using NumPy.

    Parameters
    ----------
    image : np.ndarray
        2D grayscale array of shape (H, W).
    kernel_size : int, default=3
        Side length of neighborhood window. Must be an odd positive integer (3, 5, 7, ...).
    mode : str, default='reflect'
        Boundary padding mode ('reflect', 'edge', 'constant').

    Returns
    -------
    np.ndarray
        Filtered 2D grayscale image with same dtype and shape.
    """
    validate_image(image, require_2d=True)

    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"Kernel size must be an odd positive integer (3, 5, 7, ...); got {kernel_size}")

    if kernel_size == 1:
        return image.copy()

    h, w = image.shape
    pad = kernel_size // 2

    # Map padding mode
    np_mode = "edge" if mode == "replicate" else mode

    # Pad image to handle boundaries
    padded = np.pad(image, ((pad, pad), (pad, pad)), mode=np_mode)

    # Use NumPy sliding_window_view for zero-copy strided window extraction
    # Shape of windows: (H, W, kernel_size, kernel_size)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel_size, kernel_size))

    # Flatten the 2D window into a 1D vector of length K*K for each pixel
    # Shape becomes: (H, W, K*K)
    num_elements = kernel_size * kernel_size
    flattened_windows = windows.reshape(h, w, num_elements)

    # For an odd number of elements, the median is the element at rank (K^2 // 2)
    # np.partition efficiently partitions array so the median is at index (num_elements // 2)
    # in O(N) linear time without requiring a full O(N log N) sort.
    median_idx = num_elements // 2
    partitioned = np.partition(flattened_windows, kth=median_idx, axis=-1)
    filtered = partitioned[..., median_idx]

    return filtered.astype(image.dtype)


def apply_median_filter(
    image: np.ndarray,
    kernel_size: int = 3,
    mode: str = "reflect",
) -> np.ndarray:
    """
    Apply median filtering supporting both 2D grayscale and 3D RGB images.

    For 3D images, filtering is applied per color channel.

    Parameters
    ----------
    image : np.ndarray
        Input image (uint8 or float64, 2D or 3D).
    kernel_size : int, default=3
        Odd window size.
    mode : str, default='reflect'
        Padding mode.

    Returns
    -------
    np.ndarray
        Median filtered image.
    """
    validate_image(image)

    if image.ndim == 2:
        return apply_median_filter_grayscale(image, kernel_size=kernel_size, mode=mode)

    # Multi-channel 3D image
    channels = [
        apply_median_filter_grayscale(image[:, :, c], kernel_size=kernel_size, mode=mode)
        for c in range(image.shape[2])
    ]
    return np.stack(channels, axis=-1)
