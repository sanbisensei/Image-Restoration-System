"""
Peak Signal-to-Noise Ratio (PSNR) and Mean Squared Error (MSE) Evaluation Metrics.

Academic Background:
1. Mean Squared Error (MSE):
   Measures the average squared difference between the ground-truth clean image f(x, y)
   and the restored image f_hat(x, y):
       MSE = (1 / (M * N * C)) * sum_{x, y, c} ( f(x, y, c) - f_hat(x, y, c) )^2

2. Peak Signal-to-Noise Ratio (PSNR):
   A logarithmic ratio between the maximum possible signal power (MAX_I) and the
   corrupting noise power (MSE), expressed in decibels (dB):
       PSNR = 10 * log10( MAX_I^2 / MSE ) = 20 * log10( MAX_I / sqrt(MSE) )

Range Conventions:
- For uint8 images in [0, 255]: MAX_I = 255.0.
- For floating-point images normalized in [0.0, 1.0]: MAX_I = 1.0.

Identical Image Case:
When MSE = 0.0 (identical images), PSNR is mathematically infinite (+inf).
"""

from typing import Tuple, Union
import numpy as np
from src.utils.validation import validate_image
from src.utils.image_helpers import to_float


def compute_mse(image_true: np.ndarray, image_test: np.ndarray) -> float:
    """
    Compute the Mean Squared Error (MSE) between two images.

    Parameters
    ----------
    image_true : np.ndarray
        Ground-truth reference image.
    image_test : np.ndarray
        Test image (degraded or restored).

    Returns
    -------
    float
        Mean squared error value (>= 0.0).
    """
    validate_image(image_true)
    validate_image(image_test)

    if image_true.shape != image_test.shape:
        raise ValueError(
            f"Images must have identical shapes for MSE; got {image_true.shape} vs {image_test.shape}"
        )

    # Convert to float64 to prevent integer overflow during subtraction and squaring
    img1 = image_true.astype(np.float64)
    img2 = image_test.astype(np.float64)

    diff = img1 - img2
    mse = float(np.mean(diff ** 2))
    return mse


def compute_psnr(
    image_true: np.ndarray,
    image_test: np.ndarray,
    data_range: Union[float, int, None] = None,
) -> float:
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).

    Parameters
    ----------
    image_true : np.ndarray
        Ground-truth reference image.
    image_test : np.ndarray
        Test image (degraded or restored).
    data_range : float or int, optional
        The dynamic range of the image (MAX_I).
        - If None and image is uint8: data_range = 255.0.
        - If None and image is floating-point: data_range = 1.0.

    Returns
    -------
    float
        PSNR value in decibels (dB). Returns float('inf') if images are identical.
    """
    validate_image(image_true)
    validate_image(image_test)

    if image_true.shape != image_test.shape:
        raise ValueError(
            f"Images must have identical shapes for PSNR; got {image_true.shape} vs {image_test.shape}"
        )

    # Determine dynamic range MAX_I if not explicitly specified
    if data_range is None:
        if np.issubdtype(image_true.dtype, np.floating) and np.issubdtype(image_test.dtype, np.floating):
            max_i = 1.0
        elif image_true.dtype == np.uint8 and image_test.dtype == np.uint8:
            max_i = 255.0
        else:
            # Mixed types: normalize both to [0.0, 1.0]
            image_true = to_float(image_true)
            image_test = to_float(image_test)
            max_i = 1.0
    else:
        max_i = float(data_range)

    mse = compute_mse(image_true, image_test)

    # Handle exact identity case: MSE == 0.0 -> PSNR = +infinity
    if mse == 0.0:
        return float("inf")

    # PSNR = 10 * log10(MAX_I^2 / MSE)
    psnr = 10.0 * np.log10((max_i ** 2) / mse)
    return float(psnr)
