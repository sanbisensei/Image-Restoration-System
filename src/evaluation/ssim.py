"""
Structural Similarity Index Measure (SSIM) Evaluation Metric.

Academic Background:
Formulated by Wang, Bovik, Sheikh, and Simoncelli (IEEE TIP 2004), SSIM addresses
the limitations of MSE and PSNR, which fail to capture human perceptual image quality.
Instead of measuring error magnitudes, SSIM compares three fundamental structural components:
1. Luminance:
       l(x, y) = (2 * mu_x * mu_y + C1) / (mu_x^2 + mu_y^2 + C1)
2. Contrast:
       c(x, y) = (2 * sigma_x * sigma_y + C2) / (sigma_x^2 + sigma_y^2 + C2)
3. Structure:
       s(x, y) = (sigma_xy + C3) / (sigma_x * sigma_y + C3)

Combined Index:
Setting alpha = beta = gamma = 1 and C3 = C2 / 2:
    SSIM(x, y) = [ (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2) ] /
                 [ (mu_x^2 + mu_y^2 + C1) * (sigma_x^2 + sigma_y^2 + C2) ]

Constants:
- C1 = (K1 * L)^2, where K1 = 0.01
- C2 = (K2 * L)^2, where K2 = 0.03
- L is the dynamic range of pixel values (255 for uint8, 1.0 for float).

Implementation Transparency (Academic Rule):
1. Custom Implementation: `compute_custom_ssim` computes local statistics using an 11x11
   Gaussian window (sigma=1.5) convolved via our custom NumPy 2D convolution engine.
2. Reference Implementation: `compute_reference_ssim` wraps scikit-image's official
   `skimage.metrics.structural_similarity` to benchmark and cross-validate our custom code.
"""

from typing import Optional, Union, Tuple
import numpy as np
from skimage.metrics import structural_similarity as skimage_ssim

from src.spatial.gaussian_filter import create_gaussian_kernel
from src.spatial.convolution import convolve2d_grayscale
from src.utils.validation import validate_image
from src.utils.image_helpers import ensure_grayscale, to_float, to_uint8


def compute_custom_ssim(
    image_true: np.ndarray,
    image_test: np.ndarray,
    data_range: Optional[float] = None,
    win_size: int = 11,
    win_sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """
    Compute Structural Similarity Index (SSIM) from scratch using custom NumPy convolution.

    Parameters
    ----------
    image_true : np.ndarray
        Ground-truth reference image.
    image_test : np.ndarray
        Test image (degraded or restored).
    data_range : float, optional
        Dynamic range of pixels L (1.0 for float, 255 for uint8).
    win_size : int, default=11
        Side length of Gaussian window.
    win_sigma : float, default=1.5
        Standard deviation of Gaussian window.
    k1, k2 : float, default=0.01, 0.03
        Stability constants.

    Returns
    -------
    float
        Mean SSIM index in range [-1.0, 1.0], where 1.0 indicates perfect structural identity.
    """
    validate_image(image_true)
    validate_image(image_test)

    if image_true.shape != image_test.shape:
        raise ValueError(f"Image shapes must match; got {image_true.shape} vs {image_test.shape}")

    # Determine dynamic range L
    if data_range is None:
        if np.issubdtype(image_true.dtype, np.floating) and np.issubdtype(image_test.dtype, np.floating):
            l_val = 1.0
        elif image_true.dtype == np.uint8 and image_test.dtype == np.uint8:
            l_val = 255.0
        else:
            image_true = to_float(image_true)
            image_test = to_float(image_test)
            l_val = 1.0
    else:
        l_val = float(data_range)

    c1 = (k1 * l_val) ** 2
    c2 = (k2 * l_val) ** 2

    # Standard 11x11 Gaussian weighting window
    window = create_gaussian_kernel(kernel_size=win_size, sigma=win_sigma)

    def _ssim_2d(im1: np.ndarray, im2: np.ndarray) -> float:
        x = im1.astype(np.float64)
        y = im2.astype(np.float64)

        # Local means
        mu_x = convolve2d_grayscale(x, window, mode="reflect", flip_kernel=False)
        mu_y = convolve2d_grayscale(y, window, mode="reflect", flip_kernel=False)

        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y

        # Local variances and covariance
        # Var(x) = E[x^2] - (E[x])^2
        sigma_x_sq = convolve2d_grayscale(x ** 2, window, mode="reflect", flip_kernel=False) - mu_x_sq
        sigma_y_sq = convolve2d_grayscale(y ** 2, window, mode="reflect", flip_kernel=False) - mu_y_sq
        sigma_xy = convolve2d_grayscale(x * y, window, mode="reflect", flip_kernel=False) - mu_xy

        # SSIM map formula
        numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
        denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)

        ssim_map = numerator / denominator
        return float(np.mean(ssim_map))

    if image_true.ndim == 2:
        return _ssim_2d(image_true, image_test)

    # Multi-channel RGB: compute per channel and average
    channel_ssims = [
        _ssim_2d(image_true[:, :, c], image_test[:, :, c])
        for c in range(image_true.shape[2])
    ]
    return float(np.mean(channel_ssims))


def compute_reference_ssim(
    image_true: np.ndarray,
    image_test: np.ndarray,
    data_range: Optional[float] = None,
) -> float:
    """
    Compute SSIM using scikit-image's official reference implementation.

    Used strictly as an independent validation benchmark to verify our custom SSIM.
    """
    validate_image(image_true)
    validate_image(image_test)

    if data_range is None:
        if np.issubdtype(image_true.dtype, np.floating):
            d_range = 1.0
        else:
            d_range = 255.0
    else:
        d_range = float(data_range)

    channel_axis = 2 if image_true.ndim == 3 else None

    score = skimage_ssim(
        image_true,
        image_test,
        data_range=d_range,
        channel_axis=channel_axis,
    )
    return float(score)


def compute_ssim(
    image_true: np.ndarray,
    image_test: np.ndarray,
    use_reference: bool = False,
) -> float:
    """
    Compute SSIM score. By default uses our custom implementation.
    """
    if use_reference:
        return compute_reference_ssim(image_true, image_test)
    return compute_custom_ssim(image_true, image_test)
