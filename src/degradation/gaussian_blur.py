"""
Gaussian Blur Degradation Model.

Academic Background:
Gaussian blur models optical defocusing, atmospheric turbulence, and optical
diffraction in camera lenses. The Point Spread Function (PSF) of an isotropic
Gaussian optical system is:
    h(x, y) = (1 / (2 * pi * sigma^2)) * exp(- (x^2 + y^2) / (2 * sigma^2))

Key Properties:
1. Symmetry: The kernel is radially symmetric (isotropic).
2. Separability: 2D Gaussian filter can be decomposed as two 1D filters:
   G(x, y) = G(x) * G(y).
3. Energy Conservation: The sum of all discrete kernel weights must equal 1.0
   so the global average brightness of the image is strictly preserved.
"""

from typing import Tuple
import numpy as np
from src.spatial.convolution import convolve2d
from src.spatial.gaussian_filter import create_gaussian_kernel
from src.utils.validation import validate_image
from src.utils.image_helpers import to_float, to_uint8


def apply_gaussian_blur(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.0,
    mode: str = "reflect",
) -> np.ndarray:
    """
    Apply Gaussian blur degradation using our custom convolution engine.

    Parameters
    ----------
    image : np.ndarray
        Input image (uint8 or float64, grayscale or RGB).
    kernel_size : int, default=5
        Side length of Gaussian kernel (odd integer).
    sigma : float, default=1.0
        Spread of Gaussian blur.
    mode : str, default='reflect'
        Boundary padding mode for convolution ('reflect', 'edge', 'constant').

    Returns
    -------
    np.ndarray
        Degraded blurred image matching input dtype and shape.
    """
    validate_image(image)

    was_uint8 = (image.dtype == np.uint8)
    img_float = to_float(image)

    # 1. Create Gaussian blur kernel
    kernel = create_gaussian_kernel(kernel_size=kernel_size, sigma=sigma)

    # 2. Convolve image with Gaussian PSF using custom 2D convolution
    blurred_float = convolve2d(img_float, kernel, mode=mode, flip_kernel=True)

    # 3. Clip to [0.0, 1.0] to guard against minor floating-point boundary rounding
    blurred_float = np.clip(blurred_float, 0.0, 1.0)

    if was_uint8:
        return to_uint8(blurred_float)
    return blurred_float
