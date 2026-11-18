"""
Gaussian Noise Degradation Model.

Academic Background:
Gaussian noise is an additive noise model that arises in digital images due to
thermal motion of electrons in camera sensors, amplifier noise, and poor illumination.
The degraded image g(x, y) is modeled as:
    g(x, y) = f(x, y) + eta(x, y)
where eta(x, y) is a zero-mean normally distributed random variable:
    p(eta) = (1 / (sqrt(2 * pi) * sigma)) * exp(- (eta - mu)^2 / (2 * sigma^2))

Key Properties:
1. Additive: Noise is added independently to each pixel value.
2. Spatial Independence: Noise values at different spatial coordinates are statistically independent.
3. Clipped Boundary: Real sensor outputs are bounded; values below 0 or above 1 are clipped.
"""

from typing import Optional, Union
import numpy as np
from src.utils.validation import validate_image
from src.utils.image_helpers import to_float, to_uint8


def add_gaussian_noise(
    image: np.ndarray,
    sigma: float = 0.05,
    mean: float = 0.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Add additive Gaussian white noise to an image.

    Parameters
    ----------
    image : np.ndarray
        Input image (uint8 or float64, grayscale or RGB).
    sigma : float, default=0.05
        Standard deviation of the Gaussian distribution (relative to [0.0, 1.0] scale).
        Typical levels:
        - Low: sigma = 0.02
        - Medium: sigma = 0.05
        - High: sigma = 0.15
    mean : float, default=0.0
        Mean of the Gaussian noise.
    seed : int, optional
        Random seed for exact reproducibility.

    Returns
    -------
    np.ndarray
        Degraded noisy image matching input dtype and range.
    """
    validate_image(image)

    if sigma < 0.0:
        raise ValueError(f"Noise standard deviation (sigma) must be non-negative; got {sigma}")

    # Determine input type to preserve exact format upon return
    was_uint8 = (image.dtype == np.uint8)

    # Work in normalized [0.0, 1.0] float representation
    img_float = to_float(image)

    # Initialize random number generator
    rng = np.random.default_rng(seed)

    # Generate Gaussian distributed noise array
    noise = rng.normal(loc=mean, scale=sigma, size=img_float.shape)

    # Additive noise model: g = f + eta
    noisy_float = img_float + noise

    # Clip to valid dynamic range [0.0, 1.0]
    noisy_float = np.clip(noisy_float, 0.0, 1.0)

    if was_uint8:
        return to_uint8(noisy_float)
    return noisy_float
