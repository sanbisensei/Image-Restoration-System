"""
Salt-and-Pepper (Impulse) Noise Degradation Model.

Academic Background:
Salt-and-pepper noise (also called bipolar impulse noise) manifests as randomly
scattered black and white pixels across the image. It is caused by:
1. Defective sensor pixels (dead or hot pixels).
2. Hardware faults in analog-to-digital converters (ADC).
3. Transmission bit errors across noisy digital communication channels.

Mathematical Formulation:
Let s_min = 0 (Pepper) and s_max = 255 or 1.0 (Salt).
For each pixel coordinate (x, y):
             { s_min  with probability P_pepper = density * (1 - salt_ratio)
    g(x, y) = { s_max  with probability P_salt   = density * salt_ratio
             { f(x, y) with probability 1 - density
"""

from typing import Optional
import numpy as np
from src.utils.validation import validate_image
from src.utils.image_helpers import to_float, to_uint8


def add_salt_and_pepper_noise(
    image: np.ndarray,
    density: float = 0.05,
    salt_ratio: float = 0.5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Apply salt-and-pepper (impulse) noise to an image.

    Parameters
    ----------
    image : np.ndarray
        Input image (uint8 or float, grayscale or RGB).
    density : float, default=0.05
        Proportion of image pixels to corrupt with impulse noise (in range [0.0, 1.0]).
        Typical levels:
        - Low: density = 0.02 (2% corrupted)
        - Medium: density = 0.08 (8% corrupted)
        - High: density = 0.20 (20% corrupted)
    salt_ratio : float, default=0.5
        Proportion of corrupted pixels that become white (salt) vs black (pepper).
        0.5 corresponds to equal probability of salt and pepper.
    seed : int, optional
        Random seed for exact reproducibility.

    Returns
    -------
    np.ndarray
        Degraded image with impulse noise, matching input dtype and range.
    """
    validate_image(image)

    if not (0.0 <= density <= 1.0):
        raise ValueError(f"Noise density must be between 0.0 and 1.0; got {density}")
    if not (0.0 <= salt_ratio <= 1.0):
        raise ValueError(f"Salt ratio must be between 0.0 and 1.0; got {salt_ratio}")

    if density == 0.0:
        return image.copy()

    was_uint8 = (image.dtype == np.uint8)
    img_float = to_float(image)
    h, w = img_float.shape[:2]

    rng = np.random.default_rng(seed)

    # Random matrix for spatial locations
    rand_matrix = rng.uniform(0.0, 1.0, size=(h, w))

    # Output copy
    degraded = img_float.copy()

    # Thresholds:
    # 0.0 to p_pepper -> Pepper (0.0)
    # p_pepper to (p_pepper + p_salt) -> Salt (1.0)
    p_pepper = density * (1.0 - salt_ratio)
    p_salt = density * salt_ratio

    pepper_mask = rand_matrix < p_pepper
    salt_mask = (rand_matrix >= p_pepper) & (rand_matrix < (p_pepper + p_salt))

    if image.ndim == 2:
        degraded[pepper_mask] = 0.0
        degraded[salt_mask] = 1.0
    else:
        # For 3D RGB, apply impulse to all color channels at the selected pixel locations
        for c in range(3):
            degraded[:, :, c][pepper_mask] = 0.0
            degraded[:, :, c][salt_mask] = 1.0

    if was_uint8:
        return to_uint8(degraded)
    return degraded
