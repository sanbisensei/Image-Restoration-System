"""
Motion Blur Degradation Model.

Academic Background:
Linear motion blur occurs when there is relative translation between the camera
and the scene during the sensor exposure time (shutter interval).
A point source is smeared into a straight line segment of length L at angle theta.

Point Spread Function (PSF):
In continuous domain:
    h(x, y) = (1 / L) if (x, y) lies on the segment of length L at angle theta, 0 otherwise.

Discrete Formulation:
We construct an odd-sized discrete kernel grid (K x K), trace the linear path
from -L/2 to +L/2 centered at (K//2, K//2), and distribute energy onto grid
cells using sub-pixel interpolation so arbitrary angles (0° to 360°) render
cleanly without jagged rasterization artifacts.
The discrete kernel is normalized to sum to 1.0 (energy conservation).
"""

from typing import Optional
import numpy as np
from src.spatial.convolution import convolve2d
from src.utils.validation import validate_image
from src.utils.image_helpers import to_float, to_uint8


def create_motion_blur_kernel(
    length: int = 9,
    angle: float = 0.0,
    kernel_size: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a directional motion blur kernel mathematically from scratch.

    Parameters
    ----------
    length : int, default=9
        Displacement length of motion in pixels (must be >= 1).
    angle : float, default=0.0
        Angle of motion in degrees counter-clockwise from horizontal.
        - 0 degrees: Pure horizontal motion.
        - 90 degrees: Pure vertical motion.
        - 45 degrees: Diagonal motion.
    kernel_size : int, optional
        Size of square kernel grid. If None, automatically chosen as the smallest
        odd integer >= length.

    Returns
    -------
    np.ndarray
        2D float64 kernel normalized so that sum(kernel) == 1.0.
    """
    if length < 1:
        raise ValueError(f"Motion blur length must be at least 1 pixel; got {length}")

    # Determine odd kernel size to house the motion vector
    if kernel_size is None:
        k_size = max(int(length), 3)
        if k_size % 2 == 0:
            k_size += 1
    else:
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be an odd positive integer; got {kernel_size}")
        if kernel_size < length:
            raise ValueError(f"kernel_size ({kernel_size}) cannot be smaller than motion length ({length})")
        k_size = kernel_size

    kernel = np.zeros((k_size, k_size), dtype=np.float64)
    center = k_size // 2

    # Angle in radians
    # In standard image coordinates, y increases downward.
    # To match standard trigonometric intuition (positive angle = upward counter-clockwise),
    # dy = -sin(theta) and dx = cos(theta).
    theta_rad = np.deg2rad(angle)
    dx = np.cos(theta_rad)
    dy = -np.sin(theta_rad)

    # Sub-pixel sampling along the motion path to avoid discretization gaps
    num_samples = max(int(length * 20), 100)
    t_vals = np.linspace(-length / 2.0, length / 2.0, num_samples)

    for t in t_vals:
        # Continuous coordinates relative to top-left of kernel
        cx = center + t * dx
        cy = center + t * dy

        # Bilinear splatting to the 4 nearest integer grid nodes
        x0, y0 = int(np.floor(cx)), int(np.floor(cy))
        x1, y1 = x0 + 1, y0 + 1

        fx = cx - x0
        fy = cy - y0

        # Boundary check and accumulation
        if 0 <= y0 < k_size and 0 <= x0 < k_size:
            kernel[y0, x0] += (1.0 - fx) * (1.0 - fy)
        if 0 <= y0 < k_size and 0 <= x1 < k_size:
            kernel[y0, x1] += fx * (1.0 - fy)
        if 0 <= y1 < k_size and 0 <= x0 < k_size:
            kernel[y1, x0] += (1.0 - fx) * fy
        if 0 <= y1 < k_size and 0 <= x1 < k_size:
            kernel[y1, x1] += fx * fy

    # Normalize kernel to conserve image energy
    kernel_sum = np.sum(kernel)
    if kernel_sum > 0:
        kernel = kernel / kernel_sum
    else:
        # Fallback to impulse at center
        kernel[center, center] = 1.0

    return kernel


def apply_motion_blur(
    image: np.ndarray,
    length: int = 9,
    angle: float = 0.0,
    kernel_size: Optional[int] = None,
    mode: str = "reflect",
) -> np.ndarray:
    """
    Apply directional linear motion blur using our custom convolution engine.

    Parameters
    ----------
    image : np.ndarray
        Input image (uint8 or float64, grayscale or RGB).
    length : int, default=9
        Motion trajectory length in pixels.
    angle : float, default=0.0
        Motion orientation in degrees.
    kernel_size : int, optional
        Kernel grid size.
    mode : str, default='reflect'
        Convolution boundary padding mode.

    Returns
    -------
    np.ndarray
        Motion-blurred image matching input dtype and shape.
    """
    validate_image(image)

    was_uint8 = (image.dtype == np.uint8)
    img_float = to_float(image)

    # 1. Create directional motion blur kernel
    kernel = create_motion_blur_kernel(length=length, angle=angle, kernel_size=kernel_size)

    # 2. Convolve image with motion blur PSF
    blurred_float = convolve2d(img_float, kernel, mode=mode, flip_kernel=True)

    # 3. Clip to [0.0, 1.0]
    blurred_float = np.clip(blurred_float, 0.0, 1.0)

    if was_uint8:
        return to_uint8(blurred_float)
    return blurred_float
