"""
Custom 2D Convolution Engine for Digital Image Processing.

Academic Implementation Rule:
This module implements discrete 2D convolution from mathematical foundations
using pure NumPy. It does NOT call cv2.filter2D, scipy.ndimage.convolve, or
scipy.signal.convolve2d.

Mathematical Background:
For a 2D continuous/discrete image f(x, y) and filter kernel h(s, t) of size
Kh x Kw with center (ch, cw):
    (f * h)(x, y) = sum_{s} sum_{t} f(x - s, y - t) * h(s, t)

Notice that in true convolution, the kernel is flipped horizontally and
vertically (h(-s, -t)) before sliding across the image. When the kernel is
not flipped, the operation is Cross-Correlation. For symmetric kernels
(such as Gaussian filters), convolution and correlation produce identical results,
but for directional motion blur kernels, flipping is mathematically critical.

Boundary Handling (Padding):
To produce an output of identical size to the input without border clipping,
the image is padded by pad_h = Kh // 2 and pad_w = Kw // 2.
Modes supported:
- 'reflect': Mirrors pixels at the edge (d c b a | a b c d | d c b a).
             Standard in DIP to avoid dark borders.
- 'replicate' ('edge'): Repeats edge pixels (a a a a | a b c d | d d d d).
- 'constant': Pads with zeros (0 0 0 0 | a b c d | 0 0 0 0).
"""

from typing import Union
import numpy as np
from src.utils.validation import validate_image


def convolve2d_grayscale(
    image: np.ndarray,
    kernel: np.ndarray,
    mode: str = "reflect",
    flip_kernel: bool = True,
) -> np.ndarray:
    """
    Perform 2D discrete spatial convolution on a single-channel grayscale image.

    Parameters
    ----------
    image : np.ndarray
        2D grayscale array of shape (H, W).
    kernel : np.ndarray
        2D filter kernel of shape (Kh, Kw). Dimensions must be odd positive integers.
    mode : str, default='reflect'
        Padding mode ('reflect', 'edge', 'constant').
    flip_kernel : bool, default=True
        Whether to flip kernel along both axes (true mathematical convolution).
        Set to False for 2D cross-correlation.

    Returns
    -------
    np.ndarray
        Convolved 2D image of shape (H, W) with dtype float64.
    """
    validate_image(image, require_2d=True)

    if not isinstance(kernel, np.ndarray) or kernel.ndim != 2:
        raise ValueError(f"Kernel must be a 2D numpy array, got {type(kernel).__name__} with shape {getattr(kernel, 'shape', None)}")

    kh, kw = kernel.shape
    if kh <= 0 or kw <= 0:
        raise ValueError("Kernel dimensions must be positive.")
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError(f"Kernel dimensions must be odd integers; got shape ({kh}, {kw})")

    h, w = image.shape
    pad_h = kh // 2
    pad_w = kw // 2

    # Map friendly mode names to numpy pad modes
    np_mode = "edge" if mode == "replicate" else mode

    # Pad image to maintain (H, W) output shape
    padded = np.pad(image.astype(np.float64), ((pad_h, pad_h), (pad_w, pad_w)), mode=np_mode)

    # Flip kernel for true convolution: h(-s, -t)
    k = np.flip(kernel) if flip_kernel else kernel.copy()

    # Vectorized accumulation over kernel offsets:
    # Instead of slow 4-nested Python loops over every pixel, we accumulate
    # 2D spatial slices for each of the (kh * kw) kernel taps.
    output = np.zeros((h, w), dtype=np.float64)

    for i in range(kh):
        for j in range(kw):
            weight = k[i, j]
            if weight != 0.0:
                output += weight * padded[i : i + h, j : j + w]

    return output


def convolve2d(
    image: np.ndarray,
    kernel: np.ndarray,
    mode: str = "reflect",
    flip_kernel: bool = True,
) -> np.ndarray:
    """
    Perform 2D convolution supporting both 2D grayscale and 3D RGB images.

    For 3D images (H, W, 3), convolution is applied independently per channel.

    Parameters
    ----------
    image : np.ndarray
        2D (H, W) or 3D (H, W, 3) image array.
    kernel : np.ndarray
        2D filter kernel with odd dimensions.
    mode : str, default='reflect'
        Padding mode ('reflect', 'edge', 'constant').
    flip_kernel : bool, default=True
        True for convolution, False for correlation.

    Returns
    -------
    np.ndarray
        Convolved image with float64 dtype and same spatial dimensions.
    """
    validate_image(image)

    if image.ndim == 2:
        return convolve2d_grayscale(image, kernel, mode=mode, flip_kernel=flip_kernel)

    # Multi-channel 3D image
    channels = [
        convolve2d_grayscale(image[:, :, c], kernel, mode=mode, flip_kernel=flip_kernel)
        for c in range(image.shape[2])
    ]
    return np.stack(channels, axis=-1)
