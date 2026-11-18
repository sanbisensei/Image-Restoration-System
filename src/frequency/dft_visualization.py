"""
2D Discrete Fourier Transform (DFT) Computation and Visualization.

Academic Background:
The Discrete Fourier Transform decomposes a spatial image f(x, y) into a sum of
complex sinusoidal basis functions of varying spatial frequencies:
    F(u, v) = sum_{x=0}^{M-1} sum_{y=0}^{N-1} f(x, y) * exp(-j * 2*pi * (u*x/M + v*y/N))

Key DIP Concepts:
1. Spatial Domain vs Frequency Domain:
   - Spatial domain describes pixel intensities at coordinate positions (x, y).
   - Frequency domain describes the rate of change of pixel intensities (u, v).
2. Low Frequencies:
   - Represent smooth regions, continuous backgrounds, and overall brightness.
   - Concentrate the overwhelming majority of total image energy.
3. High Frequencies:
   - Represent rapid transitions, sharp edges, fine textures, and noise.
4. Zero-Frequency Shift (fftshift):
   - Raw FFT places the zero-frequency (DC) component at index (0, 0) (top-left).
   - fftshift re-indexes so DC is at the geometric center (M/2, N/2), making
     symmetry and frequency distribution intuitive to inspect.
5. Logarithmic Scaling:
   - The DC component is often 10^5 to 10^7 times larger than high-frequency components.
   - Without compression, only a bright center dot is visible on a black screen.
   - Dynamic range compression: S(u, v) = log(1 + |F(u, v)|).
"""

from typing import Optional, Tuple, Union
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src.utils.validation import validate_image
from src.utils.image_helpers import ensure_grayscale, to_float, to_uint8


def compute_dft2(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the 2D Discrete Fourier Transform and its centered zero-frequency shift.

    Parameters
    ----------
    image : np.ndarray
        Input image (2D grayscale or 3D RGB). If RGB, converted to grayscale.

    Returns
    -------
    tuple of (F, F_shift)
        - F: Raw 2D complex Fourier transform array of shape (M, N).
        - F_shift: Centered 2D complex Fourier transform with zero-frequency at (M//2, N//2).
    """
    validate_image(image)
    gray = ensure_grayscale(image)
    img_float = to_float(gray)

    # Compute 2D Fast Fourier Transform
    f_transform = np.fft.fft2(img_float)

    # Shift zero-frequency component to center of spectrum
    f_shift = np.fft.fftshift(f_transform)

    return f_transform, f_shift


def compute_magnitude_spectrum(
    f_shift: np.ndarray,
    log_scale: bool = True,
) -> np.ndarray:
    """
    Calculate the magnitude spectrum |F(u, v)| from a centered Fourier transform.

    Parameters
    ----------
    f_shift : np.ndarray
        Centered 2D complex Fourier transform.
    log_scale : bool, default=True
        Whether to apply logarithmic compression: S = log(1 + |F|).

    Returns
    -------
    np.ndarray
        Normalized magnitude spectrum in range [0.0, 1.0].
    """
    if not np.iscomplexobj(f_shift):
        raise ValueError("Input to compute_magnitude_spectrum must be a complex Fourier array.")

    # Magnitude |F(u, v)| = sqrt(Re^2 + Im^2)
    magnitude = np.abs(f_shift)

    if log_scale:
        spectrum = np.log1p(magnitude)  # log(1 + |F|)
    else:
        spectrum = magnitude

    # Normalize to [0.0, 1.0] for consistent visualization
    s_min, s_max = np.min(spectrum), np.max(spectrum)
    if s_max > s_min:
        spectrum_norm = (spectrum - s_min) / (s_max - s_min)
    else:
        spectrum_norm = np.zeros_like(spectrum, dtype=np.float64)

    return spectrum_norm


def compute_phase_spectrum(f_shift: np.ndarray) -> np.ndarray:
    """
    Calculate the phase spectrum phi(u, v) = arctan2(Im, Re).

    Returns
    -------
    np.ndarray
        Phase angle array with values in [-pi, +pi].
    """
    if not np.iscomplexobj(f_shift):
        raise ValueError("Input must be a complex Fourier array.")
    return np.angle(f_shift)


def visualize_spectrum(
    image: np.ndarray,
    title: str = "Image & Fourier Magnitude Spectrum",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Display a side-by-side plot comparing spatial domain image and frequency magnitude spectrum.

    Parameters
    ----------
    image : np.ndarray
        Spatial image to visualize.
    title : str
        Figure main title.
    save_path : str or Path, optional
        Destination path to save figure.
    show : bool, default=True
        Whether to invoke plt.show().

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    validate_image(image)
    gray = ensure_grayscale(image)

    _, f_shift = compute_dft2(gray)
    magnitude_spectrum = compute_magnitude_spectrum(f_shift, log_scale=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Spatial Domain
    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255 if gray.dtype == np.uint8 else 1.0)
    axes[0].set_title("Spatial Domain f(x, y)", fontsize=11, pad=8)
    axes[0].axis("off")

    # Frequency Domain (Magnitude Spectrum)
    axes[1].imshow(magnitude_spectrum, cmap="inferno", vmin=0.0, vmax=1.0)
    axes[1].set_title("Frequency Magnitude Spectrum log(1+|F|)", fontsize=11, pad=8)
    axes[1].axis("off")

    plt.suptitle(title, fontsize=13, y=0.98)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=160, bbox_inches="tight")

    if show:
        plt.show()

    return fig
