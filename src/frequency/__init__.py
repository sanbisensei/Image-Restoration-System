"""
Frequency domain processing package:
- 2D Discrete Fourier Transform & Magnitude Spectrum Visualization
- Point Spread Function (PSF) & Optical Transfer Function (OTF)
- Inverse Filtering Restoration
- Wiener Filtering Restoration
"""

from src.frequency.dft_visualization import (
    compute_dft2,
    compute_magnitude_spectrum,
    compute_phase_spectrum,
    visualize_spectrum,
)
from src.frequency.psf import psf_to_otf, otf_to_psf
from src.frequency.inverse_filter import apply_inverse_filter
from src.frequency.wiener_filter import apply_wiener_filter

__all__ = [
    "compute_dft2",
    "compute_magnitude_spectrum",
    "compute_phase_spectrum",
    "visualize_spectrum",
    "psf_to_otf",
    "otf_to_psf",
    "apply_inverse_filter",
    "apply_wiener_filter",
]
