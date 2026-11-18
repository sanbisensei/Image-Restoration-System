"""
Point Spread Function (PSF) and Optical Transfer Function (OTF) Utilities.

Academic Background:
1. Point Spread Function (PSF):
   The spatial impulse response h(x, y) of the optical degradation system.
   It represents how an idealized point source of light is blurred and spread.
2. Optical Transfer Function (OTF):
   The frequency-domain representation H(u, v) = FFT2{h(x, y)}.
   - Magnitude |H(u, v)| is the Modulation Transfer Function (MTF), describing
     how different spatial frequencies are attenuated by the blur.
   - Phase of H(u, v) is the Phase Transfer Function (PTF), describing phase shifts.

Origin Alignment in FFT:
A spatial blur kernel is centered at (Kh // 2, Kw // 2).
In the discrete Fourier transform, the spatial origin (0, 0) corresponds to index [0, 0].
To prevent spatial coordinate shifts during frequency-domain restoration, the PSF must
be circularly shifted (rolled) by (-Kh // 2, -Kw // 2) before computing the 2D FFT.
"""

from typing import Tuple
import numpy as np


def psf_to_otf(psf: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert a spatial Point Spread Function (PSF) to an Optical Transfer Function (OTF).

    Pads the PSF to target_shape (M, N) and circularly rolls the center to index [0, 0]
    so that applying H(u, v) in the frequency domain incurs zero spatial phase shift.

    Parameters
    ----------
    psf : np.ndarray
        2D spatial blur kernel of shape (Kh, Kw).
    target_shape : tuple of (M, N)
        Spatial dimensions of the target image to restore.

    Returns
    -------
    np.ndarray
        Complex 2D array of shape (M, N) representing H(u, v) in the unshifted frequency domain.
    """
    if not isinstance(psf, np.ndarray) or psf.ndim != 2:
        raise ValueError(f"PSF must be a 2D numpy array, got {getattr(psf, 'shape', None)}")

    kh, kw = psf.shape
    m, n = target_shape

    if kh > m or kw > n:
        raise ValueError(f"PSF dimensions ({kh}, {kw}) cannot exceed target image shape ({m}, {n})")

    # Pad PSF into target shape array
    padded_psf = np.zeros((m, n), dtype=np.float64)
    padded_psf[:kh, :kw] = psf

    # Circular shift to move the PSF center (kh // 2, kw // 2) to [0, 0]
    shift_h = -(kh // 2)
    shift_w = -(kw // 2)
    aligned_psf = np.roll(padded_psf, shift=(shift_h, shift_w), axis=(0, 1))

    # Compute Optical Transfer Function H(u, v)
    otf = np.fft.fft2(aligned_psf)

    return otf


def otf_to_psf(otf: np.ndarray, original_kernel_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert an Optical Transfer Function (OTF) back to its spatial PSF kernel.

    Parameters
    ----------
    otf : np.ndarray
        Complex 2D frequency array H(u, v).
    original_kernel_shape : tuple of (Kh, Kw)
        Original dimensions of the spatial blur kernel.

    Returns
    -------
    np.ndarray
        2D float64 spatial PSF kernel of shape (Kh, Kw).
    """
    kh, kw = original_kernel_shape
    # Inverse FFT to spatial domain
    spatial = np.fft.ifft2(otf).real

    # Reverse circular roll
    shift_h = kh // 2
    shift_w = kw // 2
    unrolled = np.roll(spatial, shift=(shift_h, shift_w), axis=(0, 1))

    # Crop kernel back to original footprint
    psf = unrolled[:kh, :kw]
    psf_sum = np.sum(psf)
    if psf_sum > 0:
        psf = psf / psf_sum

    return psf
