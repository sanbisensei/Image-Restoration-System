"""
Inverse Filtering Restoration in Frequency Domain.

Academic Background:
In frequency domain, degradation by blur H(u, v) and additive noise N(u, v) is:
    G(u, v) = H(u, v) * F(u, v) + N(u, v)

The idealized inverse filter divides the degraded spectrum by the blur OTF:
    F_hat(u, v) = G(u, v) / H(u, v) = F(u, v) + (N(u, v) / H(u, v))

Why Inverse Filtering Fails (Critical Academic Concept):
1. Division by Near-Zero:
   Blur acts as a low-pass filter. At high spatial frequencies, |H(u, v)| approaches zero.
2. Catastrophic Noise Amplification:
   The term N(u, v) / H(u, v) causes tiny sensor noise (e.g. 10^-4) to be divided by
   extremely small values (e.g. 10^-6), amplifying the noise by factors of millions.
   The resulting restored image is completely overwhelmed by high-frequency ringing / snow.
3. When it CAN work:
   Only in synthetic noise-free environments (N = 0) where H is known with absolute precision.

Stabilization Strategies Supported:
1. Thresholded Inverse Filter: Frequencies where |H| < threshold have gain set to 0.
2. Radial Low-Pass Cutoff (D0): Restoration is restricted to low frequencies where |H| is large.
"""

from typing import Optional, Union
import numpy as np

from src.frequency.psf import psf_to_otf
from src.utils.validation import validate_image
from src.utils.image_helpers import ensure_grayscale, to_float, to_uint8


def apply_inverse_filter(
    image: np.ndarray,
    psf: np.ndarray,
    threshold: Optional[float] = None,
    cutoff_radius: Optional[float] = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Restore a blurred image using frequency-domain Inverse Filtering.

    Parameters
    ----------
    image : np.ndarray
        Degraded input image (grayscale or RGB, uint8 or float64).
    psf : np.ndarray
        Point Spread Function (spatial blur kernel).
    threshold : float, optional
        Magnitude stabilization threshold (relative to max|H|).
        Frequencies where |H(u, v)| < threshold * max(|H|) are attenuated to prevent
        explosive noise division.
        If None, direct unconstrained inverse filtering is performed (showing noise amplification).
    cutoff_radius : float, optional
        Radial frequency cutoff D0 (in pixels from DC). Frequencies with distance > D0
        from the center are zeroed out.
    eps : float, default=1e-8
        Small epsilon to prevent division by exact 0.0 in unconstrained mode.

    Returns
    -------
    np.ndarray
        Restored image matching input dtype and shape.
    """
    validate_image(image)

    was_uint8 = (image.dtype == np.uint8)

    def _restore_channel(channel_2d: np.ndarray) -> np.ndarray:
        m, n = channel_2d.shape
        img_flt = to_float(channel_2d)

        # 1. Fourier transform of degraded channel G(u, v)
        g_fft = np.fft.fft2(img_flt)

        # 2. Optical Transfer Function H(u, v)
        h_otf = psf_to_otf(psf, (m, n))
        h_mag = np.abs(h_otf)
        max_h = np.max(h_mag) if np.max(h_mag) > 0 else 1.0

        # 3. Construct Inverse Filter Transfer Function H_inv(u, v)
        if threshold is not None:
            # Thresholded inverse filter
            abs_thresh = threshold * max_h
            h_inv = np.zeros_like(h_otf, dtype=np.complex128)
            valid_mask = h_mag >= abs_thresh
            h_inv[valid_mask] = 1.0 / h_otf[valid_mask]
        else:
            # Direct inverse with epsilon division protection
            safe_h = np.where(h_mag < eps, eps * np.sign(h_otf.real + 1e-12), h_otf)
            h_inv = 1.0 / safe_h

        # 4. Optional radial low-pass cutoff
        if cutoff_radius is not None:
            # Frequency grid centered at (m // 2, n // 2)
            u = np.arange(m) - (m // 2)
            v = np.arange(n) - (n // 2)
            uu, vv = np.meshgrid(u, v, indexing="ij")
            dist = np.sqrt(uu**2 + vv**2)
            # Roll distance grid to match unshifted FFT layout
            dist_unshifted = np.fft.ifftshift(dist)
            h_inv[dist_unshifted > cutoff_radius] = 0.0

        # 5. Frequency multiplication: F_hat(u, v) = G(u, v) * H_inv(u, v)
        f_hat = g_fft * h_inv

        # 6. Inverse FFT to spatial domain
        restored = np.fft.ifft2(f_hat).real

        # 7. Clip to valid [0.0, 1.0] range
        restored = np.clip(restored, 0.0, 1.0)
        return restored

    if image.ndim == 2:
        restored_float = _restore_channel(image)
    else:
        # Multi-channel RGB
        channels = [_restore_channel(image[:, :, c]) for c in range(image.shape[2])]
        restored_float = np.stack(channels, axis=-1)

    if was_uint8:
        return to_uint8(restored_float)
    return restored_float
