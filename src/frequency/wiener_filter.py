"""
Wiener Filter (Minimum Mean Square Error) Frequency Restoration.

Academic Background:
Formulated by Norbert Wiener, the Wiener filter seeks an estimate f_hat(x, y)
that minimizes the statistical mean square error between the original image
and the restored image:
    e^2 = E{ (f(x, y) - f_hat(x, y))^2 }

Mathematical Transfer Function:
    W(u, v) = H*(u, v) / ( |H(u, v)|^2 + [S_eta(u, v) / S_f(u, v)] )
where:
- H*(u, v) is the complex conjugate of the blur Optical Transfer Function.
- |H(u, v)|^2 = H(u, v) * H*(u, v) is the energy spectrum of the blur.
- S_eta(u, v) is the power spectral density of the additive noise.
- S_f(u, v) is the power spectral density of the uncorrupted image.

Parametric Constant Ratio Formulation:
When the exact power spectra are unknown, the ratio is approximated by an
adjustable constant K (Noise-to-Signal Power Ratio):
    W(u, v) = H*(u, v) / ( |H(u, v)|^2 + K )

The restored spectrum is:
    F_hat(u, v) = W(u, v) * G(u, v) = [ H*(u, v) / (|H(u, v)|^2 + K) ] * G(u, v)

Role of the K Parameter (Viva Explanation):
1. When K = 0:
   W(u, v) = H* / |H|^2 = 1 / H.
   The Wiener filter reduces exactly to the unconstrained Inverse Filter.
2. Low Frequencies (|H|^2 >> K):
   W(u, v) approx 1 / H(u, v). The filter performs full deblurring.
3. High Frequencies (|H|^2 << K):
   The denominator is dominated by K, so W(u, v) approx H* / K -> 0.
   Instead of dividing by zero and amplifying noise to infinity (like inverse filtering),
   the Wiener filter gracefully attenuates high-frequency noise.
"""

from typing import Optional, Union
import numpy as np

from src.frequency.psf import psf_to_otf
from src.utils.validation import validate_image
from src.utils.image_helpers import ensure_grayscale, to_float, to_uint8


def apply_wiener_filter(
    image: np.ndarray,
    psf: np.ndarray,
    K: float = 0.01,
) -> np.ndarray:
    """
    Restore a blurred and noisy image using the frequency-domain Wiener filter.

    Parameters
    ----------
    image : np.ndarray
        Degraded image (uint8 or float64, grayscale or RGB).
    psf : np.ndarray
        Point Spread Function (spatial blur kernel).
    K : float, default=0.01
        Noise-to-signal power ratio constant.
        - K -> 0: Behaves like inverse filter (sharp, but amplifies noise).
        - Small K (e.g. 0.001): Suitable for low noise.
        - Moderate K (e.g. 0.01 - 0.05): Optimal balance for typical sensor noise.
        - High K (e.g. 0.1 - 0.5): Heavily suppresses noise, but retains slight blur.

    Returns
    -------
    np.ndarray
        Restored image matching input dtype and shape.
    """
    validate_image(image)

    if K < 0.0:
        raise ValueError(f"Wiener parameter K must be non-negative (>= 0); got {K}")

    was_uint8 = (image.dtype == np.uint8)

    def _restore_channel(channel_2d: np.ndarray) -> np.ndarray:
        m, n = channel_2d.shape
        img_flt = to_float(channel_2d)

        # 1. Fourier transform of degraded channel G(u, v)
        g_fft = np.fft.fft2(img_flt)

        # 2. Optical Transfer Function H(u, v)
        h_otf = psf_to_otf(psf, (m, n))

        # 3. Complex conjugate H*(u, v) and power spectrum |H(u, v)|^2
        h_conj = np.conj(h_otf)
        h_power = np.abs(h_otf) ** 2

        # 4. Wiener Transfer Function: W(u, v) = H* / (|H|^2 + K)
        # Denominator is strictly positive whenever K > 0
        denominator = h_power + K
        # Protect against K = 0 exact zero divisions
        denominator = np.where(denominator == 0.0, 1e-12, denominator)
        w_filter = h_conj / denominator

        # 5. Frequency multiplication: F_hat(u, v) = W(u, v) * G(u, v)
        f_hat = w_filter * g_fft

        # 6. Inverse Fourier Transform to spatial domain
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
