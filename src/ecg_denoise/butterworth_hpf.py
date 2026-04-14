from __future__ import annotations

import numpy as np

from .iir_core import FilterSection


def design_butterworth_highpass(fs: float, cutoff_hz: float) -> FilterSection:
    """Design a 2nd-order Butterworth high-pass biquad using bilinear transform form."""
    if fs <= 0 or cutoff_hz <= 0:
        raise ValueError("fs and cutoff_hz must be positive")
    if cutoff_hz >= fs / 2.0:
        raise ValueError("cutoff_hz must be below Nyquist")

    q = 1.0 / np.sqrt(2.0)  # Butterworth damping
    w0 = 2.0 * np.pi * cutoff_hz / fs
    cos_w0 = float(np.cos(w0))
    sin_w0 = float(np.sin(w0))
    alpha = sin_w0 / (2.0 * q)

    b0 = (1.0 + cos_w0) / 2.0
    b1 = -(1.0 + cos_w0)
    b2 = (1.0 + cos_w0) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha

    b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return b, a
