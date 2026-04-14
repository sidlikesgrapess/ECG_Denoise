from __future__ import annotations

import numpy as np

from .iir_core import FilterSection


def design_notch_iir(fs: float, notch_hz: float, bandwidth_hz: float = 2.0) -> FilterSection:
    """Design a 2nd-order IIR notch filter centered at notch_hz."""
    if fs <= 0 or notch_hz <= 0 or bandwidth_hz <= 0:
        raise ValueError("fs, notch_hz, and bandwidth_hz must be positive")
    if notch_hz >= fs / 2.0:
        raise ValueError("notch_hz must be below Nyquist")

    w0 = 2.0 * np.pi * notch_hz / fs
    r = 1.0 - (np.pi * bandwidth_hz / fs)
    r = float(np.clip(r, 0.8, 0.9999))

    cos_w0 = float(np.cos(w0))
    b = np.array([1.0, -2.0 * cos_w0, 1.0], dtype=np.float64)
    a = np.array([1.0, -2.0 * r * cos_w0, r * r], dtype=np.float64)
    return b, a
