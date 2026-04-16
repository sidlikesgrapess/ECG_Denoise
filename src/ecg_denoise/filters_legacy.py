from __future__ import annotations

import numpy as np

from .butterworth_hpf import design_butterworth_highpass
from .dc_removal import remove_dc_mean
from .iir_core import FilterSection, apply_iir_filter, cascade_iir
from .kaiser_fir import apply_fir_filter, design_kaiser_lowpass_fir
from .notch_iir import design_notch_iir


def design_notch(fs: float, notch_hz: float, bandwidth_hz: float = 2.0) -> FilterSection:
    """Backward-compatible alias to the dedicated notch-IIR module."""
    return design_notch_iir(fs, notch_hz, bandwidth_hz)


def design_first_order_highpass(fs: float, cutoff_hz: float) -> FilterSection:
    """Legacy name kept for compatibility; implemented by Butterworth HPF module."""
    return design_butterworth_highpass(fs, cutoff_hz)


def design_first_order_lowpass(fs: float, cutoff_hz: float) -> FilterSection:
    """
    Legacy first-order low-pass kept for compatibility.
    """
    if fs <= 0 or cutoff_hz <= 0:
        raise ValueError("fs and cutoff_hz must be positive")

    wc = 2.0 * np.pi * cutoff_hz
    denom = (2.0 * fs) + wc

    b = np.array([wc / denom, wc / denom], dtype=np.float64)
    a = np.array([1.0, (wc - (2.0 * fs)) / denom], dtype=np.float64)
    return b, a
