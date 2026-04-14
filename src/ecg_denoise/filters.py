from __future__ import annotations

from typing import Iterable

import numpy as np


FilterSection = tuple[np.ndarray, np.ndarray]


def apply_iir_filter(signal: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Apply an IIR filter with direct-form difference equation."""
    x = np.asarray(signal, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)

    if b.ndim != 1 or a.ndim != 1 or b.size == 0 or a.size == 0:
        raise ValueError("Filter coefficients must be non-empty 1D arrays")
    if np.isclose(a[0], 0.0):
        raise ValueError("a[0] cannot be zero")

    if not np.isclose(a[0], 1.0):
        b = b / a[0]
        a = a / a[0]

    y = np.zeros_like(x)
    for n in range(x.size):
        acc = 0.0

        for k in range(b.size):
            if n - k >= 0:
                acc += b[k] * x[n - k]

        for k in range(1, a.size):
            if n - k >= 0:
                acc -= a[k] * y[n - k]

        y[n] = acc

    return y


def cascade_iir(signal: np.ndarray, sections: Iterable[FilterSection]) -> np.ndarray:
    y = np.asarray(signal, dtype=np.float64)
    for b, a in sections:
        y = apply_iir_filter(y, b, a)
    return y


def design_first_order_highpass(fs: float, cutoff_hz: float) -> FilterSection:
    """
    First-order high-pass from analog H(s)=s/(s+wc) via bilinear transform.
    """
    if fs <= 0 or cutoff_hz <= 0:
        raise ValueError("fs and cutoff_hz must be positive")

    wc = 2.0 * np.pi * cutoff_hz
    denom = (2.0 * fs) + wc

    b = np.array([(2.0 * fs) / denom, -(2.0 * fs) / denom], dtype=np.float64)
    a = np.array([1.0, (wc - (2.0 * fs)) / denom], dtype=np.float64)
    return b, a


def design_first_order_lowpass(fs: float, cutoff_hz: float) -> FilterSection:
    """
    First-order low-pass from analog H(s)=wc/(s+wc) via bilinear transform.
    """
    if fs <= 0 or cutoff_hz <= 0:
        raise ValueError("fs and cutoff_hz must be positive")

    wc = 2.0 * np.pi * cutoff_hz
    denom = (2.0 * fs) + wc

    b = np.array([wc / denom, wc / denom], dtype=np.float64)
    a = np.array([1.0, (wc - (2.0 * fs)) / denom], dtype=np.float64)
    return b, a


def design_notch(fs: float, notch_hz: float, bandwidth_hz: float = 2.0) -> FilterSection:
    """
    2nd-order notch filter:
      H(z) = (1 - 2cos(w0)z^-1 + z^-2) / (1 - 2r cos(w0)z^-1 + r^2 z^-2)
    """
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
