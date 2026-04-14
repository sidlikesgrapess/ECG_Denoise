from __future__ import annotations

from typing import Iterable

import numpy as np


FilterSection = tuple[np.ndarray, np.ndarray]


def apply_iir_filter(signal: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Apply an IIR filter section with direct-form difference equation."""
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
