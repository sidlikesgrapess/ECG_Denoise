from __future__ import annotations

import numpy as np


def remove_dc_mean(signal: np.ndarray) -> np.ndarray:
    """Remove DC component by subtracting the segment mean."""
    x = np.asarray(signal, dtype=np.float64)
    if x.size == 0:
        return x.copy()
    return x - float(np.mean(x))
