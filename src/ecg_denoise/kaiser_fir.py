from __future__ import annotations

import math

import numpy as np


def _kaiser_beta(attenuation_db: float) -> float:
    if attenuation_db > 50.0:
        return 0.1102 * (attenuation_db - 8.7)
    if attenuation_db >= 21.0:
        return 0.5842 * ((attenuation_db - 21.0) ** 0.4) + 0.07886 * (attenuation_db - 21.0)
    return 0.0


def _estimate_num_taps(fs: float, transition_hz: float, attenuation_db: float) -> int:
    delta_w = 2.0 * np.pi * transition_hz / fs
    if delta_w <= 0:
        raise ValueError("transition_hz must be positive")

    taps = int(math.ceil((attenuation_db - 8.0) / (2.285 * delta_w))) + 1
    taps = max(taps, 5)
    if taps % 2 == 0:
        taps += 1
    return taps


def design_kaiser_lowpass_fir(
    fs: float,
    cutoff_hz: float,
    *,
    transition_hz: float = 8.0,
    attenuation_db: float = 60.0,
) -> np.ndarray:
    """Design a low-pass FIR using a Kaiser windowed-sinc method."""
    if fs <= 0 or cutoff_hz <= 0:
        raise ValueError("fs and cutoff_hz must be positive")
    if cutoff_hz >= fs / 2.0:
        raise ValueError("cutoff_hz must be below Nyquist")

    num_taps = _estimate_num_taps(fs, transition_hz, attenuation_db)
    beta = _kaiser_beta(attenuation_db)

    m = np.arange(num_taps, dtype=np.float64) - ((num_taps - 1) / 2.0)
    cutoff_norm = cutoff_hz / fs
    ideal = 2.0 * cutoff_norm * np.sinc(2.0 * cutoff_norm * m)
    window = np.kaiser(num_taps, beta)

    taps = ideal * window
    taps /= np.sum(taps)
    return taps.astype(np.float64)


def apply_fir_filter(signal: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Apply FIR filter taps and return same-length output."""
    x = np.asarray(signal, dtype=np.float64)
    h = np.asarray(taps, dtype=np.float64)
    if h.ndim != 1 or h.size == 0:
        raise ValueError("taps must be a non-empty 1D array")
    return np.convolve(x, h, mode="same")
