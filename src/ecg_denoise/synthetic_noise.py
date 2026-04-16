from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticNoiseConfig:
    breathing_hz: float = 0.25
    breathing_amplitude_mv: float = 0.12
    powerline_hz: float = 60.0
    powerline_amplitude_mv: float = 0.08
    muscle_center_hz: float = 90.0
    muscle_amplitude_mv: float = 0.10
    burst_count: int = 5
    burst_width_sec: float = 0.20


def _muscle_envelope(num_samples: int, fs: float, cfg: SyntheticNoiseConfig, rng: np.random.Generator) -> np.ndarray:
    duration_sec = num_samples / fs
    centers = rng.uniform(0.0, duration_sec, size=max(cfg.burst_count, 1))
    t = np.arange(num_samples, dtype=np.float64) / fs
    sigma = max(cfg.burst_width_sec, 1e-3)

    envelope = np.zeros(num_samples, dtype=np.float64)
    for center in centers:
        envelope += np.exp(-0.5 * ((t - center) / sigma) ** 2)

    peak = np.max(envelope)
    if peak > 0:
        envelope /= peak
    return envelope


def add_synthetic_noise(
    clean_signal_mv: np.ndarray,
    fs: float,
    *,
    config: SyntheticNoiseConfig | None = None,
    seed: int = 1234,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    # Riyan Dhiren Shah: synthetic noise addition for breathing, powerline, and muscle artifacts.
    x = np.asarray(clean_signal_mv, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("clean_signal_mv must be a non-empty 1D array")
    if fs <= 0:
        raise ValueError("fs must be positive")

    cfg = config if config is not None else SyntheticNoiseConfig()
    t = np.arange(x.size, dtype=np.float64) / fs
    rng = np.random.default_rng(seed)

    breathing = cfg.breathing_amplitude_mv * (
        np.sin(2.0 * np.pi * cfg.breathing_hz * t)
        + 0.3 * np.sin(2.0 * np.pi * (2.0 * cfg.breathing_hz) * t + 0.7)
    )

    phase_1 = rng.uniform(0.0, 2.0 * np.pi)
    phase_2 = rng.uniform(0.0, 2.0 * np.pi)
    powerline = cfg.powerline_amplitude_mv * (
        np.sin(2.0 * np.pi * cfg.powerline_hz * t + phase_1)
        + 0.25 * np.sin(2.0 * np.pi * (2.0 * cfg.powerline_hz) * t + phase_2)
    )

    envelope = _muscle_envelope(x.size, fs, cfg, rng)
    white = rng.standard_normal(x.size)
    ma_len = max(int(round(0.03 * fs)), 3)
    ma_kernel = np.ones(ma_len, dtype=np.float64) / ma_len
    white_low = np.convolve(white, ma_kernel, mode="same")
    white_high = white - white_low
    white_high /= max(np.std(white_high), 1e-12)

    phase_3 = rng.uniform(0.0, 2.0 * np.pi)
    phase_4 = rng.uniform(0.0, 2.0 * np.pi)
    carrier = (
        0.7 * np.sin(2.0 * np.pi * cfg.muscle_center_hz * t + phase_3)
        + 0.3 * np.sin(2.0 * np.pi * (cfg.muscle_center_hz + 25.0) * t + phase_4)
    )
    muscle = cfg.muscle_amplitude_mv * envelope * (0.5 * carrier + 0.5 * white_high)

    total_noise = breathing + powerline + muscle
    noisy = x + total_noise
    components = {
        "breathing": breathing,
        "powerline": powerline,
        "muscle": muscle,
        "total": total_noise,
    }
    return noisy, components