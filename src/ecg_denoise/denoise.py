from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .filters import FilterSection, cascade_iir, design_first_order_highpass, design_first_order_lowpass, design_notch


@dataclass(frozen=True)
class DenoiseConfig:
    highpass_cutoff_hz: float = 0.5
    lowpass_cutoff_hz: float = 40.0
    powerline_hz: float = 60.0
    notch_bandwidth_hz: float = 2.0
    include_second_harmonic_notch: bool = True


def build_filter_sections(fs: float, config: DenoiseConfig) -> list[FilterSection]:
    sections: list[FilterSection] = [
        design_first_order_highpass(fs, config.highpass_cutoff_hz),
        design_notch(fs, config.powerline_hz, config.notch_bandwidth_hz),
    ]

    second_harmonic_hz = 2.0 * config.powerline_hz
    if config.include_second_harmonic_notch and second_harmonic_hz < (fs / 2.0):
        sections.append(design_notch(fs, second_harmonic_hz, config.notch_bandwidth_hz))

    sections.append(design_first_order_lowpass(fs, config.lowpass_cutoff_hz))
    return sections


def denoise_ecg(
    signal: np.ndarray,
    fs: float,
    config: DenoiseConfig | None = None,
) -> tuple[np.ndarray, list[FilterSection]]:
    """Denoise ECG using a hand-built cascaded transfer-function pipeline."""
    cfg = config if config is not None else DenoiseConfig()
    sections = build_filter_sections(fs, cfg)
    filtered = cascade_iir(np.asarray(signal, dtype=np.float64), sections)
    return filtered, sections
