from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .butterworth_hpf import design_butterworth_highpass
from .dc_removal import remove_dc_mean
from .iir_core import FilterSection, cascade_iir
from .kaiser_fir import apply_fir_filter, design_kaiser_lowpass_fir
from .notch_iir import design_notch_iir


@dataclass(frozen=True)
class DenoiseConfig:
    highpass_cutoff_hz: float = 0.5
    powerline_hz: float = 60.0
    notch_bandwidth_hz: float = 2.0
    include_second_harmonic_notch: bool = True
    kaiser_cutoff_hz: float = 40.0
    kaiser_transition_hz: float = 8.0
    kaiser_attenuation_db: float = 60.0
    zero_phase: bool = True
    edge_pad_seconds: float = 1.5


def build_filter_sections(fs: float, config: DenoiseConfig) -> list[FilterSection]:
    sections: list[FilterSection] = [
        design_butterworth_highpass(fs, config.highpass_cutoff_hz),
        design_notch_iir(fs, config.powerline_hz, config.notch_bandwidth_hz),
    ]

    second_harmonic_hz = 2.0 * config.powerline_hz
    if config.include_second_harmonic_notch and second_harmonic_hz < (fs / 2.0):
        sections.append(design_notch_iir(fs, second_harmonic_hz, config.notch_bandwidth_hz))

    return sections


def _reflect_pad(signal: np.ndarray, pad_len: int) -> tuple[np.ndarray, int]:
    x = np.asarray(signal, dtype=np.float64)
    if x.size <= 1 or pad_len <= 0:
        return x, 0

    # Reflection padding reduces startup transients at segment edges.
    effective_pad = min(pad_len, x.size - 1)
    padded = np.pad(x, (effective_pad, effective_pad), mode="reflect")
    return padded, effective_pad


def _zero_phase_cascade(signal: np.ndarray, sections: list[FilterSection]) -> np.ndarray:
    forward = cascade_iir(signal, sections)
    backward = cascade_iir(forward[::-1], sections)
    return backward[::-1]


def denoise_ecg(
    signal: np.ndarray,
    fs: float,
    config: DenoiseConfig | None = None,
) -> tuple[np.ndarray, list[FilterSection]]:
    """Denoise ECG: DC removal -> Butterworth HPF -> Notch IIR -> Kaiser FIR."""
    cfg = config if config is not None else DenoiseConfig()
    sections = build_filter_sections(fs, cfg)

    x = remove_dc_mean(np.asarray(signal, dtype=np.float64))
    if cfg.zero_phase:
        pad_len = max(0, int(round(cfg.edge_pad_seconds * fs)))
        padded, used_pad = _reflect_pad(x, pad_len)
        iir_out = _zero_phase_cascade(padded, sections)
        if used_pad > 0:
            iir_out = iir_out[used_pad:-used_pad]
    else:
        iir_out = cascade_iir(x, sections)

    kaiser_taps = design_kaiser_lowpass_fir(
        fs,
        cfg.kaiser_cutoff_hz,
        transition_hz=cfg.kaiser_transition_hz,
        attenuation_db=cfg.kaiser_attenuation_db,
    )
    filtered = apply_fir_filter(iir_out, kaiser_taps)

    return filtered, sections
