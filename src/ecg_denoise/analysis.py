from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def _band_power(signal: np.ndarray, fs: float, low_hz: float, high_hz: float) -> float:
    x = np.asarray(signal, dtype=np.float64)
    if x.size < 2 or high_hz <= low_hz:
        return 0.0

    x = x - np.mean(x)
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    power = (np.abs(spectrum) ** 2) / x.size

    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(power[mask], freqs[mask]))


def _db_ratio(raw_value: float, denoised_value: float) -> float:
    eps = 1e-12
    return float(10.0 * np.log10((raw_value + eps) / (denoised_value + eps)))


def compute_noise_metrics(
    raw_signal: np.ndarray,
    denoised_signal: np.ndarray,
    fs: float,
    *,
    powerline_hz: float = 60.0,
) -> dict[str, float]:
    """Compute simple spectral metrics to quantify denoising effectiveness."""
    nyquist = fs / 2.0
    high_band_low = min(45.0, nyquist - 1.0)
    high_band_high = max(high_band_low + 0.5, nyquist - 0.1)
    line_low = max(powerline_hz - 1.0, 0.0)
    line_high = min(powerline_hz + 1.0, nyquist)

    baseline_raw = _band_power(raw_signal, fs, 0.0, 0.5)
    baseline_denoised = _band_power(denoised_signal, fs, 0.0, 0.5)

    hf_raw = _band_power(raw_signal, fs, high_band_low, high_band_high)
    hf_denoised = _band_power(denoised_signal, fs, high_band_low, high_band_high)

    line_raw = _band_power(raw_signal, fs, line_low, line_high)
    line_denoised = _band_power(denoised_signal, fs, line_low, line_high)

    residual = np.asarray(raw_signal, dtype=np.float64) - np.asarray(denoised_signal, dtype=np.float64)
    residual_rms = float(np.sqrt(np.mean(residual**2)))
    raw_std = float(np.std(raw_signal))
    residual_pct = float((100.0 * residual_rms / raw_std) if raw_std > 0 else 0.0)

    noise = raw_signal - denoised_signal

    signal_power = np.mean(denoised_signal ** 2)
    noise_power = np.mean(noise ** 2)

    snr_db = 10 * np.log10((signal_power + 1e-12) / (noise_power + 1e-12))
    return {
        "snr_db": snr_db,
        "baseline_reduction_db": _db_ratio(baseline_raw, baseline_denoised),
        "high_freq_reduction_db": _db_ratio(hf_raw, hf_denoised),
        "powerline_reduction_db": _db_ratio(line_raw, line_denoised),
        "residual_rms_mv": residual_rms,
        "residual_vs_raw_std_pct": residual_pct,
        }



def plot_ecg_signals(raw_signal, denoised_signal, fs):
    t = np.arange(len(raw_signal)) / fs

    plt.figure(figsize=(12, 5))

    plt.plot(t, raw_signal, label="Raw ECG", alpha=0.5)
    plt.plot(t, denoised_signal, label="Denoised ECG", linewidth=2)

    plt.title("ECG Signal Denoising")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()

    plt.show()

    import scipy.signal as signal

def plot_pole_zero(b, a):
    z, p, _ = signal.tf2zpk(b, a)

    plt.figure()
    plt.scatter(np.real(z), np.imag(z), label="Zeros", marker='o')
    plt.scatter(np.real(p), np.imag(p), label="Poles", marker='x')

    circle = plt.Circle((0, 0), 1, fill=False)
    plt.gca().add_artist(circle)

    plt.title("Pole-Zero Plot")
    plt.legend()
    plt.grid()
    plt.axis('equal')

    plt.show()