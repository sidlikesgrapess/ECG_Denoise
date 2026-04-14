from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ecg_denoise.butterworth_hpf import design_butterworth_highpass
from ecg_denoise.dc_removal import remove_dc_mean
from ecg_denoise.iir_core import apply_iir_filter
from ecg_denoise.kaiser_fir import apply_fir_filter, design_kaiser_lowpass_fir
from ecg_denoise.notch_iir import design_notch_iir


class FilterTests(unittest.TestCase):
    @staticmethod
    def _tone_amplitude(signal: np.ndarray, fs: float, freq_hz: float) -> float:
        t = np.arange(signal.size, dtype=np.float64) / fs
        s = np.sin(2.0 * np.pi * freq_hz * t)
        c = np.cos(2.0 * np.pi * freq_hz * t)
        a = (2.0 / signal.size) * float(np.dot(signal, s))
        b = (2.0 / signal.size) * float(np.dot(signal, c))
        return float(np.sqrt((a * a) + (b * b)))

    def test_dc_removal_zero_mean(self) -> None:
        fs = 360.0
        t = np.arange(int(fs * 2), dtype=np.float64) / fs
        x = 0.8 + 0.2 * np.sin(2.0 * np.pi * 5.0 * t)
        y = remove_dc_mean(x)
        self.assertLess(abs(float(np.mean(y))), 1e-10)

    def test_butterworth_highpass_rejects_dc(self) -> None:
        fs = 360.0
        x = np.ones(int(fs * 5), dtype=np.float64)
        b, a = design_butterworth_highpass(fs, cutoff_hz=0.5)
        y = apply_iir_filter(x, b, a)
        steady_state = float(np.mean(np.abs(y[-int(fs) :])))
        self.assertLess(steady_state, 0.03)

    def test_notch_attenuates_powerline(self) -> None:
        fs = 360.0
        t = np.arange(int(fs * 3), dtype=np.float64) / fs
        x = np.sin(2.0 * np.pi * 60.0 * t)

        b, a = design_notch_iir(fs, notch_hz=60.0, bandwidth_hz=2.0)
        y = apply_iir_filter(x, b, a)

        rms_in = float(np.sqrt(np.mean(x**2)))
        rms_out = float(np.sqrt(np.mean(y**2)))
        self.assertLess(rms_out, 0.35 * rms_in)

    def test_kaiser_lowpass_suppresses_high_frequency(self) -> None:
        fs = 360.0
        t = np.arange(int(fs * 4), dtype=np.float64) / fs
        x = np.sin(2.0 * np.pi * 5.0 * t) + 0.7 * np.sin(2.0 * np.pi * 90.0 * t)

        taps = design_kaiser_lowpass_fir(fs, cutoff_hz=40.0, transition_hz=8.0, attenuation_db=60.0)
        y = apply_fir_filter(x, taps)

        amp_5_in = self._tone_amplitude(x, fs, 5.0)
        amp_5_out = self._tone_amplitude(y, fs, 5.0)
        amp_90_in = self._tone_amplitude(x, fs, 90.0)
        amp_90_out = self._tone_amplitude(y, fs, 90.0)

        self.assertGreater(amp_5_out, 0.85 * amp_5_in)
        self.assertLess(amp_90_out, 0.2 * amp_90_in)


if __name__ == "__main__":
    unittest.main()
