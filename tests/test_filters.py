from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ecg_denoise.filters import apply_iir_filter, design_first_order_highpass, design_first_order_lowpass, design_notch


class FilterTests(unittest.TestCase):
    def test_highpass_rejects_dc(self) -> None:
        fs = 360.0
        x = np.ones(int(fs * 5), dtype=np.float64)
        b, a = design_first_order_highpass(fs, cutoff_hz=0.5)
        y = apply_iir_filter(x, b, a)
        steady_state = float(np.mean(np.abs(y[-int(fs) :])))
        self.assertLess(steady_state, 0.05)

    def test_notch_attenuates_powerline(self) -> None:
        fs = 360.0
        t = np.arange(int(fs * 3), dtype=np.float64) / fs
        x = np.sin(2.0 * np.pi * 60.0 * t)

        b, a = design_notch(fs, notch_hz=60.0, bandwidth_hz=2.0)
        y = apply_iir_filter(x, b, a)

        rms_in = float(np.sqrt(np.mean(x**2)))
        rms_out = float(np.sqrt(np.mean(y**2)))
        self.assertLess(rms_out, 0.35 * rms_in)

    def test_lowpass_preserves_low_frequency_shape(self) -> None:
        fs = 360.0
        t = np.arange(int(fs * 3), dtype=np.float64) / fs
        x = np.sin(2.0 * np.pi * 5.0 * t)

        b, a = design_first_order_lowpass(fs, cutoff_hz=40.0)
        y = apply_iir_filter(x, b, a)

        corr = float(np.corrcoef(x[50:], y[50:])[0, 1])
        self.assertGreater(corr, 0.9)


if __name__ == "__main__":
    unittest.main()
