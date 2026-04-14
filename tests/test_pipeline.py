from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ecg_denoise import DenoiseConfig, denoise_ecg, load_record_segment


class PipelineTests(unittest.TestCase):
    def test_short_segment_pipeline_runs(self) -> None:
        dataset_dir = ROOT / "dataset" / "mit-bih-arrhythmia-database-1.0.0" / "mit-bih-arrhythmia-database-1.0.0"
        t, raw_mv, fs, _header = load_record_segment(
            dataset_dir,
            "100",
            channel=0,
            start_sec=60.0,
            duration_sec=5.0,
            to_millivolts=True,
        )
        denoised_mv, sections = denoise_ecg(raw_mv, fs)

        self.assertEqual(raw_mv.shape, denoised_mv.shape)
        self.assertEqual(raw_mv.shape, t.shape)
        self.assertGreater(len(sections), 0)
        self.assertTrue(np.isfinite(denoised_mv).all())

    def test_pipeline_runs_in_both_application_modes(self) -> None:
        dataset_dir = ROOT / "dataset" / "mit-bih-arrhythmia-database-1.0.0" / "mit-bih-arrhythmia-database-1.0.0"
        _t, raw_mv, fs, _header = load_record_segment(
            dataset_dir,
            "100",
            channel=0,
            start_sec=60.0,
            duration_sec=5.0,
            to_millivolts=True,
        )

        denoised_zero_phase, _ = denoise_ecg(raw_mv, fs, config=DenoiseConfig(zero_phase=True, edge_pad_seconds=1.0))
        denoised_causal, _ = denoise_ecg(raw_mv, fs, config=DenoiseConfig(zero_phase=False))

        self.assertEqual(raw_mv.shape, denoised_zero_phase.shape)
        self.assertEqual(raw_mv.shape, denoised_causal.shape)
        self.assertTrue(np.isfinite(denoised_zero_phase).all())
        self.assertTrue(np.isfinite(denoised_causal).all())


if __name__ == "__main__":
    unittest.main()
