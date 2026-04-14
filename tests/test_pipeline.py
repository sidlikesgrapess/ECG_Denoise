from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ecg_denoise import denoise_ecg, load_record_segment


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


if __name__ == "__main__":
    unittest.main()
