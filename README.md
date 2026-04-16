# MIT-BIH ECG Denoising Project (Pure Signals and Systems)

This project implements ECG denoising on MIT-BIH records using hand-built transfer-function filters and difference equations.

The implementation uses short test windows (default 10 seconds) from six 30-minute records:

- 100
- 101
- 102
- 200
- 201
- 202

No external denoising library is used.

## Implemented Features

- MIT-BIH header parser for `.hea` files
- MIT format 212 waveform decoder for `.dat` files
- Physical unit conversion to mV using ADC baseline and gain from headers
- Pure transfer-function denoising pipeline:
  - DC removal (mean subtraction)
  - Butterworth high-pass filter (IIR)
  - Powerline interference removal (IIR notch, optional 2nd harmonic)
  - Kaiser-window FIR low-pass smoothing
- Optional zero-phase application of the IIR cascade (forward-backward)
- Reflection edge padding to reduce boundary transients on short windows
- Quantitative spectral noise metrics
- Plot generation and CSV export for short segments
- Automated tests for filters and end-to-end short-segment pipeline

## What Each File Does

### Core package (`src/ecg_denoise`)

- `mitdb_io.py`: Reads MIT-BIH headers and decodes waveform samples, then loads a selected segment.
- `dc_removal.py`: Removes DC offset by subtracting segment mean.
- `butterworth_hpf.py`: Designs Butterworth high-pass IIR section for baseline-wander suppression.
- `notch_iir.py`: Designs notch IIR section used for 60 Hz and optional 120 Hz suppression.
- `kaiser_fir.py`: Designs and applies Kaiser-window FIR low-pass smoothing.
- `iir_core.py`: Applies one IIR section and cascades multiple sections via difference equations.
- `denoise.py`: Builds the full denoising chain (DC -> HPF -> notch(es) -> Kaiser FIR).
- `analysis.py`: Computes metrics (including SNR) and contains plotting utilities.
- `synthetic_noise.py`: Adds synthetic breathing, powerline, and muscle noise for controlled demos.
- `__init__.py`: Public exports for the package.

### Scripts (`scripts`)

- `run_denoise_demo.py`: Main run for the six default records; saves per-record comparison and stage residual plots plus summary metrics.
- `run_record100_synthetic_demo.py`: Record 100 synthetic-noise injection and denoising demo.
- `pipeline.md`: Pipeline notes and contributor mapping.

### Tests (`tests`)

- `test_filters.py`: Unit-level checks for filter design and behavior.
- `test_pipeline.py`: End-to-end checks of the short-segment denoising flow.

## Setup

1. Activate your existing venv.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Short-Segment Denoising

Default run (6 records, 10-second segment from 60 seconds):

```bash
python scripts/run_denoise_demo.py
```

Example with CSV export:

```bash
python scripts/run_denoise_demo.py --save-csv
```

Run in strict one-pass causal mode (same filters, no reverse pass):

```bash
python scripts/run_denoise_demo.py --causal-only --save-csv
```

Change edge padding used for short-segment stability:

```bash
python scripts/run_denoise_demo.py --edge-pad-sec 2.0
```

Adjust Kaiser FIR parameters:

```bash
python scripts/run_denoise_demo.py --kaiser-cutoff-hz 35 --kaiser-transition-hz 6 --kaiser-atten-db 60
```

Results are saved in `outputs/`:

- `record_<id>_segment.png`
- `record_<id>_stage_residuals.png`
- `record_<id>_segment.csv` (if `--save-csv`)
- `summary_metrics.csv`

## Run Tests

```bash
python -m unittest discover -s tests -v
```

## Notes For GitHub Upload

- A `requirements.txt` file is included.
- `.gitignore` excludes:
  - local venv
  - generated outputs
  - local MIT-BIH dataset mirror
