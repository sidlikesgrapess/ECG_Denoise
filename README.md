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
  - Baseline wander removal (1st-order high-pass)
  - Powerline interference removal (2nd-order notch, optional 2nd harmonic)
  - High-frequency smoothing (1st-order low-pass)
- Quantitative spectral noise metrics
- Plot generation and CSV export for short segments
- Automated tests for filters and end-to-end short-segment pipeline

## Filter Transfer Functions

The filters are implemented directly from transfer functions.

### High-pass (baseline wander removal)

Analog prototype:

H(s) = s / (s + wc)

After bilinear transform (s = 2fs(1-z^-1)/(1+z^-1)), the digital coefficients become:

- b = [2fs/(2fs+wc), -2fs/(2fs+wc)]
- a = [1, (wc-2fs)/(2fs+wc)]

Difference equation:

y[n] = b0 x[n] + b1 x[n-1] - a1 y[n-1]

### Low-pass (muscle noise suppression)

Analog prototype:

H(s) = wc / (s + wc)

Digital coefficients:

- b = [wc/(2fs+wc), wc/(2fs+wc)]
- a = [1, (wc-2fs)/(2fs+wc)]

### Notch (powerline)

Digital notch section:

H(z) = (1 - 2cos(w0)z^-1 + z^-2) / (1 - 2r cos(w0)z^-1 + r^2 z^-2)

where w0 = 2pi f0 / fs.

## Project Structure

```
src/ecg_denoise/
  analysis.py
  denoise.py
  filters.py
  mitdb_io.py
scripts/
  run_denoise_demo.py
tests/
  test_filters.py
  test_pipeline.py
requirements.txt
README.md
```

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

Results are saved in `outputs/`:

- `record_<id>_segment.png`
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
