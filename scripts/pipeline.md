# ECG Denoising Pipeline

This document describes the current denoising chain, outputs, and ownership notes used in this project.

## 1. Input Segment

- Load a short ECG segment from MIT-BIH.
- Convert digital ADC samples to millivolts (mV).
- See [../src/ecg_denoise/mitdb_io.py](../src/ecg_denoise/mitdb_io.py)

## 2. DC Removal

- Remove constant offset by subtracting the segment mean.
- See [../src/ecg_denoise/dc_removal.py](../src/ecg_denoise/dc_removal.py)

Formula:

$$
y[n] = x[n] - \mu_x
$$

Where:

- $x[n]$: input sample
- $\mu_x$: mean of the segment
- $y[n]$: DC-removed output

## 3. Butterworth High-Pass Filter (IIR)

- Apply a 2nd-order Butterworth high-pass section.
- Purpose: suppress baseline wander (very low-frequency drift).
- See [../src/ecg_denoise/butterworth_hpf.py](../src/ecg_denoise/butterworth_hpf.py)

This stage is implemented as a biquad with coefficients computed from:

- sampling frequency $f_s$
- cutoff frequency $f_c$
- Butterworth damping $Q = 1/\sqrt{2}$

## 4. Notch Filter (IIR)

- Apply a 2nd-order notch at powerline frequency (typically 60 Hz).
- Optionally apply a second notch at the 2nd harmonic (120 Hz) if below Nyquist.
- See [../src/ecg_denoise/notch_iir.py](../src/ecg_denoise/notch_iir.py)

Transfer function:

$$
H(z) = \frac{1 - 2\cos(\omega_0)z^{-1} + z^{-2}}
{1 - 2r\cos(\omega_0)z^{-1} + r^2 z^{-2}}
$$

Where:

- $\omega_0 = 2\pi f_0 / f_s$
- $r$ controls notch bandwidth

## 5. IIR Application Mode

IIR stages (HPF + Notch) can run in either mode:

- **Causal one-pass**: normal forward filtering
- **Zero-phase**: forward then reverse filtering (with reflection padding)

Reflection padding reduces short-window edge transients before filtering.

- See [../src/ecg_denoise/denoise.py](../src/ecg_denoise/denoise.py)
- See [../src/ecg_denoise/iir_core.py](../src/ecg_denoise/iir_core.py)

## 6. Kaiser FIR Low-Pass Filter

- Design low-pass FIR taps using Kaiser-windowed sinc.
- Apply FIR to suppress remaining high-frequency noise.
- See [../src/ecg_denoise/kaiser_fir.py](../src/ecg_denoise/kaiser_fir.py)

Design idea:

$$
h[n] = h_{ideal}[n] \cdot w_{kaiser}[n]
$$

Then normalize taps and convolve with the signal.

## 7. Output and Residual

- **Denoised signal**: final output after all stages
- **Residual signal**:

$$
r[n] = x_{raw}[n] - x_{denoised}[n]
$$

Residual shows what the pipeline removed.

- See [../src/ecg_denoise/analysis.py](../src/ecg_denoise/analysis.py)
- See [run_denoise_demo.py](run_denoise_demo.py)

## 8. Per-Record Output Images (Current)

For each record in the default set (`100, 101, 102, 200, 201, 202`), the main demo now saves two images:

1. `record_<id>_segment.png`
- Panel 1: raw vs denoised
- Panel 2: residual

2. `record_<id>_stage_residuals.png`
- Residual after Stage 1: Butterworth HPF
- Residual after Stage 2: Notch at powerline (default 60 Hz)
- Residual after Stage 3: Notch at 2nd harmonic (120 Hz)
- Residual after Stage 4: Kaiser LPF (final)

Outputs are written to `../outputs/` by [run_denoise_demo.py](run_denoise_demo.py).

## 9. Synthetic Noise Demo (Record 100)

The synthetic demo script injects three noise sources into record 100 and then denoises:

- Breathing (low-frequency baseline drift)
- Powerline interference (60 Hz + harmonic)
- Muscle artifact bursts (high-frequency)

Relevant files:

- [../src/ecg_denoise/synthetic_noise.py](../src/ecg_denoise/synthetic_noise.py)
- [run_record100_synthetic_demo.py](run_record100_synthetic_demo.py)

Synthetic demo outputs include:

- `record_100_synthetic_denoise.png`
- `record_100_synthetic_components.png`

## Stage Order Summary

1. Input segment (mV)
2. DC removal
3. Butterworth HPF (IIR)
4. Notch IIR (plus optional 2nd harmonic notch)
5. Kaiser low-pass FIR
6. Final denoised output + residual analysis

## Contributor Mapping (Code Links)

- Siddartha Guha: Butterworth HPF design in [../src/ecg_denoise/butterworth_hpf.py](../src/ecg_denoise/butterworth_hpf.py#L9)
- Rudra Ajit Singh: Notch filter design in [../src/ecg_denoise/notch_iir.py](../src/ecg_denoise/notch_iir.py#L9)
- Anuvi Pareek: Kaiser FIR LPF design/windowing/DTFT analysis in [../src/ecg_denoise/kaiser_fir.py](../src/ecg_denoise/kaiser_fir.py#L35)
- Riyan Dhiren Shah: dataset loading and segment extraction in [../src/ecg_denoise/mitdb_io.py](../src/ecg_denoise/mitdb_io.py#L151), synthetic noise addition in [../src/ecg_denoise/synthetic_noise.py](../src/ecg_denoise/synthetic_noise.py#L43)
- Madhura Bhattu: SNR computation and pole-zero stability plotting in [../src/ecg_denoise/analysis.py](../src/ecg_denoise/analysis.py#L43), [../src/ecg_denoise/analysis.py](../src/ecg_denoise/analysis.py#L98), and final pole-zero visualization in [run_denoise_demo.py](run_denoise_demo.py#L216)
