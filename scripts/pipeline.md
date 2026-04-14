# ECG Denoising Pipeline

This document describes the full denoising chain used in this project.

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

## Stage Order Summary

1. Input segment (mV)
2. DC removal
3. Butterworth HPF (IIR)
4. Notch IIR (plus optional 2nd harmonic notch)
5. Kaiser low-pass FIR
6. Final denoised output + residual analysis
