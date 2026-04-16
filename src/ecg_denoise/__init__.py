"""ECG denoising toolkit for short MIT-BIH experiments."""

from .analysis import compute_noise_metrics
from .butterworth_hpf import design_butterworth_highpass
from .dc_removal import remove_dc_mean
from .denoise import DenoiseConfig, denoise_ecg
from .kaiser_fir import apply_fir_filter, design_kaiser_lowpass_fir
from .mitdb_io import ChannelInfo, RecordHeader, available_records, load_record_segment, parse_header
from .notch_iir import design_notch_iir
from .synthetic_noise import SyntheticNoiseConfig, add_synthetic_noise

__all__ = [
    "ChannelInfo",
    "RecordHeader",
    "DenoiseConfig",
    "available_records",
    "compute_noise_metrics",
    "design_butterworth_highpass",
    "design_kaiser_lowpass_fir",
    "design_notch_iir",
    "denoise_ecg",
    "load_record_segment",
    "parse_header",
    "remove_dc_mean",
    "apply_fir_filter",
    "SyntheticNoiseConfig",
    "add_synthetic_noise",
]
