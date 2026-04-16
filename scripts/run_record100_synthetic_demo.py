from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ecg_denoise import (
    DenoiseConfig,
    SyntheticNoiseConfig,
    add_synthetic_noise,
    compute_noise_metrics,
    denoise_ecg,
    load_record_segment,
)


def _default_dataset_dir() -> Path:
    return ROOT / "dataset" / "mit-bih-arrhythmia-database-1.0.0" / "mit-bih-arrhythmia-database-1.0.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record-100 synthetic-noise demo: breathing + powerline + muscle artifacts followed by denoising."
    )
    parser.add_argument("--dataset-dir", type=Path, default=_default_dataset_dir())
    parser.add_argument("--record", type=str, default="100")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--start-sec", type=float, default=60.0)
    parser.add_argument("--duration-sec", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--powerline-hz", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def save_synthetic_demo_plot(
    output_path: Path,
    time_s: np.ndarray,
    clean_mv: np.ndarray,
    noisy_mv: np.ndarray,
    denoised_mv: np.ndarray,
    record_id: str,
    lead_name: str,
) -> None:
    residual = noisy_mv - denoised_mv
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)

    axes[0].plot(time_s, clean_mv, color="tab:blue", linewidth=0.9, alpha=0.9, label="Clean")
    axes[0].plot(time_s, noisy_mv, color="tab:orange", linewidth=0.8, alpha=0.75, label="Synthetic noisy")
    axes[0].set_ylabel("Amplitude (mV)")
    axes[0].set_title(f"Record {record_id} ({lead_name}) - Synthetic Noise Injection")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_s, noisy_mv, color="tab:orange", linewidth=0.8, alpha=0.7, label="Synthetic noisy")
    axes[1].plot(time_s, denoised_mv, color="tab:green", linewidth=1.0, label="Denoised")
    axes[1].set_ylabel("Amplitude (mV)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_s, residual, color="tab:red", linewidth=0.8, label="Residual (noisy - denoised)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Residual (mV)")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def save_noise_components_plot(
    output_path: Path,
    time_s: np.ndarray,
    components: dict[str, np.ndarray],
    record_id: str,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    component_order = ["breathing", "powerline", "muscle", "total"]
    colors = ["tab:purple", "tab:orange", "tab:brown", "tab:red"]

    for idx, name in enumerate(component_order):
        axes[idx].plot(time_s, components[name], color=colors[idx], linewidth=0.8)
        axes[idx].set_ylabel("mV")
        axes[idx].set_title(name.capitalize())
        axes[idx].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Record {record_id} - Synthetic Noise Components")
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    time_s, clean_mv, fs, header = load_record_segment(
        args.dataset_dir,
        args.record,
        channel=args.channel,
        start_sec=args.start_sec,
        duration_sec=args.duration_sec,
        to_millivolts=True,
    )

    noise_cfg = SyntheticNoiseConfig(powerline_hz=args.powerline_hz)
    noisy_mv, components = add_synthetic_noise(clean_mv, fs, config=noise_cfg, seed=args.seed)

    denoise_cfg = DenoiseConfig(powerline_hz=args.powerline_hz)
    denoised_mv, _ = denoise_ecg(noisy_mv, fs, config=denoise_cfg)

    metrics = compute_noise_metrics(noisy_mv, denoised_mv, fs, powerline_hz=args.powerline_hz)
    print("\n=== SYNTHETIC NOISE DENOISE METRICS ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.3f}")

    lead_name = header.channels[args.channel].lead_name
    plot_path = args.output_dir / f"record_{args.record}_synthetic_denoise.png"
    save_synthetic_demo_plot(plot_path, time_s, clean_mv, noisy_mv, denoised_mv, args.record, lead_name)

    component_path = args.output_dir / f"record_{args.record}_synthetic_components.png"
    save_noise_components_plot(component_path, time_s, components, args.record)

    print(f"Saved synthetic-noise demo plots to: {args.output_dir}")


if __name__ == "__main__":
    main()