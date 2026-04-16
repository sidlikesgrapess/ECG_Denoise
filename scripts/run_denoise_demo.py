from __future__ import annotations
from src.ecg_denoise.analysis import compute_noise_metrics, plot_ecg_signals

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ecg_denoise import DenoiseConfig, compute_noise_metrics, denoise_ecg, load_record_segment


DEFAULT_RECORDS = ["100", "101", "102", "200", "201", "202"]


def _default_dataset_dir() -> Path:
    return ROOT / "dataset" / "mit-bih-arrhythmia-database-1.0.0" / "mit-bih-arrhythmia-database-1.0.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run short-segment ECG denoising on MIT-BIH records with pure transfer-function filters."
    )
    parser.add_argument("--dataset-dir", type=Path, default=_default_dataset_dir())
    parser.add_argument("--records", nargs="+", default=DEFAULT_RECORDS)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--start-sec", type=float, default=60.0)
    parser.add_argument("--duration-sec", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--powerline-hz", type=float, default=60.0)
    parser.add_argument("--highpass-hz", type=float, default=0.5)
    parser.add_argument("--lowpass-hz", "--kaiser-cutoff-hz", dest="kaiser_cutoff_hz", type=float, default=40.0)
    parser.add_argument("--kaiser-transition-hz", type=float, default=8.0)
    parser.add_argument("--kaiser-atten-db", type=float, default=60.0)
    parser.add_argument("--notch-bandwidth-hz", type=float, default=2.0)
    parser.add_argument("--causal-only", action="store_true")
    parser.add_argument("--edge-pad-sec", type=float, default=1.5)
    parser.add_argument("--save-csv", action="store_true")
    return parser.parse_args()


def save_overlay_plot(
    output_path: Path,
    time_s: np.ndarray,
    raw_mv: np.ndarray,
    denoised_mv: np.ndarray,
    record_id: str,
    lead_name: str,
) -> None:
    residual = raw_mv - denoised_mv
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, constrained_layout=True)

    axes[0].plot(time_s, raw_mv, color="tab:blue", linewidth=0.9, alpha=0.8, label="Raw")
    axes[0].plot(time_s, denoised_mv, color="tab:green", linewidth=1.0, label="Denoised")
    axes[0].set_ylabel("Amplitude (mV)")
    axes[0].set_title(f"Record {record_id} ({lead_name})")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_s, residual, color="tab:red", linewidth=0.8)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Residual (mV)")
    axes[1].grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = DenoiseConfig(
        highpass_cutoff_hz=args.highpass_hz,
        powerline_hz=args.powerline_hz,
        notch_bandwidth_hz=args.notch_bandwidth_hz,
        kaiser_cutoff_hz=args.kaiser_cutoff_hz,
        kaiser_transition_hz=args.kaiser_transition_hz,
        kaiser_attenuation_db=args.kaiser_atten_db,
        zero_phase=not args.causal_only,
        edge_pad_seconds=max(0.0, args.edge_pad_sec),
    )

    summary_rows: list[dict[str, float | str]] = []

    for record_id in args.records:
        time_s, raw_mv, fs, header = load_record_segment(
            args.dataset_dir,
            record_id,
            channel=args.channel,
            start_sec=args.start_sec,
            duration_sec=args.duration_sec,
            to_millivolts=True,
        )
        denoised_mv, _sections = denoise_ecg(raw_mv, fs, config=cfg)
        metrics = compute_noise_metrics(raw_mv, denoised_mv, fs, powerline_hz=args.powerline_hz)
        print("\n=== METRICS ===")
        for k, v in metrics.items():
            print(f"{k}: {v:.3f}")

        from src.ecg_denoise.analysis import plot_ecg_signals
        plot_ecg_signals(raw_mv, denoised_mv, fs)
        
        lead_name = header.channels[args.channel].lead_name
        plot_path = args.output_dir / f"record_{record_id}_segment.png"
        save_overlay_plot(plot_path, time_s, raw_mv, denoised_mv, record_id, lead_name)

        if args.save_csv:
            csv_path = args.output_dir / f"record_{record_id}_segment.csv"
            data = np.column_stack((time_s, raw_mv, denoised_mv, raw_mv - denoised_mv))
            np.savetxt(
                csv_path,
                data,
                delimiter=",",
                header="time_s,raw_mv,denoised_mv,residual_mv",
                comments="",
            )

        summary_row: dict[str, float | str] = {
            "record": record_id,
            "lead": lead_name,
            "fs_hz": fs,
            "segment_seconds": args.duration_sec,
            **metrics,
        }
        summary_rows.append(summary_row)

        print(
            f"record={record_id} "
            f"baseline_db={metrics['baseline_reduction_db']:.2f} "
            f"line_db={metrics['powerline_reduction_db']:.2f} "
            f"hf_db={metrics['high_freq_reduction_db']:.2f}"
        )

    summary_path = args.output_dir / "summary_metrics.csv"
    fieldnames = [
        "record",
        "lead",
        "fs_hz",
        "segment_seconds",
        "baseline_reduction_db",
        "powerline_reduction_db",
        "high_freq_reduction_db",
        "residual_rms_mv",
        "residual_vs_raw_std_pct",
    ]

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print(f"Saved plots and metrics to: {args.output_dir}")


if __name__ == "__main__":
    main()
