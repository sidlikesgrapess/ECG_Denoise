from __future__ import annotations

import argparse
import csv
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

from ecg_denoise import DenoiseConfig, compute_noise_metrics, denoise_ecg, load_record_segment
from ecg_denoise.dc_removal import remove_dc_mean
from ecg_denoise.denoise import build_filter_sections
from ecg_denoise.iir_core import FilterSection, cascade_iir
from ecg_denoise.kaiser_fir import apply_fir_filter, design_kaiser_lowpass_fir


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
    comparison_label: str | None = None,
) -> None:
    residual = raw_mv - denoised_mv
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, constrained_layout=True)

    axes[0].plot(time_s, raw_mv, color="tab:blue", linewidth=0.9, alpha=0.8, label="Raw")
    axes[0].plot(time_s, denoised_mv, color="tab:green", linewidth=1.0, label="Denoised")
    axes[0].set_ylabel("Amplitude (mV)")
    if comparison_label:
        axes[0].set_title(f"Record {record_id} ({lead_name}) - {comparison_label}")
    else:
        axes[0].set_title(f"Record {record_id} ({lead_name})")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_s, residual, color="tab:red", linewidth=0.8)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Residual (mV)")
    axes[1].grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _reflect_pad(signal: np.ndarray, pad_len: int) -> tuple[np.ndarray, int]:
    x = np.asarray(signal, dtype=np.float64)
    if x.size <= 1 or pad_len <= 0:
        return x, 0

    used_pad = min(pad_len, x.size - 1)
    return np.pad(x, (used_pad, used_pad), mode="reflect"), used_pad


def _apply_iir_chain(
    signal: np.ndarray,
    sections: list[FilterSection],
    fs: float,
    *,
    zero_phase: bool,
    edge_pad_seconds: float,
) -> np.ndarray:
    if not sections:
        return np.asarray(signal, dtype=np.float64)

    x = np.asarray(signal, dtype=np.float64)
    if zero_phase:
        pad_len = max(0, int(round(edge_pad_seconds * fs)))
        padded, used_pad = _reflect_pad(x, pad_len)
        forward = cascade_iir(padded, sections)
        backward = cascade_iir(forward[::-1], sections)
        y = backward[::-1]
        if used_pad > 0:
            y = y[used_pad:-used_pad]
        return y

    return cascade_iir(x, sections)


def compute_stage_outputs(raw_mv: np.ndarray, fs: float, cfg: DenoiseConfig) -> dict[str, np.ndarray]:
    x = remove_dc_mean(np.asarray(raw_mv, dtype=np.float64))
    sections = build_filter_sections(fs, cfg)

    stage1 = _apply_iir_chain(
        x,
        sections[:1],
        fs,
        zero_phase=cfg.zero_phase,
        edge_pad_seconds=cfg.edge_pad_seconds,
    )
    stage2 = _apply_iir_chain(
        x,
        sections[:2],
        fs,
        zero_phase=cfg.zero_phase,
        edge_pad_seconds=cfg.edge_pad_seconds,
    )
    stage3 = _apply_iir_chain(
        x,
        sections,
        fs,
        zero_phase=cfg.zero_phase,
        edge_pad_seconds=cfg.edge_pad_seconds,
    )

    taps = design_kaiser_lowpass_fir(
        fs,
        cfg.kaiser_cutoff_hz,
        transition_hz=cfg.kaiser_transition_hz,
        attenuation_db=cfg.kaiser_attenuation_db,
    )
    stage4 = apply_fir_filter(stage3, taps)

    return {
        "stage1": stage1,
        "stage2": stage2,
        "stage3": stage3,
        "stage4": stage4,
    }


def save_stage_residuals_plot(
    output_path: Path,
    time_s: np.ndarray,
    raw_mv: np.ndarray,
    stage_outputs: dict[str, np.ndarray],
    record_id: str,
    lead_name: str,
    powerline_hz: float,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True, constrained_layout=True)

    stage_labels = [
        "Residual after Stage 1: Butterworth HPF",
        f"Residual after Stage 2: Notch {powerline_hz:g} Hz",
        f"Residual after Stage 3: Notch {2.0 * powerline_hz:g} Hz",
        "Residual after Stage 4: Kaiser LPF (final)",
    ]
    stage_keys = ["stage1", "stage2", "stage3", "stage4"]

    for idx, key in enumerate(stage_keys):
        residual = np.asarray(raw_mv, dtype=np.float64) - np.asarray(stage_outputs[key], dtype=np.float64)
        axes[idx].plot(time_s, residual, color="tab:red", linewidth=0.8)
        axes[idx].set_ylabel("Residual (mV)")
        axes[idx].set_title(stage_labels[idx])
        axes[idx].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Record {record_id} ({lead_name}) - Stage Residual Progression")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_snr_plot(output_path: Path, summary_rows: list[dict[str, float | str]]) -> None:
    records = [str(row["record"]) for row in summary_rows]
    snr_values = [float(row["snr_db"]) for row in summary_rows]

    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    colors = ["tab:green" if value >= 0.0 else "tab:red" for value in snr_values]
    ax.bar(records, snr_values, color=colors, alpha=0.85)
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_title("SNR by Record")
    ax.set_xlabel("Record ID")
    ax.set_ylabel("SNR (dB)")
    ax.grid(True, axis="y", alpha=0.3)

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_pole_zero_stability_plot(
    output_path: Path,
    sections: list[tuple[np.ndarray, np.ndarray]],
    powerline_hz: float,
) -> None:
    # Madhura Bhattu: final visualization for pole-zero stability.
    fig, ax = plt.subplots(figsize=(6.5, 6.5), constrained_layout=True)

    theta = np.linspace(0.0, 2.0 * np.pi, 800)
    ax.plot(np.cos(theta), np.sin(theta), linestyle="--", linewidth=1.0, color="black", alpha=0.75, label="Unit circle")

    max_radius = 0.0
    all_stable = True
    section_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for idx, (b, a) in enumerate(sections):
        zeros = np.roots(b) if len(b) > 1 else np.array([], dtype=np.complex128)
        poles = np.roots(a) if len(a) > 1 else np.array([], dtype=np.complex128)
        color = section_colors[idx % len(section_colors)]
        if idx == 0:
            section_id = "HPF"
        elif idx == 1:
            section_id = f"N{powerline_hz:g}"
        elif idx == 2:
            section_id = f"N{2.0 * powerline_hz:g}"
        else:
            section_id = f"S{idx + 1}"

        if zeros.size > 0:
            ax.scatter(
                np.real(zeros),
                np.imag(zeros),
                marker="o",
                facecolors="none",
                edgecolors=color,
                s=70,
                linewidths=1.4,
                label=f"{section_id} z",
            )

        if poles.size > 0:
            ax.scatter(
                np.real(poles),
                np.imag(poles),
                marker="x",
                color=color,
                s=70,
                linewidths=1.6,
                label=f"{section_id} p",
            )

            pole_radii = np.abs(poles)
            max_radius = max(max_radius, float(np.max(pole_radii)))
            all_stable = all_stable and bool(np.all(pole_radii < 1.0))

    plot_radius = max(1.1, max_radius + 0.15)
    ax.set_xlim(-plot_radius, plot_radius)
    ax.set_ylim(-plot_radius, plot_radius)
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.axvline(0.0, color="gray", linewidth=0.8)
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("Real")
    ax.set_ylabel("Imag")

    status = "Stable" if all_stable else "Unstable"
    ax.set_title(f"Pole-Zero Stability ({status}, max |p| = {max_radius:.4f})")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
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
    reference_sections: list[tuple[np.ndarray, np.ndarray]] | None = None

    for record_id in args.records:
        time_s, raw_mv, fs, header = load_record_segment(
            args.dataset_dir,
            record_id,
            channel=args.channel,
            start_sec=args.start_sec,
            duration_sec=args.duration_sec,
            to_millivolts=True,
        )
        denoised_mv, sections = denoise_ecg(raw_mv, fs, config=cfg)
        if reference_sections is None:
            reference_sections = sections
        metrics = compute_noise_metrics(raw_mv, denoised_mv, fs, powerline_hz=args.powerline_hz)
        print("\n=== METRICS ===")
        for k, v in metrics.items():
            print(f"{k}: {v:.3f}")

        lead_name = header.channels[args.channel].lead_name
        plot_path = args.output_dir / f"record_{record_id}_segment.png"
        save_overlay_plot(
            plot_path,
            time_s,
            raw_mv,
            denoised_mv,
            record_id,
            lead_name,
            comparison_label="All stages",
        )

        stage_outputs = compute_stage_outputs(raw_mv, fs, cfg)
        stage_residuals_path = args.output_dir / f"record_{record_id}_stage_residuals.png"
        save_stage_residuals_plot(
            stage_residuals_path,
            time_s,
            raw_mv,
            stage_outputs,
            record_id,
            lead_name,
            args.powerline_hz,
        )

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
        "snr_db",
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

    snr_plot_path = args.output_dir / "snr_by_record.png"
    save_snr_plot(snr_plot_path, summary_rows)

    if reference_sections:
        pz_plot_path = args.output_dir / "pole_zero_stability.png"
        save_pole_zero_stability_plot(pz_plot_path, reference_sections, args.powerline_hz)

    print(f"Saved plots and metrics to: {args.output_dir}")


if __name__ == "__main__":
    main()
