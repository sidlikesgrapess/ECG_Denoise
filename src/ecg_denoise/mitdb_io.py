from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ChannelInfo:
    file_name: str
    fmt: int
    gain: float
    adc_resolution_bits: int
    adc_zero: float
    first_value: int
    checksum: int
    block_size: int
    lead_name: str


@dataclass(frozen=True)
class RecordHeader:
    record_name: str
    num_channels: int
    fs: float
    num_samples: int
    channels: list[ChannelInfo]


def _safe_float(token: str, default: float) -> float:
    cleaned = token.strip()
    if "/" in cleaned:
        cleaned = cleaned.split("/", maxsplit=1)[0]
    if "(" in cleaned:
        cleaned = cleaned.split("(", maxsplit=1)[0]
    if cleaned == "":
        return default
    try:
        return float(cleaned)
    except ValueError:
        return default


def _safe_int(token: str, default: int) -> int:
    try:
        return int(float(token))
    except ValueError:
        return default


def parse_header(header_path: str | Path) -> RecordHeader:
    """Parse a MIT-BIH .hea file for metadata needed by the pipeline."""
    path = Path(header_path)
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Header file is empty: {path}")

    top = lines[0].split()
    if len(top) < 3:
        raise ValueError(f"Invalid header first line in {path}")

    record_name = top[0]
    num_channels = _safe_int(top[1], 2)
    fs = _safe_float(top[2], 360.0)
    num_samples = _safe_int(top[3], -1) if len(top) > 3 else -1

    channels: list[ChannelInfo] = []
    for channel_idx in range(num_channels):
        if channel_idx + 1 >= len(lines):
            break
        fields = lines[channel_idx + 1].split()
        if len(fields) < 2:
            continue

        gain = _safe_float(fields[2], 200.0) if len(fields) > 2 else 200.0
        if gain == 0:
            gain = 200.0

        channels.append(
            ChannelInfo(
                file_name=fields[0],
                fmt=_safe_int(fields[1], 212),
                gain=gain,
                adc_resolution_bits=_safe_int(fields[3], 11) if len(fields) > 3 else 11,
                adc_zero=_safe_float(fields[4], 0.0) if len(fields) > 4 else 0.0,
                first_value=_safe_int(fields[5], 0) if len(fields) > 5 else 0,
                checksum=_safe_int(fields[6], 0) if len(fields) > 6 else 0,
                block_size=_safe_int(fields[7], 0) if len(fields) > 7 else 0,
                lead_name=" ".join(fields[8:]) if len(fields) > 8 else f"ch{channel_idx}",
            )
        )

    if not channels:
        raise ValueError(f"No channel information found in {path}")

    return RecordHeader(
        record_name=record_name,
        num_channels=num_channels,
        fs=fs,
        num_samples=num_samples,
        channels=channels,
    )


def _decode_format_212(dat_path: str | Path) -> np.ndarray:
    """
    Decode MIT-BIH signal format 212.

    Each time sample contains 2 channels packed into 3 bytes:
    ch0 = low 8 bits from byte0 + high 4 bits from low nibble of byte1
    ch1 = low 8 bits from byte2 + high 4 bits from high nibble of byte1
    """
    raw = Path(dat_path).read_bytes()
    if len(raw) < 3:
        raise ValueError(f"Signal file too short: {dat_path}")

    leftover = len(raw) % 3
    if leftover:
        raw = raw[: len(raw) - leftover]

    packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)

    ch0 = (((packed[:, 1] & 0x0F).astype(np.int32)) << 8) | packed[:, 0].astype(np.int32)
    ch1 = (((packed[:, 1] >> 4).astype(np.int32)) << 8) | packed[:, 2].astype(np.int32)

    samples = np.stack([ch0, ch1], axis=1)
    samples[samples >= 2048] -= 4096
    return samples.astype(np.float64)


def available_records(dataset_dir: str | Path) -> list[str]:
    base = Path(dataset_dir)
    record_names = []
    for header_file in sorted(base.glob("*.hea")):
        rec = header_file.stem
        if (base / f"{rec}.dat").exists():
            record_names.append(rec)
    return record_names


def load_record_segment(
    dataset_dir: str | Path,
    record_id: str,
    *,
    channel: int = 0,
    start_sec: float = 0.0,
    duration_sec: float = 10.0,
    to_millivolts: bool = True,
) -> tuple[np.ndarray, np.ndarray, float, RecordHeader]:
    # Riyan Dhiren Shah: dataset loading and clean segment extraction.
    """Load a short segment from one MIT-BIH record."""
    if duration_sec <= 0:
        raise ValueError("duration_sec must be positive")

    base = Path(dataset_dir)
    header_path = base / f"{record_id}.hea"
    dat_path = base / f"{record_id}.dat"

    if not header_path.exists() or not dat_path.exists():
        raise FileNotFoundError(f"Record {record_id} not found in {base}")

    header = parse_header(header_path)
    if channel < 0 or channel >= len(header.channels):
        raise ValueError(f"Invalid channel index {channel} for record {record_id}")

    fmt = header.channels[channel].fmt
    if fmt != 212:
        raise NotImplementedError(f"Only MIT format 212 is supported, got {fmt}")

    all_samples = _decode_format_212(dat_path)
    if header.num_samples > 0:
        all_samples = all_samples[: header.num_samples, :]

    fs = header.fs
    start_idx = max(int(round(start_sec * fs)), 0)
    length = max(int(round(duration_sec * fs)), 1)
    end_idx = min(start_idx + length, all_samples.shape[0])
    if start_idx >= end_idx:
        raise ValueError("Requested segment is out of bounds")

    segment = all_samples[start_idx:end_idx, channel].copy()

    if to_millivolts:
        chan = header.channels[channel]
        segment = (segment - chan.adc_zero) / chan.gain

    time_s = np.arange(segment.size, dtype=np.float64) / fs + (start_idx / fs)
    return time_s, segment, fs, header
