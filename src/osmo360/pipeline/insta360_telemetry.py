from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .manifest import ManifestError, ROOT

DEFAULT_TELEMETRY_PARSER = (
    ROOT / "work/tools/telemetry-parser-0.3.0/gyro2bb"
)

TELEMETRY_PARSER_VERSION = "0.3.0"
TELEMETRY_PARSER_COMMIT = "c0110c546e1fdc30014e6556ff49d81c4ca94821"
TELEMETRY_PARSER_SHA256 = "594559960f108cc36132ec6d69288e71750130477444e44693e9a6430883b482"


@dataclass(frozen=True)
class ImuSamples:
    timestamp_ns: np.ndarray
    source_timestamp_ns: np.ndarray
    angular_velocity: np.ndarray
    linear_acceleration: np.ndarray
    valid: np.ndarray
    provenance: dict[str, Any]


def telemetry_parser_path() -> Path:
    configured = os.environ.get("INSTAUMI_X5_TELEMETRY_PARSER")
    path = Path(configured) if configured else DEFAULT_TELEMETRY_PARSER
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ManifestError(
            "X5 telemetry parser is missing or not executable: " + str(path)
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != TELEMETRY_PARSER_SHA256:
        raise ManifestError(
            f"X5 telemetry parser hash mismatch: expected "
            f"{TELEMETRY_PARSER_SHA256}, found {digest}"
        )
    return path


def _read_blackbox_csv(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    metadata: dict[str, Any] = {}
    values: list[list[float]] = []
    in_samples = False
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            if row[0] == "loopIteration":
                expected = [
                    "loopIteration", "time", "gyroADC[0]", "gyroADC[1]",
                    "gyroADC[2]", "accSmooth[0]", "accSmooth[1]",
                    "accSmooth[2]",
                ]
                if row != expected:
                    raise ManifestError(f"unexpected X5 IMU CSV columns: {row}")
                in_samples = True
                continue
            if not in_samples:
                if len(row) >= 2:
                    raw = ",".join(row[1:])
                    try:
                        metadata[row[0]] = json.loads(raw)
                    except json.JSONDecodeError:
                        metadata[row[0]] = raw
                continue
            if len(row) != 8:
                raise ManifestError(f"malformed X5 IMU sample row: {row[:2]}")
            values.append([float(value) for value in row[1:]])
    if not values:
        raise ManifestError(f"X5 telemetry parser produced no IMU samples: {path}")
    return metadata, np.asarray(values, dtype=np.float64)


def extract_x5_imu(
    source: Path,
    scratch: Path,
    *,
    source_start_s: float,
    duration_s: float,
    expected_serial: str,
) -> ImuSamples:
    if source_start_s < 0 or duration_s <= 0:
        raise ValueError("invalid X5 IMU extraction window")
    source = source.resolve(strict=True)
    scratch.mkdir(parents=True, exist_ok=True)
    link = scratch / "source.insv"
    csv_path = link.with_suffix(link.suffix + ".csv")
    link.unlink(missing_ok=True)
    csv_path.unlink(missing_ok=True)
    link.symlink_to(source)
    parser = telemetry_parser_path()
    try:
        process = subprocess.run(
            [str(parser), str(link)], capture_output=True, text=True, timeout=120,
        )
        if process.returncode:
            raise ManifestError(
                f"X5 telemetry parser failed ({process.returncode}): "
                + process.stderr.strip()[:500]
            )
        if "Detected camera: Insta360 Insta360 X5" not in process.stdout:
            raise ManifestError(
                "telemetry parser did not identify an Insta360 X5: "
                + process.stdout.strip()[:300]
            )
        metadata, samples = _read_blackbox_csv(csv_path)
    finally:
        link.unlink(missing_ok=True)
        csv_path.unlink(missing_ok=True)

    serial = str(metadata.get("serial_number", ""))
    if serial != expected_serial:
        raise ManifestError(
            f"X5 IMU serial mismatch: expected {expected_serial}, found {serial}"
        )
    relative_time_s = samples[:, 0] / 1_000_000.0
    end_s = source_start_s + duration_s
    keep = (relative_time_s >= source_start_s) & (relative_time_s < end_s)
    if not np.any(keep):
        raise ManifestError("X5 IMU has no samples in the aligned video window")
    samples = samples[keep]
    relative_time_s = relative_time_s[keep]

    # gyro2bb writes telemetry-parser's normalized vectors in Betaflight column
    # order: [-gz, gy, gx] and [-az, ay, ax] * 2048. Undo that output-only
    # permutation before converting to canonical SI units.
    blackbox_gyro = samples[:, 1:4]
    angular_velocity_deg_s = np.column_stack((
        blackbox_gyro[:, 2], blackbox_gyro[:, 1], -blackbox_gyro[:, 0],
    ))
    angular_velocity = np.deg2rad(angular_velocity_deg_s)
    blackbox_acceleration = samples[:, 4:7]
    linear_acceleration = np.column_stack((
        blackbox_acceleration[:, 2],
        blackbox_acceleration[:, 1],
        -blackbox_acceleration[:, 0],
    )) / 2048.0
    timestamp_ns = np.rint(
        (relative_time_s - source_start_s) * 1_000_000_000.0
    ).astype(np.int64)

    try:
        first_frame_us = int(metadata["first_frame_timestamp"])
    except (KeyError, TypeError, ValueError) as error:
        raise ManifestError("X5 telemetry is missing first_frame_timestamp") from error
    gyro_offset_us = float(metadata.get("gyro_timestamp", 0.0)) * 1000.0
    raw_time_us = samples[:, 0] + first_frame_us + gyro_offset_us
    source_timestamp_ns = np.rint(raw_time_us * 1000.0).astype(np.int64)
    finite = (
        np.isfinite(angular_velocity).all(axis=1)
        & np.isfinite(linear_acceleration).all(axis=1)
    )
    valid = finite.astype(np.uint8)
    if not np.all(np.diff(timestamp_ns) >= 0):
        raise ManifestError("aligned X5 IMU timestamps are not monotonic")
    if not np.all(np.diff(source_timestamp_ns) >= 0):
        raise ManifestError("source X5 IMU timestamps are not monotonic")
    if not np.any(valid):
        raise ManifestError("all X5 IMU samples are invalid")

    return ImuSamples(
        timestamp_ns=timestamp_ns,
        source_timestamp_ns=source_timestamp_ns,
        angular_velocity=angular_velocity,
        linear_acceleration=linear_acceleration,
        valid=valid,
        provenance={
            "parser": "telemetry-parser/gyro2bb",
            "parser_version": TELEMETRY_PARSER_VERSION,
            "parser_commit": TELEMETRY_PARSER_COMMIT,
            "parser_sha256": TELEMETRY_PARSER_SHA256,
            "normalization": {
                "axis_mapping": "telemetry-parser X5 normalized; gyro2bb Betaflight columns inverted",
                "source_gyro_unit": "deg/s",
                "source_acceleration_encoding": "m/s^2 * 2048",
            },
            "camera_model": metadata.get("camera_type"),
            "camera_serial": serial,
            "firmware_version": metadata.get("fw_version"),
            "first_frame_timestamp_us": first_frame_us,
            "gyro_timestamp_ms": metadata.get("gyro_timestamp"),
            "gyro_config": metadata.get("gyro_cfg_info"),
            "source": str(source),
        },
    )
