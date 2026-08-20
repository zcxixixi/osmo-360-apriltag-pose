#!/usr/bin/env python3
"""Fuse sparse AprilGrid XYZ with dense DJI IMU attitude for visualization."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.signal import medfilt, savgol_filter

from imu_relative_pose import quat_to_matrix, rotation_to_rpy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("visual_pose_csv", type=Path)
    parser.add_argument("imu_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--fps", type=float, default=60000 / 1001)
    return parser.parse_args()


def smooth_visual_positions(positions: np.ndarray) -> tuple[np.ndarray, int]:
    """Suppress isolated PnP position spikes, then smooth the visual knots."""
    median = np.column_stack([medfilt(positions[:, axis], kernel_size=5) for axis in range(3)])
    residual = np.linalg.norm(positions - median, axis=1)
    outliers = residual > 0.12
    cleaned = positions.copy()
    cleaned[outliers] = median[outliers]
    window = min(9, len(cleaned) if len(cleaned) % 2 else len(cleaned) - 1)
    if window >= 5:
        cleaned = np.column_stack(
            [savgol_filter(cleaned[:, axis], window, 2, mode="interp") for axis in range(3)]
        )
    return cleaned, int(outliers.sum())


def main() -> int:
    args = parse_args()
    if args.fps <= 0:
        raise SystemExit("fps must be positive")
    with args.visual_pose_csv.open(encoding="utf-8", newline="") as handle:
        visual_rows = [
            row for row in csv.DictReader(handle)
            if row.get("quality_status") == "valid" and row.get("camera_x_m")
        ]
    with args.imu_csv.open(encoding="utf-8", newline="") as handle:
        imu_rows = list(csv.DictReader(handle))
    if len(visual_rows) < 4 or not imu_rows:
        raise SystemExit("not enough visual or IMU samples")

    visual_times = np.asarray([float(row["timestamp"]) for row in visual_rows])
    visual_positions = np.asarray(
        [[float(row[f"camera_{axis}_m"]) for axis in "xyz"] for row in visual_rows]
    )
    visual_positions, outlier_count = smooth_visual_positions(visual_positions)
    frame_times = np.arange(len(imu_rows), dtype=np.float64) / args.fps
    interpolated = np.column_stack(
        [
            np.interp(
                frame_times, visual_times, visual_positions[:, axis],
                left=visual_positions[0, axis], right=visual_positions[-1, axis],
            )
            for axis in range(3)
        ]
    )
    interpolated -= interpolated[0]

    imu_rotations = [
        quat_to_matrix(
            np.asarray([float(row[key]) for key in ("qw", "qx", "qy", "qz")])
        )
        for row in imu_rows
    ]
    start_to_world = imu_rotations[0].T
    fields = (
        "frame", "timestamp", "camera_x_m", "camera_y_m", "camera_z_m",
        "roll_deg", "pitch_deg", "yaw_deg", "quality_status", "fusion_status",
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (timestamp, position, rotation) in enumerate(
            zip(frame_times, interpolated, imu_rotations)
        ):
            relative_rotation = start_to_world @ rotation
            roll, pitch, yaw = rotation_to_rpy(relative_rotation)
            nearest_visual_age = float(np.min(np.abs(visual_times - timestamp)))
            status = "visual_anchor" if nearest_visual_age <= 0.075 else "interpolated"
            writer.writerow(
                {
                    "frame": index,
                    "timestamp": f"{timestamp:.6f}",
                    "camera_x_m": f"{position[0]:.7f}",
                    "camera_y_m": f"{position[1]:.7f}",
                    "camera_z_m": f"{position[2]:.7f}",
                    "roll_deg": f"{roll:.6f}",
                    "pitch_deg": f"{pitch:.6f}",
                    "yaw_deg": f"{yaw:.6f}",
                    "quality_status": "valid",
                    "fusion_status": status,
                }
            )
    print(f"{args.output_csv.resolve()} visual={len(visual_rows)} outliers={outlier_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
