#!/usr/bin/env python3
"""Convert PanoForge per-frame DJI IMU quaternions to start-relative pose CSV."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("imu_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--fps", type=float, default=60000 / 1001)
    return parser.parse_args()


def quat_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = quaternion / np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_to_rpy(rotation: np.ndarray) -> np.ndarray:
    sy = math.hypot(rotation[0, 0], rotation[1, 0])
    if sy >= 1e-8:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])


def main() -> int:
    args = parse_args()
    if args.fps <= 0:
        raise SystemExit("fps must be positive")
    with args.imu_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("IMU CSV is empty")
    rotations = [
        quat_to_matrix(np.array([float(row[key]) for key in ("qw", "qx", "qy", "qz")]))
        for row in rows
    ]
    start_to_world = rotations[0].T
    fields = (
        "frame", "timestamp", "camera_x_m", "camera_y_m", "camera_z_m",
        "roll_deg", "pitch_deg", "yaw_deg", "quality_status",
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, rotation in enumerate(rotations):
            relative = start_to_world @ rotation
            roll, pitch, yaw = rotation_to_rpy(relative)
            writer.writerow(
                {
                    "frame": index,
                    "timestamp": f"{index / args.fps:.6f}",
                    "camera_x_m": "0.0",
                    "camera_y_m": "0.0",
                    "camera_z_m": "0.0",
                    "roll_deg": f"{roll:.6f}",
                    "pitch_deg": f"{pitch:.6f}",
                    "yaw_deg": f"{yaw:.6f}",
                    "quality_status": "valid",
                }
            )
    print(args.output_csv.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
