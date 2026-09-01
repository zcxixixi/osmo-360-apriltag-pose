#!/usr/bin/env python3
"""Reject raw-fisheye PnP jumps and recover short visibility gaps.

Direct AprilTag measurements and temporally recovered poses remain explicitly
separated. The effective ratio may be used for visualization, while training
quality gates can still require direct_measurement=true.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("pose", type=Path)
    parser.add_argument("--initial-pose", type=Path, required=True,
                        help="independent trajectory used only for jump auditing")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-interpolation-gap-s", type=float, default=3.0)
    parser.add_argument("--position-innovation-mm", type=float, default=35.0)
    parser.add_argument("--rotation-innovation-deg", type=float, default=15.0)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(newline="", encoding="utf-8")))


def pose_arrays(rows: list[dict[str, str]]):
    selected = [row for row in rows if row.get("camera_x_m") and row.get("qw")]
    time = np.asarray([float(row["timestamp"]) for row in selected])
    position = np.asarray([[float(row[key]) for key in
                            ("camera_x_m", "camera_y_m", "camera_z_m")]
                           for row in selected])
    rotation = Rotation.from_quat([[float(row[key]) for key in
                                    ("qx", "qy", "qz", "qw")]
                                   for row in selected])
    return selected, time, position, rotation


def interpolate_reference(rows: list[dict[str, str]], query: np.ndarray):
    _, time, position, rotation = pose_arrays(rows)
    p = np.column_stack([np.interp(query, time, position[:, axis]) for axis in range(3)])
    q = Slerp(time, rotation)(np.clip(query, time[0], time[-1]))
    return p, q


def detect_jump_indices(rows: list[dict[str, str]], reference_rows: list[dict[str, str]],
                        args: argparse.Namespace):
    selected, time, position, rotation = pose_arrays(rows)
    ref_position, ref_rotation = interpolate_reference(reference_rows, time)
    jumps: set[int] = set()
    evidence = []
    if len(time) < 3:
        return jumps, evidence
    dt = np.diff(time)
    direct_step = np.linalg.norm(np.diff(position, axis=0), axis=1)
    ref_step = np.linalg.norm(np.diff(ref_position, axis=0), axis=1)
    direct_angle = np.degrees((rotation[:-1].inv() * rotation[1:]).magnitude())
    ref_angle = np.degrees((ref_rotation[:-1].inv() * ref_rotation[1:]).magnitude())
    for index in range(len(dt)):
        if dt[index] > 0.15:
            continue
        position_innovation_mm = (direct_step[index] - ref_step[index]) * 1000.0
        rotation_innovation_deg = direct_angle[index] - ref_angle[index]
        if ((position_innovation_mm > args.position_innovation_mm and
             direct_step[index] * 1000.0 > 45.0) or
                (rotation_innovation_deg > args.rotation_innovation_deg and
                 direct_angle[index] > 22.0)):
            jumps.add(index + 1)
            evidence.append({
                "time_s": float(time[index + 1]),
                "kind": "relative_step_innovation",
                "raw_translation_step_mm": float(direct_step[index] * 1000.0),
                "reference_translation_step_mm": float(ref_step[index] * 1000.0),
                "raw_rotation_step_deg": float(direct_angle[index]),
                "reference_rotation_step_deg": float(ref_angle[index]),
            })
    # A one-frame planar PnP excursion produces a large acceleration which is
    # absent from the independent reference trajectory.
    for index in range(1, len(time) - 1):
        if time[index] - time[index - 1] > 0.15 or time[index + 1] - time[index] > 0.15:
            continue
        raw_accel_mm = np.linalg.norm(position[index + 1] - 2 * position[index] + position[index - 1]) * 1000.0
        ref_accel_mm = np.linalg.norm(ref_position[index + 1] - 2 * ref_position[index] + ref_position[index - 1]) * 1000.0
        raw_rot_accel = abs(direct_angle[index] - direct_angle[index - 1])
        ref_rot_accel = abs(ref_angle[index] - ref_angle[index - 1])
        if ((raw_accel_mm - ref_accel_mm > args.position_innovation_mm and raw_accel_mm > 50.0) or
                (raw_rot_accel - ref_rot_accel > args.rotation_innovation_deg and raw_rot_accel > 22.0)):
            jumps.add(index)
            evidence.append({
                "time_s": float(time[index]),
                "kind": "second_difference_innovation",
                "raw_translation_second_difference_mm": float(raw_accel_mm),
                "reference_translation_second_difference_mm": float(ref_accel_mm),
                "raw_rotation_step_difference_deg": float(raw_rot_accel),
                "reference_rotation_step_difference_deg": float(ref_rot_accel),
            })
    return jumps, evidence


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.pose)
    reference_rows = read_rows(args.initial_pose)
    valid_rows, valid_time, valid_position, valid_rotation = pose_arrays(rows)
    jump_indices, evidence = detect_jump_indices(rows, reference_rows, args)
    jump_times = {float(valid_time[index]) for index in jump_indices}
    accepted = [row for row in valid_rows if float(row["timestamp"]) not in jump_times]
    _, accepted_time, accepted_position, accepted_rotation = pose_arrays(accepted)
    slerp = Slerp(accepted_time, accepted_rotation)
    output = []
    recovered_count = 0
    direct_count = 0
    for row in rows:
        result = dict(row)
        time_s = float(row["timestamp"])
        direct = bool(row.get("camera_x_m")) and time_s not in jump_times
        if direct:
            result["direct_measurement"] = "true"
            result["temporal_gap_s"] = "0.000000"
            result["quality_status"] = "valid"
            direct_count += 1
        else:
            before = np.flatnonzero(accepted_time < time_s)
            after = np.flatnonzero(accepted_time > time_s)
            recoverable = bool(len(before) and len(after))
            if recoverable:
                left = before[-1]; right = after[0]
                gap = accepted_time[right] - accepted_time[left]
                recoverable = gap <= args.max_interpolation_gap_s + 1e-9
            if recoverable:
                alpha = (time_s - accepted_time[left]) / gap
                position = accepted_position[left] * (1.0 - alpha) + accepted_position[right] * alpha
                rotation = slerp([time_s])[0]
                quaternion = rotation.as_quat()
                for key, value in zip(("camera_x_m", "camera_y_m", "camera_z_m"), position):
                    result[key] = f"{value:.9f}"
                for key, value in zip(("qx", "qy", "qz", "qw"), quaternion):
                    result[key] = f"{value:.12f}"
                result["quality_status"] = "valid"
                result["measurement_source"] = "raw_fisheye_temporal_interpolation"
                result["direct_measurement"] = "false"
                result["temporal_gap_s"] = f"{gap:.6f}"
                result["angular_rmse_deg"] = ""
                recovered_count += 1
            else:
                for key in ("camera_x_m", "camera_y_m", "camera_z_m", "qx", "qy", "qz", "qw"):
                    result[key] = ""
                result["quality_status"] = "pnp_jump_rejected" if time_s in jump_times else "invalid"
                result["direct_measurement"] = "false"
                result["temporal_gap_s"] = ""
        output.append(result)
    fieldnames = list(rows[0])
    for field in ("direct_measurement", "temporal_gap_s"):
        if field not in fieldnames:
            fieldnames.append(field)
    with (args.output_dir / "pose.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader(); writer.writerows(output)
    total = len(output)
    effective = direct_count + recovered_count
    second_half_evidence = [item for item in evidence if item["time_s"] >= float(rows[-1]["timestamp"]) / 2.0]
    summary = {
        "schema_version": "raw-fisheye-pose-recovery/1.0",
        "source": str(args.pose.resolve()),
        "direct_measurements": direct_count,
        "direct_measurement_ratio": direct_count / total,
        "temporally_recovered": recovered_count,
        "effective_poses": effective,
        "effective_pose_ratio": effective / total,
        "target_effective_ratio": 0.85,
        "target_passed": effective / total >= 0.85,
        "max_interpolation_gap_s": args.max_interpolation_gap_s,
        "pnp_jump_rejected_count": len(jump_times),
        "pnp_jump_times_s": sorted(jump_times),
        "second_half_pnp_jump_evidence": second_half_evidence,
        "warning": "temporally recovered poses are not direct AprilTag measurements",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
