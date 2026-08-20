#!/usr/bin/env python3
"""Compare an AprilGrid camera trajectory with motion-capture ground truth.

Ground-truth CSV columns:
  timestamp,x,y,z,qx,qy,qz,qw

The quaternion describes body-to-world orientation.  By default the mocap
body is assumed to be the camera.  Supply --body-to-camera with a JSON 4x4
matrix when the tracked rigid-body origin differs from the camera optical
frame.  The estimated CSV is the pose.csv emitted by osmo_360_offline.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class Trajectory:
    times: np.ndarray
    positions: np.ndarray
    rotations: Rotation


def _stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def load_estimate(path: Path) -> tuple[Trajectory, list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    valid = [
        row
        for row in rows
        if row.get("quality_status") == "valid" and row.get("camera_x_m")
    ]
    if len(valid) < 4:
        raise ValueError("estimated pose CSV has fewer than four valid poses")
    times = np.asarray([float(row["timestamp"]) for row in valid])
    positions = np.asarray(
        [[float(row[f"camera_{axis}_m"]) for axis in "xyz"] for row in valid]
    )
    rpy = np.asarray(
        [[float(row[f"{axis}_deg"]) for axis in ("roll", "pitch", "yaw")] for row in valid]
    )
    return Trajectory(times, positions, Rotation.from_euler("xyz", rpy, degrees=True)), rows


def load_mocap(path: Path, unit: str, body_to_camera: np.ndarray) -> Trajectory:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"mocap CSV must contain {sorted(required)}")
    scale = 0.001 if unit == "mm" else 1.0
    times = np.asarray([float(row["timestamp"]) for row in rows])
    positions = scale * np.asarray(
        [[float(row[axis]) for axis in "xyz"] for row in rows]
    )
    rotations = Rotation.from_quat(
        [[float(row[key]) for key in ("qx", "qy", "qz", "qw")] for row in rows]
    )
    order = np.argsort(times)
    times, positions, rotations = times[order], positions[order], rotations[order]
    if np.any(np.diff(times) <= 0):
        raise ValueError("mocap timestamps must be unique and increasing")
    lever = body_to_camera[:3, 3]
    body_to_camera_rotation = Rotation.from_matrix(body_to_camera[:3, :3])
    camera_positions = positions + rotations.apply(lever)
    camera_rotations = rotations * body_to_camera_rotation
    return Trajectory(times, camera_positions, camera_rotations)


def interpolate_trajectory(trajectory: Trajectory, times: np.ndarray) -> Trajectory:
    if times.min() < trajectory.times[0] or times.max() > trajectory.times[-1]:
        raise ValueError("requested timestamps exceed mocap coverage")
    positions = np.column_stack(
        [np.interp(times, trajectory.times, trajectory.positions[:, axis]) for axis in range(3)]
    )
    rotations = Slerp(trajectory.times, trajectory.rotations)(times)
    return Trajectory(times, positions, rotations)


def rigid_alignment(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kabsch SE(3) alignment from source points into target coordinates."""
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    u, _singular, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def _motion_signature(trajectory: Trajectory) -> tuple[np.ndarray, np.ndarray]:
    dt = np.diff(trajectory.times)
    linear = np.linalg.norm(np.diff(trajectory.positions, axis=0), axis=1) / dt
    relative = trajectory.rotations[:-1].inv() * trajectory.rotations[1:]
    angular = np.linalg.norm(relative.as_rotvec(), axis=1) / dt
    return linear, angular


def estimate_time_offset(
    estimate: Trajectory, mocap: Trajectory, search_seconds: float
) -> tuple[float, float]:
    """Return offset added to estimate timestamps and its motion correlation."""
    median_dt = float(np.median(np.diff(estimate.times)))
    step = max(median_dt, 0.002)
    offsets = np.arange(-search_seconds, search_seconds + step / 2, step)
    estimate_linear, estimate_angular = _motion_signature(estimate)
    best = (0.0, -math.inf)
    for offset in offsets:
        shifted = estimate.times + offset
        if shifted[0] < mocap.times[0] or shifted[-1] > mocap.times[-1]:
            continue
        sampled = interpolate_trajectory(mocap, shifted)
        mocap_linear, mocap_angular = _motion_signature(sampled)
        scores = []
        for left, right in (
            (estimate_linear, mocap_linear),
            (estimate_angular, mocap_angular),
        ):
            if np.std(left) > 1e-8 and np.std(right) > 1e-8:
                scores.append(float(np.corrcoef(left, right)[0, 1]))
        score = float(np.mean(scores)) if scores else -math.inf
        if score > best[1]:
            best = (float(offset), score)
    if not math.isfinite(best[1]):
        raise ValueError("cannot estimate time offset; check overlapping timestamps")
    return best


def _relative_errors(
    estimate: Trajectory, truth: Trajectory, delta: int
) -> tuple[np.ndarray, np.ndarray]:
    if delta <= 0 or delta >= len(estimate.times):
        return np.array([]), np.array([])
    est_delta_r = estimate.rotations[:-delta].inv() * estimate.rotations[delta:]
    gt_delta_r = truth.rotations[:-delta].inv() * truth.rotations[delta:]
    rotation_error = gt_delta_r.inv() * est_delta_r
    est_local_t = estimate.rotations[:-delta].inv().apply(
        estimate.positions[delta:] - estimate.positions[:-delta]
    )
    gt_local_t = truth.rotations[:-delta].inv().apply(
        truth.positions[delta:] - truth.positions[:-delta]
    )
    return np.linalg.norm(est_local_t - gt_local_t, axis=1), np.degrees(rotation_error.magnitude())


def loss_events(rows: list[dict[str, str]], time_offset: float) -> list[dict[str, float | int]]:
    events: list[dict[str, float | int]] = []
    start = None
    for index, row in enumerate(rows):
        valid = row.get("quality_status") == "valid" and bool(row.get("camera_x_m"))
        if not valid and start is None:
            start = index
        if valid and start is not None:
            start_time = float(rows[start]["timestamp"]) + time_offset
            end_time = float(row["timestamp"]) + time_offset
            events.append(
                {
                    "start_frame": int(rows[start].get("frame", start)),
                    "recovery_frame": int(row.get("frame", index)),
                    "start_time_s": start_time,
                    "recovery_time_s": end_time,
                    "duration_s": end_time - start_time,
                }
            )
            start = None
    return events


def parse_interval(value: str) -> tuple[float, float]:
    try:
        start, end = (float(item) for item in value.split(":", 1))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("interval must be START:END seconds") from exc
    if end <= start:
        raise argparse.ArgumentTypeError("interval end must exceed start")
    return start, end


def evaluate(
    estimate: Trajectory,
    mocap: Trajectory,
    rows: list[dict[str, str]],
    time_offset: float,
    rpe_delta: int,
    static_intervals: list[tuple[float, float]],
) -> tuple[dict, dict[str, np.ndarray]]:
    shifted_times = estimate.times + time_offset
    inside = (shifted_times >= mocap.times[0]) & (shifted_times <= mocap.times[-1])
    estimate = Trajectory(
        estimate.times[inside], estimate.positions[inside], estimate.rotations[inside]
    )
    shifted_times = shifted_times[inside]
    truth = interpolate_trajectory(mocap, shifted_times)
    align_r, align_t = rigid_alignment(truth.positions, estimate.positions)
    aligned_truth = Trajectory(
        truth.times,
        (align_r @ truth.positions.T).T + align_t,
        Rotation.from_matrix(align_r) * truth.rotations,
    )
    position_error = np.linalg.norm(estimate.positions - aligned_truth.positions, axis=1)
    orientation_error = np.degrees(
        (aligned_truth.rotations.inv() * estimate.rotations).magnitude()
    )
    rpe_t, rpe_r = _relative_errors(estimate, aligned_truth, rpe_delta)
    events = loss_events(rows, time_offset)
    for event in events:
        recovery_time = float(event["recovery_time_s"])
        nearest = int(np.argmin(np.abs(shifted_times - recovery_time)))
        event["recovery_position_error_m"] = float(position_error[nearest])
        event["recovery_orientation_error_deg"] = float(orientation_error[nearest])
    path_length = float(np.linalg.norm(np.diff(aligned_truth.positions, axis=0), axis=1).sum())
    endpoint_relative_error = float(
        np.linalg.norm(
            (estimate.positions[-1] - estimate.positions[0])
            - (aligned_truth.positions[-1] - aligned_truth.positions[0])
        )
    )
    static = []
    for start, end in static_intervals:
        mask = (estimate.times >= start) & (estimate.times <= end)
        if mask.sum() < 2:
            continue
        local_positions = estimate.positions[mask]
        local_rotations = estimate.rotations[mask]
        static.append(
            {
                "start_s": start,
                "end_s": end,
                "samples": int(mask.sum()),
                "position_radius_p95_m": float(
                    np.percentile(
                        np.linalg.norm(local_positions - local_positions.mean(axis=0), axis=1), 95
                    )
                ),
                "orientation_drift_max_deg": float(
                    np.degrees((local_rotations[0].inv() * local_rotations).magnitude()).max()
                ),
            }
        )
    report = {
        "matched_visual_samples": int(len(estimate.times)),
        "se3_alignment": {"rotation": align_r.tolist(), "translation_m": align_t.tolist()},
        "position_ate_m": _stats(position_error),
        "orientation_error_deg": _stats(orientation_error),
        "translation_rpe_m": _stats(rpe_t),
        "rotation_rpe_deg": _stats(rpe_r),
        "path_length_m": path_length,
        "endpoint_relative_error_m": endpoint_relative_error,
        "drift_percent_of_path": 100 * endpoint_relative_error / path_length if path_length else 0.0,
        "loss_events": events,
        "longest_loss_s": max((float(event["duration_s"]) for event in events), default=0.0),
        "static_intervals": static,
    }
    series = {
        "times": estimate.times,
        "position_error": position_error,
        "orientation_error": orientation_error,
        "estimate_positions": estimate.positions,
        "truth_positions": aligned_truth.positions,
    }
    return report, series


def write_outputs(output_dir: Path, report: dict, series: dict[str, np.ndarray]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mocap_evaluation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (output_dir / "mocap_errors.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("timestamp", "position_error_m", "orientation_error_deg"))
        writer.writerows(
            zip(series["times"], series["position_error"], series["orientation_error"])
        )
    fig = plt.figure(figsize=(13, 8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    axis3d = fig.add_subplot(grid[:, 0], projection="3d")
    axis3d.plot(*series["truth_positions"].T, label="Mocap truth", linewidth=2)
    axis3d.plot(*series["estimate_positions"].T, label="AprilGrid estimate", linewidth=1.2)
    axis3d.set_xlabel("X [m]")
    axis3d.set_ylabel("Y [m]")
    axis3d.set_zlabel("Z [m]")
    axis3d.legend()
    pos_axis = fig.add_subplot(grid[0, 1])
    pos_axis.plot(series["times"], 1000 * series["position_error"])
    pos_axis.set_ylabel("Position error [mm]")
    pos_axis.grid(alpha=0.3)
    rot_axis = fig.add_subplot(grid[1, 1], sharex=pos_axis)
    rot_axis.plot(series["times"], series["orientation_error"])
    rot_axis.set_xlabel("Time [s]")
    rot_axis.set_ylabel("Orientation error [deg]")
    rot_axis.grid(alpha=0.3)
    fig.savefig(output_dir / "mocap_evaluation.png", dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("estimate_csv", type=Path)
    parser.add_argument("mocap_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mocap-unit", choices=("m", "mm"), default="m")
    parser.add_argument("--body-to-camera", type=Path, help="JSON 4x4 T_body_camera")
    parser.add_argument("--time-offset", type=float, help="seconds added to estimate timestamps")
    parser.add_argument("--search-time-offset", type=float, default=1.0)
    parser.add_argument("--rpe-delta", type=int, default=1)
    parser.add_argument("--static-interval", action="append", type=parse_interval, default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    body_to_camera = np.eye(4)
    if args.body_to_camera:
        body_to_camera = np.asarray(
            json.loads(args.body_to_camera.read_text(encoding="utf-8")), dtype=float
        )
        if body_to_camera.shape != (4, 4):
            raise SystemExit("--body-to-camera must contain a JSON 4x4 matrix")
    estimate, rows = load_estimate(args.estimate_csv)
    mocap = load_mocap(args.mocap_csv, args.mocap_unit, body_to_camera)
    if args.time_offset is None:
        time_offset, correlation = estimate_time_offset(
            estimate, mocap, args.search_time_offset
        )
    else:
        time_offset, correlation = args.time_offset, None
    report, series = evaluate(
        estimate, mocap, rows, time_offset, args.rpe_delta, args.static_interval
    )
    report["time_alignment"] = {
        "estimate_timestamp_offset_s": time_offset,
        "motion_correlation": correlation,
    }
    report["inputs"] = {
        "estimate_csv": str(args.estimate_csv.resolve()),
        "mocap_csv": str(args.mocap_csv.resolve()),
        "mocap_unit": args.mocap_unit,
        "body_to_camera": body_to_camera.tolist(),
    }
    write_outputs(args.output_dir, report, series)
    print(args.output_dir.resolve())
    print(json.dumps({
        "position_ate_rmse_m": report["position_ate_m"].get("rmse"),
        "orientation_rmse_deg": report["orientation_error_deg"].get("rmse"),
        "longest_loss_s": report["longest_loss_s"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
