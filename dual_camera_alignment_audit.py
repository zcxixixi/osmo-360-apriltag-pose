#!/usr/bin/env python3
"""Synchronize and audit two 6DoF camera trajectories in the left frame."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


@dataclass(frozen=True)
class Trajectory:
    times: np.ndarray
    positions: np.ndarray
    rotations: Rotation
    frames: np.ndarray
    states: np.ndarray
    direct: np.ndarray


@dataclass(frozen=True)
class SampledTrajectory:
    positions: np.ndarray
    rotations: Rotation
    valid: np.ndarray
    nearest_indices: np.ndarray


@dataclass(frozen=True)
class CaptureInterval:
    start: datetime
    duration_s: float

    @property
    def end(self) -> datetime:
        return self.start + timedelta(seconds=self.duration_s)


def capture_overlap_s(left: CaptureInterval, right: CaptureInterval) -> float:
    """Return wall-clock overlap; non-positive means the recordings are not a pair."""
    return (min(left.end, right.end) - max(left.start, right.start)).total_seconds()


def probe_capture_interval(path: Path) -> CaptureInterval:
    """Read camera recording UTC start and duration from its original container."""
    ffprobe = (
        Path(__file__).resolve().parent
        / "work/tools/ffmpeg-master-latest-linux64-gpl/bin/ffprobe"
    )
    if not ffprobe.is_file():
        raise ValueError(f"missing bundled ffprobe: {ffprobe}")
    process = subprocess.run(
        [str(ffprobe), "-v", "error", "-show_entries",
         "stream=duration:stream_tags=creation_time", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=60, check=False,
    )
    if process.returncode:
        raise ValueError(f"ffprobe failed for {path}: {process.stderr.strip()[:300]}")
    streams = json.loads(process.stdout).get("streams", [])
    starts = [
        stream.get("tags", {}).get("creation_time") for stream in streams
        if stream.get("tags", {}).get("creation_time")
    ]
    durations = [
        float(stream["duration"]) for stream in streams
        if stream.get("duration") not in (None, "N/A")
    ]
    if not starts or not durations:
        raise ValueError(f"source has no creation_time/duration metadata: {path}")
    start = datetime.fromisoformat(starts[0].replace("Z", "+00:00"))
    return CaptureInterval(start, max(durations))


def uuid4_text(value: str | None = None) -> str:
    parsed = uuid.uuid4() if value is None else uuid.UUID(value)
    if parsed.version != 4:
        raise ValueError("capture pair ID must be UUIDv4")
    return str(parsed)


def at_most(value: float, limit: float) -> bool:
    """Inclusive numeric threshold with a negligible floating-point tolerance."""
    return value <= limit + 1e-9


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in (None, "") else math.nan


def load_trajectory(path: Path) -> Trajectory:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty trajectory: {path}")
    fields = set(rows[0])
    timestamp_field = "timestamp_s" if "timestamp_s" in fields else "timestamp"
    if {"optimized_x_m", "optimized_qw"} <= fields:
        position_fields = [f"optimized_{axis}_m" for axis in "xyz"]
        quaternion_fields = [f"optimized_q{axis}" for axis in "xyzw"]
        euler_fields = None
    elif {"relative_x_m", "relative_qw"} <= fields:
        position_fields = [f"relative_{axis}_m" for axis in "xyz"]
        quaternion_fields = [f"relative_q{axis}" for axis in "xyzw"]
        euler_fields = None
    elif {"camera_x_m", "yaw_deg"} <= fields:
        position_fields = [f"camera_{axis}_m" for axis in "xyz"]
        quaternion_fields = None
        euler_fields = ["roll_deg", "pitch_deg", "yaw_deg"]
    else:
        raise ValueError(f"unsupported 6DoF CSV schema: {path}")

    parsed: list[tuple[float, np.ndarray, np.ndarray, int, str, bool]] = []
    for row in rows:
        timestamp = _float(row, timestamp_field)
        position = np.asarray([_float(row, field) for field in position_fields])
        if quaternion_fields:
            quaternion = np.asarray([_float(row, field) for field in quaternion_fields])
        else:
            euler = np.asarray([_float(row, field) for field in euler_fields or ()])
            quaternion = (
                Rotation.from_euler("xyz", euler, degrees=True).as_quat()
                if np.isfinite(euler).all() else np.full(4, np.nan)
            )
        norm = np.linalg.norm(quaternion)
        if not (math.isfinite(timestamp) and np.isfinite(position).all()
                and np.isfinite(quaternion).all() and norm > 1e-9):
            continue
        state = row.get("state") or row.get("quality_status") or "VALID"
        direct = row.get("direct_measurement") == "1" or (
            row.get("measurement_source", "direct") in ("", "direct")
            and row.get("quality_status", "valid") == "valid"
        )
        parsed.append((
            timestamp, position, quaternion / norm,
            int(row.get("frame") or len(parsed)), state, direct,
        ))
    if len(parsed) < 10:
        raise ValueError(f"fewer than 10 valid poses in {path}")
    parsed.sort(key=lambda item: item[0])
    unique = []
    for item in parsed:
        if not unique or item[0] > unique[-1][0]:
            unique.append(item)
    return Trajectory(
        np.asarray([item[0] for item in unique]),
        np.asarray([item[1] for item in unique]),
        Rotation.from_quat(np.asarray([item[2] for item in unique])),
        np.asarray([item[3] for item in unique], dtype=int),
        np.asarray([item[4] for item in unique], dtype=object),
        np.asarray([item[5] for item in unique], dtype=bool),
    )


def _motion_signature(trajectory: Trajectory, baseline_s: float = 0.20):
    times, positions, rotations = trajectory.times, trajectory.positions, trajectory.rotations
    left, right = [], []
    tolerance = max(0.03, baseline_s * 0.25)
    for index, timestamp in enumerate(times):
        insertion = int(np.searchsorted(times, timestamp + baseline_s))
        candidates = [candidate for candidate in (insertion - 1, insertion)
                      if index < candidate < len(times)]
        if not candidates:
            continue
        other = min(candidates, key=lambda candidate: abs(times[candidate] - timestamp - baseline_s))
        if abs(times[other] - timestamp - baseline_s) <= tolerance:
            left.append(index)
            right.append(other)
    left = np.asarray(left, dtype=int)
    right = np.asarray(right, dtype=int)
    dt = times[right] - times[left]
    signature_times = (times[left] + times[right]) / 2.0
    linear = np.linalg.norm(positions[right] - positions[left], axis=1) / dt
    angular = (rotations[left].inv() * rotations[right]).magnitude() / dt
    return signature_times, linear, angular


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 20 or np.std(left) < 1e-9 or np.std(right) < 1e-9:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def estimate_time_offset(
    left: Trajectory, right: Trajectory, initial_offset_s: float = 0.0,
    search_radius_s: float = 3.0,
) -> tuple[float, float, float, dict[str, float], list[tuple[float, float]]]:
    """Return offset where right_timestamp = left_timestamp + offset."""
    lt, ll, la = _motion_signature(left)
    rt, rl, ra = _motion_signature(right)
    max_gap = 0.10

    def score(offset: float) -> tuple[float, float, float]:
        query = lt + offset
        insertion = np.searchsorted(rt, query)
        insertion = np.clip(insertion, 1, len(rt) - 1)
        before = insertion - 1
        valid = (
            (query >= rt[0]) & (query <= rt[-1])
            & ((rt[insertion] - rt[before]) <= max_gap)
        )
        if valid.sum() < 20:
            return -math.inf, math.nan, math.nan
        right_linear = np.interp(query[valid], rt, rl)
        right_angular = np.interp(query[valid], rt, ra)
        linear = _correlation(ll[valid], right_linear)
        angular = _correlation(la[valid], right_angular)
        finite = [value for value in (linear, angular) if math.isfinite(value)]
        return (float(np.mean(finite)) if finite else -math.inf, linear, angular)

    coarse_offsets = np.arange(
        initial_offset_s - search_radius_s,
        initial_offset_s + search_radius_s + 0.005, 0.01,
    )
    coarse = [(float(offset), *score(float(offset))) for offset in coarse_offsets]
    best_coarse = max(coarse, key=lambda item: item[1])
    if not math.isfinite(best_coarse[1]):
        raise ValueError("cannot synchronize trajectories; insufficient shared motion")
    fine_offsets = np.arange(best_coarse[0] - 0.02, best_coarse[0] + 0.0201, 0.001)
    fine = [(float(offset), *score(float(offset))) for offset in fine_offsets]
    best = max(fine, key=lambda item: item[1])
    near = [item[0] for item in fine if item[1] >= best[1] - 0.02]
    uncertainty = (max(near) - min(near)) / 2.0 if near else math.inf
    return best[0], best[1], uncertainty, {
        "linear_correlation": best[2], "angular_correlation": best[3],
    }, [(item[0], item[1]) for item in coarse]


def sample_trajectory(
    trajectory: Trajectory, query_times: np.ndarray, max_gap_s: float,
) -> SampledTrajectory:
    right = np.searchsorted(trajectory.times, query_times)
    right = np.clip(right, 1, len(trajectory.times) - 1)
    left = right - 1
    valid = (
        (query_times >= trajectory.times[0])
        & (query_times <= trajectory.times[-1])
        & ((trajectory.times[right] - trajectory.times[left]) <= max_gap_s)
    )
    positions = np.full((len(query_times), 3), np.nan)
    rotation_matrices = np.repeat(np.eye(3)[None], len(query_times), axis=0)
    if valid.any():
        for axis in range(3):
            positions[valid, axis] = np.interp(
                query_times[valid], trajectory.times, trajectory.positions[:, axis]
            )
        rotation_matrices[valid] = Slerp(
            trajectory.times, trajectory.rotations
        )(query_times[valid]).as_matrix()
    before_distance = np.abs(query_times - trajectory.times[left])
    after_distance = np.abs(trajectory.times[right] - query_times)
    nearest = np.where(before_distance <= after_distance, left, right)
    return SampledTrajectory(
        positions, Rotation.from_matrix(rotation_matrices), valid, nearest
    )


def estimate_spatial_alignment(
    left_positions: np.ndarray, left_rotations: Rotation,
    right_positions: np.ndarray, right_rotations: Rotation,
    eligible: np.ndarray,
) -> tuple[Rotation, np.ndarray, np.ndarray]:
    mask = eligible.copy()
    if mask.sum() < 10:
        raise ValueError("fewer than 10 synchronized poses for spatial alignment")
    for _ in range(5):
        coordinate_rotations = left_rotations[mask] * right_rotations[mask].inv()
        right_to_left_rotation = coordinate_rotations.mean()
        translation_samples = (
            left_positions[mask]
            - right_to_left_rotation.apply(right_positions[mask])
        )
        translation = np.median(translation_samples, axis=0)
        aligned_positions = right_to_left_rotation.apply(right_positions) + translation
        aligned_rotations = right_to_left_rotation * right_rotations
        position_error = np.linalg.norm(left_positions - aligned_positions, axis=1)
        orientation_error = np.degrees(
            (left_rotations.inv() * aligned_rotations).magnitude()
        )
        position_med = np.median(position_error[mask])
        position_mad = np.median(np.abs(position_error[mask] - position_med))
        angle_med = np.median(orientation_error[mask])
        angle_mad = np.median(np.abs(orientation_error[mask] - angle_med))
        refined = eligible & (
            position_error <= max(position_med + 4.0 * 1.4826 * position_mad, 0.002)
        ) & (
            orientation_error <= max(angle_med + 4.0 * 1.4826 * angle_mad, 0.2)
        )
        if refined.sum() < 10 or np.array_equal(refined, mask):
            break
        mask = refined
    return right_to_left_rotation, translation, mask


def estimate_rigid_camera_extrinsic(
    left_positions: np.ndarray, left_rotations: Rotation,
    right_positions: np.ndarray, right_rotations: Rotation,
    eligible: np.ndarray,
) -> tuple[Rotation, np.ndarray, np.ndarray]:
    """Estimate fixed ``T_left_right`` for two cameras on one rigid body."""
    mask = eligible.copy()
    if mask.sum() < 10:
        raise ValueError("fewer than 10 synchronized poses for rigid extrinsic alignment")
    for _ in range(5):
        left_to_right_rotation = (left_rotations[mask].inv() * right_rotations[mask]).mean()
        baseline_samples = left_rotations[mask].inv().apply(
            right_positions[mask] - left_positions[mask]
        )
        baseline = np.median(baseline_samples, axis=0)
        reconstructed_rotations = right_rotations * left_to_right_rotation.inv()
        reconstructed_positions = right_positions - reconstructed_rotations.apply(baseline)
        position_error = np.linalg.norm(left_positions - reconstructed_positions, axis=1)
        orientation_error = np.degrees(
            (left_rotations.inv() * reconstructed_rotations).magnitude()
        )
        position_med = np.median(position_error[mask])
        position_mad = np.median(np.abs(position_error[mask] - position_med))
        angle_med = np.median(orientation_error[mask])
        angle_mad = np.median(np.abs(orientation_error[mask] - angle_med))
        refined = eligible & (
            position_error <= max(position_med + 4.0 * 1.4826 * position_mad, 0.002)
        ) & (
            orientation_error <= max(angle_med + 4.0 * 1.4826 * angle_mad, 0.2)
        )
        if refined.sum() < 10 or np.array_equal(refined, mask):
            break
        mask = refined
    return left_to_right_rotation, baseline, mask


def _stats(values: np.ndarray) -> dict[str, float | int | None]:
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "rmse": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(len(values)),
        "rmse": float(np.sqrt(np.mean(values ** 2))),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _identity(path: Path, asset_id: str, role: str, pair_id: str) -> dict:
    stat = path.stat()
    return {
        "asset_id": asset_id, "capture_pair_id": pair_id, "role": role,
        "path": str(path.resolve()), "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _write_plot(
    path: Path, left_positions: np.ndarray, right_raw: np.ndarray,
    right_aligned: np.ndarray, times: np.ndarray,
    position_error: np.ndarray, orientation_error: np.ndarray,
    pair_id: str,
) -> None:
    fig = plt.figure(figsize=(15, 5.5), facecolor="#101318")
    grid = fig.add_gridspec(1, 3, width_ratios=(1.3, 1, 1))
    axis = fig.add_subplot(grid[0, 0], projection="3d")
    axis.plot(*left_positions.T, color="#43a5ff", label="LEFT reference", linewidth=2)
    axis.plot(*right_raw.T, color="#777777", label="RIGHT raw", alpha=0.5)
    axis.plot(*right_aligned.T, color="#ff9f43", label="RIGHT aligned", linewidth=1.7)
    colors = ("#ff4d4d", "#55d66b", "#4d7dff")
    for index, (label, color) in enumerate(zip("XYZ", colors)):
        direction = np.zeros(3); direction[index] = 0.15
        axis.quiver(0, 0, 0, *direction, color=color, linewidth=2)
        axis.text(*direction, label, color=color)
    axis.set_title("Right trajectory transformed into LEFT frame", color="white")
    axis.legend(fontsize=8)
    axis.set_xlabel("X m"); axis.set_ylabel("Y m"); axis.set_zlabel("Z m")
    for plot_axis, values, title, color, unit in (
        (fig.add_subplot(grid[0, 1]), position_error, "Position residual", "#ff9f43", "m"),
        (fig.add_subplot(grid[0, 2]), orientation_error, "Orientation residual", "#bd8cff", "deg"),
    ):
        plot_axis.plot(times, values, color=color, linewidth=1)
        plot_axis.set_title(title, color="white")
        plot_axis.set_xlabel("LEFT time (s)"); plot_axis.set_ylabel(unit)
        plot_axis.grid(alpha=0.2)
    for plot_axis in fig.axes:
        plot_axis.set_facecolor("#171b22")
        plot_axis.tick_params(colors="#d4d8df")
        for label in (plot_axis.xaxis.label, plot_axis.yaxis.label):
            label.set_color("#d4d8df")
    fig.suptitle(f"Dual-camera alignment audit  |  pair {pair_id}", color="white")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align a right 6DoF trajectory into the left-camera frame and audit the pair"
    )
    parser.add_argument("left_video", type=Path)
    parser.add_argument("left_trajectory", type=Path)
    parser.add_argument("right_video", type=Path)
    parser.add_argument("right_trajectory", type=Path)
    parser.add_argument(
        "--left-source", type=Path,
        help="original camera container for wall-clock pairing audit",
    )
    parser.add_argument(
        "--right-source", type=Path,
        help="original camera container for wall-clock pairing audit",
    )
    parser.add_argument(
        "--require-wall-clock-overlap", action="store_true",
        help="reject non-overlapping creation_time metadata (only for clock-synchronized cameras)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capture-pair-id", help="existing UUIDv4; generated when omitted")
    parser.add_argument("--initial-time-offset", type=float, default=0.0)
    parser.add_argument("--search-radius", type=float, default=3.0)
    parser.add_argument(
        "--fixed-time-offset", type=float,
        help="skip trajectory correlation and use an externally measured offset",
    )
    parser.add_argument(
        "--external-sync-correlation", type=float, default=1.0,
        help="quality score for --fixed-time-offset (for example audio correlation)",
    )
    parser.add_argument(
        "--external-sync-uncertainty", type=float, default=0.001,
        help="seconds of uncertainty for --fixed-time-offset",
    )
    parser.add_argument("--max-interpolation-gap", type=float, default=0.10)
    parser.add_argument(
        "--alignment-model", choices=("rigid-rig", "world-frame", "shared-world"),
        default="rigid-rig",
        help=("rigid-rig estimates a fixed camera-to-camera extrinsic; "
              "world-frame fits a left-multiplication; shared-world preserves "
              "trajectories already expressed in the same tag-map frame"),
    )
    parser.add_argument("--min-alignment-samples", type=int, default=30)
    parser.add_argument("--max-position-p95-m", type=float, default=0.05)
    parser.add_argument("--max-orientation-p95-deg", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (
        args.left_video, args.left_trajectory,
        args.right_video, args.right_trajectory,
    ):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if args.search_radius <= 0 or args.max_interpolation_gap <= 0:
        raise SystemExit("invalid synchronization parameters")
    if (args.left_source is None) != (args.right_source is None):
        raise SystemExit("--left-source and --right-source must be supplied together")
    try:
        pair_id = uuid4_text(args.capture_pair_id)
        if args.left_source is not None and args.right_source is not None:
            for source in (args.left_source, args.right_source):
                if not source.is_file():
                    raise ValueError(f"missing original source: {source}")
            left_capture = probe_capture_interval(args.left_source)
            right_capture = probe_capture_interval(args.right_source)
            overlap = capture_overlap_s(left_capture, right_capture)
            if overlap <= 0:
                message = (
                    "recordings do not overlap in wall-clock time: "
                    f"left={left_capture.start.isoformat()}..{left_capture.end.isoformat()}, "
                    f"right={right_capture.start.isoformat()}..{right_capture.end.isoformat()}"
                )
                if args.require_wall_clock_overlap:
                    raise ValueError(message)
                print(
                    "WARNING: " + message
                    + "; camera clocks may be unsynchronized, continuing with signal-based sync",
                )
        left = load_trajectory(args.left_trajectory)
        right = load_trajectory(args.right_trajectory)
        if args.fixed_time_offset is None:
            offset, correlation, uncertainty, components, offset_curve = estimate_time_offset(
                left, right, args.initial_time_offset, args.search_radius
            )
            sync_method = "trajectory_motion_correlation"
        else:
            if not (-1.0 <= args.external_sync_correlation <= 1.0):
                raise ValueError("external sync correlation must be within [-1, 1]")
            if args.external_sync_uncertainty < 0:
                raise ValueError("external sync uncertainty must be non-negative")
            offset = args.fixed_time_offset
            correlation = args.external_sync_correlation
            uncertainty = args.external_sync_uncertainty
            components = {
                "linear_correlation": None,
                "angular_correlation": None,
            }
            offset_curve = [(offset, correlation)]
            sync_method = "external_fixed_offset"
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    query_right_times = left.times + offset
    sampled = sample_trajectory(right, query_right_times, args.max_interpolation_gap)
    matched = sampled.valid
    both_direct = matched & left.direct & right.direct[sampled.nearest_indices]
    fit_source = "both_direct" if both_direct.sum() >= args.min_alignment_samples else "all_valid"
    eligible = both_direct if fit_source == "both_direct" else matched
    try:
        if args.alignment_model == "shared-world":
            rotation = Rotation.identity()
            translation = np.zeros(3, dtype=float)
            inliers = eligible.copy()
        elif args.alignment_model == "rigid-rig":
            rotation, translation, inliers = estimate_rigid_camera_extrinsic(
                left.positions, left.rotations,
                sampled.positions, sampled.rotations, eligible,
            )
        else:
            rotation, translation, inliers = estimate_spatial_alignment(
                left.positions, left.rotations,
                sampled.positions, sampled.rotations, eligible,
            )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.alignment_model == "rigid-rig":
        aligned_rotations = sampled.rotations[matched] * rotation.inv()
        aligned_positions = (
            sampled.positions[matched] - aligned_rotations.apply(translation)
        )
        alignment_definition = "T_board_right = T_board_left * T_left_right"
        matrix_name = "T_left_right"
    else:
        aligned_positions = rotation.apply(sampled.positions[matched]) + translation
        aligned_rotations = rotation * sampled.rotations[matched]
        if args.alignment_model == "shared-world":
            alignment_definition = "pose_shared = pose_tagmap (identity; no cross-trajectory fit)"
            matrix_name = "T_shared_from_right_tagmap"
        else:
            alignment_definition = "pose_left = T_left_from_right * pose_right"
            matrix_name = "T_left_from_right"
    left_positions = left.positions[matched]
    left_rotations = left.rotations[matched]
    position_error = np.linalg.norm(left_positions - aligned_positions, axis=1)
    orientation_error = np.degrees(
        (left_rotations.inv() * aligned_rotations).magnitude()
    )
    position_stats = _stats(position_error)
    orientation_stats = _stats(orientation_error)
    checks = {
        "sync_correlation_at_least_0_80": correlation >= 0.80,
        # The uncertainty grid is evaluated in 1 ms increments. A tiny
        # tolerance prevents an exact 20 ms boundary from failing due to
        # binary floating-point representation.
        "sync_uncertainty_at_most_20ms": at_most(uncertainty, 0.020),
        "enough_alignment_samples": int(inliers.sum()) >= args.min_alignment_samples,
        "position_p95_within_limit": (
            args.alignment_model == "shared-world" or
            position_stats["p95"] is not None
            and position_stats["p95"] <= args.max_position_p95_m
        ),
        "orientation_p95_within_limit": (
            args.alignment_model == "shared-world" or
            orientation_stats["p95"] is not None
            and orientation_stats["p95"] <= args.max_orientation_p95_deg
        ),
    }
    status = "PASS" if all(checks.values()) else "REVIEW"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    left_asset_id, right_asset_id, run_id = (
        str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())
    )
    left_identity = _identity(args.left_video, left_asset_id, "left_reference", pair_id)
    right_identity = _identity(args.right_video, right_asset_id, "right", pair_id)
    matrix = np.eye(4)
    matrix[:3, :3] = rotation.as_matrix()
    matrix[:3, 3] = translation
    report = {
        "schema_version": "dual-camera-alignment-audit/v1",
        "capture_pair_id": pair_id,
        "alignment_run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reference_frame": (
            "shared_tagmap" if args.alignment_model == "shared-world" else "left_camera"
        ),
        "left": {**left_identity, "trajectory": str(args.left_trajectory.resolve())},
        "right": {**right_identity, "trajectory": str(args.right_trajectory.resolve())},
        "time_alignment": {
            "definition": "right_timestamp = left_timestamp + offset_s",
            "method": sync_method,
            "offset_s": offset, "correlation": correlation,
            "uncertainty_s": uncertainty, **components,
        },
        "spatial_alignment": {
            "model": args.alignment_model,
            "definition": alignment_definition,
            matrix_name: matrix.tolist(),
            "translation_m": translation.tolist(),
            "rotation_xyzw": rotation.as_quat().tolist(),
            "fit_source": fit_source,
            "fit_samples": int(eligible.sum()), "inlier_samples": int(inliers.sum()),
        },
        "matched_samples": int(matched.sum()),
        "position_residual_m": position_stats,
        "orientation_residual_deg": orientation_stats,
        "quality_checks": checks,
        "status": status,
    }
    (args.output_dir / "alignment_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    pair_manifest = {
        "schema_version": "dual-camera-capture-pair/v1",
        "capture_pair_id": pair_id,
        "declared_simultaneous_capture": True,
        "reference_role": "left_reference",
        "assets": [left_identity, right_identity],
        "alignment_report": "alignment_report.json",
        "aligned_trajectory": "aligned_trajectories.csv",
    }
    (args.output_dir / "capture_pair.json").write_text(
        json.dumps(pair_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for name, identity in (("left_asset.json", left_identity), ("right_asset.json", right_identity)):
        (args.output_dir / name).write_text(
            json.dumps(identity, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    with (args.output_dir / "time_offset_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(("offset_s", "motion_correlation"))
        writer.writerows(offset_curve)
    matched_indices = np.flatnonzero(matched)
    right_quaternions = sampled.rotations[matched].as_quat()
    aligned_quaternions = aligned_rotations.as_quat()
    left_quaternions = left_rotations.as_quat()
    with (args.output_dir / "aligned_trajectories.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "capture_pair_id", "left_frame", "left_timestamp_s", "right_timestamp_s",
            *[f"left_{axis}_m" for axis in "xyz"], *[f"left_q{axis}" for axis in "xyzw"],
            *[f"right_raw_{axis}_m" for axis in "xyz"], *[f"right_raw_q{axis}" for axis in "xyzw"],
            *[f"right_aligned_{axis}_m" for axis in "xyz"],
            *[f"right_aligned_q{axis}" for axis in "xyzw"],
            "position_residual_m", "orientation_residual_deg",
            "left_state", "right_state", "both_direct",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for output_index, left_index in enumerate(matched_indices):
            right_index = sampled.nearest_indices[left_index]
            row = {
                "capture_pair_id": pair_id,
                "left_frame": int(left.frames[left_index]),
                "left_timestamp_s": f"{left.times[left_index]:.6f}",
                "right_timestamp_s": f"{query_right_times[left_index]:.6f}",
                "position_residual_m": f"{position_error[output_index]:.8f}",
                "orientation_residual_deg": f"{orientation_error[output_index]:.6f}",
                "left_state": left.states[left_index], "right_state": right.states[right_index],
                "both_direct": int(left.direct[left_index] and right.direct[right_index]),
            }
            for prefix, position, quaternion in (
                ("left", left_positions[output_index], left_quaternions[output_index]),
                ("right_raw", sampled.positions[left_index], right_quaternions[output_index]),
                ("right_aligned", aligned_positions[output_index], aligned_quaternions[output_index]),
            ):
                row.update({f"{prefix}_{axis}_m": f"{position[index]:.8f}"
                            for index, axis in enumerate("xyz")})
                row.update({f"{prefix}_q{axis}": f"{quaternion[index]:.9f}"
                            for index, axis in enumerate("xyzw")})
            writer.writerow(row)
    _write_plot(
        args.output_dir / "alignment_audit.png",
        left_positions, sampled.positions[matched], aligned_positions,
        left.times[matched], position_error, orientation_error, pair_id,
    )
    print(json.dumps({
        "capture_pair_id": pair_id, "status": status,
        "time_offset_s": offset, "sync_correlation": correlation,
        "output_dir": str(args.output_dir.resolve()),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
