#!/usr/bin/env python3
"""Held-out Insta360 AprilTag trajectory evaluation against Motive ground truth."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class PoseSeries:
    times: np.ndarray
    positions: np.ndarray
    rotations: Rotation
    frames: np.ndarray


@dataclass
class MotiveData:
    all_times: np.ndarray
    positions: np.ndarray
    quaternions: np.ndarray
    valid: np.ndarray
    frames: np.ndarray
    quarantined_ranges: list[tuple[int, int]]
    metadata: dict


def stats(values: np.ndarray) -> dict[str, float | int]:
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


def _false_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[True, mask.astype(bool), True]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(left), int(right - 1)) for left, right in changes.reshape(-1, 2)]


def visual_audit(
    rows: list[dict[str, str]], min_tags: int,
    accepted_sources: tuple[str, ...] = ("", "direct"),
) -> tuple[dict, set[int]]:
    times = np.asarray([float(row["timestamp"]) for row in rows], dtype=float)
    geometrically_valid = np.asarray([
        row.get("quality_status") == "valid"
        and bool(row.get("camera_x_m"))
        and int(row.get("detected_tag_count") or 0) >= min_tags
        for row in rows
    ])
    sources = np.asarray([row.get("measurement_source", "direct") for row in rows])
    direct = geometrically_valid & np.isin(sources, ("", "direct"))
    accepted = geometrically_valid & np.isin(sources, accepted_sources)
    nominal_dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    lost_runs = _false_runs(accepted)
    durations = [times[right] - times[left] + nominal_dt for left, right in lost_runs]
    recovery_indices = {
        right + 1 for left, right in lost_runs if right + 1 < len(rows) and accepted[right + 1]
    }
    noninitial = [
        duration for (left, _right), duration in zip(lost_runs, durations) if left > 0
    ]
    return {
        "sampled_frames": int(len(rows)),
        "direct_multi_tag_frames": int(direct.sum()),
        "direct_coverage_ratio": float(direct.mean()) if len(direct) else 0.0,
        "accepted_pose_frames": int(accepted.sum()),
        "accepted_coverage_ratio": float(accepted.mean()) if len(accepted) else 0.0,
        "accepted_measurement_sources": list(accepted_sources),
        "lost_intervals": int(len(lost_runs)),
        "longest_lost_s": float(max(durations, default=0.0)),
        "longest_lost_after_first_lock_s": float(max(noninitial, default=0.0)),
        "total_lost_s": float(sum(durations)),
        "recovery_count": int(len(recovery_indices)),
    }, {int(rows[index]["frame"]) for index in recovery_indices}


def transform(rotation: Rotation, translation: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = rotation.as_matrix()
    result[:3, 3] = translation
    return result


def matrix_to_vector(value: np.ndarray) -> np.ndarray:
    return np.r_[Rotation.from_matrix(value[:3, :3]).as_rotvec(), value[:3, 3]]


def vector_to_matrix(value: np.ndarray) -> np.ndarray:
    return transform(Rotation.from_rotvec(value[:3]), value[3:6])


def pose_matrices(series: PoseSeries) -> np.ndarray:
    values = np.repeat(np.eye(4)[None, ...], len(series.times), axis=0)
    values[:, :3, :3] = series.rotations.as_matrix()
    values[:, :3, 3] = series.positions
    return values


def parse_motive(path: Path, max_speed_m_s: float = 10.0) -> MotiveData:
    """Parse Motive's multi-row CSV and quarantine discontinuous identity swaps."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError("empty Motive CSV")
    metadata = {rows[0][i]: rows[0][i + 1] for i in range(0, len(rows[0]) - 1, 2)}
    header_index = next(
        (index for index, row in enumerate(rows) if len(row) >= 2 and row[0] == "Frame" and row[1] == "Time (Seconds)"),
        None,
    )
    if header_index is None:
        raise ValueError("Motive Frame/Time header not found")
    declared = int(float(metadata.get("Total Exported Frames", 0)))
    parsed: dict[int, tuple[float, np.ndarray, np.ndarray, bool]] = {}
    for row in rows[header_index + 1 :]:
        if not row or not row[0].strip():
            continue
        try:
            frame = int(row[0])
            timestamp = float(row[1])
        except (ValueError, IndexError):
            continue
        complete = len(row) >= 9 and all(item.strip() for item in row[2:9])
        if complete:
            quaternion = np.asarray([float(item) for item in row[2:6]], dtype=float)
            position = np.asarray([float(item) for item in row[6:9]], dtype=float) * 0.001
            norm = float(np.linalg.norm(quaternion))
            complete = np.isfinite(position).all() and math.isfinite(norm) and norm > 0.5
            if complete:
                quaternion /= norm
        else:
            quaternion = np.full(4, np.nan)
            position = np.full(3, np.nan)
        parsed[frame] = (timestamp, position, quaternion, complete)
    total = declared or (max(parsed) + 1)
    nominal_dt = 1.0 / float(metadata.get("Export Frame Rate", 120.0))
    frames = np.arange(total, dtype=int)
    times = np.asarray([parsed.get(i, (i * nominal_dt, None, None, False))[0] for i in frames])
    positions = np.full((total, 3), np.nan)
    quaternions = np.full((total, 4), np.nan)
    valid = np.zeros(total, dtype=bool)
    for frame, (_time, position, quaternion, complete) in parsed.items():
        if frame >= total:
            continue
        positions[frame] = position
        quaternions[frame] = quaternion
        valid[frame] = complete

    metadata["raw_valid_frames"] = int(valid.sum())
    raw_missing_runs = _false_runs(valid)
    metadata["longest_raw_missing_run_frames"] = int(
        max((right - left + 1 for left, right in raw_missing_runs), default=0)
    )
    # A Motive rigid-body identity swap enters and later leaves through two
    # physically impossible jumps. Pair such boundaries and quarantine the
    # entire alternate branch rather than merely deleting the boundary frames.
    valid_indices = np.flatnonzero(valid)
    boundaries: list[int] = []
    for left, right in zip(valid_indices[:-1], valid_indices[1:]):
        dt = times[right] - times[left]
        if 0 < dt <= 0.1:
            speed = float(np.linalg.norm(positions[right] - positions[left]) / dt)
            if speed > max_speed_m_s:
                boundaries.append(right)
    quarantined: list[tuple[int, int]] = []
    for start, exit_frame in zip(boundaries[0::2], boundaries[1::2]):
        end = exit_frame - 1
        if 2 <= end - start + 1 <= int(10.0 / nominal_dt):
            valid[start : end + 1] = False
            quarantined.append((int(start), int(end)))
    return MotiveData(times, positions, quaternions, valid, frames, quarantined, metadata)


def write_normalized_mocap(path: Path, motive: MotiveData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("frame", "timestamp", "x_m", "y_m", "z_m", "qx", "qy", "qz", "qw", "valid"))
        for index in range(len(motive.frames)):
            if motive.valid[index]:
                writer.writerow((
                    int(motive.frames[index]), f"{motive.all_times[index]:.6f}",
                    *[f"{value:.9f}" for value in motive.positions[index]],
                    *[f"{value:.9f}" for value in motive.quaternions[index]], 1,
                ))
            else:
                writer.writerow((int(motive.frames[index]), f"{motive.all_times[index]:.6f}", "", "", "", "", "", "", "", 0))


def load_visual(
    path: Path, min_tags: int = 2,
    accepted_sources: tuple[str, ...] = ("", "direct"),
) -> tuple[PoseSeries, list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row for row in rows
        if row.get("quality_status") == "valid"
        and row.get("camera_x_m")
        and int(row.get("detected_tag_count") or 0) >= min_tags
        and row.get("measurement_source", "direct") in accepted_sources
    ]
    if len(selected) < 20:
        raise ValueError("fewer than 20 direct multi-tag visual poses")
    return PoseSeries(
        np.asarray([float(row["timestamp"]) for row in selected]),
        np.asarray([[float(row[f"camera_{axis}_m"]) for axis in "xyz"] for row in selected]),
        Rotation.from_euler(
            "xyz",
            [[float(row[f"{axis}_deg"]) for axis in ("roll", "pitch", "yaw")] for row in selected],
            degrees=True,
        ),
        np.asarray([int(row["frame"]) for row in selected]),
    ), rows


def motive_series(motive: MotiveData) -> PoseSeries:
    indices = np.flatnonzero(motive.valid)
    return PoseSeries(
        motive.all_times[indices], motive.positions[indices],
        Rotation.from_quat(motive.quaternions[indices]), motive.frames[indices],
    )


def interpolate_motive(motive: MotiveData, query_times: np.ndarray, max_gap_s: float = 0.05) -> tuple[PoseSeries, np.ndarray]:
    source = motive_series(motive)
    right = np.searchsorted(source.times, query_times)
    right = np.clip(right, 1, len(source.times) - 1)
    left = right - 1
    valid = (
        (query_times >= source.times[0]) & (query_times <= source.times[-1])
        & ((source.times[right] - source.times[left]) <= max_gap_s)
    )
    positions = np.full((len(query_times), 3), np.nan)
    rotations_matrix = np.repeat(np.eye(3)[None, ...], len(query_times), axis=0)
    if valid.any():
        accepted = query_times[valid]
        for axis in range(3):
            positions[valid, axis] = np.interp(accepted, source.times, source.positions[:, axis])
        rotations_matrix[valid] = Slerp(source.times, source.rotations)(accepted).as_matrix()
    return PoseSeries(query_times, positions, Rotation.from_matrix(rotations_matrix), np.full(len(query_times), -1)), valid


def motion_signature(series: PoseSeries, max_dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dt = np.diff(series.times)
    keep = (dt > 0) & (dt <= max_dt)
    midpoint = (series.times[:-1] + series.times[1:]) / 2.0
    linear = np.linalg.norm(np.diff(series.positions, axis=0), axis=1) / dt
    relative = series.rotations[:-1].inv() * series.rotations[1:]
    angular = relative.magnitude() / dt
    return midpoint[keep], linear[keep], angular[keep]


def _interp_signature(query: np.ndarray, times: np.ndarray, values: np.ndarray, max_gap: float) -> tuple[np.ndarray, np.ndarray]:
    right = np.searchsorted(times, query)
    right = np.clip(right, 1, len(times) - 1)
    left = right - 1
    valid = (query >= times[0]) & (query <= times[-1]) & ((times[right] - times[left]) <= max_gap)
    output = np.full(len(query), np.nan)
    output[valid] = np.interp(query[valid], times, values)
    return output, valid


def estimate_time_offset(
    visual: PoseSeries,
    motive: MotiveData,
    initial_offset: float,
    search_radius: float = 1.0,
) -> tuple[float, float, float, dict[str, float], list[tuple[float, float, float, float]]]:
    visual_dt = np.diff(visual.times)
    nominal_visual_dt = float(np.median(visual_dt[visual_dt > 0]))
    # Synchronization needs motion, not single-frame PnP noise. Compare poses
    # over a 200 ms baseline (or one frame for slower inputs), pairing only
    # nearby direct measurements so LOST sections cannot bridge the signal.
    baseline = max(0.20, nominal_visual_dt)
    tolerance = max(0.03, baseline * 0.20)
    pair_left, pair_right = [], []
    for index, timestamp in enumerate(visual.times):
        insertion = int(np.searchsorted(visual.times, timestamp + baseline))
        candidates = [item for item in (insertion - 1, insertion)
                      if index < item < len(visual.times)]
        if not candidates:
            continue
        other = min(candidates, key=lambda item: abs(visual.times[item] - timestamp - baseline))
        if abs(visual.times[other] - timestamp - baseline) <= tolerance:
            pair_left.append(index)
            pair_right.append(other)
    left = np.asarray(pair_left, dtype=int)
    right = np.asarray(pair_right, dtype=int)
    interval_dt = visual.times[right] - visual.times[left]
    visual_linear = np.linalg.norm(
        visual.positions[right] - visual.positions[left], axis=1
    ) / interval_dt
    visual_angular = (
        visual.rotations[left].inv() * visual.rotations[right]
    ).magnitude() / interval_dt

    def score(offset: float) -> tuple[float, float, float]:
        count = len(left)
        query = np.r_[visual.times[left] + offset, visual.times[right] + offset]
        sampled, valid_query = interpolate_motive(motive, query, max_gap_s=0.05)
        valid = valid_query[:count] & valid_query[count:]
        if valid.sum() < 100:
            return -math.inf, -math.inf, -math.inf
        mocap_linear = np.linalg.norm(
            sampled.positions[count:][valid] - sampled.positions[:count][valid], axis=1
        ) / interval_dt[valid]
        mocap_angular = (
            sampled.rotations[:count][valid].inv()
            * sampled.rotations[count:][valid]
        ).magnitude() / interval_dt[valid]
        linear = float(np.corrcoef(visual_linear[valid], mocap_linear)[0, 1])
        angular = float(np.corrcoef(visual_angular[valid], mocap_angular)[0, 1])
        # Both channels must agree. Equal weighting prevents a stabilized video
        # (good translation, suppressed attitude) from passing synchronization.
        return (linear + angular) / 2.0, linear, angular

    coarse_offsets = np.arange(initial_offset - search_radius, initial_offset + search_radius + 0.005, 0.01)
    coarse = [(float(offset), *score(float(offset))) for offset in coarse_offsets]
    best_coarse = max(coarse, key=lambda item: item[1])
    if abs(best_coarse[0] - initial_offset) >= search_radius - 0.1:
        expanded_radius = max(2.0, search_radius * 2.0)
        coarse_offsets = np.arange(
            initial_offset - expanded_radius, initial_offset + expanded_radius + 0.005, 0.01
        )
        coarse = [(float(offset), *score(float(offset))) for offset in coarse_offsets]
        best_coarse = max(coarse, key=lambda item: item[1])
    fine_offsets = np.arange(best_coarse[0] - 0.02, best_coarse[0] + 0.0201, 0.001)
    fine = [(float(offset), *score(float(offset))) for offset in fine_offsets]
    best = max(fine, key=lambda item: item[1])
    near = [row[0] for row in fine if row[1] >= best[1] - 0.02]
    uncertainty = (max(near) - min(near)) / 2.0 if near else math.inf
    return best[0], best[1], uncertainty, {
        "linear_correlation": best[2], "angular_correlation": best[3],
    }, fine


def average_transform(values: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = Rotation.from_matrix(values[:, :3, :3]).mean().as_matrix()
    result[:3, 3] = np.median(values[:, :3, 3], axis=0)
    return result


def calibrate_extrinsics(world_body: np.ndarray, tag_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    if len(world_body) < 20:
        raise ValueError("not enough calibration poses")
    indices = np.unique(np.linspace(0, len(world_body) - 1, min(240, len(world_body))).round().astype(int))
    wb = world_body[indices]
    tc = tag_camera[indices]
    body_to_camera_candidates = []
    for method in (cv2.CALIB_HAND_EYE_TSAI, cv2.CALIB_HAND_EYE_PARK, cv2.CALIB_HAND_EYE_HORAUD):
        try:
            rotation, translation = cv2.calibrateHandEye(
                list(wb[:, :3, :3]), list(wb[:, :3, 3]),
                list(np.transpose(tc[:, :3, :3], (0, 2, 1))),
                list(-(np.transpose(tc[:, :3, :3], (0, 2, 1)) @ tc[:, :3, 3, None])[:, :, 0]),
                method=method,
            )
            candidate = np.eye(4)
            raw_rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
            # OpenCV can return a reflection for poorly excited hand-eye data.
            # A reflection is not a rigid transform and must never seed the
            # SE(3) optimizer.
            if np.linalg.det(raw_rotation) <= 0.0:
                continue
            u, _, vt = np.linalg.svd(raw_rotation)
            candidate[:3, :3] = u @ vt
            candidate[:3, 3] = np.asarray(translation).reshape(3)
            if np.isfinite(candidate).all():
                body_to_camera_candidates.append(candidate)
        except cv2.error:
            continue
    if not body_to_camera_candidates:
        body_to_camera_candidates = [np.eye(4)]

    best_initial = None
    for body_camera in body_to_camera_candidates:
        world_tag_samples = np.asarray([
            left @ body_camera @ np.linalg.inv(right) for left, right in zip(wb, tc)
        ])
        world_tag = average_transform(world_tag_samples)
        residuals = []
        for left, right in zip(wb, tc):
            error = np.linalg.inv(left @ body_camera) @ world_tag @ right
            residuals.append(np.r_[error[:3, 3], Rotation.from_matrix(error[:3, :3]).as_rotvec()])
        value = float(np.median(np.linalg.norm(np.asarray(residuals), axis=1)))
        if best_initial is None or value < best_initial[0]:
            best_initial = (value, body_camera, world_tag)
    assert best_initial is not None
    initial = np.r_[matrix_to_vector(best_initial[1]), matrix_to_vector(best_initial[2])]

    def residual(parameters: np.ndarray) -> np.ndarray:
        body_camera = vector_to_matrix(parameters[:6])
        world_tag = vector_to_matrix(parameters[6:12])
        output = []
        for left, right in zip(wb, tc):
            error = np.linalg.inv(left @ body_camera) @ world_tag @ right
            output.extend(error[:3, 3] / 0.02)
            output.extend(Rotation.from_matrix(error[:3, :3]).as_rotvec() / math.radians(2.0))
        return np.asarray(output)

    fit = least_squares(residual, initial, loss="huber", f_scale=1.0, max_nfev=1000)
    body_camera = vector_to_matrix(fit.x[:6])
    world_tag = vector_to_matrix(fit.x[6:12])
    return body_camera, world_tag, {
        "success": bool(fit.success), "cost": float(fit.cost),
        "optimality": float(fit.optimality), "calibration_samples": int(len(wb)),
    }


def trajectory_errors(
    world_body: np.ndarray, tag_camera: np.ndarray,
    body_camera: np.ndarray, world_tag: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    truth = world_body @ body_camera
    estimate = world_tag @ tag_camera
    position = np.linalg.norm(estimate[:, :3, 3] - truth[:, :3, 3], axis=1)
    relative_rotation = np.transpose(truth[:, :3, :3], (0, 2, 1)) @ estimate[:, :3, :3]
    orientation = np.degrees(Rotation.from_matrix(relative_rotation).magnitude())
    return position, orientation, truth, estimate


def relative_pose_errors(truth: np.ndarray, estimate: np.ndarray, times: np.ndarray, delta_s: float) -> tuple[np.ndarray, np.ndarray]:
    translation, rotation = [], []
    tolerance = max(0.011, delta_s * 0.03)
    for index, timestamp in enumerate(times):
        target = timestamp + delta_s
        right = int(np.searchsorted(times, target))
        candidates = [item for item in (right - 1, right) if index < item < len(times)]
        if not candidates:
            continue
        other = min(candidates, key=lambda item: abs(times[item] - target))
        if abs(times[other] - target) > tolerance:
            continue
        truth_delta = np.linalg.inv(truth[index]) @ truth[other]
        estimate_delta = np.linalg.inv(estimate[index]) @ estimate[other]
        error = np.linalg.inv(truth_delta) @ estimate_delta
        translation.append(float(np.linalg.norm(error[:3, 3])))
        rotation.append(float(np.degrees(Rotation.from_matrix(error[:3, :3]).magnitude())))
    return np.asarray(translation), np.asarray(rotation)


def bootstrap_rmse(values: np.ndarray, times: np.ndarray, iterations: int = 1000) -> dict[str, float]:
    if len(values) < 2:
        return {"low": math.nan, "high": math.nan}
    block_ids = np.floor((times - times.min()) / 1.0).astype(int)
    blocks = [np.flatnonzero(block_ids == block) for block in np.unique(block_ids)]
    rng = np.random.default_rng(42)
    sampled = []
    for _ in range(iterations):
        indices = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        sampled.append(float(np.sqrt(np.mean(values[indices] ** 2))))
    return {"low": float(np.percentile(sampled, 2.5)), "high": float(np.percentile(sampled, 97.5))}


def write_plot(path: Path, times: np.ndarray, truth: np.ndarray, estimate: np.ndarray, position: np.ndarray, orientation: np.ndarray, split_time: float) -> None:
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    axis = fig.add_subplot(grid[:, 0], projection="3d")
    axis.plot(*truth[:, :3, 3].T, label="OptiTrack camera truth", linewidth=2)
    axis.plot(*estimate[:, :3, 3].T, label="Insta360 AprilTag", linewidth=1.2)
    axis.set(xlabel="X [m]", ylabel="Y [m]", zlabel="Z [m]")
    axis.legend()
    p_axis = fig.add_subplot(grid[0, 1])
    p_axis.plot(times, 1000.0 * position)
    p_axis.axvline(split_time, color="#fbbf24", linestyle="--", label="test begins")
    p_axis.set_ylabel("Position error [mm]")
    p_axis.grid(alpha=0.25)
    p_axis.legend()
    r_axis = fig.add_subplot(grid[1, 1], sharex=p_axis)
    r_axis.plot(times, orientation)
    r_axis.axvline(split_time, color="#fbbf24", linestyle="--")
    r_axis.set(xlabel="Video time [s]", ylabel="Orientation error [deg]")
    r_axis.grid(alpha=0.25)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("visual_csv", type=Path)
    parser.add_argument("motive_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-time-offset", type=float, default=-3.852)
    parser.add_argument("--search-radius", type=float, default=1.0)
    parser.add_argument("--calibration-fraction", type=float, default=0.30)
    parser.add_argument("--min-tags", type=int, default=2)
    parser.add_argument("--min-test-samples", type=int, default=200)
    parser.add_argument(
        "--include-optical-flow", action="store_true",
        help="evaluate LK-tracked measurements in addition to direct decodes",
    )
    args = parser.parse_args()
    if not 0.1 <= args.calibration_fraction <= 0.6:
        raise SystemExit("calibration fraction must be in 0.1..0.6")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    accepted_sources = (
        ("", "direct", "optical_flow")
        if args.include_optical_flow else ("", "direct")
    )
    visual, visual_rows = load_visual(
        args.visual_csv, args.min_tags, accepted_sources
    )
    visual_quality, recovery_frames = visual_audit(
        visual_rows, args.min_tags, accepted_sources
    )
    motive = parse_motive(args.motive_csv)
    write_normalized_mocap(args.output_dir / "mocap_normalized.csv", motive)
    offset, correlation, uncertainty, correlation_components, offset_curve = estimate_time_offset(
        visual, motive, args.initial_time_offset, args.search_radius
    )
    shifted = visual.times + offset
    interpolated, mocap_ok = interpolate_motive(motive, shifted)
    visual_matched = PoseSeries(
        visual.times[mocap_ok], visual.positions[mocap_ok], visual.rotations[mocap_ok], visual.frames[mocap_ok]
    )
    mocap_matched = PoseSeries(
        shifted[mocap_ok], interpolated.positions[mocap_ok], interpolated.rotations[mocap_ok], visual.frames[mocap_ok]
    )
    if len(visual_matched.times) < 40:
        raise SystemExit("fewer than 40 synchronized visual/mocap samples; diagnostic fit is impossible")
    overlap_start, overlap_end = visual_matched.times.min(), visual_matched.times.max()
    split_time = overlap_start + args.calibration_fraction * (overlap_end - overlap_start)
    calibration = visual_matched.times <= split_time
    test = visual_matched.times > split_time
    if test.sum() < 20:
        raise SystemExit(f"only {int(test.sum())} held-out samples; diagnostic fit is impossible")
    tc = pose_matrices(visual_matched)
    wb = pose_matrices(mocap_matched)
    body_camera, world_tag, fit_info = calibrate_extrinsics(wb[calibration], tc[calibration])
    position_all, orientation_all, truth_all, estimate_all = trajectory_errors(wb, tc, body_camera, world_tag)
    position, orientation = position_all[test], orientation_all[test]
    truth, estimate = truth_all[test], estimate_all[test]
    test_times = visual_matched.times[test]
    rpe20_t, rpe20_r = relative_pose_errors(truth, estimate, test_times, 0.02)
    rpe1_t, rpe1_r = relative_pose_errors(truth, estimate, test_times, 1.0)
    truth_displacement = truth[-1, :3, 3] - truth[0, :3, 3]
    estimate_displacement = estimate[-1, :3, 3] - estimate[0, :3, 3]
    endpoint_drift = float(np.linalg.norm(estimate_displacement - truth_displacement))
    path_length = float(np.linalg.norm(np.diff(truth[:, :3, 3], axis=0), axis=1).sum())
    recovery_test = test & np.asarray([
        int(frame) in recovery_frames for frame in visual_matched.frames
    ])
    publishable = correlation >= 0.80 and uncertainty <= 0.020 and int(test.sum()) >= args.min_test_samples
    failed_requirements = []
    if correlation < 0.80:
        failed_requirements.append("combined linear/angular motion correlation below 0.80")
    if uncertainty > 0.020:
        failed_requirements.append("time-offset uncertainty above 20 ms")
    if int(test.sum()) < args.min_test_samples:
        failed_requirements.append("too few held-out direct multi-tag samples")

    report = {
        "publishable_accuracy": publishable,
        "result_status": "FORMAL_ACCURACY" if publishable else "DIAGNOSTIC_ONLY",
        "failed_requirements": failed_requirements,
        "inputs": {"visual_csv": str(args.visual_csv.resolve()), "motive_csv": str(args.motive_csv.resolve())},
        "motive": {
            "declared_frames": int(len(motive.frames)),
            "valid_raw_frames": int(motive.metadata["raw_valid_frames"]),
            "valid_after_quarantine": int(motive.valid.sum()),
            "longest_raw_missing_run_frames": int(motive.metadata["longest_raw_missing_run_frames"]),
            "quarantined_ranges": [list(value) for value in motive.quarantined_ranges],
        },
        "visual": visual_quality,
        "time_alignment": {
            "initial_offset_s": args.initial_time_offset, "optimized_offset_s": offset,
            "motion_correlation": correlation, "uncertainty_s": uncertainty,
            **correlation_components,
        },
        "split": {
            "calibration_fraction": args.calibration_fraction, "video_split_time_s": split_time,
            "calibration_samples": int(calibration.sum()), "test_samples": int(test.sum()),
        },
        "hand_eye": {
            **fit_info, "T_body_camera": body_camera.tolist(), "T_world_tagmap": world_tag.tolist(),
        },
        "position_ate_m": stats(position),
        "position_ate_rmse_95ci_m": bootstrap_rmse(position, test_times),
        "orientation_error_deg": stats(orientation),
        "orientation_rmse_95ci_deg": bootstrap_rmse(orientation, test_times),
        "rpe_20ms": {"translation_m": stats(rpe20_t), "rotation_deg": stats(rpe20_r)},
        "rpe_1s": {"translation_m": stats(rpe1_t), "rotation_deg": stats(rpe1_r)},
        "path_length_m": path_length, "endpoint_drift_m": endpoint_drift,
        "endpoint_drift_percent": 100.0 * endpoint_drift / path_length if path_length else 0.0,
        "recovery_error": {
            "count": int(recovery_test.sum()),
            "position_m": stats(position_all[recovery_test]),
            "orientation_deg": stats(orientation_all[recovery_test]),
        },
        "visual_direct_pose_samples": int(len(visual.times)),
        "requirements": {"min_correlation": 0.80, "max_sync_uncertainty_s": 0.020, "min_test_samples": args.min_test_samples},
    }
    (args.output_dir / "mocap_evaluation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (args.output_dir / "matched_errors.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "video_frame", "video_time_s", "mocap_time_s", "split", "measurement_status",
            "position_error_m", "orientation_error_deg",
            "truth_x_m", "truth_y_m", "truth_z_m", "truth_qx", "truth_qy", "truth_qz", "truth_qw",
            "estimate_x_m", "estimate_y_m", "estimate_z_m",
            "estimate_qx", "estimate_qy", "estimate_qz", "estimate_qw",
        ))
        truth_quaternions = Rotation.from_matrix(truth_all[:, :3, :3]).as_quat()
        estimate_quaternions = Rotation.from_matrix(estimate_all[:, :3, :3]).as_quat()
        for original in range(len(visual_matched.times)):
            split_label = "test" if test[original] else "calibration"
            status = "RECOVERED_DIRECT" if int(visual_matched.frames[original]) in recovery_frames else "MEASURED"
            writer.writerow((
                int(visual_matched.frames[original]), f"{visual_matched.times[original]:.6f}",
                f"{mocap_matched.times[original]:.6f}", split_label, status,
                f"{position_all[original]:.9f}", f"{orientation_all[original]:.6f}",
                *truth_all[original, :3, 3], *truth_quaternions[original],
                *estimate_all[original, :3, 3], *estimate_quaternions[original],
            ))
    with (args.output_dir / "time_offset_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("offset_s", "combined_correlation", "linear_correlation", "angular_correlation"))
        writer.writerows(offset_curve)
    write_plot(
        args.output_dir / "mocap_evaluation.png", test_times, truth, estimate,
        position, orientation, split_time,
    )
    print(json.dumps({
        "output": str(args.output_dir.resolve()), "publishable": publishable,
        "offset_s": offset, "correlation": correlation,
        "position_rmse_mm": 1000.0 * report["position_ate_m"]["rmse"],
        "orientation_rmse_deg": report["orientation_error_deg"]["rmse"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
