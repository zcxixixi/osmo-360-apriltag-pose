#!/usr/bin/env python3
"""Build a deterministic 60 fps dual-gripper animation timeline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation, Slerp

from render_trajectory_overlay_video import kalman_rts_filter


def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("aligned_csv", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("layout", type=Path)
    parser.add_argument("left_extrinsic", type=Path)
    parser.add_argument("right_extrinsic", type=Path)
    parser.add_argument("left_angles", type=Path)
    parser.add_argument("right_angles", type=Path)
    parser.add_argument("left_pose", type=Path)
    parser.add_argument("right_pose", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--start-s", type=float)
    parser.add_argument("--end-s", type=float)
    parser.add_argument("--swap-sides", action="store_true")
    parser.add_argument("--view1-imu", type=Path)
    parser.add_argument("--view2-imu", type=Path)
    parser.add_argument(
        "--attitude-mode", choices=("visual", "imu-full", "imu-level"), default="visual",
        help="imu-level keeps the gripper at its confirmed starting tilt and uses IMU heading only",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_extrinsic(path: Path) -> tuple[np.ndarray, Rotation]:
    data = json.loads(path.read_text(encoding="utf-8"))
    framed = data.get("camera_to_tcp")
    if not isinstance(framed, dict):
        raise ValueError(
            f"{path} contains only the deprecated camera->base fields; "
            "regenerate it with estimate_gripper_extrinsic.py before rendering TCP data"
        )
    if (
        framed.get("parent_frame") != "panorama_camera"
        or framed.get("child_frame") not in {"gripper_tcp", "left_tcp", "right_tcp"}
    ):
        raise ValueError(f"ambiguous camera_to_tcp frame direction in {path}")
    translation = np.asarray(framed["translation_m"], float)
    rotation = Rotation.from_quat(np.asarray(framed["quaternion_xyzw"], float))
    if translation.shape != (3,):
        raise ValueError(f"invalid extrinsic translation in {path}")
    return translation, rotation


def continuous_quaternions(quaternions: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternions, float).copy()
    for index in range(1, len(result)):
        if np.dot(result[index - 1], result[index]) < 0:
            result[index] *= -1
    return result


def smooth_quaternions(quaternions: np.ndarray, radius: int = 12) -> np.ndarray:
    """Symmetric quaternion-space smoothing without Euler-angle wrapping."""
    source = continuous_quaternions(quaternions)
    weights = np.exp(-0.5 * (np.arange(-radius, radius + 1) / max(radius / 2, 1)) ** 2)
    output = np.empty_like(source)
    for index in range(len(source)):
        lo, hi = max(0, index - radius), min(len(source), index + radius + 1)
        window = source[lo:hi].copy()
        reference = source[index]
        window[np.sum(window * reference, axis=1) < 0] *= -1
        local = weights[(lo - index + radius):(hi - index + radius)]
        value = np.sum(window * local[:, None], axis=0)
        output[index] = value / np.linalg.norm(value)
    return continuous_quaternions(output)


def suppress_position_spikes(positions: np.ndarray, window: int = 9) -> tuple[np.ndarray, int]:
    """Replace isolated PnP jumps while preserving sustained physical motion."""
    if len(positions) < window:
        return positions.copy(), 0
    baseline = median_filter(positions, size=(window, 1), mode="nearest")
    residual = np.linalg.norm(positions - baseline, axis=1)
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    threshold = max(0.030, median + 6.0 * 1.4826 * mad)
    rejected = residual > threshold
    # Do not erase a sustained segment: only short isolated runs are spikes.
    padded = np.pad(rejected.astype(np.int8), (1, 1))
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1)
    isolated = np.zeros(len(positions), dtype=bool)
    for start, end in zip(starts, ends):
        if end - start <= 3:
            isolated[start:end] = True
    good = ~isolated
    if good.sum() < 2:
        return positions.copy(), 0
    repaired = positions.copy()
    indices = np.arange(len(positions))
    for axis in range(3):
        repaired[isolated, axis] = np.interp(
            indices[isolated], indices[good], positions[good, axis]
        )
    return repaired, int(isolated.sum())


def load_track(path: Path, prefix: str, extrinsic_path: Path) -> tuple[np.ndarray, np.ndarray, Rotation]:
    rows = read_rows(path)
    times = np.asarray([float(row["left_timestamp_s"]) for row in rows])
    positions = np.asarray([[float(row[f"{prefix}_{axis}_m"]) for axis in "xyz"] for row in rows])
    rotations = Rotation.from_quat(np.asarray([
        [float(row[f"{prefix}_q{axis}"]) for axis in "xyzw"] for row in rows
    ]))
    translation, gripper_rotation = load_extrinsic(extrinsic_path)
    positions = positions + rotations.apply(translation)
    rotations = rotations * gripper_rotation
    positions, _ = suppress_position_spikes(positions)
    # Reuse the project's constant-velocity Kalman + RTS implementation for XYZ only.
    filtered = kalman_rts_filter(
        np.column_stack((times, positions, np.zeros((len(times), 3)))),
        measurement_noise=0.020,
        accel_noise=0.30,
        angle_noise=1.0,
        angular_accel_noise=1.0,
    )
    return times, filtered[:, 1:4], rotations


def rebase(
    positions: np.ndarray, rotations: Rotation, target: dict[str, list[float]],
) -> tuple[np.ndarray, Rotation]:
    target_position = np.asarray(target["translation_m"], float)
    target_rotation = Rotation.from_euler("xyz", target["rotation_rpy_deg"], degrees=True)
    relative_positions = rotations[0].inv().apply(positions - positions[0])
    return (
        target_position + target_rotation.apply(relative_positions),
        target_rotation * rotations[0].inv() * rotations,
    )


def rebase_shared_world(
    left_positions: np.ndarray, left_rotations: Rotation,
    right_positions: np.ndarray, right_rotations: Rotation,
    left_target: dict[str, list[float]], right_target: dict[str, list[float]],
) -> tuple[np.ndarray, Rotation, np.ndarray, Rotation]:
    """Place two tracks while preserving their common tag-map translation axes.

    Translation uses one shared world-to-animation rotation, so equal world
    motion cannot point in opposite directions merely because the grippers
    start with opposite local orientations. Attitude still receives a
    per-gripper start-pose offset.
    """
    left_target_p = np.asarray(left_target["translation_m"], float)
    right_target_p = np.asarray(right_target["translation_m"], float)
    left_target_r = Rotation.from_euler("xyz", left_target["rotation_rpy_deg"], degrees=True)
    right_target_r = Rotation.from_euler("xyz", right_target["rotation_rpy_deg"], degrees=True)
    # The printed AprilGrid is vertical: its +Y direction is physical up in
    # these captures and +Z points along the wall normal. Convert that shared
    # board frame to a conventional animation frame with +Z up.
    shared_world_rotation = Rotation.from_euler("x", 90.0, degrees=True)
    left_rebased_p = left_target_p + shared_world_rotation.apply(left_positions - left_positions[0])
    right_rebased_p = right_target_p + shared_world_rotation.apply(right_positions - right_positions[0])
    left_rebased_r = left_target_r * left_rotations[0].inv() * left_rotations
    right_rebased_r = right_target_r * right_rotations[0].inv() * right_rotations
    return left_rebased_p, left_rebased_r, right_rebased_p, right_rebased_r


def load_angles(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = read_rows(path)
    times = np.asarray([float(row["time_s"]) for row in rows])
    opening = np.asarray([float(row["opening_angle_deg"]) for row in rows])
    neutral = abs(np.degrees(np.arctan2(50.568, 63.276) - np.arctan2(-50.745, 63.134)))
    joints, direct = [], []
    for row, angle in zip(rows, opening):
        if row.get("joint1_deg") not in (None, ""):
            joints.append([float(row["joint1_deg"]), float(row["joint2_deg"])])
            direct.append(float(row.get("measured", "1")) >= 0.5)
        else:
            travel = (neutral - angle) / 2.0
            joints.append([-travel, travel])
            raw = row.get("raw_angle_deg", "nan")
            direct.append(np.isfinite(float(raw)))
    return times, opening, np.asarray(joints), np.asarray(direct, bool)


def load_pose_status(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = read_rows(path)
    times, states = [], []
    for row in rows:
        times.append(float(row["timestamp"]))
        source = row.get("measurement_source", "")
        quality = row.get("quality_status", "")
        states.append("MEASURED" if source == "direct" and quality == "valid" else
                      "FILTERED" if source == "optical_flow" and quality == "valid" else "LOST")
    return np.asarray(times), np.asarray(states, object)


def load_imu_track(path: Path) -> tuple[np.ndarray, Rotation]:
    """Load DJI body-to-world quaternions on their source-video clock."""
    rows = read_rows(path)
    info_path = path.with_name("source_info.json")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    fps = float(info["fps"])
    times = np.asarray([float(row["frame"]) / fps for row in rows])
    quaternions = np.asarray([
        [float(row[key]) for key in ("qx", "qy", "qz", "qw")] for row in rows
    ])
    return times, Rotation.from_quat(continuous_quaternions(quaternions))


def resample_imu_attitude(
    imu_times: np.ndarray,
    imu_rotations: Rotation,
    sample_times: np.ndarray,
    target: dict[str, list[float]],
    mode: str,
) -> Rotation:
    """Rebase DJI attitude onto the manually confirmed initial model pose.

    DJI quaternions rotate body vectors into a gravity-aligned world frame. In
    ``imu-level`` mode only their continuous world heading is retained. This is
    appropriate for captures where the operator states the gripper remained
    flat: translation stays fully 3-D, but planar PnP ambiguity cannot stand
    the rendered gripper on edge.
    """
    clipped = np.clip(sample_times, imu_times[0], imu_times[-1])
    sampled_q = Slerp(imu_times, imu_rotations)(clipped).as_quat()
    sampled = Rotation.from_quat(smooth_quaternions(sampled_q))
    target_rotation = Rotation.from_euler("xyz", target["rotation_rpy_deg"], degrees=True)
    if mode == "imu-full":
        return target_rotation * sampled[0].inv() * sampled
    headings = np.unwrap(sampled.as_euler("ZYX")[:, 0])
    heading_delta = headings - headings[0]
    world_yaw = Rotation.from_rotvec(
        np.column_stack((np.zeros(len(headings)), np.zeros(len(headings)), heading_delta))
    )
    return world_yaw * target_rotation


def pose_state(times: np.ndarray, states: np.ndarray, now: float) -> str:
    distances = np.abs(times - now)
    index = int(np.argmin(distances))
    if distances[index] > 0.50:
        return "LOST"
    return str(states[index]) if distances[index] <= 0.12 else "FILTERED"


def round_list(values: np.ndarray, digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in values]


def main() -> int:
    args = args_parser()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    layout = json.loads(args.layout.read_text(encoding="utf-8"))
    if args.swap_sides:
        left_t, left_p, left_r = load_track(args.aligned_csv, "right_raw", args.right_extrinsic)
        right_t, right_p, right_r = load_track(args.aligned_csv, "left", args.left_extrinsic)
    else:
        left_t, left_p, left_r = load_track(args.aligned_csv, "left", args.left_extrinsic)
        right_t, right_p, right_r = load_track(args.aligned_csv, "right_raw", args.right_extrinsic)
    available_start, available_end = max(left_t[0], right_t[0]), min(left_t[-1], right_t[-1])
    start = max(available_start, args.start_s) if args.start_s is not None else available_start
    end = min(available_end, args.end_s) if args.end_s is not None else available_end
    if end <= start:
        raise ValueError(f"invalid timeline interval {start:.3f}..{end:.3f}s")
    frame_times = np.arange(int(np.floor((end - start) * args.fps)) + 1) / args.fps + start

    def resample(times: np.ndarray, positions: np.ndarray, rotations: Rotation):
        sampled_p = np.column_stack([
            PchipInterpolator(times, positions[:, axis])(frame_times) for axis in range(3)
        ])
        sampled_q = Slerp(times, rotations)(np.clip(frame_times, times[0], times[-1])).as_quat()
        return sampled_p, Rotation.from_quat(smooth_quaternions(sampled_q))

    left_p, left_r = resample(left_t, left_p, left_r)
    right_p, right_r = resample(right_t, right_p, right_r)
    center = layout["grippers_in_center_frame"]
    left_p, left_r, right_p, right_r = rebase_shared_world(
        left_p, left_r, right_p, right_r, center["left"], center["right"]
    )

    offset = float(report["time_alignment"]["offset_s"])
    if args.attitude_mode != "visual":
        if args.view1_imu is None or args.view2_imu is None:
            raise ValueError(f"{args.attitude_mode} requires --view1-imu and --view2-imu")
        view1_imu_t, view1_imu_r = load_imu_track(args.view1_imu)
        view2_imu_t, view2_imu_r = load_imu_track(args.view2_imu)
        if args.swap_sides:
            left_r = resample_imu_attitude(
                view2_imu_t, view2_imu_r, frame_times + offset, center["left"], args.attitude_mode
            )
            right_r = resample_imu_attitude(
                view1_imu_t, view1_imu_r, frame_times, center["right"], args.attitude_mode
            )
        else:
            left_r = resample_imu_attitude(
                view1_imu_t, view1_imu_r, frame_times, center["left"], args.attitude_mode
            )
            right_r = resample_imu_attitude(
                view2_imu_t, view2_imu_r, frame_times + offset, center["right"], args.attitude_mode
            )
    if args.swap_sides:
        left_angle_t, left_opening, left_joints, left_direct = load_angles(args.right_angles)
        right_angle_t, right_opening, right_joints, right_direct = load_angles(args.left_angles)
        left_status_t, left_status = load_pose_status(args.right_pose)
        right_status_t, right_status = load_pose_status(args.left_pose)
        left_clock_offset, right_clock_offset = offset, 0.0
    else:
        left_angle_t, left_opening, left_joints, left_direct = load_angles(args.left_angles)
        right_angle_t, right_opening, right_joints, right_direct = load_angles(args.right_angles)
        left_status_t, left_status = load_pose_status(args.left_pose)
        right_status_t, right_status = load_pose_status(args.right_pose)
        left_clock_offset, right_clock_offset = 0.0, offset

    frames = []
    for index, now in enumerate(frame_times):
        relative_t = now - start
        left_sample_t = now + left_clock_offset
        right_sample_t = now + right_clock_offset
        left_angle_index = int(np.argmin(np.abs(left_angle_t - left_sample_t)))
        right_angle_index = int(np.argmin(np.abs(right_angle_t - right_sample_t)))
        frame = {"t": round(relative_t, 6)}
        for side, position, quaternion, opening, joints, angle_direct, status_t, statuses, status_now in (
            ("left", left_p[index], left_r[index].as_quat(),
             np.interp(left_sample_t, left_angle_t, left_opening),
             [np.interp(left_sample_t, left_angle_t, left_joints[:, j]) for j in range(2)],
             left_direct[left_angle_index], left_status_t, left_status, left_sample_t),
            ("right", right_p[index], right_r[index].as_quat(),
             np.interp(right_sample_t, right_angle_t, right_opening),
             [np.interp(right_sample_t, right_angle_t, right_joints[:, j]) for j in range(2)],
             right_direct[right_angle_index], right_status_t, right_status, right_sample_t),
        ):
            frame[side] = {
                "p": round_list(position),
                "q": round_list(quaternion),
                "opening": round(float(opening), 3),
                "joints": [round(float(value), 3) for value in joints],
                "pose_state": pose_state(status_t, statuses, status_now),
                "angle_state": "MEASURED" if angle_direct else "RECOVERED",
            }
        frames.append(frame)

    all_positions = np.vstack((left_p, right_p))
    low, high = np.percentile(all_positions, [1, 99], axis=0)
    payload = {
        "schema_version": "dual-gripper-animation/v1",
        "capture_pair_id": report["capture_pair_id"],
        "layout_calibration_id": layout["calibration_id"],
        "reference_frame": "manual_layout_center",
        "source_interval_s": {"start": round(float(start), 6), "end": round(float(end), 6)},
        "fps": args.fps,
        "duration_s": round(float(frame_times[-1] - start), 6),
        "sync": report["time_alignment"],
        "side_mapping": {
            "left": "view2" if args.swap_sides else "view1",
            "right": "view1" if args.swap_sides else "view2",
        },
        "coordinate_mapping": {
            "animation_x": "+tagmap_x",
            "animation_y": "-tagmap_z",
            "animation_z": "+tagmap_y",
        },
        "attitude": {
            "mode": args.attitude_mode,
            "source": "dji_imu_perframe" if args.attitude_mode != "visual" else "aprilgrid_pnp",
            "level_constraint": args.attitude_mode == "imu-level",
        },
        "bounds_m": {"low": round_list(low), "high": round_list(high)},
        "frames": frames,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")

    initial_errors = {}
    for side, positions, rotations in (("left", left_p, left_r), ("right", right_p, right_r)):
        target_p = np.asarray(center[side]["translation_m"])
        target_r = Rotation.from_euler("xyz", center[side]["rotation_rpy_deg"], degrees=True)
        initial_errors[side] = {
            "position_mm": float(np.linalg.norm(positions[0] - target_p) * 1000),
            "attitude_deg": float((target_r.inv() * rotations[0]).magnitude() * 180 / np.pi),
        }
    print(json.dumps({
        "output": str(args.output.resolve()), "frames": len(frames),
        "duration_s": payload["duration_s"], "initial_pose_error": initial_errors,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
