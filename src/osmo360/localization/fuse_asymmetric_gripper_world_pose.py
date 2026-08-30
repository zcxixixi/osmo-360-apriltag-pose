#!/usr/bin/env python3
"""Fuse a strong wall camera with an instance-tracked opposite BaseTag.

This is deliberately asymmetric: capture visibility, not the semantic word
``left`` or ``right``, decides which factor is usable.  The strong camera is
localized by the immutable 200 mm wall map.  Its observation of the physical
20 mm BaseTag supplies the weak gripper's position; target IMU supplies that
gripper's continuous attitude.  A same-ID screen copy is never allowed to
replace the instance track.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from osmo360.calibration.calibrate_basetag_reciprocal import Transform
from osmo360.datasets.vla_dataset_export import smooth_positions, smooth_rotations
from osmo360.localization.world_frames import compile_world_tag_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strong-camera-csv", type=Path, required=True)
    parser.add_argument("--weak-cross-base-csv", type=Path, required=True)
    parser.add_argument("--weak-instance-cache", type=Path, required=True)
    parser.add_argument("--weak-tag-id", type=int, required=True)
    parser.add_argument(
        "--weak-imu-csv", type=Path,
        help="deprecated compatibility option; attitude comes from the audited cross-pose CSV",
    )
    parser.add_argument("--hardware", type=Path, required=True)
    parser.add_argument("--world-map", type=Path, required=True)
    parser.add_argument("--weak-role", choices=("left", "right"), required=True)
    parser.add_argument("--strong-role", choices=("left", "right"), required=True)
    parser.add_argument("--maximum-interpolation-gap-s", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def transform_from_row(row: dict[str, str], prefix: str) -> Transform:
    return Transform(
        np.asarray([float(row[f"{prefix}_{axis}_m"]) for axis in "xyz"]),
        Rotation.from_quat([float(row[key]) for key in ("qx", "qy", "qz", "qw")]),
    )


def camera_to_base(hardware: dict, role: str) -> Transform:
    robot = hardware["robots"][role]
    tcp = robot["camera_to_eef_reference"]
    camera_tcp = Transform(
        np.asarray(tcp["translation_m"], dtype=float),
        Rotation.from_quat(tcp["quaternion_xyzw"]),
    )
    base_tcp = Transform(
        np.asarray(hardware["eef_reference"]["base_to_tcp_translation_m"], dtype=float),
        Rotation.identity(),
    )
    return camera_tcp.compose(base_tcp.inverse())


def read_camera(path: Path) -> tuple[np.ndarray, list[Transform], list[str]]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    accepted = {"valid", "tracked", "filtered"}
    rows = [row for row in rows if row["quality_status"] in accepted]
    times = np.asarray([float(row["timestamp"]) for row in rows])
    poses = [transform_from_row(row, "camera") for row in rows]
    states = [row["quality_status"] for row in rows]
    return times, poses, states


def read_base(path: Path) -> tuple[np.ndarray, np.ndarray, Rotation]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    times = np.asarray([float(row["timestamp"]) for row in rows])
    positions = np.asarray([
        [float(row[f"base_{axis}_m"]) for axis in "xyz"] for row in rows
    ])
    rotations = Rotation.from_quat([
        [float(row[key]) for key in ("qx", "qy", "qz", "qw")] for row in rows
    ])
    return times, positions, rotations


def nearest_distance(source: np.ndarray, query: np.ndarray) -> np.ndarray:
    where = np.searchsorted(source, query)
    lo = np.clip(where - 1, 0, len(source) - 1)
    hi = np.clip(where, 0, len(source) - 1)
    return np.minimum(np.abs(query - source[lo]), np.abs(query - source[hi]))


def audit_instance_cache(path: Path, tag_id: int) -> dict:
    """Require evidence that the requested physical Tag instance was locked.

    A plain detector cache is not sufficient: a monitor can display the same
    family/ID and silently replace the gripper-mounted Tag.  The optical-flow
    augmenter labels a locked physical instance as ``lk_instance_track``.
    """
    with np.load(path, allow_pickle=False) as cache:
        ids = np.asarray(cache["tag_id"], dtype=int)
        frames = np.asarray(cache["frame_index"], dtype=int)
        sources = np.asarray(cache["detection_source"]).astype(str)
    selected = ids == int(tag_id)
    if not np.any(selected):
        raise ValueError(f"instance cache has no Tag ID {tag_id}: {path}")
    selected_sources = sources[selected]
    tracked = selected_sources == "lk_instance_track"
    if not np.any(tracked):
        raise ValueError(
            f"Tag ID {tag_id} has no lk_instance_track evidence; refusing a "
            f"same-ID screen-copy ambiguity: {path}"
        )
    unique, counts = np.unique(selected_sources, return_counts=True)
    return {
        "path": str(path.resolve()),
        "tag_id": int(tag_id),
        "observation_count": int(selected.sum()),
        "frame_count": int(np.unique(frames[selected]).size),
        "instance_tracked_frame_count": int(np.unique(frames[selected][tracked]).size),
        "source_counts": {str(k): int(v) for k, v in zip(unique, counts)},
    }


def resample_bounded_base(
    times: np.ndarray,
    positions: np.ndarray,
    rotations: Rotation,
    query: np.ndarray,
    maximum_gap_s: float,
) -> tuple[np.ndarray, Rotation, np.ndarray]:
    """Resample an audited base trajectory without inventing attitude.

    The previous fusion discarded the cross-pose orientation and naively applied
    raw IMU deltas in the base frame.  That omitted the calibrated IMU/body/camera
    transform chain and produced a large, time-varying orientation regression.
    Keep the already audited world-frame rotations and fail closed across long
    observation gaps instead.
    """
    times = np.asarray(times, dtype=float)
    query = np.asarray(query, dtype=float)
    if len(times) < 2 or len(positions) != len(times) or len(rotations) != len(times):
        raise ValueError("base trajectory needs at least two matching pose samples")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("base trajectory timestamps must be strictly increasing")
    if maximum_gap_s <= 0.0:
        raise ValueError("maximum interpolation gap must be positive")

    where = np.searchsorted(times, query, side="left")
    lo = np.clip(where - 1, 0, len(times) - 1)
    hi = np.clip(where, 0, len(times) - 1)
    exact = np.isclose(query, times[hi], atol=1e-9, rtol=0.0)
    inside = (query >= times[0]) & (query <= times[-1])
    bracket_gap = times[hi] - times[lo]
    trusted = inside & (exact | (bracket_gap <= maximum_gap_s))

    sampled_position = np.column_stack([
        np.interp(query, times, positions[:, axis]) for axis in range(3)
    ])
    clipped_query = np.clip(query, times[0], times[-1])
    sampled_rotation = Slerp(times, rotations)(clipped_query)
    return sampled_position, sampled_rotation, trusted


def write_pose(path: Path, times: np.ndarray, poses: list[Transform],
               role: str, states: list[str], sources: list[str], child: str,
               map_hash: str, detected_ids: str) -> None:
    fields = [
        "frame", "timestamp", f"{child}_x_m", f"{child}_y_m", f"{child}_z_m",
        "qx", "qy", "qz", "qw", "parent_frame", "child_frame",
        "measurement_source", "quality_status",
        "direct_measurement", "tag_map_sha256", "detected_ids",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (time_s, pose, state, source) in enumerate(
            zip(times, poses, states, sources)
        ):
            quaternion = pose.r.as_quat()
            writer.writerow({
                "frame": index, "timestamp": f"{time_s:.9f}",
                f"{child}_x_m": pose.p[0], f"{child}_y_m": pose.p[1],
                f"{child}_z_m": pose.p[2],
                "qx": quaternion[0], "qy": quaternion[1],
                "qz": quaternion[2], "qw": quaternion[3],
                "parent_frame": "tag_map",
                "child_frame": "panorama_camera" if child == "camera" else f"{role}_base_link",
                "measurement_source": source, "quality_status": state,
                "direct_measurement": "true" if source == "raw_fisheye_unit_bearing_wall" else "false",
                "tag_map_sha256": map_hash, "detected_ids": detected_ids,
            })


def main() -> int:
    args = parse_args()
    if args.weak_role == args.strong_role:
        raise ValueError("weak and strong roles must differ")
    hardware = json.loads(args.hardware.read_text(encoding="utf-8"))
    world_map = compile_world_tag_map(args.world_map)
    if str(world_map.get("calibration_status", "")).upper() != "VERIFIED":
        raise ValueError("fusion output requires a VERIFIED world map")
    map_hash = world_map["tag_map_sha256"]
    detected_ids = " ".join(str(tag["id"]) for tag in world_map["tags"])
    instance_audit = audit_instance_cache(args.weak_instance_cache, args.weak_tag_id)
    weak_cb = camera_to_base(hardware, args.weak_role)
    strong_cb = camera_to_base(hardware, args.strong_role)

    strong_t, strong_camera, strong_states = read_camera(args.strong_camera_csv)
    weak_t, weak_position, weak_cross_rotation = read_base(args.weak_cross_base_csv)
    distance = nearest_distance(weak_t, strong_t)
    weak_resampled, weak_rotation, valid = resample_bounded_base(
        weak_t, weak_position, weak_cross_rotation, strong_t,
        args.maximum_interpolation_gap_s,
    )
    weak_filtered, weak_rejected, weak_filter_audit = smooth_positions(
        weak_resampled, strong_t
    )
    weak_rotation = smooth_rotations(weak_rotation, radius=2)
    weak_base = [
        Transform(position, rotation)
        for position, rotation in zip(weak_filtered, weak_rotation)
    ]
    strong_base = [pose.compose(strong_cb) for pose in strong_camera]
    strong_positions = np.asarray([pose.p for pose in strong_base])
    strong_filtered, strong_rejected, strong_filter_audit = smooth_positions(
        strong_positions, strong_t
    )
    strong_rotations = smooth_rotations(
        Rotation.concatenate([pose.r for pose in strong_base]), radius=2
    )
    strong_base = [
        Transform(position, rotation)
        for position, rotation in zip(strong_filtered, strong_rotations)
    ]

    weak_camera = [pose.compose(weak_cb.inverse()) for pose in weak_base]
    strong_camera_fused = [pose.compose(strong_cb.inverse()) for pose in strong_base]
    weak_states = ["tracked" if keep else "interpolated_untrusted" for keep in valid]
    weak_sources = [
        "direct_opposite_basetag_raw_fisheye"
        if keep else "untrusted_long_gap_interpolation"
        for keep in valid
    ]
    strong_sources = ["raw_fisheye_unit_bearing_wall"] * len(strong_t)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_pose(
        args.output_dir / f"{args.weak_role}_base_pose.csv", strong_t,
        weak_base, args.weak_role, weak_states, weak_sources, "base", map_hash, detected_ids,
    )
    write_pose(
        args.output_dir / f"{args.strong_role}_base_pose.csv", strong_t,
        strong_base, args.strong_role, strong_states, strong_sources, "base", map_hash, detected_ids,
    )
    write_pose(
        args.output_dir / f"{args.weak_role}_camera_pose.csv", strong_t,
        weak_camera, args.weak_role, weak_states, weak_sources, "camera", map_hash, detected_ids,
    )
    write_pose(
        args.output_dir / f"{args.strong_role}_camera_pose.csv", strong_t,
        strong_camera_fused, args.strong_role, strong_states, strong_sources, "camera", map_hash, detected_ids,
    )

    separation = np.linalg.norm(
        np.asarray([pose.p for pose in weak_base])
        - np.asarray([pose.p for pose in strong_base]), axis=1
    )
    map_audit_path = Path(str(world_map.get("calibration_audit", "")))
    if map_audit_path and not map_audit_path.is_absolute():
        map_audit_path = (args.world_map.parent / map_audit_path).resolve()
    map_audit = (
        json.loads(map_audit_path.read_text(encoding="utf-8"))
        if str(map_audit_path) and map_audit_path.is_file() else {}
    )
    holdout = hardware.get("holdout_gate", {})
    map_verified = bool(
        map_audit.get("status") == "VERIFIED"
        and map_audit.get("training_ready") is True
        and map_audit.get("sim3_used") is False
        and map_audit.get("gates", {}).get("translation_pass") is True
        and map_audit.get("gates", {}).get("rotation_pass") is True
    )
    hardware_verified = bool(
        str(hardware.get("calibration_status", "")).startswith("HOLDOUT_PASS")
        and all(
            float(holdout.get(role_key, {}).get("position_p95_mm", np.inf))
            <= float(holdout.get("maximum_position_p95_mm", 0.0))
            and float(holdout.get(role_key, {}).get("rotation_p95_deg", np.inf))
            <= float(holdout.get("maximum_rotation_p95_deg", 0.0))
            for role_key in ("physical_left_SGG", "physical_right_JCY")
        )
    )
    instance_verified = bool(
        instance_audit["instance_tracked_frame_count"] > 0
        and instance_audit["observation_count"] > 0
    )
    coverage_verified = bool(float(valid.mean()) >= 0.90)
    motion_verified = bool(
        weak_filter_audit["rejected_ratio"] <= 0.08
        and strong_filter_audit["rejected_ratio"] <= 0.08
        and weak_filter_audit["filtered_max_speed_mps"] <= 1.5
        and strong_filter_audit["filtered_max_speed_mps"] <= 1.5
    )
    training_ready = bool(
        map_verified and hardware_verified and instance_verified
        and coverage_verified and motion_verified
    )
    report = {
        "schema_version": "asymmetric-gripper-world-fusion/1.0",
        "status": "VERIFIED" if training_ready else "DIAGNOSTIC",
        "weak_role": args.weak_role,
        "strong_role": args.strong_role,
        "weak_position_source": "opposite physical 20mm BaseTag instance track",
        "weak_attitude_source": "audited cross-pose world rotation (bounded SLERP only)",
        "strong_pose_source": "raw-fisheye 200mm wall map + frozen camera_to_base",
        "screen_same_id_used": False,
        "physical_basetag_instance_audit": instance_audit,
        "world_map": str(args.world_map.resolve()),
        "world_map_sha256": map_hash,
        "world_map_audit": str(map_audit_path) if str(map_audit_path) else None,
        "contact_constraint_used": False,
        "synthetic_frames_used": False,
        "coverage": {
            "count": len(strong_t),
            "weak_trusted": int(valid.sum()),
            "weak_ratio": float(valid.mean()),
            "maximum_nearest_measurement_gap_s": float(distance.max()),
            "maximum_allowed_interpolation_gap_s": args.maximum_interpolation_gap_s,
            "untrusted_long_gap_frames": int((~valid).sum()),
        },
        "filters": {
            "weak_position": weak_filter_audit,
            "strong_position": strong_filter_audit,
            "weak_rejected_frames": np.flatnonzero(weak_rejected).tolist(),
            "strong_rejected_frames": np.flatnonzero(strong_rejected).tolist(),
        },
        "base_separation_m": {
            "min": float(separation.min()),
            "median": float(np.median(separation)),
            "p95": float(np.quantile(separation, 0.95)),
            "max": float(separation.max()),
        },
        "quality_gates": {
            "world_map_multicapture_holdout_verified": map_verified,
            "camera_to_tcp_holdout_verified": hardware_verified,
            "physical_basetag_instance_locked": instance_verified,
            "tracked_coverage_at_least_90pct": coverage_verified,
            "motion_limits_pass": motion_verified,
        },
        "training_ready": training_ready,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
