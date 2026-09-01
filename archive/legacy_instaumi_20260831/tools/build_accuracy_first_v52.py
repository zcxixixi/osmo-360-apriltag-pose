#!/usr/bin/env python3
"""Build unsmoothed v52 dual-gripper poses with fail-closed geometry gates."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from osmo360.calibration.calibrate_basetag_reciprocal import Transform
from osmo360.localization.fuse_asymmetric_gripper_world_pose import camera_to_base
from osmo360.rig_revision import load_rig_revision, sha256


QUATERNION_KEYS = ("qx", "qy", "qz", "qw")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rig-revision", type=Path, required=True)
    parser.add_argument("--left-pose", type=Path, required=True)
    parser.add_argument("--right-pose", type=Path, required=True)
    parser.add_argument("--right-instance-cache", type=Path, required=True)
    parser.add_argument("--display-pose-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def valid_camera_pose(row: dict[str, str] | None) -> Transform | None:
    if not row or row.get("quality_status") != "valid" or not row.get("camera_x_m"):
        return None
    return Transform(
        np.asarray([float(row[f"camera_{axis}_m"]) for axis in "xyz"]),
        Rotation.from_quat([float(row[key]) for key in QUATERNION_KEYS]),
    )


def detected_ids(row: dict[str, str]) -> set[int]:
    return {int(value) for value in row.get("detected_ids", "").split() if value}


def has_all_panel_groups(row: dict[str, str], groups: list[set[int]]) -> bool:
    ids = detected_ids(row)
    return all(bool(ids & group) for group in groups)


def nearest_row(
    rows: list[dict[str, str]], times: np.ndarray, query: float, maximum_delta_s: float
) -> dict[str, str] | None:
    if not len(times):
        return None
    index = int(np.argmin(np.abs(times - query)))
    return rows[index] if abs(float(times[index]) - query) <= maximum_delta_s else None


def pose_source_serial(path: Path) -> str:
    summary_path = path.parent / "summary.json"
    if not summary_path.is_file():
        raise ValueError(f"pose summary is missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cache_path = Path(summary["observation_cache"])
    cache_summary = cache_path.with_suffix(".json")
    if not cache_summary.is_file():
        raise ValueError(f"observation cache summary is missing: {cache_summary}")
    return str(json.loads(cache_summary.read_text(encoding="utf-8"))["camera_serial"])


def verify_pose_map(path: Path, expected_map_sha256: str) -> None:
    _, rows = read_csv(path)
    hashes = {row.get("tag_map_sha256") for row in rows if row.get("tag_map_sha256")}
    if hashes != {expected_map_sha256}:
        raise ValueError(
            f"pose Tag-map hash mismatch for {path}: expected {expected_map_sha256}, got {hashes}"
        )


def load_locked_bearings(
    path: Path, expected_serial: str, expected_tag_id: int
) -> tuple[np.ndarray, np.ndarray]:
    summary_path = path.with_suffix(".json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("camera_serial") != expected_serial:
        raise ValueError("instance cache camera serial does not match physical right role")
    tracking = summary.get("tracking", {})
    if tracking.get("instance_track_ids") != [expected_tag_id]:
        raise ValueError("instance cache does not lock the physical left BaseTag")
    with np.load(path, allow_pickle=False) as cache:
        selected = (
            (cache["tag_id"].astype(int) == expected_tag_id)
            & (cache["detection_source"].astype(str) == "lk_instance_track")
        )
        times = cache["common_time_s"][selected].astype(float)
        rays = cache["rays_camera"][selected].mean(axis=1).astype(float)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    _, unique = np.unique(np.round(times, 9), return_index=True)
    return times[unique], rays[unique]


def gate_reason(
    left_row: dict[str, str], right_row: dict[str, str],
    *, panel_groups: list[set[int]], maximum_rmse_deg: float,
) -> str | None:
    if valid_camera_pose(left_row) is None:
        return "left_pose_unavailable"
    if valid_camera_pose(right_row) is None:
        return "right_pose_unavailable"
    if not has_all_panel_groups(left_row, panel_groups):
        return "left_single_panel_depth_untrusted"
    if not has_all_panel_groups(right_row, panel_groups):
        return "right_single_panel_depth_untrusted"
    if float(left_row["angular_rmse_deg"]) > maximum_rmse_deg:
        return "left_angular_rmse_rejected"
    if float(right_row["angular_rmse_deg"]) > maximum_rmse_deg:
        return "right_angular_rmse_rejected"
    return None


def transform_fields(prefix: str, pose: Transform) -> dict[str, float]:
    quaternion = pose.r.as_quat()
    return {
        f"{prefix}_x_m": float(pose.p[0]),
        f"{prefix}_y_m": float(pose.p[1]),
        f"{prefix}_z_m": float(pose.p[2]),
        f"{prefix}_qx": float(quaternion[0]),
        f"{prefix}_qy": float(quaternion[1]),
        f"{prefix}_qz": float(quaternion[2]),
        f"{prefix}_qw": float(quaternion[3]),
    }


def longest_run(rows: list[dict[str, Any]]) -> int:
    best = current = 0
    previous_time: float | None = None
    for row in rows:
        trusted = row["quality_status"] == "direct_trusted"
        contiguous = previous_time is not None and row["timestamp"] - previous_time <= 0.05
        current = current + 1 if trusted and contiguous else 1 if trusted else 0
        best = max(best, current)
        previous_time = row["timestamp"]
    return best


def reject_adjacent_jumps(
    rows: list[dict[str, Any]], maximum_step_m: float
) -> int:
    rejected: set[int] = set()
    candidate_indices = [
        index for index, row in enumerate(rows)
        if row["quality_status"] == "direct_trusted"
    ]
    for first_index, second_index in zip(candidate_indices, candidate_indices[1:]):
        first = rows[first_index]
        second = rows[second_index]
        if second["timestamp"] - first["timestamp"] > 0.05:
            continue
        left_step = np.linalg.norm(
            np.asarray([second[f"left_base_{axis}_m"] for axis in "xyz"])
            - np.asarray([first[f"left_base_{axis}_m"] for axis in "xyz"])
        )
        right_step = np.linalg.norm(
            np.asarray([second[f"right_base_{axis}_m"] for axis in "xyz"])
            - np.asarray([first[f"right_base_{axis}_m"] for axis in "xyz"])
        )
        if max(left_step, right_step) > maximum_step_m:
            rejected.update((first_index, second_index))
    for index in rejected:
        rows[index]["quality_status"] = "position_jump_rejected"
        rows[index]["trusted"] = False
    return len(rejected)


def main() -> int:
    args = parse_args()
    bundle = load_rig_revision(args.rig_revision)
    revision = bundle["revision"]
    hardware = bundle["hardware"]
    geometry = bundle["geometry"]
    policy = bundle["policy"]
    expected_map_sha256 = bundle["world_map"]["tag_map_sha256"]
    verify_pose_map(args.left_pose, expected_map_sha256)
    verify_pose_map(args.right_pose, expected_map_sha256)
    expected_serials = {
        role: hardware["robots"][role]["camera_serial"] for role in ("left", "right")
    }
    if pose_source_serial(args.left_pose) != expected_serials["left"]:
        raise ValueError("left pose source serial does not match rig revision")
    if pose_source_serial(args.right_pose) != expected_serials["right"]:
        raise ValueError("right pose source serial does not match rig revision")

    _, left_rows = read_csv(args.left_pose)
    _, right_rows = read_csv(args.right_pose)
    left_rows = sorted(left_rows, key=lambda row: float(row["timestamp"]))
    right_rows = sorted(right_rows, key=lambda row: float(row["timestamp"]))
    right_times = np.asarray([float(row["timestamp"]) for row in right_rows])
    bearing_times, bearing_rays = load_locked_bearings(
        args.right_instance_cache,
        expected_serials["right"],
        int(hardware["robots"]["left"]["base_tag_id"]),
    )
    panel_groups = [set(map(int, group)) for group in policy["wall_panel_id_groups"]]
    left_camera_base = camera_to_base(hardware, "left")
    right_camera_base = camera_to_base(hardware, "right")
    base_tag = geometry["base_to_tag"]
    base_tag_transform = Transform(
        np.asarray(base_tag["translation_m"], dtype=float),
        Rotation.from_quat(base_tag["quaternion_xyzw"]),
    )
    base_tcp = geometry["base_to_tcp"]
    base_tcp_transform = Transform(
        np.asarray(base_tcp["translation_m"], dtype=float),
        Rotation.from_quat(base_tcp["quaternion_xyzw"]),
    )

    rows: list[dict[str, Any]] = []
    pre_gate_bearing_errors: list[float] = []
    for left_row in left_rows:
        timestamp = float(left_row["timestamp"])
        right_row = nearest_row(right_rows, right_times, timestamp, 0.018)
        if right_row is None:
            continue
        row: dict[str, Any] = {
            "frame": int(left_row["frame"]),
            "timestamp": timestamp,
            "parent_frame": bundle["world_map"]["world_frame"],
            "pose_reference": "base_link",
            "action_reference": "tcp",
            "revision_id": revision["revision_id"],
            "revision_sha256": bundle["revision_sha256"],
            "left_detected_ids": left_row.get("detected_ids", ""),
            "right_detected_ids": right_row.get("detected_ids", ""),
            "left_angular_rmse_deg": left_row.get("angular_rmse_deg", ""),
            "right_angular_rmse_deg": right_row.get("angular_rmse_deg", ""),
            "cross_bearing_error_deg": "",
            "trusted": False,
        }
        reason = gate_reason(
            left_row, right_row,
            panel_groups=panel_groups,
            maximum_rmse_deg=float(policy["maximum_pose_angular_rmse_deg"]),
        )
        left_camera = valid_camera_pose(left_row)
        right_camera = valid_camera_pose(right_row)
        if left_camera is not None and right_camera is not None:
            left_base = left_camera.compose(left_camera_base)
            right_base = right_camera.compose(right_camera_base)
            left_tcp = left_base.compose(base_tcp_transform)
            right_tcp = right_base.compose(base_tcp_transform)
            row.update(transform_fields("left_base", left_base))
            row.update(transform_fields("right_base", right_base))
            row.update(transform_fields("left_tcp", left_tcp))
            row.update(transform_fields("right_tcp", right_tcp))
            row["raw_tcp_separation_m"] = float(np.linalg.norm(right_tcp.p - left_tcp.p))
            if reason is None:
                bearing_index = int(np.argmin(np.abs(bearing_times - timestamp)))
                if abs(float(bearing_times[bearing_index]) - timestamp) > 0.018:
                    reason = "cross_bearing_unavailable"
                else:
                    world_left_tag = left_base.compose(base_tag_transform)
                    predicted = right_camera.r.inv().apply(world_left_tag.p - right_camera.p)
                    predicted /= np.linalg.norm(predicted)
                    error = float(np.degrees(np.arccos(np.clip(
                        predicted @ bearing_rays[bearing_index], -1.0, 1.0
                    ))))
                    row["cross_bearing_error_deg"] = error
                    pre_gate_bearing_errors.append(error)
                    if error > float(policy["maximum_cross_bearing_error_deg"]):
                        reason = "cross_bearing_rejected"
        if reason is None:
            row["quality_status"] = "direct_trusted"
            row["trusted"] = True
        else:
            row["quality_status"] = reason
        rows.append(row)

    jump_rejected = reject_adjacent_jumps(
        rows, float(policy["maximum_trusted_position_step_m"])
    )
    trusted_rows = [row for row in rows if row["quality_status"] == "direct_trusted"]
    trusted_bearing = np.asarray([
        float(row["cross_bearing_error_deg"]) for row in trusted_rows
    ])
    reasons = Counter(row["quality_status"] for row in rows)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fields = [
        "frame", "timestamp", "parent_frame", "pose_reference", "action_reference",
        "revision_id", "revision_sha256", "quality_status", "trusted",
        "left_detected_ids", "right_detected_ids", "left_angular_rmse_deg",
        "right_angular_rmse_deg", "cross_bearing_error_deg", "raw_tcp_separation_m",
    ] + [
        f"{prefix}_{key}" for prefix in ("left_base", "right_base", "left_tcp", "right_tcp")
        for key in ("x_m", "y_m", "z_m", "qx", "qy", "qz", "qw")
    ]
    trajectory_path = output / "accuracy_first_raw_trajectory.csv"
    with trajectory_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    display_paths: dict[str, str] | None = None
    if args.display_pose_dir:
        display_paths = {
            role: str((args.display_pose_dir.resolve() / f"{role}_base_pose.csv"))
            for role in ("left", "right")
        }
        for path in map(Path, display_paths.values()):
            if not path.is_file():
                raise ValueError(f"display-only pose is missing: {path}")
    report = {
        "schema_version": "accuracy-first-dual-gripper-v52/1.0",
        "status": "DIAGNOSTIC_NOT_TRAINING_READY",
        "training_ready": False,
        "revision_id": revision["revision_id"],
        "revision_sha256": bundle["revision_sha256"],
        "inputs": {
            "left_pose": str(args.left_pose.resolve()),
            "left_pose_sha256": sha256(args.left_pose),
            "right_pose": str(args.right_pose.resolve()),
            "right_pose_sha256": sha256(args.right_pose),
            "right_instance_cache": str(args.right_instance_cache.resolve()),
            "right_instance_cache_sha256": sha256(args.right_instance_cache),
        },
        "raw_metric_trajectory": str(trajectory_path),
        "display_only_poses": display_paths,
        "metric_smoothing_used": False,
        "interpolation_used": False,
        "dual_fisheye_position_fill_used": False,
        "hidden_contact_constraint_used": False,
        "counts": {
            "aligned_frames": len(rows),
            "trusted_frames": len(trusted_rows),
            "trusted_ratio": len(trusted_rows) / len(rows) if rows else 0.0,
            "jump_rejected_frames": jump_rejected,
            "quality_status": dict(sorted(reasons.items())),
        },
        "longest_trusted_run_frames": longest_run(rows),
        "longest_trusted_run_s": longest_run(rows) / 30.0,
        "cross_bearing_error_deg": {
            "pre_gate_median": float(np.median(pre_gate_bearing_errors)) if pre_gate_bearing_errors else None,
            "pre_gate_p95": float(np.quantile(pre_gate_bearing_errors, 0.95)) if pre_gate_bearing_errors else None,
            "trusted_median": float(np.median(trusted_bearing)) if len(trusted_bearing) else None,
            "trusted_p95": float(np.quantile(trusted_bearing, 0.95)) if len(trusted_bearing) else None,
            "trusted_max": float(np.max(trusted_bearing)) if len(trusted_bearing) else None,
        },
        "policy": policy,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
