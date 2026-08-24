#!/usr/bin/env python3
"""Build an audited UMI/VLA episode from synchronized camera trajectories.

The exporter intentionally permits a DRAFT package while physical calibration is
pending.  It only writes the UMI-compatible replay buffer when all required
hardware calibration and quality gates pass (or --allow-unready is explicit).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import py360convert
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation, Slerp

from world_frames import RigidTransform, compile_world_tag_map


SCHEMA_VERSION = "vla-episode/1.1"


def _number(row: dict[str, str], names: tuple[str, ...], default: float = math.nan) -> float:
    for name in names:
        try:
            value = float(row.get(name, ""))
            if math.isfinite(value):
                return value
        except (TypeError, ValueError):
            pass
    return default


def _resolve(base: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _rot6d(rotations: Rotation) -> np.ndarray:
    matrices = rotations.as_matrix()
    return matrices[:, :, :2].transpose(0, 2, 1).reshape(-1, 6).astype(np.float32)


def _longest_false_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in mask:
        current = 0 if value else current + 1
        longest = max(longest, current)
    return longest


def _true_runs(mask: np.ndarray, minimum_length: int) -> list[tuple[int, int]]:
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends)
            if end - start >= minimum_length]


@dataclass
class PoseSeries:
    time: np.ndarray
    position: np.ndarray
    rotation: Rotation
    direct: np.ndarray
    tracked: np.ndarray
    rmse_px: np.ndarray
    parent_frame: str = ""
    child_frame: str = ""
    tag_map_sha256: str = ""
    detected_ids: frozenset[int] = frozenset()


def load_pose_csv(path: Path) -> PoseSeries:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        raise ValueError(f"pose CSV needs at least two rows: {path}")

    times, positions, quaternions, direct, tracked, rmses = [], [], [], [], [], []
    parent_frames: set[str] = set()
    child_frames: set[str] = set()
    tag_map_hashes: set[str] = set()
    detected_ids: set[int] = set()
    for row in rows:
        t = _number(row, ("timestamp_s", "timestamp", "time_s", "time"))
        p = [
            _number(row, ("optimized_x_m", "camera_x_m", "x_m", "x")),
            _number(row, ("optimized_y_m", "camera_y_m", "y_m", "y")),
            _number(row, ("optimized_z_m", "camera_z_m", "z_m", "z")),
        ]
        q = [
            _number(row, ("optimized_qx", "qx")),
            _number(row, ("optimized_qy", "qy")),
            _number(row, ("optimized_qz", "qz")),
            _number(row, ("optimized_qw", "qw")),
        ]
        if not np.all(np.isfinite(q)):
            euler = [
                _number(row, ("optimized_roll_deg", "roll_deg", "roll")),
                _number(row, ("optimized_pitch_deg", "pitch_deg", "pitch")),
                _number(row, ("optimized_yaw_deg", "yaw_deg", "yaw")),
            ]
            if not np.all(np.isfinite(euler)):
                continue
            q = Rotation.from_euler("xyz", euler, degrees=True).as_quat().tolist()
        if not math.isfinite(t) or not np.all(np.isfinite(p)):
            continue
        source = row.get("measurement_source", "").strip().lower()
        source_kind = source.rsplit(":", 1)[-1]
        source_is_trusted = not source.startswith("secondary_map:")
        quality = row.get("quality_status", row.get("state", "valid")).strip().lower()
        explicit = row.get("direct_measurement", "").strip().lower()
        is_direct = explicit in {"1", "true", "yes"} if explicit else source_kind in {"", "direct", "measured"}
        is_direct = is_direct and source_is_trusted
        is_direct = is_direct and quality not in {"invalid", "lost", "searching", "predicted"}
        is_tracked = quality in {"valid", "tracked", "filtered"} and source_kind in {
            "", "direct", "measured", "optical_flow", "flow"
        } and source_is_trusted
        parent = row.get("parent_frame", "").strip()
        child = row.get("child_frame", "").strip()
        tag_hash = row.get("tag_map_sha256", "").strip()
        if parent:
            parent_frames.add(parent)
        if child:
            child_frames.add(child)
        if tag_hash:
            tag_map_hashes.add(tag_hash)
        for value in row.get("detected_ids", "").replace(",", " ").split():
            try:
                detected_ids.add(int(value))
            except ValueError:
                pass
        times.append(t); positions.append(p); quaternions.append(q)
        direct.append(is_direct); tracked.append(is_tracked)
        rmses.append(_number(row, ("reprojection_rmse_px",)))

    if len(parent_frames) > 1 or len(child_frames) > 1 or len(tag_map_hashes) > 1:
        raise ValueError(f"pose CSV mixes coordinate-frame metadata: {path}")

    order = np.argsort(times)
    time = np.asarray(times, dtype=float)[order]
    keep = np.concatenate(([True], np.diff(time) > 1e-8))
    return PoseSeries(
        time=time[keep],
        position=np.asarray(positions, dtype=float)[order][keep],
        rotation=Rotation.from_quat(np.asarray(quaternions, dtype=float)[order][keep]),
        direct=np.asarray(direct, dtype=bool)[order][keep],
        tracked=np.asarray(tracked, dtype=bool)[order][keep],
        rmse_px=np.asarray(rmses, dtype=float)[order][keep],
        parent_frame=next(iter(parent_frames), ""),
        child_frame=next(iter(child_frames), ""),
        tag_map_sha256=next(iter(tag_map_hashes), ""),
        detected_ids=frozenset(detected_ids),
    )


def resample_pose(series: PoseSeries, query_time: np.ndarray) -> tuple[np.ndarray, Rotation, np.ndarray, np.ndarray]:
    if query_time[0] < series.time[0] or query_time[-1] > series.time[-1]:
        raise ValueError("episode time is outside trajectory coverage")
    position = np.column_stack([
        np.interp(query_time, series.time, series.position[:, axis]) for axis in range(3)
    ])
    rotation = Slerp(series.time, series.rotation)(query_time)
    right = np.searchsorted(series.time, query_time, side="left")
    right = np.clip(right, 0, len(series.time) - 1)
    left = np.clip(right - 1, 0, len(series.time) - 1)
    nearest = np.where(
        np.abs(series.time[right] - query_time) < np.abs(series.time[left] - query_time), right, left
    )
    typical_dt = float(np.median(np.diff(series.time)))
    direct = series.direct[nearest] & (np.abs(series.time[nearest] - query_time) <= typical_dt * 0.75 + 1e-6)
    tracked = series.tracked[nearest] & (np.abs(series.time[nearest] - query_time) <= typical_dt * 0.75 + 1e-6)
    return position, rotation, direct, tracked


def suppress_position_spikes(positions: np.ndarray, window: int = 9) -> tuple[np.ndarray, np.ndarray]:
    """Repair short PnP excursions without erasing sustained physical motion."""
    source = np.asarray(positions, dtype=float)
    rejected = np.zeros(len(source), dtype=bool)
    if len(source) < window:
        return source.copy(), rejected
    baseline = median_filter(source, size=(window, 1), mode="nearest")
    residual = np.linalg.norm(source - baseline, axis=1)
    center = float(np.median(residual))
    mad = float(np.median(np.abs(residual - center)))
    candidates = residual > max(0.030, center + 6.0 * 1.4826 * mad)
    padded = np.pad(candidates.astype(np.int8), (1, 1))
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1)
    for start, end in zip(starts, ends):
        if end - start <= 3:
            rejected[start:end] = True
    good = ~rejected
    if good.sum() >= 2:
        indices = np.arange(len(source))
        repaired = source.copy()
        for axis in range(3):
            repaired[rejected, axis] = np.interp(indices[rejected], indices[good], source[good, axis])
        return repaired, rejected
    return source.copy(), np.zeros(len(source), dtype=bool)


def kalman_rts_scalar(measurements: np.ndarray, dt: float, acceleration_sigma: float = 0.5,
                      measurement_sigma: float = 0.015) -> np.ndarray:
    """Constant-velocity Kalman filter with an offline RTS backward pass."""
    transition = np.asarray([[1.0, dt], [0.0, 1.0]])
    process = acceleration_sigma**2 * np.asarray([
        [dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2],
    ])
    observation = np.asarray([[1.0, 0.0]])
    measurement_var = measurement_sigma**2
    count = len(measurements)
    filtered_state = np.zeros((count, 2)); filtered_covariance = np.zeros((count, 2, 2))
    predicted_state = np.zeros((count, 2)); predicted_covariance = np.zeros((count, 2, 2))
    state = np.asarray([float(measurements[0]), 0.0]); covariance = np.diag([measurement_var, 1.0])
    for index, measurement in enumerate(measurements):
        if index:
            state = transition @ state
            covariance = transition @ covariance @ transition.T + process
        predicted_state[index] = state; predicted_covariance[index] = covariance
        innovation = float(measurement) - (observation @ state).item()
        innovation_covariance = (observation @ covariance @ observation.T).item() + measurement_var
        gain = covariance @ observation.T / innovation_covariance
        state = state + gain[:, 0] * innovation
        covariance = (np.eye(2) - gain @ observation) @ covariance
        filtered_state[index] = state; filtered_covariance[index] = covariance
    smoothed = filtered_state.copy()
    for index in range(count - 2, -1, -1):
        gain = filtered_covariance[index] @ transition.T @ np.linalg.inv(predicted_covariance[index + 1])
        smoothed[index] += gain @ (smoothed[index + 1] - predicted_state[index + 1])
    return smoothed[:, 0]


def smooth_positions(positions: np.ndarray, timeline: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    repaired, rejected = suppress_position_spikes(positions)
    dt = float(np.median(np.diff(timeline)))
    filtered = np.column_stack([
        kalman_rts_scalar(repaired[:, axis], dt) for axis in range(3)
    ])
    raw_speed = np.linalg.norm(np.diff(positions, axis=0), axis=1) / dt
    speed = np.linalg.norm(np.diff(filtered, axis=0), axis=1) / dt
    acceleration = np.linalg.norm(np.diff(filtered, n=2, axis=0), axis=1) / (dt * dt)
    audit = {
        "rejected_frames": int(rejected.sum()),
        "rejected_ratio": float(rejected.mean()),
        "raw_max_speed_mps": float(raw_speed.max(initial=0.0)),
        "filtered_max_speed_mps": float(speed.max(initial=0.0)),
        "filtered_p99_acceleration_mps2": float(np.quantile(acceleration, 0.99)) if len(acceleration) else 0.0,
    }
    return filtered, rejected, audit


def smooth_rotations(rotations: Rotation, radius: int = 2) -> Rotation:
    source = rotations.as_quat().copy()
    for index in range(1, len(source)):
        if np.dot(source[index - 1], source[index]) < 0:
            source[index] *= -1
    result = np.empty_like(source)
    for index in range(len(source)):
        lo, hi = max(0, index - radius), min(len(source), index + radius + 1)
        window = source[lo:hi].copy(); reference = source[index]
        window[np.sum(window * reference, axis=1) < 0] *= -1
        weights = np.exp(-0.5 * ((np.arange(lo, hi) - index) / max(radius / 1.5, 1.0)) ** 2)
        value = np.sum(window * weights[:, None], axis=0)
        result[index] = value / np.linalg.norm(value)
    return Rotation.from_quat(result)


def _compose_hardware_camera_to_tcp(robot: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve the physical camera→base→TCP chain and reject conflicting aliases."""
    direct = robot.get("camera_to_tcp")
    camera_base = robot.get("camera_to_base")
    base_tcp = robot.get("base_to_tcp")
    if camera_base is None and base_tcp is None:
        return direct
    if camera_base is None or base_tcp is None:
        raise ValueError("camera_to_base and base_to_tcp must be supplied together")
    first = RigidTransform.from_dict(camera_base)
    second = RigidTransform.from_dict(base_tcp)
    composed = first.compose(second)
    if first.parent_frame != "panorama_camera" or first.child_frame != "base_link":
        raise ValueError("hardware chain must start panorama_camera→base_link")
    if second.parent_frame != "base_link" or not second.child_frame.endswith("tcp"):
        raise ValueError("hardware chain must end base_link→*_tcp")
    result = composed.to_dict()
    if direct is not None:
        alias = RigidTransform.from_dict(direct)
        translation_error = np.linalg.norm(alias.translation_m - composed.translation_m)
        angle_error = (alias.rotation.inv() * composed.rotation).magnitude()
        if translation_error > 1e-6 or angle_error > 1e-6:
            raise ValueError("camera_to_tcp conflicts with camera_to_base @ base_to_tcp")
    return result


def apply_camera_to_tcp(position: np.ndarray, rotation: Rotation,
                        calibration: dict[str, Any] | None) -> tuple[np.ndarray, Rotation]:
    if not calibration:
        return position.copy(), rotation
    transform = RigidTransform.from_dict(calibration)
    if transform.parent_frame != "panorama_camera" or not transform.child_frame.endswith("tcp"):
        raise ValueError("expected explicit panorama_camera→*_tcp transform")
    return (
        position + rotation.apply(np.broadcast_to(transform.translation_m, position.shape)),
        rotation * transform.rotation,
    )


def load_gripper(path: Path | None, query_time: np.ndarray,
                 mapping: dict[str, Any] | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if path is None:
        nan = np.full(len(query_time), np.nan, dtype=np.float32)
        return nan, nan.copy(), np.zeros(len(query_time), dtype=bool)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    times = np.asarray([_number(row, ("time_s", "timestamp_s", "timestamp")) for row in rows])
    angles = np.asarray([_number(row, ("opening_angle_deg", "angle_deg")) for row in rows])
    measured = np.asarray([
        (row.get("measured", "").strip().lower() in {"1", "true", "yes"})
        if row.get("measured", "").strip()
        else _number(row, ("confidence",), 0.0) > 0.12
        for row in rows
    ])
    valid = np.isfinite(times) & np.isfinite(angles)
    times, angles, measured = times[valid], angles[valid], measured[valid]
    if len(times) < 2:
        raise ValueError(f"gripper CSV needs two valid angle samples: {path}")
    angle = np.interp(query_time, times, angles).astype(np.float32)
    idx = np.clip(np.searchsorted(times, query_time), 0, len(times) - 1)
    angle_direct = measured[idx]
    width = np.full(len(query_time), np.nan, dtype=np.float32)
    if mapping:
        kind = mapping.get("type", "linear")
        if kind == "linear":
            width[:] = np.interp(
                angle,
                [mapping["closed_angle_deg"], mapping["open_angle_deg"]],
                [mapping["closed_width_m"], mapping["open_width_m"]],
            )
        elif kind == "piecewise_linear":
            width[:] = np.interp(angle, mapping["angle_deg"], mapping["width_m"])
        else:
            raise ValueError(f"unsupported gripper mapping: {kind}")
    return angle, width, angle_direct


def extract_rgb(video: Path, query_time: np.ndarray, output_hw: tuple[int, int],
                view: dict[str, Any]) -> np.ndarray:
    capture = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    requested = np.rint(query_time * fps).astype(int)
    frames: list[np.ndarray] = []
    last_index = -1
    frame = None
    for wanted in requested:
        if wanted < last_index or wanted - last_index > max(5, int(fps)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(wanted))
            ok, frame = capture.read()
            last_index = int(wanted)
        else:
            ok = True
            while last_index < wanted:
                ok, frame = capture.read()
                last_index += 1
                if not ok:
                    break
        if not ok or frame is None:
            capture.release()
            raise RuntimeError(f"failed to decode frame {wanted} from {video}")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        perspective = py360convert.e2p(
            rgb, fov_deg=float(view.get("fov_deg", 110.0)),
            u_deg=float(view.get("yaw_deg", 0.0)), v_deg=float(view.get("pitch_deg", 0.0)),
            out_hw=output_hw, mode="bilinear",
        )
        frames.append(np.asarray(perspective, dtype=np.uint8))
    capture.release()
    return np.stack(frames)


def _hardware_robot(hardware: dict[str, Any], name: str) -> dict[str, Any]:
    return hardware.get("robots", {}).get(name, {})


def _camera_serial_from_calibration(path: Path | None) -> str:
    """Return the physical camera serial recorded by a factory calibration file."""
    if path is None or not path.is_file():
        return ""
    value = _load_json(path).get("serial", "")
    return str(value).strip() if value is not None else ""


def build_episode(spec_path: Path, output_dir: Path, skip_rgb: bool = False,
                  allow_unready: bool = False,
                  hardware_override: Path | None = None) -> dict[str, Any]:
    spec_path = spec_path.resolve(); base = spec_path.parent
    spec = _load_json(spec_path)
    hardware_path = (
        hardware_override.resolve()
        if hardware_override is not None
        else _resolve(base, spec.get("hardware_config"))
    )
    hardware = _load_json(hardware_path) if hardware_path else {"calibration_status": "pending", "robots": {}}
    robots = spec.get("robots", [])
    if not robots:
        raise ValueError("episode spec has no robots")
    start, end = float(spec["start_s"]), float(spec["end_s"])
    hz = float(spec.get("frequency_hz", 20.0))
    if not 0 <= start < end or hz <= 0:
        raise ValueError("invalid episode time range or frequency")
    timeline = start + np.arange(int(math.floor((end - start) * hz)), dtype=float) / hz
    episode_id = spec.get("episode_id") or str(uuid.uuid4())
    arrays: dict[str, np.ndarray] = {"timestamp_s": (timeline - start).astype(np.float64)}
    checks: list[dict[str, Any]] = []
    per_robot: list[dict[str, Any]] = []
    coordinate = spec.get("coordinate_frame", {})
    world_mode = coordinate.get("mode") == "world"
    world_frame = str(coordinate.get("frame_id", "")).strip()
    world_map_path = _resolve(base, coordinate.get("tag_map"))
    world_map = compile_world_tag_map(world_map_path) if world_map_path else None
    expected_map_hash = world_map["tag_map_sha256"] if world_map else ""
    expected_ids = {
        int(tag["id"]) for tag in world_map.get("tags", [])
    } if world_map else set()
    calibration_status = str(
        world_map.get("calibration_status", "") if world_map else ""
    ).upper()
    calibration_final = calibration_status in {"CALIBRATED", "FROZEN", "VERIFIED"}
    if len(robots) > 1:
        checks.extend([
            {"name": "coordinate.world_mode", "pass": world_mode, "value": coordinate.get("mode")},
            {"name": "coordinate.world_frame_declared", "pass": bool(world_frame), "value": world_frame},
            {"name": "coordinate.global_tag_map_present", "pass": world_map is not None,
             "value": str(world_map_path) if world_map_path else None},
            {"name": "coordinate.global_tag_map_calibrated", "pass": calibration_final,
             "value": calibration_status or None},
        ])
        if world_map and world_frame:
            checks.append({
                "name": "coordinate.world_frame_matches_map",
                "pass": world_map.get("world_frame") == world_frame,
                "value": world_map.get("world_frame"),
            })
    world_positions: dict[str, np.ndarray] = {}

    for index, robot in enumerate(robots):
        name = robot.get("name", f"robot{index}")
        local_time = timeline + float(robot.get("source_time_offset_s", 0.0))
        pose_path = _resolve(base, robot.get("trajectory_csv"))
        if pose_path is None or not pose_path.is_file():
            raise ValueError(f"missing trajectory for {name}")
        pose_series = load_pose_csv(pose_path)
        if world_mode:
            checks.extend([
                {"name": f"{name}.pose_parent_frame_matches", "pass": pose_series.parent_frame == world_frame,
                 "value": pose_series.parent_frame or None},
                {"name": f"{name}.pose_child_frame_is_camera",
                 "pass": pose_series.child_frame == "panorama_camera",
                 "value": pose_series.child_frame or None},
                {"name": f"{name}.tag_map_hash_matches",
                 "pass": bool(expected_map_hash) and pose_series.tag_map_sha256 == expected_map_hash,
                 "value": pose_series.tag_map_sha256 or None},
                {"name": f"{name}.detected_ids_in_global_map",
                 "pass": bool(pose_series.detected_ids) and pose_series.detected_ids <= expected_ids,
                 "value": sorted(pose_series.detected_ids)},
            ])
        position, rotation, direct, tracked = resample_pose(pose_series, local_time)
        hw = _hardware_robot(hardware, name)
        serial_binding_declared = "camera_serial" in robot or "camera_calibration" in robot
        declared_serial = str(robot.get("camera_serial", "")).strip()
        hardware_serial = str(hw.get("camera_serial", "")).strip()
        calibration_path = _resolve(base, robot.get("camera_calibration"))
        actual_serial = _camera_serial_from_calibration(calibration_path)
        if serial_binding_declared:
            checks.extend([
                {"name": f"{name}.camera_serial_declared", "pass": bool(declared_serial),
                 "value": declared_serial or None},
                {"name": f"{name}.camera_calibration_present",
                 "pass": calibration_path is not None and calibration_path.is_file(),
                 "value": str(calibration_path) if calibration_path else None},
                {"name": f"{name}.camera_serial_matches_hardware",
                 "pass": bool(declared_serial and hardware_serial) and declared_serial == hardware_serial,
                 "value": {"episode": declared_serial or None, "hardware": hardware_serial or None}},
                {"name": f"{name}.camera_serial_matches_calibration",
                 "pass": bool(declared_serial and actual_serial) and declared_serial == actual_serial,
                 "value": {"episode": declared_serial or None, "calibration": actual_serial or None}},
                {"name": f"{name}.hardware_camera_serial_matches_calibration",
                 "pass": bool(hardware_serial and actual_serial) and hardware_serial == actual_serial,
                 "value": {"hardware": hardware_serial or None, "calibration": actual_serial or None}},
                {"name": f"{name}.camera_source_view_declared",
                 "pass": bool(str(hw.get("source_view", "")).strip()),
                 "value": str(hw.get("source_view", "")).strip() or None},
                {"name": f"{name}.camera_mount_revision_declared",
                 "pass": bool(str(hw.get("mount_revision", "")).strip()),
                 "value": str(hw.get("mount_revision", "")).strip() or None},
            ])
        extrinsic = _compose_hardware_camera_to_tcp(hw)
        position, rotation = apply_camera_to_tcp(position, rotation, extrinsic)
        raw_position = position.copy()
        position, rejected, motion_audit = smooth_positions(position, timeline)
        rotation = smooth_rotations(rotation)
        direct = direct & ~rejected
        trusted = tracked & ~rejected
        recovered = ~trusted
        origin_p, origin_r = position[0].copy(), rotation[0]
        relative_p = origin_r.inv().apply(position - origin_p)
        relative_r = origin_r.inv() * rotation
        raw_relative_p = origin_r.inv().apply(raw_position - origin_p)
        angle, width, angle_direct = load_gripper(
            _resolve(base, robot.get("gripper_csv")), local_time, hw.get("gripper_width_calibration")
        )
        stored_p = position if world_mode else relative_p
        stored_raw_p = raw_position if world_mode else raw_relative_p
        stored_r = rotation if world_mode else relative_r
        arrays[f"robot{index}_eef_pos"] = stored_p.astype(np.float32)
        arrays[f"robot{index}_eef_pos_raw"] = stored_raw_p.astype(np.float32)
        arrays[f"robot{index}_eef_rot_axis_angle"] = stored_r.as_rotvec().astype(np.float32)
        arrays[f"robot{index}_eef_rot_6d"] = _rot6d(stored_r)
        arrays[f"robot{index}_eef_delta_from_start_pos"] = relative_p.astype(np.float32)
        arrays[f"robot{index}_eef_delta_from_start_rot_axis_angle"] = relative_r.as_rotvec().astype(np.float32)
        arrays[f"robot{index}_eef_delta_from_start_rot_6d"] = _rot6d(relative_r)
        arrays[f"robot{index}_gripper_width"] = width[:, None]
        arrays[f"robot{index}_gripper_angle_deg"] = angle[:, None]
        arrays[f"robot{index}_pose_measured"] = direct
        arrays[f"robot{index}_pose_tracked"] = trusted
        arrays[f"robot{index}_pose_recovered"] = recovered
        arrays[f"robot{index}_pose_outlier_rejected"] = rejected
        arrays[f"robot{index}_gripper_measured"] = angle_direct
        arrays[f"robot{index}_demo_start_pose"] = np.tile(
            np.r_[stored_p[0], stored_r[0].as_rotvec()], (len(timeline), 1)).astype(np.float32)
        arrays[f"robot{index}_demo_end_pose"] = np.tile(
            np.r_[stored_p[-1], stored_r[-1].as_rotvec()], (len(timeline), 1)).astype(np.float32)

        video_path = _resolve(base, robot.get("video"))
        if not skip_rgb:
            if video_path is None or not video_path.is_file():
                raise ValueError(f"missing video for {name}")
            size = robot.get("observation", {}).get("size", [224, 224])
            arrays[f"camera{index}_rgb"] = extract_rgb(
                video_path, local_time, (int(size[0]), int(size[1])), robot.get("observation", {})
            )
        direct_ratio = float(np.mean(direct))
        tracked_ratio = float(np.mean(trusted))
        longest_gap = _longest_false_run(trusted)
        redetect_interval = int(robot.get("detector_redetect_interval_frames", 1))
        refresh_success = min(1.0, float(np.mean(pose_series.direct)) * redetect_interval)
        quality_config = spec.get("quality", {})
        minimum_tracked = float(quality_config.get("min_tracked_pose_ratio", 0.90))
        minimum_refresh = float(quality_config.get("min_direct_refresh_success", 0.70))
        maximum_gap = int(quality_config.get("max_tracked_gap_frames", 10))
        maximum_speed = float(quality_config.get("max_filtered_speed_mps", 1.5))
        maximum_acceleration = float(quality_config.get("max_filtered_p99_acceleration_mps2", 6.0))
        maximum_rejected = float(quality_config.get("max_outlier_rejected_ratio", 0.08))
        checks.extend([
            {"name": f"{name}.camera_to_tcp_verified", "pass": bool(extrinsic and hw.get("camera_to_tcp_verified"))},
            {"name": f"{name}.physical_extrinsic_chain_explicit", "pass": bool(
                hw.get("camera_to_base") and hw.get("base_to_tcp")
            ) if world_mode else True},
            {"name": f"{name}.mount_revision_has_no_display_patch", "pass": not any(
                token in str(hw.get("mount_revision", "")).lower()
                for token in ("flat", "table", "shared-a", "shared_a")
            ), "value": hw.get("mount_revision")},
            {"name": f"{name}.camera_to_tcp_direction_explicit", "pass": bool(
                extrinsic
                and extrinsic.get("parent_frame") == "panorama_camera"
                and extrinsic.get("child_frame") == f"{name}_tcp"
            ) if world_mode else True,
             "value": {
                 "parent_frame": extrinsic.get("parent_frame") if extrinsic else None,
                 "child_frame": extrinsic.get("child_frame") if extrinsic else None,
             }},
            {"name": f"{name}.gripper_width_verified", "pass": bool(hw.get("gripper_width_calibration") and hw.get("gripper_width_verified"))},
            {"name": f"{name}.tracked_pose_ratio>={minimum_tracked:.2f}", "pass": tracked_ratio >= minimum_tracked, "value": tracked_ratio},
            {"name": f"{name}.direct_refresh_success>={minimum_refresh:.2f}", "pass": refresh_success >= minimum_refresh, "value": refresh_success},
            {"name": f"{name}.longest_tracked_gap<={maximum_gap}", "pass": longest_gap <= maximum_gap, "value": longest_gap},
            {"name": f"{name}.filtered_max_speed<={maximum_speed:.2f}mps", "pass": motion_audit["filtered_max_speed_mps"] <= maximum_speed, "value": motion_audit["filtered_max_speed_mps"]},
            {"name": f"{name}.filtered_p99_acceleration<={maximum_acceleration:.2f}mps2", "pass": motion_audit["filtered_p99_acceleration_mps2"] <= maximum_acceleration, "value": motion_audit["filtered_p99_acceleration_mps2"]},
            {"name": f"{name}.outlier_rejected_ratio<={maximum_rejected:.2f}", "pass": motion_audit["rejected_ratio"] <= maximum_rejected, "value": motion_audit["rejected_ratio"]},
            {"name": f"{name}.rgb_present", "pass": not skip_rgb},
        ])
        finite_rmse = pose_series.rmse_px[np.isfinite(pose_series.rmse_px)]
        median_rmse = float(np.median(finite_rmse)) if len(finite_rmse) else math.inf
        p95_rmse = float(np.quantile(finite_rmse, 0.95)) if len(finite_rmse) else math.inf
        if world_mode:
            checks.extend([
                {"name": f"{name}.pnp_median_rmse<=1.5px", "pass": median_rmse <= 1.5,
                 "value": median_rmse if math.isfinite(median_rmse) else None},
                {"name": f"{name}.pnp_p95_rmse<=3.0px", "pass": p95_rmse <= 3.0,
                 "value": p95_rmse if math.isfinite(p95_rmse) else None},
            ])
        if world_mode:
            world_positions[name] = stored_p.copy()
        per_robot.append({
            "name": name, "direct_pose_ratio": direct_ratio,
            "tracked_pose_ratio": tracked_ratio, "direct_refresh_success": refresh_success,
            "longest_tracked_gap_frames": longest_gap, "motion_audit": motion_audit,
            "parent_frame": pose_series.parent_frame or None,
            "child_frame": pose_series.child_frame or None,
            "tag_map_sha256": pose_series.tag_map_sha256 or None,
            "pnp_median_rmse_px": median_rmse if math.isfinite(median_rmse) else None,
            "pnp_p95_rmse_px": p95_rmse if math.isfinite(p95_rmse) else None,
            "camera_serial": declared_serial or None,
            "camera_calibration": str(calibration_path) if calibration_path else None,
            "calibration_camera_serial": actual_serial or None,
            "hardware_camera_serial": hardware_serial or None,
            "source_view": hw.get("source_view"),
            "mount_revision": hw.get("mount_revision"),
        })

    workspace = spec.get("workspace", {})
    if world_mode and len(world_positions) > 1 and workspace.get("type") == "tabletop":
        convention = world_map.get("frame_convention", {}) if world_map else {}
        up = np.asarray(workspace.get("up_vector", convention.get("up_vector", [0, 0, 1])), float)
        if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) < 1e-9:
            raise ValueError("tabletop workspace needs a finite non-zero up_vector")
        up /= np.linalg.norm(up)
        heights = {name: values @ up for name, values in world_positions.items()}
        medians = {name: float(np.median(values)) for name, values in heights.items()}
        maximum_median_difference = float(workspace.get(
            "max_robot_median_height_difference_m",
            workspace.get("max_robot_median_z_difference_m", 0.25),
        ))
        plane_calibrated = workspace.get("table_plane_status") in {"CALIBRATED", "VERIFIED"}
        plane_offset = workspace.get("table_plane_offset_m")
        checks.append({
            "name": "coordinate.table_plane_calibrated",
            "pass": plane_calibrated and plane_offset is not None,
            "value": workspace.get("table_plane_status", "NOT_CALIBRATED"),
        })
        if plane_calibrated and plane_offset is not None:
            minimum_clearance = float(workspace.get("minimum_tcp_clearance_m", -0.10))
            offset = float(plane_offset)
            checks.append({
                "name": "coordinate.tcp_not_below_table",
                "pass": all(float(np.quantile(values, 0.01)) - offset >= minimum_clearance
                            for values in heights.values()),
                "value": {name: float(np.quantile(values, 0.01)) - offset
                          for name, values in heights.items()},
            })
        checks.append({
            "name": "coordinate.robot_workspace_height_consistent",
            "pass": max(medians.values()) - min(medians.values()) <= maximum_median_difference,
            "value": {"axis": up.tolist(), "median_projection_m": medians},
        })

    action_parts = []
    for index in range(len(robots)):
        action_parts.extend([
            arrays[f"robot{index}_eef_pos"], arrays[f"robot{index}_eef_rot_6d"],
            arrays[f"robot{index}_gripper_width"],
        ])
    arrays["action"] = np.concatenate(action_parts, axis=1).astype(np.float32)
    arrays["action_valid"] = np.logical_and.reduce([
        arrays[f"robot{index}_pose_tracked"] for index in range(len(robots))
    ])
    minimum_segment_frames = int(spec.get("quality", {}).get("min_training_segment_frames", max(10, round(hz))))
    training_runs = _true_runs(arrays["action_valid"], minimum_segment_frames)
    training_indices = np.concatenate([
        np.arange(start_index, end_index) for start_index, end_index in training_runs
    ]) if training_runs else np.asarray([], dtype=int)
    default_minimum_training_frames = min(200, max(60, int(round(len(timeline) * 0.60))))
    minimum_training_frames = int(spec.get("quality", {}).get(
        "min_training_frames", default_minimum_training_frames))
    instruction = str(spec.get("task", {}).get("instruction", "")).strip()
    checks.append({"name": "task.instruction_present", "pass": bool(instruction)})
    checks.append({"name": "episode.frames>=60", "pass": len(timeline) >= 60, "value": len(timeline)})
    checks.append({
        "name": f"episode.continuous_trusted_training_frames>={minimum_training_frames}",
        "pass": len(training_indices) >= minimum_training_frames,
        "value": int(len(training_indices)),
    })
    sync_uncertainty = spec.get("sync", {}).get("uncertainty_s")
    checks.append({
        "name": "sync.uncertainty<=20ms", "pass": sync_uncertainty is not None and float(sync_uncertainty) <= 0.020,
        "value": sync_uncertainty,
    })
    ready = all(check["pass"] for check in checks)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "episode_arrays.npz", **arrays)
    metadata = {
        "schema_version": SCHEMA_VERSION, "episode_id": episode_id,
        "created_at": datetime.now(timezone.utc).isoformat(), "task": spec.get("task", {}),
        "frequency_hz": hz, "frames": len(timeline), "duration_s": float(len(timeline) / hz),
        "training_frames": int(len(training_indices)), "training_episodes": len(training_runs),
        "training_segments": [[start_index, end_index] for start_index, end_index in training_runs],
        "robots": per_robot, "hardware_config": str(hardware_path) if hardware_path else None,
        "training_ready": ready,
        "status": (
            "READY" if ready else
            "PROVISIONAL_COMMON_WORLD_NOT_TRAINING_READY"
            if world_mode and not calibration_final else
            "DRAFT_HARDWARE_OR_QUALITY_PENDING"
        ),
        "tag_visibility_policy": spec.get("tag_visibility_policy", "unspecified"),
        "coordinate_frame": {
            "mode": "world" if world_mode else "per_robot_start_local",
            "frame_id": world_frame if world_mode else None,
            "tag_map": str(world_map_path) if world_map_path else None,
            "tag_map_sha256": expected_map_hash or None,
            "calibration_status": calibration_status or None,
            "frame_convention": world_map.get("frame_convention") if world_map else None,
            "world_origin": world_map.get("world_origin", world_map.get("origin")) if world_map else None,
            "world_axes": world_map.get("world_axes", world_map.get("axes")) if world_map else None,
            "workspace": spec.get("workspace", {}) if world_mode else None,
        },
    }
    report = {"training_ready": ready, "checks": checks, "failed": [c["name"] for c in checks if not c["pass"]]}
    (output_dir / "episode_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "quality_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "episode_spec.snapshot.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    critical_prefixes = ("coordinate.",)
    critical_failures = [
        check["name"] for check in checks
        if not check["pass"] and (
            check["name"].startswith(critical_prefixes)
            or ".pose_parent_frame" in check["name"]
            or ".pose_child_frame" in check["name"]
            or ".tag_map_hash" in check["name"]
            or ".detected_ids_in_global_map" in check["name"]
            or ".camera_to_tcp_direction_explicit" in check["name"]
            or ".physical_extrinsic_chain_explicit" in check["name"]
            or ".mount_revision_has_no_display_patch" in check["name"]
            or ".camera_serial" in check["name"]
            or ".hardware_camera_serial" in check["name"]
            or ".camera_calibration_present" in check["name"]
            or ".camera_source_view_declared" in check["name"]
            or ".camera_mount_revision_declared" in check["name"]
        )
    ]
    report["critical_failures"] = critical_failures
    if ready:
        if not np.all(np.isfinite(arrays["action"])):
            report["umi_export"] = "blocked: action contains uncalibrated values"
        elif skip_rgb:
            report["umi_export"] = "blocked: RGB was skipped"
        elif not len(training_indices):
            report["umi_export"] = "blocked: no continuous trusted training segment"
        else:
            import zarr
            store = zarr.ZipStore(str(output_dir / "dataset.zarr.zip"), mode="w")
            root = zarr.group(store=store)
            data = root.create_group("data"); meta = root.create_group("meta")
            for key, value in arrays.items():
                if key == "timestamp_s" or key.endswith("_measured") or key.endswith("_tracked") or key.endswith("_rot_6d") or key == "action":
                    continue
                training_value = value[training_indices]
                chunks = (1,) + training_value.shape[1:] if key.endswith("_rgb") else (min(len(training_value), 1024),) + training_value.shape[1:]
                data.create_dataset(key, data=training_value, chunks=chunks)
            episode_ends = np.cumsum([end_index - start_index for start_index, end_index in training_runs])
            meta.create_dataset("episode_ends", data=np.asarray(episode_ends, dtype=np.int64))
            store.close(); report["umi_export"] = "dataset.zarr.zip"
        (output_dir / "quality_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        report["umi_export"] = (
            "blocked: invalid or uncalibrated common coordinate frame"
            if critical_failures else "blocked: quality gates failed"
        )
        if allow_unready:
            report["allow_unready_notice"] = (
                "--allow-unready keeps diagnostic NPZ output but never bypasses training Zarr gates"
            )
        stale_dataset = output_dir / "dataset.zarr.zip"
        if stale_dataset.exists():
            quarantined = output_dir / "dataset.zarr.zip.NOT_TRAINING_READY"
            if quarantined.exists():
                raise RuntimeError(
                    f"cannot quarantine stale Zarr because target exists: {quarantined}"
                )
            stale_dataset.replace(quarantined)
            report["quarantined_stale_zarr"] = str(quarantined)
    (output_dir / "quality_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an audited UMI/VLA episode")
    parser.add_argument("episode_spec", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--skip-rgb", action="store_true", help="validate signals without decoding video")
    parser.add_argument(
        "--allow-unready", action="store_true",
        help="deprecated compatibility flag; diagnostic NPZ is written but failed gates never produce Zarr",
    )
    parser.add_argument(
        "--hardware-override", type=Path,
        help="diagnostic-only hardware file; the episode spec remains unchanged",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = build_episode(
        args.episode_spec, args.output_dir, args.skip_rgb, args.allow_unready,
        args.hardware_override,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
