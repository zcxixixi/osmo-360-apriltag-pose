"""Fast self-calibration and camera tracking from cached A3 AprilGrid bearings.

The four-MP4 pipeline already converts fisheye pixels into calibrated unit
bearings.  This module uses those bearings directly: the two printed panels
define a capture-local world frame, and each camera is localized independently
in that frame.  No fixed transform between the two moving cameras is assumed.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from osmo360.localization.coordinate_frames import (
    X5_STREAM0_OPENCV_FROM_HAND_CAMERA_FLU,
)
from tools.calibrate_wall_pair_transform import PoseSample, robust_panel_transform


@dataclass(frozen=True)
class Pose:
    """``T_world_camera``: camera origin and axes expressed in world."""

    position: np.ndarray
    rotation: Rotation


@dataclass(frozen=True)
class Detection:
    tag_id: int
    rays: np.ndarray
    area_px2: float
    source: str


STREAM0_FROM_HAND_FLU = Rotation.from_matrix(
    X5_STREAM0_OPENCV_FROM_HAND_CAMERA_FLU
)


def pose_to_hand_camera_flu(pose: Pose) -> Pose:
    """Re-express only the child axes; the world frame and origin stay fixed."""
    return Pose(pose.position.copy(), pose.rotation * STREAM0_FROM_HAND_FLU)


def load_direct_tag_map(path: Path) -> tuple[dict[str, Any], dict[int, np.ndarray]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    corners = {
        int(tag["id"]): np.asarray(tag["corners_m"], dtype=np.float64)
        for tag in payload.get("tags", [])
    }
    if not corners or any(value.shape != (4, 3) for value in corners.values()):
        raise ValueError(f"invalid direct AprilTag map: {path}")
    return payload, corners


def compose(parent_child: Pose, child_grandchild: Pose) -> Pose:
    return Pose(
        parent_child.position
        + parent_child.rotation.apply(child_grandchild.position),
        parent_child.rotation * child_grandchild.rotation,
    )


def inverse(transform: Pose) -> Pose:
    rotation = transform.rotation.inv()
    return Pose(-rotation.apply(transform.position), rotation)


def _tangent_from_camera(rays: np.ndarray) -> Rotation:
    mean_ray = np.asarray(rays, dtype=np.float64).mean(axis=0)
    norm = float(np.linalg.norm(mean_ray))
    if norm < 1e-9:
        raise ValueError("bearing rays have no stable mean direction")
    mean_ray /= norm
    return Rotation.align_vectors(
        np.asarray([[0.0, 0.0, 1.0]]), mean_ray.reshape(1, 3)
    )[0]


def _from_tangent_extrinsic(
    tangent_from_camera: Rotation,
    rotation_vector: np.ndarray,
    translation: np.ndarray,
) -> Pose:
    tangent_from_world = Rotation.from_rotvec(rotation_vector.reshape(3))
    tangent_translation = translation.reshape(3)
    camera_from_world = tangent_from_camera.inv() * tangent_from_world
    camera_translation = tangent_from_camera.inv().apply(tangent_translation)
    world_from_camera = camera_from_world.inv()
    return Pose(
        -world_from_camera.apply(camera_translation),
        world_from_camera,
    )


def angular_errors(
    world_points: np.ndarray,
    rays: np.ndarray,
    pose: Pose,
) -> np.ndarray:
    predicted = pose.rotation.inv().apply(world_points - pose.position)
    predicted /= np.linalg.norm(predicted, axis=1, keepdims=True)
    cosine = np.sum(predicted * rays, axis=1)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def planar_pose_candidates(world_points: np.ndarray, rays: np.ndarray) -> list[Pose]:
    """Return both IPPE branches for one planar multi-Tag target."""
    tangent_from_camera = _tangent_from_camera(rays)
    tangent_rays = tangent_from_camera.apply(rays)
    if np.any(tangent_rays[:, 2] <= 1e-4):
        return []
    normalized = tangent_rays[:, :2] / tangent_rays[:, 2:3]
    result = cv2.solvePnPGeneric(
        world_points.astype(np.float64),
        normalized.astype(np.float64),
        np.eye(3, dtype=np.float64),
        None,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not result[0]:
        return []
    candidates = []
    for rotation_vector, translation in zip(result[1], result[2]):
        tangent_from_world = Rotation.from_rotvec(rotation_vector.reshape(3))
        points_tangent = tangent_from_world.apply(world_points) + translation.reshape(3)
        if np.all(points_tangent[:, 2] > 0):
            candidates.append(_from_tangent_extrinsic(
                tangent_from_camera, rotation_vector, translation
            ))
    return candidates


def refine_pose(world_points: np.ndarray, rays: np.ndarray, initial: Pose) -> Pose:
    """Fast OpenCV refinement in a virtual tangent camera."""
    tangent_from_camera = _tangent_from_camera(rays)
    tangent_rays = tangent_from_camera.apply(rays)
    if np.any(tangent_rays[:, 2] <= 1e-4):
        return initial
    normalized = tangent_rays[:, :2] / tangent_rays[:, 2:3]
    camera_from_world = initial.rotation.inv()
    camera_translation = -camera_from_world.apply(initial.position)
    tangent_from_world = tangent_from_camera * camera_from_world
    tangent_translation = tangent_from_camera.apply(camera_translation)
    success, rotation_vector, translation = cv2.solvePnP(
        world_points.astype(np.float64),
        normalized.astype(np.float64),
        np.eye(3, dtype=np.float64),
        None,
        tangent_from_world.as_rotvec().reshape(3, 1),
        tangent_translation.reshape(3, 1),
        True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return initial
    return _from_tangent_extrinsic(
        tangent_from_camera, rotation_vector, translation
    )


def load_cache_frames(path: Path) -> tuple[dict[int, dict[int, Detection]], dict[int, float]]:
    with np.load(path) as cache:
        frames: dict[int, dict[int, Detection]] = {}
        for index, frame_value in enumerate(cache["frame_index"]):
            frame = int(frame_value)
            tag_id = int(cache["tag_id"][index])
            detection = Detection(
                tag_id=tag_id,
                rays=np.asarray(cache["rays_camera"][index], dtype=np.float64),
                area_px2=float(cache["area_px2"][index]),
                source=str(cache["detection_source"][index]),
            )
            previous = frames.setdefault(frame, {}).get(tag_id)
            if previous is None or detection.area_px2 > previous.area_px2:
                frames[frame][tag_id] = detection
        times = {
            int(frame): float(common_time)
            for frame, common_time in zip(
                cache["timeline_frame_index"], cache["timeline_common_time_s"]
            )
        }
    return frames, times


def _panel_pose(
    detections: dict[int, Detection],
    corners: dict[int, np.ndarray],
    minimum_tags: int,
) -> tuple[Pose, float, list[int]] | None:
    ids = sorted(set(detections) & set(corners))
    if len(ids) < minimum_tags:
        return None
    points = np.concatenate([corners[tag_id] for tag_id in ids])
    rays = np.concatenate([detections[tag_id].rays for tag_id in ids])
    candidates = planar_pose_candidates(points, rays)
    if not candidates:
        return None
    scored = [
        (float(np.sqrt(np.mean(angular_errors(points, rays, pose) ** 2))), pose)
        for pose in candidates
    ]
    _, initial = min(scored, key=lambda value: value[0])
    pose = refine_pose(points, rays, initial)
    rmse = float(np.sqrt(np.mean(angular_errors(points, rays, pose) ** 2)))
    return pose, rmse, ids


def _pose_sample(
    frame: int,
    time_s: float,
    result: tuple[Pose, float, list[int]],
) -> PoseSample:
    pose, rmse, ids = result
    return PoseSample(
        frame=frame,
        timestamp_s=time_s,
        position_m=pose.position,
        orientation=pose.rotation,
        rmse_px=max(rmse, 1e-3),
        detected_ids=tuple(ids),
        measurement_source="direct_cached_raw_fisheye_unit_bearing",
    )


def calibrate_panel_pair(
    caches: dict[str, Path],
    panel_a: dict[int, np.ndarray],
    panel_b: dict[int, np.ndarray],
    *,
    minimum_tags: int = 2,
    minimum_inliers: int = 20,
    maximum_frames_per_camera: int = 50,
) -> tuple[Pose, dict[str, Any]]:
    """Estimate ``T_panel_A_panel_B`` jointly from both moving cameras."""
    primary: list[PoseSample] = []
    secondary: list[PoseSample] = []
    side_counts: dict[str, int] = {}
    for side, cache_path in caches.items():
        frames, times = load_cache_frames(cache_path)
        eligible = [
            frame for frame, detections in sorted(frames.items())
            if len(set(detections) & set(panel_a)) >= minimum_tags
            and len(set(detections) & set(panel_b)) >= minimum_tags
        ]
        if len(eligible) > maximum_frames_per_camera:
            selected_indices = np.unique(np.linspace(
                0, len(eligible) - 1, maximum_frames_per_camera
            ).round().astype(int))
            selected_frames = {eligible[index] for index in selected_indices}
        else:
            selected_frames = set(eligible)
        count = 0
        for frame, detections in sorted(frames.items()):
            if frame not in selected_frames:
                continue
            result_a = _panel_pose(detections, panel_a, minimum_tags)
            result_b = _panel_pose(detections, panel_b, minimum_tags)
            if result_a is None or result_b is None:
                continue
            # Make frame keys unique across cameras for auditing.
            audit_frame = frame + (0 if side == "left" else 1_000_000_000)
            primary.append(_pose_sample(audit_frame, times[frame], result_a))
            secondary.append(_pose_sample(audit_frame, times[frame], result_b))
            count += 1
        side_counts[side] = count
    available = len(primary)
    required = min(minimum_inliers, max(8, available // 5))
    if available < required:
        raise ValueError(
            f"only {available} joint A/B observations; need at least {required}"
        )
    rotation, translation, keep, audit = robust_panel_transform(
        primary,
        secondary,
        minimum_inliers=required,
        max_position_residual_m=0.08,
        max_orientation_residual_deg=5.0,
        expected_wall_plane_angle_deg=0.0,
        wall_plane_angle_tolerance_deg=20.0,
    )
    report = {
        "joint_observations": available,
        "joint_observations_by_camera": side_counts,
        "inliers": int(keep.sum()),
        "minimum_inliers": required,
        "panel_B_in_panel_A": {
            "translation_m": translation.tolist(),
            "quaternion_xyzw": rotation.as_quat().tolist(),
        },
        "fit": audit,
    }
    return Pose(translation, rotation), report


def build_world_map(
    pair_id: str,
    panel_a_payload: dict[str, Any],
    panel_b_payload: dict[str, Any],
    panel_a_to_b: Pose,
) -> dict[str, Any]:
    tags = []
    for tag in panel_a_payload["tags"]:
        item = dict(tag)
        item["panel"] = "grid_A"
        tags.append(item)
    for tag in panel_b_payload["tags"]:
        item = dict(tag)
        item["panel"] = "grid_B"
        item["corners_m"] = (
            panel_a_to_b.rotation.apply(np.asarray(tag["corners_m"], dtype=float))
            + panel_a_to_b.position
        ).tolist()
        tags.append(item)
    return {
        "schema_version": "world-apriltag-map/1.0",
        "map_id": f"{pair_id}-a3-self-calibrated-map",
        "world_frame": "session_grid_A",
        "physical_up_vector": [0, -1, 0],
        "calibration_status": "SELF_CALIBRATED_FROM_SAME_CAPTURE_NOT_EXTERNAL_GROUND_TRUTH",
        "tag_outer_size_m": float(panel_a_payload["tag_outer_size_m"]),
        "expected_ids": sorted(int(tag["id"]) for tag in tags),
        "panel_transform": {
            "parent_frame": "session_grid_A",
            "child_frame": "session_grid_B",
            "translation_m": panel_a_to_b.position.tolist(),
            "quaternion_xyzw": panel_a_to_b.rotation.as_quat().tolist(),
            "scale": 1.0,
        },
        "tags": tags,
    }


def _candidate_score(
    pose: Pose,
    ids: list[int],
    points: np.ndarray,
    rays: np.ndarray,
    previous: tuple[float, Pose] | None,
    now_s: float,
) -> tuple[float, np.ndarray]:
    errors = angular_errors(points, rays, pose).reshape(len(ids), 4)
    tag_errors = np.sqrt(np.mean(errors ** 2, axis=1))
    score = float(np.median(tag_errors) + 0.25 * np.percentile(tag_errors, 75))
    if previous is not None:
        previous_time, previous_pose = previous
        delta_s = max(now_s - previous_time, 1e-3)
        distance = float(np.linalg.norm(pose.position - previous_pose.position))
        angle = float(np.degrees(
            (previous_pose.rotation.inv() * pose.rotation).magnitude()
        ))
        # Only penalize motion beyond deliberately generous hand-held limits.
        score += max(0.0, distance - 3.0 * delta_s) * 10.0
        score += max(0.0, angle - 360.0 * delta_s) * 0.03
    return score, tag_errors


def _temporal_gate(
    pose: Pose,
    previous: tuple[float, Pose] | None,
    now_s: float,
    *,
    inlier_tag_count: int,
    sources: set[str],
) -> dict[str, Any]:
    """Reject implausible weak planar-PnP branches without smoothing motion.

    Two co-planar Tags carried only by LK optical flow are useful for filling
    ordinary motion, but they do not provide enough independent evidence to
    accept a sudden IPPE branch change.  Strong direct re-detections remain
    unrestricted so that fast real motion and reacquisition are preserved.
    """
    result: dict[str, Any] = {
        "delta_s": None,
        "speed_m_s": None,
        "angular_speed_deg_s": None,
        "weak_flow_only_measurement": False,
        "rejected": False,
        "reason": "not_applicable",
    }
    if previous is None:
        return result
    previous_time, previous_pose = previous
    delta_s = max(float(now_s - previous_time), 1e-3)
    distance = float(np.linalg.norm(pose.position - previous_pose.position))
    angle = float(np.degrees(
        (previous_pose.rotation.inv() * pose.rotation).magnitude()
    ))
    speed = distance / delta_s
    angular_speed = angle / delta_s
    flow_only = bool(sources) and all(source.startswith("lk_") for source in sources)
    weak = inlier_tag_count <= 2 and flow_only
    rejected = weak and (speed > 1.5 or angular_speed > 180.0)
    result.update({
        "delta_s": delta_s,
        "speed_m_s": speed,
        "angular_speed_deg_s": angular_speed,
        "weak_flow_only_measurement": weak,
        "rejected": rejected,
        "reason": (
            "sparse_flow_planar_pose_exceeds_temporal_limits"
            if rejected else "accepted"
        ),
    })
    return result


def track_cache(
    cache_path: Path,
    panel_a: dict[int, np.ndarray],
    panel_b: dict[int, np.ndarray],
    panel_a_to_b: Pose,
    *,
    minimum_tags: int = 2,
    max_angular_rmse_deg: float = 2.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frames, times = load_cache_frames(cache_path)
    world_corners = dict(panel_a)
    world_corners.update({
        tag_id: panel_a_to_b.rotation.apply(corners) + panel_a_to_b.position
        for tag_id, corners in panel_b.items()
    })
    rows: list[dict[str, Any]] = []
    previous: tuple[float, Pose] | None = None
    for frame, detections in sorted(frames.items()):
        ids = sorted(set(detections) & set(world_corners))
        if len(ids) < minimum_tags:
            continue
        now_s = times[frame]
        points = np.concatenate([world_corners[tag_id] for tag_id in ids])
        rays = np.concatenate([detections[tag_id].rays for tag_id in ids])
        candidates: list[Pose] = []
        for panel, transform in (
            (panel_a, None),
            (panel_b, panel_a_to_b),
        ):
            panel_ids = sorted(set(ids) & set(panel))
            if len(panel_ids) < minimum_tags:
                continue
            panel_points = np.concatenate([panel[tag_id] for tag_id in panel_ids])
            panel_rays = np.concatenate([detections[tag_id].rays for tag_id in panel_ids])
            for candidate in planar_pose_candidates(panel_points, panel_rays):
                candidates.append(
                    compose(transform, candidate) if transform is not None else candidate
                )
        if previous is not None:
            candidates.append(previous[1])
        if not candidates:
            continue
        initial = min(
            candidates,
            key=lambda pose: _candidate_score(
                pose, ids, points, rays, previous, now_s
            )[0],
        )
        _, tag_errors = _candidate_score(
            initial, ids, points, rays, previous, now_s
        )
        median = float(np.median(tag_errors))
        mad = float(np.median(np.abs(tag_errors - median)))
        threshold = max(1.0, median + 3.0 * max(mad, 0.15))
        inlier_ids = [
            tag_id for tag_id, error in zip(ids, tag_errors) if error <= threshold
        ]
        if len(inlier_ids) < minimum_tags:
            inlier_ids = [
                tag_id for tag_id, _ in sorted(
                    zip(ids, tag_errors), key=lambda value: value[1]
                )[:minimum_tags]
            ]
        inlier_points = np.concatenate([world_corners[tag_id] for tag_id in inlier_ids])
        inlier_rays = np.concatenate([detections[tag_id].rays for tag_id in inlier_ids])
        pose = refine_pose(inlier_points, inlier_rays, initial)
        errors = angular_errors(inlier_points, inlier_rays, pose)
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        sources = {detections[tag_id].source for tag_id in inlier_ids}
        temporal = _temporal_gate(
            pose,
            previous,
            now_s,
            inlier_tag_count=len(inlier_ids),
            sources=sources,
        )
        quality = "valid" if rmse <= max_angular_rmse_deg else "angular_rmse_rejected"
        if quality == "valid" and temporal["rejected"]:
            quality = "temporal_outlier_rejected"
        if quality == "valid":
            previous = (now_s, pose)
        output_pose = pose_to_hand_camera_flu(pose)
        quaternion = output_pose.rotation.as_quat()
        euler = output_pose.rotation.as_euler("xyz", degrees=True)
        rows.append({
            "frame": frame,
            "timestamp": f"{now_s:.9f}",
            "camera_x_m": f"{pose.position[0]:.9f}" if quality == "valid" else "",
            "camera_y_m": f"{pose.position[1]:.9f}" if quality == "valid" else "",
            "camera_z_m": f"{pose.position[2]:.9f}" if quality == "valid" else "",
            "qx": f"{quaternion[0]:.12f}" if quality == "valid" else "",
            "qy": f"{quaternion[1]:.12f}" if quality == "valid" else "",
            "qz": f"{quaternion[2]:.12f}" if quality == "valid" else "",
            "qw": f"{quaternion[3]:.12f}" if quality == "valid" else "",
            "roll_deg": f"{euler[0]:.9f}" if quality == "valid" else "",
            "pitch_deg": f"{euler[1]:.9f}" if quality == "valid" else "",
            "yaw_deg": f"{euler[2]:.9f}" if quality == "valid" else "",
            "parent_frame": "session_grid_A",
            "child_frame": "hand_camera_flu_back_x",
            "detected_tag_count": len(ids),
            "inlier_tag_count": len(inlier_ids),
            "inlier_count": len(inlier_points),
            "angular_rmse_deg": f"{rmse:.6f}",
            "detected_ids": " ".join(map(str, ids)),
            "inlier_ids": " ".join(map(str, sorted(inlier_ids))),
            "measurement_source": (
                "cached_raw_fisheye_bearing_direct"
                if all(not source.startswith("lk_") for source in sources)
                else "cached_raw_fisheye_bearing_flow_assisted"
            ),
            "temporal_delta_s": (
                "" if temporal["delta_s"] is None
                else f"{temporal['delta_s']:.9f}"
            ),
            "temporal_speed_m_s": (
                "" if temporal["speed_m_s"] is None
                else f"{temporal['speed_m_s']:.6f}"
            ),
            "temporal_angular_speed_deg_s": (
                "" if temporal["angular_speed_deg_s"] is None
                else f"{temporal['angular_speed_deg_s']:.6f}"
            ),
            "temporal_gate_reason": temporal["reason"],
            "quality_status": quality,
        })
    valid = [row for row in rows if row["quality_status"] == "valid"]
    residuals = np.asarray([float(row["angular_rmse_deg"]) for row in valid])
    positions = np.asarray([
        [float(row[key]) for key in ("camera_x_m", "camera_y_m", "camera_z_m")]
        for row in valid
    ]) if valid else np.empty((0, 3))
    step = np.linalg.norm(np.diff(positions, axis=0), axis=1) if len(positions) > 1 else np.asarray([])
    summary = {
        "cache": str(cache_path.resolve()),
        "observed_frames": len(rows),
        "valid_frames": len(valid),
        "valid_ratio": len(valid) / len(rows) if rows else 0.0,
        "temporal_outlier_rejected_frames": sum(
            row["quality_status"] == "temporal_outlier_rejected" for row in rows
        ),
        "angular_rmse_deg": {
            "median": float(np.median(residuals)) if len(residuals) else None,
            "p95": float(np.percentile(residuals, 95)) if len(residuals) else None,
            "max": float(np.max(residuals)) if len(residuals) else None,
        },
        "path_length_m": float(step.sum()),
        "position_step_m": {
            "median": float(np.median(step)) if len(step) else None,
            "p95": float(np.percentile(step, 95)) if len(step) else None,
            "max": float(np.max(step)) if len(step) else None,
        },
    }
    return rows, summary


POSE_FIELDS = [
    "frame", "timestamp", "camera_x_m", "camera_y_m", "camera_z_m",
    "qx", "qy", "qz", "qw", "roll_deg", "pitch_deg", "yaw_deg",
    "parent_frame", "child_frame", "detected_tag_count", "inlier_tag_count",
    "inlier_count", "angular_rmse_deg", "detected_ids", "inlier_ids",
    "measurement_source", "temporal_delta_s", "temporal_speed_m_s",
    "temporal_angular_speed_deg_s", "temporal_gate_reason", "quality_status",
]


def write_pose_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=POSE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_joint_pose_csv(
    path: Path,
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    *,
    map_id: str,
) -> dict[str, Any]:
    """Write synchronized left/right poses in one explicitly shared map."""
    by_side = {
        "left": {int(row["frame"]): row for row in left_rows},
        "right": {int(row["frame"]): row for row in right_rows},
    }
    frames = sorted(set(by_side["left"]) | set(by_side["right"]))
    pose_keys = (
        "camera_x_m", "camera_y_m", "camera_z_m", "qx", "qy", "qz", "qw"
    )
    fields = [
        "frame", "timestamp_s", "world_frame", "map_id",
        "joint_valid", "joint_measured",
    ]
    for side in ("left", "right"):
        fields.extend(f"{side}_{key}" for key in pose_keys)
        fields.extend((
            f"{side}_quality_status",
            f"{side}_pose_state",
            f"{side}_angular_rmse_deg",
            f"{side}_detected_tag_count",
            f"{side}_inlier_tag_count",
            f"{side}_measurement_source",
        ))
    output_rows = []
    joint_valid_count = 0
    joint_measured_count = 0
    interpolation_gaps: dict[str, list[float]] = {"left": [], "right": []}

    valid_series = {}
    for side in ("left", "right"):
        valid = [
            row for row in by_side[side].values()
            if row["quality_status"] == "valid" and row["camera_x_m"]
        ]
        valid.sort(key=lambda row: float(row["timestamp"]))
        series_times = np.asarray([float(row["timestamp"]) for row in valid])
        series_positions = np.asarray([
            [float(row[key]) for key in ("camera_x_m", "camera_y_m", "camera_z_m")]
            for row in valid
        ])
        series_rotations = Rotation.from_quat(np.asarray([
            [float(row[key]) for key in ("qx", "qy", "qz", "qw")]
            for row in valid
        ]))
        valid_series[side] = (
            series_times, series_positions, series_rotations,
            Slerp(series_times, series_rotations),
        )
    for frame in frames:
        left = by_side["left"].get(frame)
        right = by_side["right"].get(frame)
        timestamps = [
            float(row["timestamp"]) for row in (left, right) if row is not None
        ]
        if timestamps and max(timestamps) - min(timestamps) > 1e-6:
            raise ValueError(f"left/right common timestamps disagree at frame {frame}")
        joint_measured = bool(
            left is not None and right is not None
            and left["quality_status"] == "valid"
            and right["quality_status"] == "valid"
        )
        resolved = {"left": left, "right": right}
        now_s = timestamps[0]
        for side in ("left", "right"):
            source = resolved[side]
            if source is not None and source["quality_status"] == "valid":
                source = dict(source)
                source["pose_state"] = "MEASURED"
                resolved[side] = source
                continue
            series_times, series_positions, _, slerp = valid_series[side]
            if not series_times[0] <= now_s <= series_times[-1]:
                continue
            upper = int(np.searchsorted(series_times, now_s, side="right"))
            lower = max(0, upper - 1)
            upper = min(upper, len(series_times) - 1)
            position = np.asarray([
                np.interp(now_s, series_times, series_positions[:, axis])
                for axis in range(3)
            ])
            quaternion = slerp([now_s]).as_quat()[0]
            gap = float(series_times[upper] - series_times[lower])
            interpolation_gaps[side].append(gap)
            resolved[side] = {
                "camera_x_m": f"{position[0]:.9f}",
                "camera_y_m": f"{position[1]:.9f}",
                "camera_z_m": f"{position[2]:.9f}",
                "qx": f"{quaternion[0]:.12f}",
                "qy": f"{quaternion[1]:.12f}",
                "qz": f"{quaternion[2]:.12f}",
                "qw": f"{quaternion[3]:.12f}",
                "quality_status": "interpolated",
                "pose_state": "INTERPOLATED",
                "angular_rmse_deg": "",
                "detected_tag_count": "",
                "inlier_tag_count": "",
                "measurement_source": "temporal_interpolation_between_cached_bearing_poses",
            }
        joint_valid = all(
            resolved[side] is not None
            and resolved[side].get("quality_status") in {"valid", "interpolated"}
            for side in ("left", "right")
        )
        joint_valid_count += int(joint_valid)
        joint_measured_count += int(joint_measured)
        item: dict[str, Any] = {
            "frame": frame,
            "timestamp_s": f"{timestamps[0]:.9f}",
            "world_frame": "session_grid_A",
            "map_id": map_id,
            "joint_valid": str(joint_valid).lower(),
            "joint_measured": str(joint_measured).lower(),
        }
        for side, source in resolved.items():
            for key in pose_keys:
                item[f"{side}_{key}"] = "" if source is None else source[key]
            for key in (
                "quality_status", "pose_state", "angular_rmse_deg", "detected_tag_count",
                "inlier_tag_count", "measurement_source",
            ):
                item[f"{side}_{key}"] = "" if source is None else source[key]
        output_rows.append(item)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    return {
        "common_timeline_frames": len(output_rows),
        "joint_valid_frames": joint_valid_count,
        "joint_valid_ratio": (
            joint_valid_count / len(output_rows) if output_rows else 0.0
        ),
        "joint_measured_frames": joint_measured_count,
        "joint_measured_ratio": (
            joint_measured_count / len(output_rows) if output_rows else 0.0
        ),
        "maximum_interpolation_gap_s": {
            side: max(interpolation_gaps[side], default=0.0)
            for side in ("left", "right")
        },
        "world_frame": "session_grid_A",
        "map_id": map_id,
    }
