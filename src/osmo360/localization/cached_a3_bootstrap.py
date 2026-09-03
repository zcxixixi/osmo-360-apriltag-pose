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
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp

from osmo360.localization.coordinate_frames import (
    X5_STREAM0_OPENCV_FROM_HAND_CAMERA_FLU,
)
from osmo360.localization.instaumi_imu import (
    ImuAssistanceUnavailable,
    ImuSeries,
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
    lens_stream: int | None


@dataclass(frozen=True)
class PoseFit:
    """One bearing-PnP fit plus the observations that support it."""

    pose: Pose
    detected_ids: list[int]
    inlier_ids: list[int]
    angular_rmse_deg: float
    sources: set[str]
    dominant_lens_stream: int | None
    inlier_lens_stream_counts: dict[int, int]
    selected_lens_stream: int | None = None
    direct_only_priority: bool = False
    temporally_regularized: bool = False


STREAM0_FROM_HAND_FLU = Rotation.from_matrix(
    X5_STREAM0_OPENCV_FROM_HAND_CAMERA_FLU
)
MAXIMUM_TRUSTED_INTERPOLATION_GAP_S = 0.25
MAXIMUM_ABSOLUTE_SPEED_M_S = 3.0
MAXIMUM_ABSOLUTE_ANGULAR_SPEED_DEG_S = 540.0
MAXIMUM_IMU_VISUAL_ROTATION_RESIDUAL_DEG = 15.0
MINIMUM_TEMPORAL_RECOVERY_INLIER_TAGS = 5


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
        # ``NpzFile.__getitem__`` decompresses the complete member array.  Keep
        # every lookup outside the per-observation loop; otherwise a compressed
        # cache is inflated thousands of times while loading one trajectory.
        frame_indices = np.asarray(cache["frame_index"])
        tag_ids = np.asarray(cache["tag_id"])
        rays = np.asarray(cache["rays_camera"], dtype=np.float64)
        areas = np.asarray(cache["area_px2"])
        sources = np.asarray(cache["detection_source"])
        try:
            lens_streams = np.asarray(cache["lens_stream"])
        except KeyError:
            # Keep older observation caches readable. A missing provenance
            # field disables the handoff gate instead of guessing a lens.
            lens_streams = np.full(len(frame_indices), -1, dtype=np.int8)
        timeline_frames = np.asarray(cache["timeline_frame_index"])
        timeline_times = np.asarray(cache["timeline_common_time_s"], dtype=np.float64)

    frames: dict[int, dict[int, Detection]] = {}
    for frame_value, tag_value, ray, area, source, lens_stream in zip(
        frame_indices, tag_ids, rays, areas, sources, lens_streams, strict=True
    ):
        frame = int(frame_value)
        tag_id = int(tag_value)
        detection = Detection(
            tag_id=tag_id,
            rays=ray,
            area_px2=float(area),
            source=str(source),
            lens_stream=(int(lens_stream) if int(lens_stream) >= 0 else None),
        )
        previous = frames.setdefault(frame, {}).get(tag_id)
        if previous is None or detection.area_px2 > previous.area_px2:
            frames[frame][tag_id] = detection
    times = {
        int(frame): float(common_time)
        for frame, common_time in zip(
            timeline_frames, timeline_times, strict=True
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
    dominant_lens_stream: int | None = None,
    previous_dominant_lens_stream: int | None = None,
    imu_stream: ImuSeries | None = None,
    recovery_required: bool = False,
) -> dict[str, Any]:
    """Reject physically implausible or IMU-inconsistent visual branches.

    Two co-planar Tags carried only by LK optical flow are useful for filling
    ordinary motion, but they do not provide enough independent evidence to
    accept a sudden IPPE branch change. A lens handoff is likewise weak only
    when fewer than five Tag IDs support it. Five or more mutually consistent
    AprilTags remain the primary observation even when their attitude differs
    from the short-horizon gyro prediction; the IMU is an auxiliary weak-visual
    gate, not a replacement for strong AprilGrid geometry. Every solve also has
    a deliberately generous absolute motion ceiling. After such a rejection,
    the tracker only reacquires on strong geometry consistent with the last
    accepted pose.
    """
    result: dict[str, Any] = {
        "delta_s": None,
        "speed_m_s": None,
        "angular_speed_deg_s": None,
        "weak_flow_only_measurement": False,
        "lens_handoff_measurement": False,
        "absolute_motion_outlier": False,
        "recovery_required": recovery_required,
        "imu_prediction_available": False,
        "imu_prediction_reason": "not_requested",
        "imu_visual_rotation_residual_deg": None,
        "dominant_lens_stream": dominant_lens_stream,
        "previous_dominant_lens_stream": previous_dominant_lens_stream,
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
    lens_handoff = (
        dominant_lens_stream is not None
        and previous_dominant_lens_stream is not None
        and dominant_lens_stream != previous_dominant_lens_stream
    )
    weak_lens_handoff = (
        lens_handoff
        and inlier_tag_count < MINIMUM_TEMPORAL_RECOVERY_INLIER_TAGS
    )
    exceeds_motion_limit = speed > 1.5 or angular_speed > 180.0
    absolute_motion_outlier = (
        speed > MAXIMUM_ABSOLUTE_SPEED_M_S
        or angular_speed > MAXIMUM_ABSOLUTE_ANGULAR_SPEED_DEG_S
    )
    recovery_outlier = recovery_required and (
        inlier_tag_count < MINIMUM_TEMPORAL_RECOVERY_INLIER_TAGS
        or (flow_only and delta_s > MAXIMUM_TRUSTED_INTERPOLATION_GAP_S)
        or exceeds_motion_limit
    )
    imu_residual = None
    imu_reason = "stream_unavailable"
    if imu_stream is not None:
        try:
            prediction = imu_stream.predict_orientation(
                previous_time,
                pose_to_hand_camera_flu(previous_pose).rotation,
                now_s,
            )
        except (ImuAssistanceUnavailable, ValueError) as exc:
            imu_reason = str(exc)
        else:
            candidate_rotation = pose_to_hand_camera_flu(pose).rotation
            imu_residual = float(np.degrees(
                (prediction.rotation.inv() * candidate_rotation).magnitude()
            ))
            imu_reason = "available"
    imu_inconsistent = bool(
        imu_residual is not None
        and imu_residual > MAXIMUM_IMU_VISUAL_ROTATION_RESIDUAL_DEG
        and inlier_tag_count < MINIMUM_TEMPORAL_RECOVERY_INLIER_TAGS
    )
    rejected = bool(
        absolute_motion_outlier
        or recovery_outlier
        or ((weak or weak_lens_handoff) and exceeds_motion_limit)
        or imu_inconsistent
    )
    if absolute_motion_outlier:
        reason = "pose_exceeds_absolute_temporal_limits"
    elif recovery_outlier:
        reason = "temporal_recovery_requires_strong_consistent_geometry"
    elif weak_lens_handoff and exceeds_motion_limit:
        reason = "lens_handoff_pose_exceeds_temporal_limits"
    elif weak and exceeds_motion_limit:
        reason = "sparse_flow_planar_pose_exceeds_temporal_limits"
    elif imu_inconsistent:
        reason = "weak_visual_rotation_disagrees_with_imu"
    else:
        reason = "accepted"
    result.update({
        "delta_s": delta_s,
        "speed_m_s": speed,
        "angular_speed_deg_s": angular_speed,
        "weak_flow_only_measurement": weak,
        "lens_handoff_measurement": lens_handoff,
        "absolute_motion_outlier": absolute_motion_outlier,
        "imu_prediction_available": imu_residual is not None,
        "imu_prediction_reason": imu_reason,
        "imu_visual_rotation_residual_deg": imu_residual,
        "rejected": rejected,
        "reason": reason,
    })
    return result


def _inlier_lens_streams(
    detections: dict[int, Detection],
    inlier_ids: list[int],
) -> tuple[int | None, dict[int, int]]:
    """Return an unambiguous dominant calibrated lens and provenance counts."""
    counts: dict[int, int] = {}
    for tag_id in inlier_ids:
        lens_stream = detections[tag_id].lens_stream
        if lens_stream is not None:
            counts[lens_stream] = counts.get(lens_stream, 0) + 1
    if not counts:
        return None, counts
    maximum = max(counts.values())
    winners = [lens for lens, count in counts.items() if count == maximum]
    return (winners[0] if len(winners) == 1 else None), counts


def _fit_pose(
    detections: dict[int, Detection],
    world_corners: dict[int, np.ndarray],
    panel_a: dict[int, np.ndarray],
    panel_b: dict[int, np.ndarray],
    panel_a_to_b: Pose,
    previous: tuple[float, Pose] | None,
    now_s: float,
    minimum_tags: int,
    *,
    selected_lens_stream: int | None = None,
) -> PoseFit | None:
    """Fit one camera pose without mixing in observations outside ``detections``."""
    ids = sorted(set(detections) & set(world_corners))
    if len(ids) < minimum_tags:
        return None
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
        return None
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
    dominant_lens, inlier_lens_counts = _inlier_lens_streams(
        detections, inlier_ids
    )
    return PoseFit(
        pose=pose,
        detected_ids=ids,
        inlier_ids=inlier_ids,
        angular_rmse_deg=rmse,
        sources=sources,
        dominant_lens_stream=dominant_lens,
        inlier_lens_stream_counts=inlier_lens_counts,
        selected_lens_stream=selected_lens_stream,
    )


def _regularize_weak_planar_fit(
    fit: PoseFit,
    detections: dict[int, Detection],
    world_corners: dict[int, np.ndarray],
    panel_a: dict[int, np.ndarray],
    panel_b: dict[int, np.ndarray],
    previous: tuple[float, Pose] | None,
    now_s: float,
    *,
    max_angular_rmse_deg: float,
) -> PoseFit:
    """Bound poorly observable per-frame planar motion by bearing evidence.

    A single flat AprilGrid can fit a noticeably different depth/tilt pose
    after only a tiny corner perturbation, especially while a hand partially
    covers printed edges. If at least two Tags from each non-coplanar panel are
    present, their geometry already constrains this mode and no regularization
    is applied. Otherwise the improvement over the last accepted pose controls
    how much motion one 30 Hz sample may introduce. The raw fit is retained if
    a bounded pose would no longer satisfy the normal angular quality gate.
    """
    if previous is None:
        return fit
    previous_time, previous_pose = previous
    delta_s = float(now_s - previous_time)
    if not 0.0 < delta_s <= 0.1:
        return fit
    panel_a_count = len(set(fit.inlier_ids) & set(panel_a))
    panel_b_count = len(set(fit.inlier_ids) & set(panel_b))
    if panel_a_count >= 2 and panel_b_count >= 2:
        return fit

    points = np.concatenate([world_corners[tag_id] for tag_id in fit.inlier_ids])
    rays = np.concatenate([detections[tag_id].rays for tag_id in fit.inlier_ids])
    previous_rmse = float(np.sqrt(np.mean(
        angular_errors(points, rays, previous_pose) ** 2
    )))
    evidence_gain_deg = max(0.0, previous_rmse - fit.angular_rmse_deg)
    position_delta = fit.pose.position - previous_pose.position
    distance_m = float(np.linalg.norm(position_delta))
    relative_rotation = previous_pose.rotation.inv() * fit.pose.rotation
    angle_deg = float(np.degrees(relative_rotation.magnitude()))
    maximum_distance_m = delta_s * (0.15 + 3.0 * evidence_gain_deg)
    maximum_angle_deg = delta_s * (15.0 + 300.0 * evidence_gain_deg)
    position_fraction = min(1.0, maximum_distance_m / max(distance_m, 1e-12))
    rotation_fraction = min(1.0, maximum_angle_deg / max(angle_deg, 1e-12))
    if position_fraction >= 1.0 and rotation_fraction >= 1.0:
        return fit

    regularized_pose = Pose(
        previous_pose.position + position_fraction * position_delta,
        previous_pose.rotation * Rotation.from_rotvec(
            relative_rotation.as_rotvec() * rotation_fraction
        ),
    )
    regularized_rmse = float(np.sqrt(np.mean(
        angular_errors(points, rays, regularized_pose) ** 2
    )))
    if regularized_rmse > max_angular_rmse_deg:
        return fit
    return PoseFit(
        pose=regularized_pose,
        detected_ids=fit.detected_ids,
        inlier_ids=fit.inlier_ids,
        angular_rmse_deg=regularized_rmse,
        sources=fit.sources,
        dominant_lens_stream=fit.dominant_lens_stream,
        inlier_lens_stream_counts=fit.inlier_lens_stream_counts,
        selected_lens_stream=fit.selected_lens_stream,
        direct_only_priority=fit.direct_only_priority,
        temporally_regularized=True,
    )


def track_cache(
    cache_path: Path,
    panel_a: dict[int, np.ndarray],
    panel_b: dict[int, np.ndarray],
    panel_a_to_b: Pose,
    *,
    minimum_tags: int = 2,
    max_angular_rmse_deg: float = 2.0,
    imu_stream: ImuSeries | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frames, times = load_cache_frames(cache_path)
    world_corners = dict(panel_a)
    world_corners.update({
        tag_id: panel_a_to_b.rotation.apply(corners) + panel_a_to_b.position
        for tag_id, corners in panel_b.items()
    })
    rows: list[dict[str, Any]] = []
    previous: tuple[float, Pose] | None = None
    previous_observed_dominant_lens: int | None = None
    temporal_recovery_required = False
    for frame, detections in sorted(frames.items()):
        now_s = times[frame]
        detected_ids = sorted(set(detections) & set(world_corners))
        fit = _fit_pose(
            detections,
            world_corners,
            panel_a,
            panel_b,
            panel_a_to_b,
            previous,
            now_s,
            minimum_tags,
        )
        if fit is None:
            continue

        # A decoded Tag corner is a stronger observation than an LK-propagated
        # corner under partial occlusion. When at least four freshly decoded
        # IDs make a valid pose, do not let stale tracked corners pull that
        # anchor away from the printed grid.
        direct_detections = {
            tag_id: detection
            for tag_id, detection in detections.items()
            if not detection.source.startswith("lk_")
        }
        if (
            len(direct_detections) >= 4
            and any(source.startswith("lk_") for source in fit.sources)
        ):
            direct_fit = _fit_pose(
                direct_detections,
                world_corners,
                panel_a,
                panel_b,
                panel_a_to_b,
                previous,
                now_s,
                minimum_tags,
            )
            if (
                direct_fit is not None
                and direct_fit.angular_rmse_deg <= max_angular_rmse_deg
            ):
                fit = PoseFit(
                    pose=direct_fit.pose,
                    detected_ids=direct_fit.detected_ids,
                    inlier_ids=direct_fit.inlier_ids,
                    angular_rmse_deg=direct_fit.angular_rmse_deg,
                    sources=direct_fit.sources,
                    dominant_lens_stream=direct_fit.dominant_lens_stream,
                    inlier_lens_stream_counts=(
                        direct_fit.inlier_lens_stream_counts
                    ),
                    direct_only_priority=True,
                )

        # A standard central-camera PnP cannot combine rays from the two X5
        # optical centres exactly. At the overlap seam that approximation can
        # reject an otherwise strong multi-Tag frame. Only when the combined
        # fit has already failed its angular gate, retry each calibrated lens
        # independently and keep a genuinely valid single-lens solution.
        lens_streams = sorted({
            detection.lens_stream
            for detection in detections.values()
            if detection.lens_stream is not None
        })
        if fit.angular_rmse_deg > max_angular_rmse_deg and len(lens_streams) > 1:
            single_lens_fits = [
                candidate
                for lens_stream in lens_streams
                if (candidate := _fit_pose(
                    {
                        tag_id: detection
                        for tag_id, detection in detections.items()
                        if detection.lens_stream == lens_stream
                    },
                    world_corners,
                    panel_a,
                    panel_b,
                    panel_a_to_b,
                    previous,
                    now_s,
                    minimum_tags,
                    selected_lens_stream=lens_stream,
                )) is not None
            ]
            if single_lens_fits:
                recovered = min(
                    single_lens_fits,
                    key=lambda candidate: (
                        candidate.angular_rmse_deg,
                        -len(candidate.inlier_ids),
                    ),
                )
                if recovered.angular_rmse_deg <= max_angular_rmse_deg:
                    fit = recovered

        fit = _regularize_weak_planar_fit(
            fit,
            detections,
            world_corners,
            panel_a,
            panel_b,
            previous,
            now_s,
            max_angular_rmse_deg=max_angular_rmse_deg,
        )

        pose = fit.pose
        ids = fit.detected_ids
        inlier_ids = fit.inlier_ids
        rmse = fit.angular_rmse_deg
        sources = fit.sources
        dominant_lens = fit.dominant_lens_stream
        inlier_lens_counts = fit.inlier_lens_stream_counts
        temporal = _temporal_gate(
            pose,
            previous,
            now_s,
            inlier_tag_count=len(inlier_ids),
            sources=sources,
            dominant_lens_stream=dominant_lens,
            previous_dominant_lens_stream=previous_observed_dominant_lens,
            imu_stream=imu_stream,
            recovery_required=temporal_recovery_required,
        )
        quality = "valid" if rmse <= max_angular_rmse_deg else "angular_rmse_rejected"
        if quality != "valid":
            temporal["rejected"] = False
            temporal["reason"] = "not_applied_angular_rmse_rejected"
        elif temporal["rejected"]:
            quality = "temporal_outlier_rejected"
        # Advance the observation-side lens state after a geometrically valid
        # solve even when its pose is temporally rejected. Otherwise one real
        # handoff would repeatedly reject every subsequent frame on that lens.
        if rmse <= max_angular_rmse_deg and dominant_lens is not None:
            previous_observed_dominant_lens = dominant_lens
        if quality == "valid":
            previous = (now_s, pose)
            temporal_recovery_required = False
        elif temporal["rejected"]:
            temporal_recovery_required = True
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
            "detected_tag_count": len(detected_ids),
            "inlier_tag_count": len(inlier_ids),
            "inlier_count": len(inlier_ids) * 4,
            "angular_rmse_deg": f"{rmse:.6f}",
            "detected_ids": " ".join(map(str, detected_ids)),
            "inlier_ids": " ".join(map(str, sorted(inlier_ids))),
            "measurement_source": (
                (
                    "cached_raw_fisheye_bearing_direct"
                    if all(not source.startswith("lk_") for source in sources)
                    else "cached_raw_fisheye_bearing_flow_assisted"
                )
                + (
                    "_single_lens_recovery"
                    if fit.selected_lens_stream is not None
                    else ""
                )
                + ("_direct_priority" if fit.direct_only_priority else "")
                + ("_planar_regularized" if fit.temporally_regularized else "")
            ),
            "dominant_lens_stream": (
                "" if dominant_lens is None else str(dominant_lens)
            ),
            "inlier_lens_stream_counts": " ".join(
                f"{lens}:{count}"
                for lens, count in sorted(inlier_lens_counts.items())
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
            "temporal_imu_rotation_residual_deg": (
                "" if temporal["imu_visual_rotation_residual_deg"] is None
                else f"{temporal['imu_visual_rotation_residual_deg']:.6f}"
            ),
            "temporal_imu_prediction_status": temporal["imu_prediction_reason"],
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
        "single_lens_recovered_frames": sum(
            row["quality_status"] == "valid"
            and "_single_lens_recovery" in row["measurement_source"]
            for row in rows
        ),
        "planar_temporally_regularized_frames": sum(
            row["quality_status"] == "valid"
            and row["measurement_source"].endswith("_planar_regularized")
            for row in rows
        ),
        "direct_priority_frames": sum(
            row["quality_status"] == "valid"
            and "_direct_priority" in row["measurement_source"]
            for row in rows
        ),
        "temporal_outlier_rejected_frames": sum(
            row["quality_status"] == "temporal_outlier_rejected" for row in rows
        ),
        "lens_handoff_temporal_outlier_rejected_frames": sum(
            row["quality_status"] == "temporal_outlier_rejected"
            and row["temporal_gate_reason"]
                == "lens_handoff_pose_exceeds_temporal_limits"
            for row in rows
        ),
        "absolute_motion_temporal_outlier_rejected_frames": sum(
            row["quality_status"] == "temporal_outlier_rejected"
            and row["temporal_gate_reason"]
                == "pose_exceeds_absolute_temporal_limits"
            for row in rows
        ),
        "imu_inconsistent_temporal_outlier_rejected_frames": sum(
            row["quality_status"] == "temporal_outlier_rejected"
            and row["temporal_gate_reason"]
                == "weak_visual_rotation_disagrees_with_imu"
            for row in rows
        ),
        "temporal_recovery_rejected_frames": sum(
            row["quality_status"] == "temporal_outlier_rejected"
            and row["temporal_gate_reason"]
                == "temporal_recovery_requires_strong_consistent_geometry"
            for row in rows
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
    "measurement_source", "dominant_lens_stream", "inlier_lens_stream_counts",
    "temporal_delta_s", "temporal_speed_m_s",
    "temporal_angular_speed_deg_s", "temporal_imu_rotation_residual_deg",
    "temporal_imu_prediction_status", "temporal_gate_reason", "quality_status",
]


def _zero_phase_filter_joint_rows(
    rows: list[dict[str, Any]],
    *,
    window_frames: int = 9,
    polynomial_order: int = 2,
    gap_guard_s: float = 0.35,
    minimum_closed_excursion_m: float = 0.05,
    closed_excursion_ratio: float = 3.0,
) -> dict[str, Any]:
    """Suppress high-frequency pose jitter without shifting timestamps.

    The centered Savitzky-Golay pass has zero phase delay and exactly preserves
    constant-velocity/quadratic trends. Before it, a long untrusted gap is
    guarded only when its numeric bridge makes a large closed excursion and
    returns near its starting pose. That catches occlusion-driven PnP spikes
    without flattening ordinary one-way hand motion.
    """
    if window_frames < 5 or window_frames % 2 == 0:
        raise ValueError("zero-phase filter window must be an odd integer >= 5")
    if not 1 <= polynomial_order < window_frames:
        raise ValueError("invalid zero-phase filter polynomial order")
    if len(rows) < window_frames:
        return {
            "enabled": False,
            "reason": "timeline_shorter_than_filter_window",
            "timestamp_or_frame_changed": False,
        }
    timestamps = np.asarray([float(row["timestamp_s"]) for row in rows])
    frames = np.asarray([int(row["frame"]) for row in rows])
    if np.any(np.diff(timestamps) <= 0) or np.any(np.diff(frames) <= 0):
        raise ValueError("zero-phase filter requires an increasing joint timeline")
    original_timestamps = timestamps.copy()
    original_frames = frames.copy()
    audit: dict[str, Any] = {
        "enabled": True,
        "method": "closed-excursion gap guard plus centered Savitzky-Golay",
        "zero_phase": True,
        "phase_delay_frames": 0,
        "timestamp_or_frame_changed": False,
        "window_frames": window_frames,
        "polynomial_order": polynomial_order,
        "nominal_window_s": float(
            window_frames * np.median(np.diff(timestamps))
        ),
        "gap_guard_s": gap_guard_s,
        "minimum_closed_excursion_m": minimum_closed_excursion_m,
        "closed_excursion_ratio": closed_excursion_ratio,
        "sides": {},
    }
    untrusted_statuses = {
        "interpolation_untrusted", "imu_assisted_untrusted", "pose_untrusted"
    }
    for side in ("left", "right"):
        position = np.asarray([
            [float(row[f"{side}_camera_{axis}_m"]) for axis in "xyz"]
            for row in rows
        ])
        quaternion = np.asarray([
            [float(row[f"{side}_q{axis}"]) for axis in "xyzw"]
            for row in rows
        ])
        original_position = position.copy()
        original_rotation = Rotation.from_quat(quaternion.copy())
        guarded = np.zeros(len(rows), dtype=bool)
        untrusted = np.asarray([
            str(row[f"{side}_quality_status"]) in untrusted_statuses
            for row in rows
        ])
        changes = np.diff(np.r_[False, untrusted, False].astype(np.int8))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1) - 1
        repaired_intervals = []
        for start, end in zip(starts, ends, strict=True):
            if timestamps[end] - timestamps[start] <= MAXIMUM_TRUSTED_INTERPOLATION_GAP_S:
                continue
            guard_start = int(np.searchsorted(
                timestamps, timestamps[start] - gap_guard_s, side="left"
            ))
            guard_end = int(np.searchsorted(
                timestamps, timestamps[end] + gap_guard_s, side="right"
            ) - 1)
            lower = guard_start - 1
            upper = guard_end + 1
            if lower < 0 or upper >= len(rows):
                continue
            alpha = (
                (timestamps[lower:upper + 1] - timestamps[lower])
                / (timestamps[upper] - timestamps[lower])
            )[:, None]
            chord = (
                (1.0 - alpha) * position[lower]
                + alpha * position[upper]
            )
            deviation = np.linalg.norm(
                position[lower:upper + 1] - chord, axis=1
            )
            anchor_displacement = float(np.linalg.norm(
                position[upper] - position[lower]
            ))
            maximum_deviation = float(np.max(deviation))
            if maximum_deviation <= max(
                minimum_closed_excursion_m,
                closed_excursion_ratio * anchor_displacement,
            ):
                continue
            position[lower:upper + 1] = chord
            quaternion[lower:upper + 1] = Slerp(
                [timestamps[lower], timestamps[upper]],
                Rotation.from_quat([quaternion[lower], quaternion[upper]]),
            )(timestamps[lower:upper + 1]).as_quat()
            guarded[lower:upper + 1] = True
            repaired_intervals.append({
                "untrusted_start_s": float(timestamps[start]),
                "untrusted_end_s": float(timestamps[end]),
                "guarded_start_s": float(timestamps[lower]),
                "guarded_end_s": float(timestamps[upper]),
                "anchor_displacement_m": anchor_displacement,
                "maximum_closed_excursion_m": maximum_deviation,
            })

        # Quaternion sign has no physical meaning, but continuous signs are
        # required before filtering its four embedding coordinates.
        for index in range(1, len(quaternion)):
            if float(np.dot(quaternion[index - 1], quaternion[index])) < 0.0:
                quaternion[index] *= -1.0
        position = savgol_filter(
            position,
            window_frames,
            polynomial_order,
            axis=0,
            mode="interp",
        )
        quaternion = savgol_filter(
            quaternion,
            window_frames,
            polynomial_order,
            axis=0,
            mode="interp",
        )
        quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
        filtered_rotation = Rotation.from_quat(quaternion)
        position_correction = np.linalg.norm(
            position - original_position, axis=1
        )
        rotation_correction = np.degrees(
            (original_rotation.inv() * filtered_rotation).magnitude()
        )
        for index, row in enumerate(rows):
            for axis_index, axis in enumerate("xyz"):
                row[f"{side}_camera_{axis}_m"] = (
                    f"{position[index, axis_index]:.9f}"
                )
            for axis_index, axis in enumerate("xyzw"):
                row[f"{side}_q{axis}"] = f"{quaternion[index, axis_index]:.12f}"
            row[f"{side}_filter_status"] = (
                "ZERO_PHASE_GAP_GUARD_SAVGOL"
                if guarded[index]
                else "ZERO_PHASE_SAVGOL"
            )
        audit["sides"][side] = {
            "closed_excursion_intervals": repaired_intervals,
            "closed_excursion_guarded_frames": int(np.count_nonzero(guarded)),
            "position_correction_m": {
                "median": float(np.median(position_correction)),
                "p95": float(np.percentile(position_correction, 95)),
                "max": float(np.max(position_correction)),
            },
            "orientation_correction_deg": {
                "median": float(np.median(rotation_correction)),
                "p95": float(np.percentile(rotation_correction, 95)),
                "max": float(np.max(rotation_correction)),
            },
        }
    audit["timestamp_or_frame_changed"] = not (
        np.array_equal(timestamps, original_timestamps)
        and np.array_equal(frames, original_frames)
    )
    return audit


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
    maximum_interpolation_gap_s: float = MAXIMUM_TRUSTED_INTERPOLATION_GAP_S,
    maximum_paired_timestamp_delta_s: float = 0.010,
    imu_streams: dict[str, ImuSeries] | None = None,
    imu_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write synchronized left/right poses in one explicitly shared map."""
    if (
        not np.isfinite(maximum_interpolation_gap_s)
        or maximum_interpolation_gap_s <= 0
        or maximum_interpolation_gap_s > MAXIMUM_TRUSTED_INTERPOLATION_GAP_S
    ):
        raise ValueError(
            "maximum_interpolation_gap_s must be positive and no greater than 0.25 s"
        )
    if (
        not np.isfinite(maximum_paired_timestamp_delta_s)
        or maximum_paired_timestamp_delta_s < 0
    ):
        raise ValueError("maximum_paired_timestamp_delta_s must be finite and non-negative")
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
        "joint_has_pose", "joint_valid", "joint_measured",
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
            f"{side}_filter_status",
        ))
    output_rows = []
    joint_pose_count = 0
    joint_valid_count = 0
    joint_measured_count = 0
    trusted_interpolation_gaps: dict[str, list[float]] = {
        "left": [], "right": []
    }
    rejected_interpolation_gaps: dict[str, list[float]] = {
        "left": [], "right": []
    }
    untrusted_side_frames = {"left": 0, "right": 0}
    untrusted_joint_frames = 0
    held_untrusted_side_frames = {"left": 0, "right": 0}
    imu_assisted_side_frames = {"left": 0, "right": 0}
    trusted_imu_assisted_side_frames = {"left": 0, "right": 0}
    untrusted_imu_assisted_side_frames = {"left": 0, "right": 0}
    accelerometer_assisted_side_frames = {"left": 0, "right": 0}
    visual_long_gap_fallback_side_frames = {"left": 0, "right": 0}
    imu_bridge_maximum_sample_gap_s = {"left": 0.0, "right": 0.0}
    imu_bridge_maximum_endpoint_closure_deg = {"left": 0.0, "right": 0.0}
    accelerometer_bridge_maximum_deviation_m = {"left": 0.0, "right": 0.0}
    imu_fallback_reasons: dict[str, dict[str, int]] = {"left": {}, "right": {}}
    accelerometer_fallback_reasons: dict[str, dict[str, int]] = {
        "left": {}, "right": {}
    }
    paired_timestamp_deltas_s: list[float] = []
    imu_streams = {} if imu_streams is None else imu_streams

    valid_series = {}
    for side in ("left", "right"):
        valid = [
            row for row in by_side[side].values()
            if row["quality_status"] == "valid" and row["camera_x_m"]
        ]
        valid.sort(key=lambda row: float(row["timestamp"]))
        if not valid:
            raise ValueError(f"{side} trajectory has no accepted pose to recover from")
        series_times = np.asarray([float(row["timestamp"]) for row in valid])
        series_positions = np.asarray([
            [float(row[key]) for key in ("camera_x_m", "camera_y_m", "camera_z_m")]
            for row in valid
        ])
        series_rotations = (
            Rotation.from_quat(np.asarray([
                [float(row[key]) for key in ("qx", "qy", "qz", "qw")]
                for row in valid
            ]))
            if valid else None
        )
        valid_series[side] = (
            series_times, series_positions, series_rotations,
            Slerp(series_times, series_rotations) if len(valid) >= 2 else None,
        )
    for frame in frames:
        left = by_side["left"].get(frame)
        right = by_side["right"].get(frame)
        timestamps = [
            float(row["timestamp"]) for row in (left, right) if row is not None
        ]
        if len(timestamps) == 2:
            paired_delta_s = max(timestamps) - min(timestamps)
            if paired_delta_s > maximum_paired_timestamp_delta_s + 1e-12:
                raise ValueError(
                    "left/right aligned frame timestamps exceed the paired limit "
                    f"at frame {frame}: {paired_delta_s:.9f} s"
                )
            paired_timestamp_deltas_s.append(paired_delta_s)
        joint_measured = bool(
            left is not None and right is not None
            and left["quality_status"] == "valid"
            and right["quality_status"] == "valid"
        )
        resolved = {"left": left, "right": right}
        # InstaUMI frame indices are already paired in the shared dataset clock.
        # Preserve their sub-frame phase difference and publish the stable right
        # 29.97 Hz camera timeline as the canonical joint timestamp.
        now_s = float(right["timestamp"]) if right is not None else timestamps[0]
        for side in ("left", "right"):
            source = resolved[side]
            if source is not None and source["quality_status"] == "valid":
                source = dict(source)
                source["pose_state"] = "MEASURED"
                resolved[side] = source
                continue
            if source is not None:
                source = dict(source)
                source["pose_state"] = "REJECTED"
                resolved[side] = source
            series_times, series_positions, _, slerp = valid_series[side]
            if len(series_times) < 2 or not series_times[0] < now_s < series_times[-1]:
                nearest = int(np.argmin(np.abs(series_times - now_s)))
                position = series_positions[nearest]
                quaternion = valid_series[side][2][nearest].as_quat()
                held_untrusted_side_frames[side] += 1
                resolved[side] = {
                    "camera_x_m": f"{position[0]:.9f}",
                    "camera_y_m": f"{position[1]:.9f}",
                    "camera_z_m": f"{position[2]:.9f}",
                    "qx": f"{quaternion[0]:.12f}",
                    "qy": f"{quaternion[1]:.12f}",
                    "qz": f"{quaternion[2]:.12f}",
                    "qw": f"{quaternion[3]:.12f}",
                    "quality_status": "pose_untrusted",
                    "pose_state": "HELD_UNTRUSTED",
                    "angular_rmse_deg": "",
                    "detected_tag_count": "",
                    "inlier_tag_count": "",
                    "measurement_source": "nearest_accepted_pose_hold_outside_measurement_span",
                }
                continue
            upper = int(np.searchsorted(series_times, now_s, side="right"))
            lower = upper - 1
            if lower < 0 or upper >= len(series_times):
                continue
            gap = float(series_times[upper] - series_times[lower])
            source_status = (
                "missing"
                if source is None
                else str(source.get("quality_status", "rejected"))
            )
            if gap > maximum_interpolation_gap_s + 1e-12:
                rejected_interpolation_gaps[side].append(gap)
                untrusted_side_frames[side] += 1
                position = np.asarray([
                    np.interp(now_s, series_times, series_positions[:, axis])
                    for axis in range(3)
                ])
                quaternion = slerp([now_s]).as_quat()[0]
                pose_state = "INTERPOLATED_UNTRUSTED"
                quality_status = "interpolation_untrusted"
                measurement_source = (
                    "temporal_interpolation_rejected_long_gap:"
                    f"source={source_status}"
                )
                stream = imu_streams.get(side)
                if stream is None:
                    reason = "imu_stream_unavailable"
                    visual_long_gap_fallback_side_frames[side] += 1
                    imu_fallback_reasons[side][reason] = (
                        imu_fallback_reasons[side].get(reason, 0) + 1
                    )
                else:
                    try:
                        bridge = stream.bridge_orientations(
                            float(series_times[lower]),
                            valid_series[side][2][lower],
                            float(series_times[upper]),
                            valid_series[side][2][upper],
                            np.asarray([now_s]),
                        )
                    except ImuAssistanceUnavailable as exc:
                        reason = str(exc)
                        visual_long_gap_fallback_side_frames[side] += 1
                        imu_fallback_reasons[side][reason] = (
                            imu_fallback_reasons[side].get(reason, 0) + 1
                        )
                    else:
                        quaternion = bridge.rotations.as_quat()[0]
                        pose_state = "IMU_ASSISTED_UNTRUSTED"
                        quality_status = "imu_assisted_untrusted"
                        measurement_source = (
                            "visual_position_interpolation+calibrated_gyro_bridge:"
                            f"source={source_status}"
                        )
                        imu_assisted_side_frames[side] += 1
                        untrusted_imu_assisted_side_frames[side] += 1
                        imu_bridge_maximum_sample_gap_s[side] = max(
                            imu_bridge_maximum_sample_gap_s[side],
                            bridge.maximum_sample_gap_s,
                        )
                        imu_bridge_maximum_endpoint_closure_deg[side] = max(
                            imu_bridge_maximum_endpoint_closure_deg[side],
                            bridge.endpoint_closure_deg,
                        )
                        try:
                            acceleration_bridge = stream.bridge_positions(
                                float(series_times[lower]),
                                series_positions[lower],
                                valid_series[side][2][lower],
                                float(series_times[upper]),
                                series_positions[upper],
                                valid_series[side][2][upper],
                                np.asarray([now_s]),
                            )
                        except ImuAssistanceUnavailable as exc:
                            reason = str(exc)
                            accelerometer_fallback_reasons[side][reason] = (
                                accelerometer_fallback_reasons[side].get(reason, 0) + 1
                            )
                        else:
                            position = acceleration_bridge.positions_m[0]
                            measurement_source = (
                                "visual_endpoint_anchored_accelerometer_translation"
                                "+calibrated_gyro_bridge:"
                                f"source={source_status}"
                            )
                            accelerometer_assisted_side_frames[side] += 1
                            accelerometer_bridge_maximum_deviation_m[side] = max(
                                accelerometer_bridge_maximum_deviation_m[side],
                                acceleration_bridge.maximum_deviation_from_linear_m,
                            )
                resolved[side] = {
                    "camera_x_m": f"{position[0]:.9f}",
                    "camera_y_m": f"{position[1]:.9f}",
                    "camera_z_m": f"{position[2]:.9f}",
                    "qx": f"{quaternion[0]:.12f}",
                    "qy": f"{quaternion[1]:.12f}",
                    "qz": f"{quaternion[2]:.12f}",
                    "qw": f"{quaternion[3]:.12f}",
                    "quality_status": quality_status,
                    "pose_state": pose_state,
                    "angular_rmse_deg": "",
                    "detected_tag_count": "",
                    "inlier_tag_count": "",
                    "measurement_source": measurement_source,
                }
                continue
            position = np.asarray([
                np.interp(now_s, series_times, series_positions[:, axis])
                for axis in range(3)
            ])
            quaternion = slerp([now_s]).as_quat()[0]
            trusted_interpolation_gaps[side].append(gap)
            pose_state = "INTERPOLATED"
            quality_status = "interpolated"
            measurement_source = (
                "temporal_interpolation_between_cached_bearing_poses"
            )
            stream = imu_streams.get(side)
            if stream is not None:
                try:
                    bridge = stream.bridge_orientations(
                        float(series_times[lower]),
                        valid_series[side][2][lower],
                        float(series_times[upper]),
                        valid_series[side][2][upper],
                        np.asarray([now_s]),
                    )
                except ImuAssistanceUnavailable as exc:
                    reason = str(exc)
                    imu_fallback_reasons[side][reason] = (
                        imu_fallback_reasons[side].get(reason, 0) + 1
                    )
                else:
                    quaternion = bridge.rotations.as_quat()[0]
                    pose_state = "IMU_ASSISTED"
                    quality_status = "imu_assisted"
                    measurement_source = (
                        "visual_position_interpolation+calibrated_gyro_bridge:"
                        f"source={source_status}"
                    )
                    imu_assisted_side_frames[side] += 1
                    trusted_imu_assisted_side_frames[side] += 1
                    imu_bridge_maximum_sample_gap_s[side] = max(
                        imu_bridge_maximum_sample_gap_s[side],
                        bridge.maximum_sample_gap_s,
                    )
                    imu_bridge_maximum_endpoint_closure_deg[side] = max(
                        imu_bridge_maximum_endpoint_closure_deg[side],
                        bridge.endpoint_closure_deg,
                    )
                    try:
                        acceleration_bridge = stream.bridge_positions(
                            float(series_times[lower]),
                            series_positions[lower],
                            valid_series[side][2][lower],
                            float(series_times[upper]),
                            series_positions[upper],
                            valid_series[side][2][upper],
                            np.asarray([now_s]),
                        )
                    except ImuAssistanceUnavailable as exc:
                        reason = str(exc)
                        accelerometer_fallback_reasons[side][reason] = (
                            accelerometer_fallback_reasons[side].get(reason, 0) + 1
                        )
                    else:
                        position = acceleration_bridge.positions_m[0]
                        measurement_source = (
                            "visual_endpoint_anchored_accelerometer_translation"
                            "+calibrated_gyro_bridge:"
                            f"source={source_status}"
                        )
                        accelerometer_assisted_side_frames[side] += 1
                        accelerometer_bridge_maximum_deviation_m[side] = max(
                            accelerometer_bridge_maximum_deviation_m[side],
                            acceleration_bridge.maximum_deviation_from_linear_m,
                        )
            resolved[side] = {
                "camera_x_m": f"{position[0]:.9f}",
                "camera_y_m": f"{position[1]:.9f}",
                "camera_z_m": f"{position[2]:.9f}",
                "qx": f"{quaternion[0]:.12f}",
                "qy": f"{quaternion[1]:.12f}",
                "qz": f"{quaternion[2]:.12f}",
                "qw": f"{quaternion[3]:.12f}",
                "quality_status": quality_status,
                "pose_state": pose_state,
                "angular_rmse_deg": "",
                "detected_tag_count": "",
                "inlier_tag_count": "",
                "measurement_source": measurement_source,
            }
        untrusted_joint_frames += int(any(
            source is not None
            and source.get("pose_state") in {
                "INTERPOLATED_UNTRUSTED", "IMU_ASSISTED_UNTRUSTED"
            }
            for source in resolved.values()
        ))
        joint_has_pose = all(
            resolved[side] is not None
            and all(str(resolved[side].get(key, "")) for key in pose_keys)
            for side in ("left", "right")
        )
        joint_pose_count += int(joint_has_pose)
        joint_valid = all(
            resolved[side] is not None
            and resolved[side].get("quality_status")
                in {"valid", "interpolated", "imu_assisted"}
            for side in ("left", "right")
        )
        joint_valid_count += int(joint_valid)
        joint_measured_count += int(joint_measured)
        item: dict[str, Any] = {
            "frame": frame,
            "timestamp_s": f"{now_s:.9f}",
            "world_frame": "session_grid_A",
            "map_id": map_id,
            "joint_has_pose": str(joint_has_pose).lower(),
            "joint_valid": str(joint_valid).lower(),
            "joint_measured": str(joint_measured).lower(),
        }
        for side, source in resolved.items():
            for key in pose_keys:
                item[f"{side}_{key}"] = "" if source is None else source.get(key, "")
            for key in (
                "quality_status", "pose_state", "angular_rmse_deg", "detected_tag_count",
                "inlier_tag_count", "measurement_source",
            ):
                item[f"{side}_{key}"] = "" if source is None else source.get(key, "")
        output_rows.append(item)
    trajectory_filter = _zero_phase_filter_joint_rows(output_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    return {
        "common_timeline_frames": len(output_rows),
        "joint_pose_frames": joint_pose_count,
        "joint_pose_ratio": (
            joint_pose_count / len(output_rows) if output_rows else 0.0
        ),
        "joint_valid_frames": joint_valid_count,
        "joint_valid_ratio": (
            joint_valid_count / len(output_rows) if output_rows else 0.0
        ),
        "joint_measured_frames": joint_measured_count,
        "joint_measured_ratio": (
            joint_measured_count / len(output_rows) if output_rows else 0.0
        ),
        "maximum_allowed_interpolation_gap_s": maximum_interpolation_gap_s,
        "canonical_joint_timestamp_source": "right_camera_aligned_h5_timeline",
        "maximum_allowed_paired_timestamp_delta_s": (
            maximum_paired_timestamp_delta_s
        ),
        "maximum_paired_timestamp_delta_s": max(
            paired_timestamp_deltas_s, default=0.0
        ),
        "maximum_interpolation_gap_s": {
            side: max(trusted_interpolation_gaps[side], default=0.0)
            for side in ("left", "right")
        },
        "maximum_rejected_interpolation_gap_s": {
            side: max(rejected_interpolation_gaps[side], default=0.0)
            for side in ("left", "right")
        },
        "untrusted_long_gap_frames": untrusted_joint_frames,
        "untrusted_long_gap_side_frames": untrusted_side_frames,
        "held_untrusted_side_frames": held_untrusted_side_frames,
        "trajectory_filter": trajectory_filter,
        "imu_assistance": {
            **({} if imu_audit is None else imu_audit),
            "assisted_side_frames": imu_assisted_side_frames,
            "assisted_frames": sum(imu_assisted_side_frames.values()),
            "trusted_assisted_side_frames": trusted_imu_assisted_side_frames,
            "trusted_assisted_frames": sum(
                trusted_imu_assisted_side_frames.values()
            ),
            "untrusted_assisted_side_frames": untrusted_imu_assisted_side_frames,
            "untrusted_assisted_frames": sum(
                untrusted_imu_assisted_side_frames.values()
            ),
            "visual_long_gap_fallback_side_frames": (
                visual_long_gap_fallback_side_frames
            ),
            "visual_long_gap_fallback_frames": sum(
                visual_long_gap_fallback_side_frames.values()
            ),
            "bridge_maximum_sample_gap_s": imu_bridge_maximum_sample_gap_s,
            "bridge_maximum_endpoint_closure_deg": (
                imu_bridge_maximum_endpoint_closure_deg
            ),
            "fallback_reasons": imu_fallback_reasons,
            "accelerometer_assisted_side_frames": (
                accelerometer_assisted_side_frames
            ),
            "accelerometer_assisted_frames": sum(
                accelerometer_assisted_side_frames.values()
            ),
            "accelerometer_bridge_maximum_deviation_from_linear_m": (
                accelerometer_bridge_maximum_deviation_m
            ),
            "accelerometer_fallback_reasons": accelerometer_fallback_reasons,
        },
        "world_frame": "session_grid_A",
        "map_id": map_id,
    }
