#!/usr/bin/env python3
"""Recover both gripper bases directly from opposite-camera BaseTag views.

For each side the observer camera is localized from the immutable 200 mm wall
map.  Its raw-fisheye observation of the *other* gripper's 20 mm mount Tag then
places that gripper base directly in the same world frame::

    T_world_base = T_world_observer_camera @ T_camera_tag @ T_tag_base

This deliberately does not use camera-to-own-Tag or camera-to-base extrinsics.
Those quantities are unnecessary for the gripper pose and, at 6--10 cm range,
were the dominant source of the old left/right overlap error.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from osmo360.calibration.calibrate_basetag_reciprocal import Transform, interpolate_pose, load_pose, rotation_distance_deg
from tools.direct_reciprocal_world_pose_cached import stats
from osmo360.calibration.estimate_gripper_extrinsic import solve_bearing_ippe
from tools.osmo_360_offline import ImuPanoramaBridgeCalibrator, load_imu_quaternions
from tools.joint_dual_camera_pose_graph_cached import (
    FrameData,
    cache_index,
    detection_timeline,
    direct_map,
    largest_ids,
    load_initial_wall_transform,
    nearest_detection_frame,
    raw_fisheye_cache_audit,
    solve_camera_wall_candidates,
    solve_camera_wall_only,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "left-cache", "right-cache", "left-initial-pose", "right-initial-pose",
        "left-panel-map", "right-panel-map", "initial-world-map",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--left-tag-id", type=int, required=True)
    parser.add_argument("--right-tag-id", type=int, required=True)
    parser.add_argument("--tag-size-m", type=float, default=0.020)
    parser.add_argument("--tag-corner-quarter-turns", type=int, default=1)
    parser.add_argument("--left-tag-corner-quarter-turns", type=int, choices=range(4))
    parser.add_argument("--right-tag-corner-quarter-turns", type=int, choices=range(4))
    parser.add_argument("--start-common-s", type=float, required=True)
    parser.add_argument("--end-common-s", type=float, required=True)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--max-cross-basetag-center-error-deg", type=float, default=50.0)
    parser.add_argument("--max-cross-basetag-attitude-error-deg", type=float, default=180.0)
    parser.add_argument("--left-target-mount-json", type=Path)
    parser.add_argument("--right-target-mount-json", type=Path)
    parser.add_argument("--left-target-mount-quaternion", type=float, nargs=4)
    parser.add_argument("--right-target-mount-quaternion", type=float, nargs=4)
    parser.add_argument("--left-imu-csv", type=Path)
    parser.add_argument("--right-imu-csv", type=Path)
    parser.add_argument(
        "--cross-pose-mode",
        choices=("fixed-attitude-translation", "free-ippe"),
        default="fixed-attitude-translation",
        help=(
            "Solve a physical 20 mm BaseTag with a fixed target attitude or "
            "retain both free-IPPE branches for the rigid-mount/Viterbi gate."
        ),
    )
    parser.add_argument(
        "--disable-left-observes-right", action="store_true",
        help="Reject the left-camera to right-BaseTag factor.",
    )
    parser.add_argument(
        "--disable-right-observes-left", action="store_true",
        help="Reject the right-camera to left-BaseTag factor.",
    )
    parser.add_argument(
        "--enable-single-tag-imu-translation", action="store_true",
        help=(
            "Diagnostic only. Enable single-wall-Tag translation with IMU-fixed "
            "attitude. Production callers must first pass a held-out metric gate."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def base_from_observer(
    world_observer_camera: Transform,
    observer_camera_tag: Transform,
    base_to_tag: Transform,
) -> Transform:
    """Compose the explicitly directed observer-camera -> Tag -> base chain."""
    return world_observer_camera.compose(observer_camera_tag).compose(
        base_to_tag.inverse()
    )


def _solution_transform(solution: dict) -> Transform:
    return Transform(
        np.asarray(solution["translation_tag_origin_in_panorama_m"], dtype=float),
        Rotation.from_matrix(solution["rotation_tag_to_panorama"]),
    )


def _continuous_ippe(
    solutions: list[dict], previous: Transform | None, delta_s: float = 1.0 / 30.0,
) -> Transform:
    """Choose an IPPE branch without a camera-to-own-Tag prior.

    BaseTag IDs are unique in production captures.  The first observation uses
    only the raw angular fit; subsequent observations use temporal continuity
    to prevent planar mirror-branch flips.  No scene-specific position or
    contact target enters the decision.
    """
    candidates = [(_solution_transform(item), float(item["angular_rmse_deg"]))
                  for item in solutions]
    if previous is None:
        return min(candidates, key=lambda item: item[1])[0]
    # Allow real motion in proportion to the elapsed time, but retain the last
    # physical branch across an occlusion. Resetting after 150 ms made the
    # lower-RMSE mirror branch look like a new valid trajectory segment.
    position_scale = max(0.05, 2.0 * max(delta_s, 0.0))
    rotation_scale = max(20.0, 240.0 * max(delta_s, 0.0))
    return min(
        candidates,
        key=lambda item: (
            rotation_distance_deg(previous.r, item[0].r) / rotation_scale
            + np.linalg.norm(previous.p - item[0].p) / position_scale
            + item[1] / 2.0
        ),
    )[0]


def _viterbi_ippe(observations: list[tuple[float, list[dict]]]) -> list[Transform]:
    """Select one coherent IPPE branch over the complete observation track.

    A greedy selector can permanently change planar branch after an occlusion.
    This dynamic program sees both sides of every gap and minimizes physical
    motion plus the raw angular reprojection error.  It uses no task/contact
    target and is therefore shared by every capture.
    """
    if not observations:
        return []
    candidates = [
        [(_solution_transform(item), float(item["angular_rmse_deg"])) for item in solutions]
        for _, solutions in observations
    ]
    costs = [np.asarray([rmse / 2.0 for _, rmse in candidates[0]], dtype=float)]
    parents: list[np.ndarray] = []
    for index in range(1, len(candidates)):
        delta_s = max(observations[index][0] - observations[index - 1][0], 1.0 / 60.0)
        # These are permissive motion envelopes, not scene-specific priors.
        position_scale = max(0.05, 2.0 * delta_s)
        rotation_scale = max(20.0, 240.0 * delta_s)
        current = np.full(len(candidates[index]), np.inf, dtype=float)
        parent = np.full(len(candidates[index]), -1, dtype=int)
        for j, (pose, rmse) in enumerate(candidates[index]):
            for k, (previous, _) in enumerate(candidates[index - 1]):
                transition = (
                    rotation_distance_deg(previous.r, pose.r) / rotation_scale
                    + np.linalg.norm(previous.p - pose.p) / position_scale
                )
                score = costs[-1][k] + transition + rmse / 2.0
                if score < current[j]:
                    current[j], parent[j] = score, k
        costs.append(current)
        parents.append(parent)
    branch = int(np.argmin(costs[-1]))
    selected = [None] * len(candidates)
    for index in range(len(candidates) - 1, -1, -1):
        selected[index] = candidates[index][branch][0]
        if index:
            branch = int(parents[index - 1][branch])
    return selected


def _fixed_mount_ippe(
    observations: list[tuple[float, list[dict], Transform, Transform]],
    fixed_mount_rotation: Rotation | None = None,
) -> tuple[list[Transform], dict]:
    """Resolve planar branches using the target camera's rigid mount rotation.

    ``observer`` sees the BaseTag carried by ``target``.  For every candidate,
    ``R_target_camera_tag`` must be one constant rotation over the capture.
    Translation is deliberately excluded because that was the biased quantity
    in the old close-range self-Tag calibration.
    """
    if not observations:
        return [], {"count": 0}
    candidate_poses, mount_rotations = [], []
    for _, solutions, observer, target in observations:
        poses = [_solution_transform(item) for item in solutions]
        candidate_poses.append(poses)
        mount_rotations.append([
            target.r.inv() * observer.r * pose.r for pose in poses
        ])
    best_score, best_indices = np.inf, None
    # Every observed branch is a possible seed.  The true rigid cluster wins;
    # the planar mirror branch changes with viewing direction.
    seed_sets = [[fixed_mount_rotation]] if fixed_mount_rotation is not None else mount_rotations
    for seed_set in seed_sets:
        for seed in seed_set:
            indices, residuals = [], []
            for rotations in mount_rotations:
                distances = [rotation_distance_deg(seed, value) for value in rotations]
                choice = int(np.argmin(distances))
                indices.append(choice)
                residuals.append(distances[choice])
            residuals = np.asarray(residuals)
            score = float(np.median(residuals) + np.quantile(residuals, 0.9))
            if score < best_score:
                best_score, best_indices = score, indices
    selected = [poses[index] for poses, index in zip(candidate_poses, best_indices)]
    selected_mounts = [values[index] for values, index in zip(mount_rotations, best_indices)]
    # Report dispersion about the rotation medoid; this is an auditable rigid
    # mount check rather than a task-specific pose constraint.
    center = fixed_mount_rotation
    if center is None:
        center = min(
            selected_mounts,
            key=lambda candidate: sum(
                rotation_distance_deg(candidate, value) for value in selected_mounts
            ),
        )
    residuals = [rotation_distance_deg(center, value) for value in selected_mounts]
    return selected, {
        "count": len(selected),
        "median_residual_deg": float(np.median(residuals)),
        "p95_residual_deg": float(np.quantile(residuals, 0.95)),
        "max_residual_deg": float(np.max(residuals)),
        "reference": "frozen_calibration" if fixed_mount_rotation is not None else "capture_medoid",
        "reference_quaternion_xyzw": center.as_quat().tolist(),
    }


def _load_mount_rotation(path: Path | None) -> Rotation | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    transform = payload.get("camera_to_basetag", payload)
    return Rotation.from_quat(transform["quaternion_xyzw"])


def _imu_predictor(pose_path: Path, imu_path: Path | None):
    if imu_path is None:
        return None, {"status": "disabled"}
    imu = load_imu_quaternions(imu_path)
    calibrator = ImuPanoramaBridgeCalibrator(max_observations=96)
    with pose_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("quality_status") != "valid" or not row.get("qx"):
                continue
            frame = int(row["frame"])
            if frame not in imu:
                continue
            rotation = Rotation.from_quat([
                float(row[key]) for key in ("qx", "qy", "qz", "qw")
            ])
            calibrator.add_observation(frame, rotation.as_matrix(), imu[frame])

    def predict(frame: int) -> Rotation | None:
        if frame not in imu:
            return None
        matrix = calibrator.predict(imu[frame])
        return None if matrix is None else Rotation.from_matrix(matrix)

    return predict, calibrator.audit()


def _interpolate_pose_clamped(series, time_s: float) -> Transform:
    """Use a pose series only as an optimizer seed, never as a coverage gate."""
    times, positions, rotations = series
    if time_s <= times[0]:
        return Transform(positions[0].copy(), rotations[0])
    if time_s >= times[-1]:
        return Transform(positions[-1].copy(), rotations[-1])
    pose = interpolate_pose(series, time_s)
    assert pose is not None
    return pose


def _joint_branch_choice(
    left_candidates, right_candidates, lr_solutions, rl_solutions,
    left_mount: Rotation | None, right_mount: Rotation | None,
    left_prior: Transform, right_prior: Transform,
):
    """Choose both wall and cross-Tag planar branches as one rigid system."""
    left_best = min(value for value, _pose in left_candidates)
    right_best = min(value for value, _pose in right_candidates)
    visual_allowance = np.deg2rad(0.25) ** 2 / 3.0
    left_candidates = [item for item in left_candidates if item[0] <= left_best + visual_allowance]
    right_candidates = [item for item in right_candidates if item[0] <= right_best + visual_allowance]
    lr_poses = [_solution_transform(item) for item in lr_solutions] if lr_solutions else [None]
    rl_poses = [_solution_transform(item) for item in rl_solutions] if rl_solutions else [None]
    best = None
    for left_mse, left_pose in left_candidates:
        for right_mse, right_pose in right_candidates:
            for lr_pose in lr_poses:
                for rl_pose in rl_poses:
                    score = (left_mse - left_best + right_mse - right_best) / visual_allowance
                    # Frozen calibration is the strongest branch-only invariant.
                    if lr_pose is not None and right_mount is not None:
                        observed = right_pose.r.inv() * left_pose.r * lr_pose.r
                        score += rotation_distance_deg(observed, right_mount) / 4.0
                    if rl_pose is not None and left_mount is not None:
                        observed = left_pose.r.inv() * right_pose.r * rl_pose.r
                        score += rotation_distance_deg(observed, left_mount) / 4.0
                    # Weak independent continuity prior breaks exact ties only.
                    score += rotation_distance_deg(left_pose.r, left_prior.r) / 60.0
                    score += rotation_distance_deg(right_pose.r, right_prior.r) / 60.0
                    if best is None or score < best[0]:
                        best = (score, left_pose, right_pose, lr_pose, rl_pose)
    return best[1:]


def _fixed_rotation_wall_translation(
    rotation: Rotation,
    leftwall: list[tuple[np.ndarray, np.ndarray]],
    rightwall: list[tuple[np.ndarray, np.ndarray]],
    wall: Transform,
) -> tuple[Transform, float] | None:
    """Solve camera centre from raw unit rays while IMU fixes attitude."""
    points, rays = [], []
    for observations, panel in ((leftwall, None), (rightwall, wall)):
        for object_points, camera_rays in observations:
            points.append(
                object_points if panel is None else panel.r.apply(object_points) + panel.p
            )
            rays.append(rotation.apply(camera_rays))
    if not points:
        return None
    points = np.concatenate(points)
    directions = np.concatenate(rays)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    projectors = np.eye(3)[None, :, :] - directions[:, :, None] * directions[:, None, :]
    system = projectors.sum(axis=0)
    rhs = np.einsum("nij,nj->i", projectors, points)
    try:
        initial = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        return None

    def residual(center):
        predicted = points - center.reshape(1, 3)
        predicted /= np.linalg.norm(predicted, axis=1, keepdims=True)
        return (predicted - directions).reshape(-1)

    fit = least_squares(residual, initial, loss="huber", f_scale=0.002, max_nfev=100)
    if not fit.success or not np.isfinite(fit.x).all():
        return None
    predicted = points - fit.x.reshape(1, 3)
    predicted /= np.linalg.norm(predicted, axis=1, keepdims=True)
    angles = np.degrees(np.arccos(np.clip(np.sum(predicted * directions, axis=1), -1, 1)))
    return Transform(fit.x, rotation), float(np.sqrt(np.mean(angles * angles)))


def _fixed_rotation_tag_translation(rotation_tag_to_camera: Rotation,
                                    tag_points: np.ndarray,
                                    rays_camera: np.ndarray
                                    ) -> tuple[Transform, float] | None:
    """Solve a small Tag's translation while its target IMU fixes attitude."""
    directions = np.asarray(rays_camera, dtype=float)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    rotated = rotation_tag_to_camera.apply(np.asarray(tag_points, dtype=float))
    projectors = np.eye(3)[None, :, :] - directions[:, :, None] * directions[:, None, :]
    system = projectors.sum(axis=0)
    rhs = -np.einsum("nij,nj->i", projectors, rotated)
    try:
        initial = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        return None

    def residual(translation):
        predicted = rotated + translation.reshape(1, 3)
        predicted /= np.linalg.norm(predicted, axis=1, keepdims=True)
        return (predicted - directions).reshape(-1)

    fit = least_squares(residual, initial, loss="huber", f_scale=0.0015, max_nfev=80)
    if not fit.success or not np.isfinite(fit.x).all():
        return None
    predicted = rotated + fit.x.reshape(1, 3)
    if np.any(np.einsum("ij,ij->i", predicted, directions) <= 0):
        return None
    predicted /= np.linalg.norm(predicted, axis=1, keepdims=True)
    angles = np.degrees(np.arccos(np.clip(np.sum(predicted * directions, axis=1), -1, 1)))
    return Transform(fit.x, rotation_tag_to_camera), float(np.sqrt(np.mean(angles * angles)))


def _continuous_wall_candidate(candidates, previous: Transform | None,
                               delta_s: float,
                               maximum_speed_mps: float = 6.0,
                               maximum_angular_speed_dps: float = 800.0) -> Transform | None:
    """Choose a physically reachable wall branch or fail closed."""
    if not candidates:
        return None
    if previous is None:
        return candidates[0][1]
    dt = max(float(delta_s), 1.0 / 120.0)
    feasible = []
    for mse, pose in candidates:
        speed = float(np.linalg.norm(pose.p - previous.p) / dt)
        angular_speed = rotation_distance_deg(previous.r, pose.r) / dt
        if speed <= maximum_speed_mps and angular_speed <= maximum_angular_speed_dps:
            feasible.append((mse, speed, angular_speed, pose))
    if not feasible:
        return None
    best_mse = min(item[0] for item in feasible)
    return min(
        feasible,
        key=lambda item: (
            (item[0] - best_mse) / max(np.deg2rad(0.5) ** 2 / 3.0, 1e-12)
            + item[1] / maximum_speed_mps
            + item[2] / maximum_angular_speed_dps
        ),
    )[3]


def _wall_observation_rms_deg(pose: Transform, observation, panel: Transform | None) -> float:
    points, rays = observation
    world_points = points if panel is None else panel.r.apply(points) + panel.p
    predicted = pose.r.inv().apply(world_points - pose.p)
    predicted /= np.linalg.norm(predicted, axis=1, keepdims=True)
    rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
    angles = np.degrees(np.arccos(np.clip(np.sum(predicted * rays, axis=1), -1.0, 1.0)))
    return float(np.sqrt(np.mean(angles * angles)))


def _robust_wall_recovery(initial: Transform, leftwall, rightwall, wall: Transform,
                          maximum_tag_rms_deg: float = 2.5):
    """Recover a wall pose after rejecting whole-Tag outliers.

    A false positive contributes four mutually consistent corners, so a Huber
    loss on individual corners is insufficient.  Two-Tag hypotheses are
    scored by the number of complete physical Tags they explain.
    """
    observations = [("left", value) for value in leftwall] + [
        ("right", value) for value in rightwall
    ]
    if len(observations) < 3:
        return []
    hypotheses = []
    for first, second in itertools.combinations(range(len(observations)), 2):
        chosen = [observations[first], observations[second]]
        ll = [value for panel, value in chosen if panel == "left"]
        rr = [value for panel, value in chosen if panel == "right"]
        for _mse, pose in solve_camera_wall_candidates(initial, ll, rr, wall):
            residuals = [
                _wall_observation_rms_deg(
                    pose, value, None if panel == "left" else wall
                )
                for panel, value in observations
            ]
            inliers = tuple(
                index for index, residual in enumerate(residuals)
                if residual <= maximum_tag_rms_deg
            )
            if len(inliers) >= 2:
                hypotheses.append((
                    -len(inliers), float(np.median([residuals[i] for i in inliers])),
                    np.linalg.norm(pose.p - initial.p), inliers,
                ))
    if not hypotheses:
        return []
    inliers = min(hypotheses) [3]
    ll = [observations[i][1] for i in inliers if observations[i][0] == "left"]
    rr = [observations[i][1] for i in inliers if observations[i][0] == "right"]
    return solve_camera_wall_candidates(initial, ll, rr, wall)


def build_direct_frames(args, left_points, right_points, tag_points, wall):
    """Build wall and cross-Tag observations with no own-mount extrinsic."""
    left_cache = np.load(args.left_cache)
    right_cache = np.load(args.right_cache)
    left_index = cache_index(left_cache)
    right_index = cache_index(right_cache)
    left_frames, left_times = detection_timeline(left_cache, left_index)
    right_frames, right_times = detection_timeline(right_cache, right_index)
    left_series = load_pose(args.left_initial_pose)
    right_series = load_pose(args.right_initial_pose)
    left_imu_predict, left_imu_audit = _imu_predictor(
        args.left_initial_pose, args.left_imu_csv
    )
    right_imu_predict, right_imu_audit = _imu_predictor(
        args.right_initial_pose, args.right_imu_csv
    )
    timeline = np.asarray(right_cache["timeline_common_time_s"], dtype=float)
    left_common_timeline = np.asarray(left_cache["timeline_common_time_s"], dtype=float)
    left_local_timeline = np.asarray(left_cache["timeline_local_time_s"], dtype=float)
    right_common_timeline = np.asarray(right_cache["timeline_common_time_s"], dtype=float)
    right_local_timeline = np.asarray(right_cache["timeline_local_time_s"], dtype=float)
    times = timeline[
        (timeline >= args.start_common_s) & (timeline <= args.end_common_s)
    ][::args.sample_stride]
    left_turns = (
        args.left_tag_corner_quarter_turns
        if args.left_tag_corner_quarter_turns is not None
        else args.tag_corner_quarter_turns
    )
    right_turns = (
        args.right_tag_corner_quarter_turns
        if args.right_tag_corner_quarter_turns is not None
        else args.tag_corner_quarter_turns
    )
    right_mount = (
        Rotation.from_quat(args.right_target_mount_quaternion)
        if args.right_target_mount_quaternion is not None
        else _load_mount_rotation(args.right_target_mount_json)
    )
    left_mount = (
        Rotation.from_quat(args.left_target_mount_quaternion)
        if args.left_target_mount_quaternion is not None
        else _load_mount_rotation(args.left_target_mount_json)
    )
    preliminary = []
    wall_quality = {}
    lr_observations = []
    rl_observations = []
    audit = {
        "left_observes_right": {"candidate_frames": 0, "accepted_frames": 0},
        "right_observes_left": {"candidate_frames": 0, "accepted_frames": 0},
        "imu_attitude_bridge": {"left": left_imu_audit, "right": right_imu_audit},
        "imu_guided_wall_frames": {"left": 0, "right": 0},
        "single_tag_imu_translation_frames": {"left": 0, "right": 0},
        "wall_motion_gate_rejected_frames": {"left": 0, "right": 0},
        "wall_tag_group_ransac_recovered_frames": {"left": 0, "right": 0},
        "fixed_attitude_cross_translation_rmse_deg": {
            "left_observes_right": [], "right_observes_left": [],
        },
    }
    previous_left_camera = previous_right_camera = None
    previous_left_time = previous_right_time = None
    for time_s in times:
        time_s = float(time_s)
        left_local_time = float(np.interp(
            time_s, left_common_timeline, left_local_timeline
        ))
        right_local_time = float(np.interp(
            time_s, right_common_timeline, right_local_timeline
        ))
        initial_left = _interpolate_pose_clamped(left_series, left_local_time)
        initial_right = _interpolate_pose_clamped(right_series, right_local_time)
        left_frame = nearest_detection_frame(left_frames, left_times, time_s)
        right_frame = nearest_detection_frame(right_frames, right_times, time_s)
        if left_frame is None or right_frame is None:
            continue
        left_detections = largest_ids(left_cache, left_index.get(left_frame, []))
        right_detections = largest_ids(right_cache, right_index.get(right_frame, []))

        def wall_observations(cache, detections, points):
            return [
                (corners, np.asarray(cache["rays_camera"][detections[tag_id]], dtype=float))
                for tag_id, corners in points.items() if tag_id in detections
            ]

        ll = wall_observations(left_cache, left_detections, left_points)
        lrw = wall_observations(left_cache, left_detections, right_points)
        rl = wall_observations(right_cache, right_detections, left_points)
        rrw = wall_observations(right_cache, right_detections, right_points)
        # Wall localization is independent of the moving BaseTags and must be
        # resolved first.  Using the old pose CSV here mixed reference frames
        # and made the later rigid-mount branch gate meaningless.
        left_imu_rotation = left_imu_predict(left_frame) if left_imu_predict else None
        right_imu_rotation = right_imu_predict(right_frame) if right_imu_predict else None
        left_count = len(ll) + len(lrw)
        right_count = len(rl) + len(rrw)
        def direct_wall_count(cache, detections):
            return sum(
                (tag_id in left_points or tag_id in right_points)
                and (
                    "detection_source" not in cache.files
                    or str(cache["detection_source"][index]) != "lk_forward_backward"
                )
                for tag_id, index in detections.items()
            )

        left_direct_wall_count = direct_wall_count(left_cache, left_detections)
        right_direct_wall_count = direct_wall_count(right_cache, right_detections)
        left_full = left_count >= 2
        right_full = right_count >= 2
        left_lk = left_full and left_direct_wall_count < 2
        right_lk = right_full and right_direct_wall_count < 2
        left_hybrid = (
            args.enable_single_tag_imu_translation
            and left_count == 1 and left_imu_rotation is not None
        )
        right_hybrid = (
            args.enable_single_tag_imu_translation
            and right_count == 1 and right_imu_rotation is not None
        )
        left_wall_ok = left_full or left_hybrid
        right_wall_ok = right_full or right_hybrid
        if not left_wall_ok and not right_wall_ok:
            continue
        left_seed = previous_left_camera or initial_left
        right_seed = previous_right_camera or initial_right
        if left_imu_rotation is not None and previous_left_camera is None:
            left_seed = Transform(left_seed.p, left_imu_rotation)
        if right_imu_rotation is not None and previous_right_camera is None:
            right_seed = Transform(right_seed.p, right_imu_rotation)
        left_wall_candidates = (
            solve_camera_wall_candidates(left_seed, ll, lrw, wall)
            if left_full else [(0.0, left_seed)]
        )
        right_wall_candidates = (
            solve_camera_wall_candidates(right_seed, rl, rrw, wall)
            if right_full else [(0.0, right_seed)]
        )
        if left_hybrid:
            solved = _fixed_rotation_wall_translation(left_imu_rotation, ll, lrw, wall)
            if solved is None or solved[1] > 0.20:
                left_wall_ok = left_hybrid = False
            else:
                left_wall_candidates = [(0.0, solved[0])]
                audit["single_tag_imu_translation_frames"]["left"] += 1
        if right_hybrid:
            solved = _fixed_rotation_wall_translation(right_imu_rotation, rl, rrw, wall)
            if solved is None or solved[1] > 0.20:
                right_wall_ok = right_hybrid = False
            else:
                right_wall_candidates = [(0.0, solved[0])]
                audit["single_tag_imu_translation_frames"]["right"] += 1
        if left_full:
            chosen = _continuous_wall_candidate(
                left_wall_candidates, previous_left_camera,
                time_s - previous_left_time if previous_left_time is not None else 1.0 / 30.0,
            )
            soft_choice = _continuous_wall_candidate(
                left_wall_candidates, previous_left_camera,
                time_s - previous_left_time if previous_left_time is not None else 1.0 / 30.0,
                2.0, 240.0,
            )
            if chosen is None or soft_choice is None:
                recovered = _robust_wall_recovery(left_seed, ll, lrw, wall)
                robust_choice = _continuous_wall_candidate(
                    recovered, previous_left_camera,
                    time_s - previous_left_time if previous_left_time is not None else 1.0 / 30.0,
                )
                if robust_choice is not None:
                    chosen = robust_choice
                    audit["wall_tag_group_ransac_recovered_frames"]["left"] += 1
            if chosen is None:
                left_wall_ok = left_full = left_lk = False
                left_wall_candidates = [(0.0, left_seed)]
                audit["wall_motion_gate_rejected_frames"]["left"] += 1
            else:
                left_wall_candidates = [(0.0, chosen)]
        if right_full:
            chosen = _continuous_wall_candidate(
                right_wall_candidates, previous_right_camera,
                time_s - previous_right_time if previous_right_time is not None else 1.0 / 30.0,
            )
            soft_choice = _continuous_wall_candidate(
                right_wall_candidates, previous_right_camera,
                time_s - previous_right_time if previous_right_time is not None else 1.0 / 30.0,
                2.0, 240.0,
            )
            if chosen is None or soft_choice is None:
                recovered = _robust_wall_recovery(right_seed, rl, rrw, wall)
                robust_choice = _continuous_wall_candidate(
                    recovered, previous_right_camera,
                    time_s - previous_right_time if previous_right_time is not None else 1.0 / 30.0,
                )
                if robust_choice is not None:
                    chosen = robust_choice
                    audit["wall_tag_group_ransac_recovered_frames"]["right"] += 1
            if chosen is None:
                right_wall_ok = right_full = right_lk = False
                right_wall_candidates = [(0.0, right_seed)]
                audit["wall_motion_gate_rejected_frames"]["right"] += 1
            else:
                right_wall_candidates = [(0.0, chosen)]
        if not left_wall_ok and not right_wall_ok:
            continue
        lr_solutions = rl_solutions = None
        cross_lr_rays = cross_rl_rays = None
        if (not args.disable_left_observes_right
                and left_wall_ok and args.right_tag_id in left_detections):
            index = left_detections[args.right_tag_id]
            cross_lr_rays = np.roll(
                np.asarray(left_cache["rays_camera"][index], dtype=float), -right_turns, axis=0,
            )
            lr_solutions = solve_bearing_ippe(tag_points, cross_lr_rays)
        if (not args.disable_right_observes_left
                and right_wall_ok and args.left_tag_id in right_detections):
            index = right_detections[args.left_tag_id]
            cross_rl_rays = np.roll(
                np.asarray(right_cache["rays_camera"][index], dtype=float), -left_turns, axis=0,
            )
            rl_solutions = solve_bearing_ippe(tag_points, cross_rl_rays)
        # The immutable 200 mm wall map is the only metric source for camera
        # pose.  A 20 mm moving BaseTag is useful for external calibration and
        # auditing, but must never vote on the planar wall branch: doing so made
        # the weak left camera jump while the wall reprojection stayed small.
        refined_left = left_wall_candidates[0][1]
        refined_right = right_wall_candidates[0][1]
        # For the small moving Tag, fix attitude from the target IMU/frozen
        # mount and solve only translation.  Free planar IPPE depth was the
        # dominant long-gap error.
        right_attitude = refined_right.r if right_wall_ok else right_imu_rotation
        left_attitude = refined_left.r if left_wall_ok else left_imu_rotation
        if (args.cross_pose_mode == "fixed-attitude-translation"
                and lr_solutions is not None and right_mount is not None
                and right_attitude is not None):
            world_right_base = right_attitude * right_mount
            rotation = refined_left.r.inv() * world_right_base
            solved = _fixed_rotation_tag_translation(rotation, tag_points, cross_lr_rays)
            if solved is None or solved[1] > 1.5:
                lr_solutions = None
            else:
                lr_solutions = [{
                    "translation_tag_origin_in_panorama_m": solved[0].p,
                    "rotation_tag_to_panorama": solved[0].r.as_matrix(),
                    "angular_rmse_deg": solved[1],
                }]
                audit["fixed_attitude_cross_translation_rmse_deg"][
                    "left_observes_right"
                ].append(solved[1])
        if (args.cross_pose_mode == "fixed-attitude-translation"
                and rl_solutions is not None and left_mount is not None
                and left_attitude is not None):
            world_left_base = left_attitude * left_mount
            rotation = refined_right.r.inv() * world_left_base
            solved = _fixed_rotation_tag_translation(rotation, tag_points, cross_rl_rays)
            if solved is None or solved[1] > 1.5:
                rl_solutions = None
            else:
                rl_solutions = [{
                    "translation_tag_origin_in_panorama_m": solved[0].p,
                    "rotation_tag_to_panorama": solved[0].r.as_matrix(),
                    "angular_rmse_deg": solved[1],
                }]
                audit["fixed_attitude_cross_translation_rmse_deg"][
                    "right_observes_left"
                ].append(solved[1])
        if left_imu_rotation is not None and rotation_distance_deg(
            refined_left.r, left_imu_rotation
        ) <= 30.0:
            audit["imu_guided_wall_frames"]["left"] += 1
        if right_imu_rotation is not None and rotation_distance_deg(
            refined_right.r, right_imu_rotation
        ) <= 30.0:
            audit["imu_guided_wall_frames"]["right"] += 1
        if left_wall_ok:
            previous_left_camera, previous_left_time = refined_left, time_s
        if right_wall_ok:
            previous_right_camera, previous_right_time = refined_right, time_s
        wall_quality[time_s] = {
            "left": "tracked_lk" if left_lk else "direct" if left_full else "hybrid" if left_hybrid else "unavailable",
            "right": "tracked_lk" if right_lk else "direct" if right_full else "hybrid" if right_hybrid else "unavailable",
        }
        cross_lr_pose = cross_rl_pose = None
        if lr_solutions is not None:
            audit["left_observes_right"]["candidate_frames"] += 1
            solutions = lr_solutions
            lr_observations.append((time_s, solutions, refined_left, refined_right))
            cross_lr_pose = ("lr", len(lr_observations) - 1)
            audit["left_observes_right"]["accepted_frames"] += 1
        if rl_solutions is not None:
            audit["right_observes_left"]["candidate_frames"] += 1
            solutions = rl_solutions
            rl_observations.append((time_s, solutions, refined_right, refined_left))
            cross_rl_pose = ("rl", len(rl_observations) - 1)
            audit["right_observes_left"]["accepted_frames"] += 1
        preliminary.append((
            time_s, refined_left, refined_right, ll, lrw, rl, rrw,
            cross_lr_rays, cross_rl_rays, cross_lr_pose, cross_rl_pose,
        ))
    if args.cross_pose_mode == "free-ippe":
        lr_selected = _viterbi_ippe([
            (time_s, solutions)
            for time_s, solutions, _observer, _target in lr_observations
        ])
        rl_selected = _viterbi_ippe([
            (time_s, solutions)
            for time_s, solutions, _observer, _target in rl_observations
        ])

        def viterbi_mount_audit(observations, selected, reference):
            if not selected:
                return {"count": 0, "selection": "temporal_viterbi"}
            rotations = [
                target.r.inv() * observer.r * pose.r
                for (_time, _solutions, observer, target), pose
                in zip(observations, selected)
            ]
            centre = reference or min(
                rotations,
                key=lambda candidate: sum(
                    rotation_distance_deg(candidate, value) for value in rotations
                ),
            )
            residuals = np.asarray([
                rotation_distance_deg(centre, value) for value in rotations
            ])
            return {
                "count": len(selected),
                "selection": "temporal_viterbi",
                "median_residual_deg": float(np.median(residuals)),
                "p95_residual_deg": float(np.quantile(residuals, 0.95)),
                "max_residual_deg": float(np.max(residuals)),
                "reference": "frozen_calibration" if reference is not None else "capture_medoid",
                "reference_quaternion_xyzw": centre.as_quat().tolist(),
            }

        lr_mount_audit = viterbi_mount_audit(
            lr_observations, lr_selected, right_mount
        )
        rl_mount_audit = viterbi_mount_audit(
            rl_observations, rl_selected, left_mount
        )
    else:
        lr_selected, lr_mount_audit = _fixed_mount_ippe(lr_observations, right_mount)
        rl_selected, rl_mount_audit = _fixed_mount_ippe(rl_observations, left_mount)
    audit["left_observes_right"]["rigid_rotation_gate"] = lr_mount_audit
    audit["right_observes_left"]["rigid_rotation_gate"] = rl_mount_audit
    resolved = []
    for item in preliminary:
        values = list(item)
        for slot, track in ((9, lr_selected), (10, rl_selected)):
            reference = values[slot]
            if isinstance(reference, tuple):
                values[slot] = track[reference[1]]
        resolved.append(tuple(values))
    frames = [
        FrameData(
            item[0], min(4, rank * 5 // max(len(resolved), 1)), *item[1:]
        )
        for rank, item in enumerate(resolved)
    ]
    return frames, audit, wall_quality


def write_pose(path: Path, rows: list[dict], side: str) -> None:
    fields = [
        "frame", "timestamp", "base_x_m", "base_y_m", "base_z_m",
        "x_m", "y_m", "z_m",
        "qx", "qy", "qz", "qw", "parent_frame", "child_frame",
        "measurement_source", "quality_status",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            pose = row[side]
            quaternion = pose.r.as_quat()
            writer.writerow({
                "frame": round(row["time_s"] * 30.0),
                "timestamp": f'{row["time_s"]:.9f}',
                "base_x_m": pose.p[0], "base_y_m": pose.p[1],
                "base_z_m": pose.p[2],
                "x_m": pose.p[0], "y_m": pose.p[1], "z_m": pose.p[2],
                "qx": quaternion[0],
                "qy": quaternion[1], "qz": quaternion[2], "qw": quaternion[3],
                "parent_frame": "tag_map", "child_frame": f"{side}_base_link",
                "measurement_source": "direct_opposite_basetag_raw_fisheye",
                "quality_status": "valid",
            })


def _recover_short_camera_gaps(rows: list[dict], side: str,
                               maximum_gap_frames: int = 2) -> int:
    """Interpolate only bounded 1--2-frame camera gaps for filtering output."""
    key = f"{side}_camera_mode"
    index = 0
    recovered = 0
    while index < len(rows):
        if rows[index][key] != "unavailable":
            index += 1
            continue
        end = index
        while end + 1 < len(rows) and rows[end + 1][key] == "unavailable":
            end += 1
        length = end - index + 1
        if index > 0 and end + 1 < len(rows) and length <= maximum_gap_frames:
            before, after = rows[index - 1], rows[end + 1]
            t0, t1 = before["time_s"], after["time_s"]
            p0 = before[f"{side}_camera"].p
            p1 = after[f"{side}_camera"].p
            slerp = Slerp(
                [t0, t1],
                Rotation.concatenate([
                    before[f"{side}_camera"].r,
                    after[f"{side}_camera"].r,
                ]),
            )
            for current in range(index, end + 1):
                t = rows[current]["time_s"]
                alpha = (t - t0) / (t1 - t0)
                rows[current][f"{side}_camera"] = Transform(
                    p0 + alpha * (p1 - p0), slerp([t])[0]
                )
                rows[current][key] = "short_gap"
                rows[current][f"{side}_camera_valid"] = True
                recovered += 1
        index = end + 1
    return recovered


def write_camera_pose(path: Path, rows: list[dict], side: str,
                      tag_map_sha256: str, detected_ids: list[int]) -> None:
    fields = [
        "frame", "timestamp", "camera_x_m", "camera_y_m", "camera_z_m",
        "qx", "qy", "qz", "qw", "parent_frame", "child_frame",
        "measurement_source", "quality_status",
        "direct_measurement", "tag_map_sha256", "detected_ids",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            pose = row[f"{side}_camera"]
            quaternion = pose.r.as_quat()
            mode = row[f"{side}_camera_mode"]
            if mode == "direct":
                measurement_source = "raw_fisheye_unit_bearing_wall"
                quality_status = "valid"
            elif mode == "hybrid":
                measurement_source = "imu_constrained_single_tag_translation"
                quality_status = "valid_hybrid"
            elif mode == "tracked_lk":
                measurement_source = "optical_flow"
                quality_status = "tracked"
            elif mode == "short_gap":
                measurement_source = "short_gap_interpolation"
                quality_status = "filtered"
            else:
                measurement_source = "imu_or_temporal_seed"
                quality_status = "recovered"
            writer.writerow({
                "frame": round(row["time_s"] * 30.0),
                "timestamp": f'{row["time_s"]:.9f}',
                "camera_x_m": pose.p[0], "camera_y_m": pose.p[1],
                "camera_z_m": pose.p[2],
                "qx": quaternion[0], "qy": quaternion[1],
                "qz": quaternion[2], "qw": quaternion[3],
                "parent_frame": "tag_map", "child_frame": "panorama_camera",
                "measurement_source": measurement_source,
                "quality_status": quality_status,
                "direct_measurement": "true" if mode == "direct" else "false",
                "tag_map_sha256": tag_map_sha256,
                "detected_ids": " ".join(map(str, detected_ids)),
            })


def _rigid_transform_audit(transforms: list[Transform]) -> dict:
    if not transforms:
        return {"count": 0}
    rotations = [value.r for value in transforms]
    sample = rotations[::max(1, len(rotations) // 120)]
    center_rotation = min(
        sample,
        key=lambda candidate: sum(
            rotation_distance_deg(candidate, value) for value in sample
        ),
    )
    rotation_residuals = np.asarray([
        rotation_distance_deg(center_rotation, value) for value in rotations
    ])
    rotation_inliers = rotation_residuals <= 12.0
    positions = np.asarray([
        value.p for value, keep in zip(transforms, rotation_inliers) if keep
    ])
    center_position = np.median(positions, axis=0)
    position_residuals = np.linalg.norm(positions - center_position, axis=1) * 1000.0
    # Fixed hardware must form one compact 6DoF mode.  Use a seed-neighbour
    # cluster (not transitive chaining) so a slowly drifting bad depth branch
    # cannot bridge otherwise distinct modes.
    best_cluster: list[int] = []
    for index, candidate in enumerate(transforms):
        cluster = [
            other for other, value in enumerate(transforms)
            if np.linalg.norm(candidate.p - value.p) <= 0.006
            and rotation_distance_deg(candidate.r, value.r) <= 3.0
        ]
        if len(cluster) > len(best_cluster):
            best_cluster = cluster
    compact = [transforms[index] for index in best_cluster]
    compact_positions = np.asarray([value.p for value in compact])
    compact_position = np.median(compact_positions, axis=0) if compact else np.zeros(3)
    compact_rotation = (
        min(
            [value.r for value in compact],
            key=lambda candidate: sum(
                rotation_distance_deg(candidate, value.r) for value in compact
            ),
        ) if compact else Rotation.identity()
    )
    compact_dp = (
        np.linalg.norm(compact_positions - compact_position, axis=1) * 1000.0
        if compact else np.asarray([])
    )
    compact_dr = np.asarray([
        rotation_distance_deg(compact_rotation, value.r) for value in compact
    ])
    return {
        "count": len(transforms),
        "rotation_inlier_count": int(rotation_inliers.sum()),
        "translation_m": center_position.tolist(),
        "quaternion_xyzw": center_rotation.as_quat().tolist(),
        "position_residual_mm": stats(position_residuals.tolist()),
        "rotation_residual_deg": stats(rotation_residuals[rotation_inliers].tolist()),
        "tight_6mm_3deg_cluster": {
            "count": len(compact),
            "fraction": len(compact) / len(transforms),
            "translation_m": compact_position.tolist(),
            "quaternion_xyzw": compact_rotation.as_quat().tolist(),
            "position_residual_mm": stats(compact_dp.tolist()),
            "rotation_residual_deg": stats(compact_dr.tolist()),
        },
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Shared loader compatibility.  These are intentionally absent: this
    # algorithm must not depend on camera-to-own-Tag calibration.
    args.left_camera_to_tag = None
    args.right_camera_to_tag = None
    args.temporal_blocks = 5
    provenance = {
        "left": raw_fisheye_cache_audit(args.left_cache),
        "right": raw_fisheye_cache_audit(args.right_cache),
    }
    left_map = direct_map(args.left_panel_map)
    right_map = direct_map(args.right_panel_map)
    tag_map_sha256 = hashlib.sha256(args.initial_world_map.read_bytes()).hexdigest()
    detected_wall_ids = sorted(set(left_map) | set(right_map))
    half = args.tag_size_m / 2.0
    tag_points = np.asarray([
        [-half, -half, 0.0], [half, -half, 0.0],
        [half, half, 0.0], [-half, half, 0.0],
    ])
    wall = load_initial_wall_transform(args.initial_world_map)
    frames, cross_audit, wall_quality = build_direct_frames(
        args, left_map, right_map, tag_points, wall
    )
    base_to_tag = Transform(np.asarray([0.02625, 0.0, 0.0196]), Rotation.identity())
    left_base_mount_rotation = (
        Rotation.from_quat(args.left_target_mount_quaternion)
        if args.left_target_mount_quaternion is not None
        else _load_mount_rotation(args.left_target_mount_json)
    )
    right_base_mount_rotation = (
        Rotation.from_quat(args.right_target_mount_quaternion)
        if args.right_target_mount_quaternion is not None
        else _load_mount_rotation(args.right_target_mount_json)
    )
    rows: list[dict] = []
    camera_to_base = {"left": [], "right": []}
    separation: list[float] = []
    both_direct = 0
    for frame in sorted(frames, key=lambda item: item.time_s):
        world_left_camera = frame.initial_left
        world_right_camera = frame.initial_right
        # left camera observes the right BaseTag; right camera observes left.
        left_base = None
        if frame.cross_rl_pose is not None:
            if (args.cross_pose_mode == "free-ippe"
                    and left_base_mount_rotation is not None):
                # A 20 mm planar Tag provides useful metric translation but its
                # IPPE attitude can flip by ~180 degrees.  Keep the directly
                # observed Tag origin, while target IMU/wall attitude plus the
                # frozen rigid mount supplies base orientation.
                world_tag = world_right_camera.compose(frame.cross_rl_pose)
                base_rotation = world_left_camera.r * left_base_mount_rotation
                left_base = Transform(
                    world_tag.p - base_rotation.apply(base_to_tag.p),
                    base_rotation,
                )
            else:
                left_base = base_from_observer(
                    world_right_camera, frame.cross_rl_pose, base_to_tag
                )
        right_base = None
        if frame.cross_lr_pose is not None:
            if (args.cross_pose_mode == "free-ippe"
                    and right_base_mount_rotation is not None):
                world_tag = world_left_camera.compose(frame.cross_lr_pose)
                base_rotation = world_right_camera.r * right_base_mount_rotation
                right_base = Transform(
                    world_tag.p - base_rotation.apply(base_to_tag.p),
                    base_rotation,
                )
            else:
                right_base = base_from_observer(
                    world_left_camera, frame.cross_lr_pose, base_to_tag
                )
        if left_base is not None and right_base is not None:
            both_direct += 1
            separation.append(float(np.linalg.norm(left_base.p - right_base.p)))
        if left_base is not None:
            camera_to_base["left"].append(world_left_camera.inverse().compose(left_base))
        if right_base is not None:
            camera_to_base["right"].append(world_right_camera.inverse().compose(right_base))
        rows.append({
            "time_s": frame.time_s, "left": left_base, "right": right_base,
            "left_camera": world_left_camera, "right_camera": world_right_camera,
            "left_camera_valid": wall_quality[frame.time_s]["left"] in {"direct", "tracked_lk", "hybrid"},
            "right_camera_valid": wall_quality[frame.time_s]["right"] in {"direct", "tracked_lk", "hybrid"},
            "left_camera_mode": wall_quality[frame.time_s]["left"],
            "right_camera_mode": wall_quality[frame.time_s]["right"],
        })

    for side in ("left", "right"):
        cross_audit.setdefault("short_gap_recovered_frames", {})[side] = (
            _recover_short_camera_gaps(rows, side, maximum_gap_frames=2)
        )
        side_rows = [row for row in rows if row[side] is not None]
        write_pose(args.output_dir / f"{side}_base_pose.csv", side_rows, side)
        write_camera_pose(
            args.output_dir / f"{side}_camera_pose.csv", rows, side,
            tag_map_sha256, detected_wall_ids,
        )
    report = {
        "schema_version": "direct-cross-basetag-world-pose/1.0",
        "status": "DIAGNOSTIC",
        "algorithm": "wall-localized observer + opposite 20mm BaseTag",
        "equation": "T_world_base = T_world_observer_camera @ T_camera_tag @ inverse(T_base_tag)",
        "frame_count_any_side": len(rows),
        "left_direct_frames": sum(row["left"] is not None for row in rows),
        "right_direct_frames": sum(row["right"] is not None for row in rows),
        "both_direct_frames": both_direct,
        "base_separation_m": stats(separation),
        "camera_to_base_rigid_audit": {
            side: _rigid_transform_audit(values)
            for side, values in camera_to_base.items()
        },
        "wall_tag_size_m": 0.200,
        "basetag_size_m": args.tag_size_m,
        "base_to_tag": {
            "translation_m": base_to_tag.p.tolist(),
            "quaternion_xyzw": base_to_tag.r.as_quat().tolist(),
        },
        "cross_selection": cross_audit,
        "wall_localization": {
            side: {
                mode: sum(
                    row[f"{side}_camera_mode"] == mode for row in rows
                )
                for mode in ("direct", "tracked_lk", "short_gap", "hybrid", "unavailable")
            }
            for side in ("left", "right")
        },
        "single_tag_imu_translation_enabled": bool(
            args.enable_single_tag_imu_translation
        ),
        "cross_pose_mode": args.cross_pose_mode,
        "disabled_cross_factors": {
            "left_observes_right": bool(args.disable_left_observes_right),
            "right_observes_left": bool(args.disable_right_observes_left),
        },
        "metric_input_audit": provenance,
        "camera_to_own_basetag_used": False,
        "stitched_input_used": False,
        "contact_constraint_used": False,
        "training_ready": False,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
