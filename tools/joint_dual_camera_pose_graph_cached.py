#!/usr/bin/env python3
"""Joint dual-camera/wall pose graph using only cached unit-bearing observations."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from osmo360.calibration.calibrate_basetag_reciprocal import Transform, interpolate_pose, load_pose, rotation_distance_deg
from osmo360.calibration.estimate_gripper_extrinsic import solve_bearing_ippe
from tools.joint_camera_correction_cached import own_tag_transform


@dataclass
class FrameData:
    time_s: float
    block: int
    initial_left: Transform
    initial_right: Transform
    left_leftwall: list[tuple[np.ndarray, np.ndarray]]
    left_rightwall: list[tuple[np.ndarray, np.ndarray]]
    right_leftwall: list[tuple[np.ndarray, np.ndarray]]
    right_rightwall: list[tuple[np.ndarray, np.ndarray]]
    cross_lr_rays: np.ndarray | None
    cross_rl_rays: np.ndarray | None
    cross_lr_pose: Transform | None
    cross_rl_pose: Transform | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-cache", type=Path, required=True)
    parser.add_argument("--right-cache", type=Path, required=True)
    parser.add_argument("--left-initial-pose", type=Path, required=True)
    parser.add_argument("--right-initial-pose", type=Path, required=True)
    parser.add_argument("--left-panel-map", type=Path, required=True)
    parser.add_argument("--right-panel-map", type=Path, required=True)
    parser.add_argument("--initial-world-map", type=Path, required=True)
    parser.add_argument("--left-tag-id", type=int, default=3)
    parser.add_argument("--right-tag-id", type=int, default=2)
    parser.add_argument("--tag-size-m", type=float, default=0.020)
    parser.add_argument("--tag-corner-quarter-turns", type=int, default=1)
    parser.add_argument("--left-tag-corner-quarter-turns", type=int, choices=range(4))
    parser.add_argument("--right-tag-corner-quarter-turns", type=int, choices=range(4))
    parser.add_argument("--left-camera-to-tag", type=Path,
                        help="reciprocal camera-to-mounted-Tag calibration JSON")
    parser.add_argument("--right-camera-to-tag", type=Path,
                        help="reciprocal camera-to-mounted-Tag calibration JSON")
    parser.add_argument("--start-common-s", type=float, required=True)
    parser.add_argument("--end-common-s", type=float, required=True)
    parser.add_argument("--sample-stride", type=int, default=6)
    parser.add_argument("--temporal-blocks", type=int, default=5)
    parser.add_argument("--holdout-blocks", default="1,3")
    parser.add_argument("--alternations", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--max-cross-basetag-center-error-deg", type=float, default=50.0,
        help="loose raw-fisheye bearing gate used only to reject a same-ID distractor",
    )
    parser.add_argument(
        "--max-cross-basetag-attitude-error-deg", type=float, default=45.0,
        help="fixed mount-attitude gate used to reject a displayed/printed same-ID distractor",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--anchored-two-pass", action="store_true",
        help=("solve the stronger wall camera first, constrain only the weaker camera "
              "with the reciprocal BaseTag, then rebuild candidate selection once"),
    )
    return parser.parse_args()


def raw_fisheye_cache_audit(path: Path) -> dict[str, Any]:
    """Fail closed unless cached rays come from traceable raw fisheye pixels."""
    sidecar = path.with_suffix(".json") if path.suffix else Path(f"{path}.json")
    if not sidecar.is_file():
        raise ValueError(f"raw-fisheye cache sidecar is missing: {sidecar}")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    schema = metadata.get("schema_version")
    supported = {
        "fisheye-apriltag-observation-cache/1.0",
        "fisheye-apriltag-observation-cache/1.1-lk",
        "fisheye-apriltag-observation-cache/1.2-dual-lens",
    }
    if schema not in supported:
        raise ValueError(f"unsupported/non-fisheye cache schema: {sidecar}")
    tracking = metadata.get("tracking") if schema.endswith("-lk") else None
    if tracking is not None:
        if tracking.get("method") != "pyramidal LK forward/backward on raw fisheye pixels":
            raise ValueError(f"unsupported optical-flow cache: {sidecar}")
        if tracking.get("pose_interpolation_used") is not False:
            raise ValueError(f"optical-flow cache may not contain pose interpolation: {sidecar}")
        parent = Path(str(metadata.get("parent_cache", "")))
        if not parent.is_file():
            raise ValueError(f"optical-flow parent cache is missing: {parent}")
    source_size = metadata.get("source_size")
    if not (
        isinstance(source_size, list)
        and len(source_size) == 2
        and int(source_size[0]) == int(source_size[1])
    ):
        raise ValueError(f"metric input is not a square raw fisheye image: {source_size}")
    if schema.endswith("-dual-lens"):
        streams = sorted(map(int, metadata.get("streams", [])))
        if streams != [0, 1]:
            raise ValueError(f"X5 metric input requires both raw lens streams: {sidecar}")
        source_videos = [Path(value) for value in metadata.get("source_videos", [])]
        if len(source_videos) != 2 or any(not source.is_file() for source in source_videos):
            raise ValueError(f"dual-fisheye source videos are missing: {source_videos}")
        if metadata.get("calibration") != "embedded_x5_offset":
            raise ValueError(f"X5 embedded offset calibration is missing: {sidecar}")
        if not metadata.get("x5_offset") or not metadata.get("calibration_sha256"):
            raise ValueError(f"X5 offset audit fields are missing: {sidecar}")
        stream_audit: int | list[int] = streams
        source_audit = [str(source.resolve()) for source in source_videos]
        calibration_audit = "embedded_x5_offset"
    else:
        if int(metadata.get("stream", -1)) != 1:
            raise ValueError(f"legacy metric input must be raw fisheye stream 1: {sidecar}")
        source_video = Path(str(metadata.get("video", "")))
        calibration = Path(str(metadata.get("calibration", "")))
        if not source_video.is_file():
            raise ValueError(f"raw fisheye source video is missing: {source_video}")
        if not calibration.is_file():
            raise ValueError(f"factory calibration is missing: {calibration}")
        stream_audit = 1
        source_audit = str(source_video.resolve())
        calibration_audit = str(calibration.resolve())
    return {
        "measurement_input": "raw_fisheye",
        "stream": stream_audit,
        "source_video": source_audit,
        "source_size": [int(source_size[0]), int(source_size[1])],
        "camera_serial": metadata.get("camera_serial"),
        "factory_calibration": calibration_audit,
        "radial_model": metadata.get("radial_model"),
        "rectified_detection_frontend": bool(metadata.get("rectified_detection", False)),
        "cached_measurement": metadata.get(
            "cached_measurement",
            "factory-calibrated unit rays in the raw fisheye optical frame",
        ),
        "stitching_used": False,
        "synthetic_frames_used": False,
        "optical_flow_measurements": tracking,
    }


def unit(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    return value / np.linalg.norm(value, axis=-1, keepdims=True)


def direct_map(path: Path) -> dict[int, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tag_size = float(payload["tag_outer_size_m"])
    if not math.isfinite(tag_size) or tag_size <= 0:
        raise ValueError(f"wall map {path} has an invalid Tag size")
    result = {
        int(tag["id"]): np.asarray(tag["corners_m"], dtype=float)
        for tag in payload["tags"]
    }
    if any(corners.shape != (4, 3) for corners in result.values()):
        raise ValueError(f"wall map {path} must contain four 3D corners per Tag")
    return result


def cache_index(cache: np.lib.npyio.NpzFile) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    for index, frame in enumerate(cache["frame_index"]):
        result.setdefault(int(frame), []).append(index)
    return result


def detection_timeline(cache: np.lib.npyio.NpzFile,
                       index: dict[int, list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Return actual detector frames and their synchronized common times.

    `timeline_*` contains every decoded frame, while the Tag detector may run
    only every N frames.  Pairing against the decoded timeline and then doing
    an exact detection lookup silently drops one camera whenever the affine
    clock offset is not an integer multiple of N frames.
    """
    frames = np.asarray(sorted(index), dtype=int)
    times = np.asarray([float(cache["common_time_s"][index[int(frame)][0]])
                        for frame in frames], dtype=float)
    return frames, times


def nearest_detection_frame(frames: np.ndarray, times: np.ndarray, time_s: float,
                            max_delta_s: float = 0.060) -> int | None:
    if not len(times):
        return None
    candidate = int(np.searchsorted(times, time_s))
    choices = [i for i in (candidate - 1, candidate) if 0 <= i < len(times)]
    best = min(choices, key=lambda i: abs(float(times[i]) - time_s))
    if abs(float(times[best]) - time_s) > max_delta_s:
        return None
    return int(frames[best])


def largest_ids(cache: np.lib.npyio.NpzFile, indices: list[int]) -> dict[int, int]:
    result: dict[int, int] = {}
    for index in indices:
        tag_id = int(cache["tag_id"][index])
        if tag_id not in result or cache["area_px2"][index] > cache["area_px2"][result[tag_id]]:
            result[tag_id] = index
    return result


def select_expected_basetag_detection(
    cache: np.lib.npyio.NpzFile,
    indices: list[int],
    tag_id: int,
    expected_camera_to_tag: Transform,
    max_center_error_deg: float = 50.0,
) -> tuple[int | None, dict[str, Any]]:
    """Select a mounted BaseTag without silently jumping to a distractor.

    A laptop/monitor can display the same small Tag ID as a gripper.  Picking
    the largest decoded instance then makes the measured bearing jump by
    roughly 90 degrees whenever the real Tag is occluded.  Wall-only camera
    poses provide an independent, deliberately loose expected bearing: it is
    accurate enough to separate those physical instances without forcing the
    two gripper trajectories to meet.

    The gate is fail-closed.  If no decoded instance lies in the expected
    hemisphere, the cross-camera factor is omitted instead of substituting a
    screen Tag.  The 50 degree default is intentionally much looser than any
    precision gate; it only rejects a different physical object.
    """
    candidates = [index for index in indices if int(cache["tag_id"][index]) == tag_id]
    expected = unit(expected_camera_to_tag.p)
    scored: list[tuple[float, int]] = []
    for index in candidates:
        measured = unit(np.asarray(cache["rays_camera"][index], dtype=float).mean(axis=0))
        error = float(np.degrees(np.arccos(np.clip(np.dot(expected, measured), -1.0, 1.0))))
        scored.append((error, index))
    scored.sort()
    accepted = bool(scored and scored[0][0] <= max_center_error_deg)
    return (scored[0][1] if accepted else None), {
        "tag_id": int(tag_id),
        "candidate_count": len(scored),
        "best_center_error_deg": None if not scored else scored[0][0],
        "second_center_error_deg": None if len(scored) < 2 else scored[1][0],
        "max_center_error_deg": float(max_center_error_deg),
        "accepted": accepted,
    }


def pose_from_solution(solution: dict[str, Any]) -> Transform:
    return Transform(
        np.asarray(solution["translation_tag_origin_in_panorama_m"]),
        Rotation.from_matrix(solution["rotation_tag_to_panorama"]),
    )


def select_ippe_branch(solutions: list[dict[str, Any]], expected: Transform) -> Transform:
    """Choose the planar branch using the current rigid-chain prediction.

    Reprojection error alone cannot distinguish the two IPPE mirror solutions.
    The independently solved wall poses are not precise enough for final output,
    but they are more than sufficient to reject a 90/180 degree mirror branch.
    """
    candidates = [pose_from_solution(solution) for solution in solutions]
    return min(
        candidates,
        key=lambda candidate: (
            rotation_distance_deg(candidate.r, expected.r) / 20.0
            + np.linalg.norm(candidate.p - expected.p) / 0.050
        ),
    )


def basetag_pose_matches_expected(
    measured: Transform,
    expected: Transform,
    max_attitude_error_deg: float = 45.0,
) -> tuple[bool, dict[str, float | bool]]:
    """Check the rigid mount attitude after the loose centre-bearing gate.

    A screen Tag can lie less than 50 degrees from the expected centre ray,
    especially while one camera has only a single planar wall.  Its plane
    orientation, however, is unrelated to the gripper mount.  The fixed
    BaseTag attitude therefore provides an independent second gate without
    using contact, trajectory overlap, or a synthetic image.
    """
    attitude_error = rotation_distance_deg(measured.r, expected.r)
    audit: dict[str, float | bool] = {
        "attitude_error_deg": float(attitude_error),
        "position_difference_m": float(np.linalg.norm(measured.p - expected.p)),
        "max_attitude_error_deg": float(max_attitude_error_deg),
        "accepted": bool(attitude_error <= max_attitude_error_deg),
    }
    return bool(audit["accepted"]), audit


def load_initial_wall_transform(path: Path) -> Transform:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "panel_transform" in payload:
        transform = payload["panel_transform"]
    else:
        panel = next(
            item for item in payload["panels"] if 128 in item["expected_ids"]
        )
        transform = panel["T_world_map"]
    return Transform(
        np.asarray(transform["translation_m"]),
        Rotation.from_quat(transform["quaternion_xyzw"]),
    )


def encode_transform(transform: Transform) -> np.ndarray:
    return np.r_[transform.p, transform.r.as_rotvec()]


def decode_transform(value: np.ndarray, offset: int = 0) -> Transform:
    return Transform(value[offset:offset + 3], Rotation.from_rotvec(value[offset + 3:offset + 6]))


def load_camera_to_tag(path: Path) -> tuple[Transform, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload["camera_to_basetag"]
    transform = Transform(
        np.asarray(value["translation_m"], dtype=float),
        Rotation.from_quat(value["quaternion_xyzw"]),
    )
    return transform, {
        "source": str(path.resolve()),
        "calibration_status": payload.get("calibration_status"),
        "training_ready": payload.get("training_ready", False),
        "inlier_count": payload.get("audit", {}).get("inlier_count"),
        "position_residual_mm": payload.get("audit", {}).get("position_residual_mm"),
        "rotation_residual_deg": payload.get("audit", {}).get("rotation_residual_deg"),
    }


def build_frames(args: argparse.Namespace, left_points: dict[int, np.ndarray],
                 right_points: dict[int, np.ndarray], tag_points: np.ndarray
                 ) -> tuple[list[FrameData], Transform, Transform, dict[str, Any]]:
    lc = np.load(args.left_cache); rc = np.load(args.right_cache)
    li = cache_index(lc); ri = cache_index(rc)
    left_detection_frames, left_detection_times = detection_timeline(lc, li)
    right_detection_frames, right_detection_times = detection_timeline(rc, ri)
    left_series = load_pose(args.left_initial_pose); right_series = load_pose(args.right_initial_pose)
    left_turns = (args.left_tag_corner_quarter_turns
                  if getattr(args, "left_tag_corner_quarter_turns", None) is not None
                  else args.tag_corner_quarter_turns)
    right_turns = (args.right_tag_corner_quarter_turns
                   if getattr(args, "right_tag_corner_quarter_turns", None) is not None
                   else args.tag_corner_quarter_turns)
    if args.left_camera_to_tag:
        own_left, own_left_audit = load_camera_to_tag(args.left_camera_to_tag)
    else:
        own_left, own_left_audit = own_tag_transform(lc, args.left_tag_id, tag_points, left_turns)
    if args.right_camera_to_tag:
        own_right, own_right_audit = load_camera_to_tag(args.right_camera_to_tag)
    else:
        own_right, own_right_audit = own_tag_transform(rc, args.right_tag_id, tag_points, right_turns)
    timeline = rc["timeline_common_time_s"]
    times = timeline[(timeline >= args.start_common_s) & (timeline <= args.end_common_s)][::args.sample_stride]
    preliminary = []
    cross_selection_audit = {
        "left_observes_right": {"candidate_frames": 0, "accepted_frames": 0,
                                  "rejected_frames": 0, "pose_rejected_frames": 0},
        "right_observes_left": {"candidate_frames": 0, "accepted_frames": 0,
                                  "rejected_frames": 0, "pose_rejected_frames": 0},
    }
    max_cross_error = float(getattr(args, "max_cross_basetag_center_error_deg", 50.0))
    max_cross_attitude = float(getattr(
        args, "max_cross_basetag_attitude_error_deg", 45.0))
    for time_s in times:
        time_s = float(time_s)
        wl = interpolate_pose(left_series, time_s); wr = interpolate_pose(right_series, time_s)
        if wl is None or wr is None:
            continue
        lf = nearest_detection_frame(left_detection_frames, left_detection_times, time_s)
        rf = nearest_detection_frame(right_detection_frames, right_detection_times, time_s)
        if lf is None or rf is None:
            continue
        ld = largest_ids(lc, li.get(lf, [])); rd = largest_ids(rc, ri.get(rf, []))
        def observations(cache, detections, points):
            return [(P, np.asarray(cache["rays_camera"][detections[tag_id]], dtype=float))
                    for tag_id, P in points.items() if tag_id in detections]
        ll = observations(lc, ld, left_points); lrw = observations(lc, ld, right_points)
        rl = observations(rc, rd, left_points); rrw = observations(rc, rd, right_points)
        # Two decoded 200 mm wall Tags already provide eight non-collinear
        # corners.  Requiring three discarded most of the otherwise strong
        # cached frames and reduced this capture to only 13 joint samples.
        if len(ll) + len(lrw) < 2 or len(rl) + len(rrw) < 2:
            continue
        cross_lr_rays = cross_rl_rays = cross_lr_pose = cross_rl_pose = None
        expected_lr = wl.inverse().compose(wr.compose(own_right))
        expected_rl = wr.inverse().compose(wl.compose(own_left))
        lr_index, lr_audit = select_expected_basetag_detection(
            lc, li.get(lf, []), args.right_tag_id, expected_lr, max_cross_error)
        rl_index, rl_audit = select_expected_basetag_detection(
            rc, ri.get(rf, []), args.left_tag_id, expected_rl, max_cross_error)
        for key, audit in (("left_observes_right", lr_audit),
                           ("right_observes_left", rl_audit)):
            if audit["candidate_count"]:
                cross_selection_audit[key]["candidate_frames"] += 1
                if not audit["accepted"]:
                    cross_selection_audit[key]["rejected_frames"] += 1
        if lr_index is not None:
            cross_lr_rays = np.roll(np.asarray(lc["rays_camera"][lr_index], dtype=float),
                                    -right_turns, axis=0)
            cross_lr_pose = select_ippe_branch(
                solve_bearing_ippe(tag_points, cross_lr_rays), expected_lr)
            accepted, _pose_audit = basetag_pose_matches_expected(
                cross_lr_pose, expected_lr, max_cross_attitude)
            if accepted:
                cross_selection_audit["left_observes_right"]["accepted_frames"] += 1
            else:
                cross_selection_audit["left_observes_right"]["pose_rejected_frames"] += 1
                cross_lr_rays = cross_lr_pose = None
        if rl_index is not None:
            cross_rl_rays = np.roll(np.asarray(rc["rays_camera"][rl_index], dtype=float),
                                    -left_turns, axis=0)
            cross_rl_pose = select_ippe_branch(
                solve_bearing_ippe(tag_points, cross_rl_rays), expected_rl)
            accepted, _pose_audit = basetag_pose_matches_expected(
                cross_rl_pose, expected_rl, max_cross_attitude)
            if accepted:
                cross_selection_audit["right_observes_left"]["accepted_frames"] += 1
            else:
                cross_selection_audit["right_observes_left"]["pose_rejected_frames"] += 1
                cross_rl_rays = cross_rl_pose = None
        preliminary.append((time_s, wl, wr, ll, lrw, rl, rrw,
                            cross_lr_rays, cross_rl_rays, cross_lr_pose, cross_rl_pose))
    frames = []
    for rank, item in enumerate(preliminary):
        block = min(args.temporal_blocks - 1, rank * args.temporal_blocks // len(preliminary))
        frames.append(FrameData(item[0], block, *item[1:]))
    return frames, own_left, own_right, {
        "left": own_left_audit,
        "right": own_right_audit,
        "cross_basetag_selection": cross_selection_audit,
    }


def transform_points(transform: Transform, points: np.ndarray) -> np.ndarray:
    return transform.r.apply(points) + transform.p


def balance(categories: list[np.ndarray]) -> np.ndarray:
    values = [value for value in categories if len(value)]
    target = max(len(value) for value in values)
    return np.concatenate([value * np.sqrt(target / len(value)) for value in values])


def frame_residual(value: np.ndarray, frame: FrameData, wall: Transform,
                   own_left: Transform, own_right: Transform, tag_points: np.ndarray,
                   include_cross: bool) -> np.ndarray:
    world_left = decode_transform(value, 0); world_right = decode_transform(value, 6)
    categories: list[np.ndarray] = []
    for camera, observations in ((world_left, frame.left_leftwall), (world_right, frame.right_leftwall)):
        residual = []
        for points, rays in observations:
            residual.append((unit(camera.r.inv().apply(points - camera.p)) - rays).ravel())
        categories.append(np.concatenate(residual) if residual else np.empty(0))
    for camera, observations in ((world_left, frame.left_rightwall), (world_right, frame.right_rightwall)):
        residual = []
        for points, rays in observations:
            world_points = transform_points(wall, points)
            residual.append((unit(camera.r.inv().apply(world_points - camera.p)) - rays).ravel())
        categories.append(np.concatenate(residual) if residual else np.empty(0))
    if include_cross and frame.cross_lr_rays is not None:
        observer_tag = world_left.inverse().compose(world_right.compose(own_right))
        predicted = unit(observer_tag.r.apply(tag_points) + observer_tag.p)
        categories.append((predicted - frame.cross_lr_rays).ravel())
    if include_cross and frame.cross_rl_rays is not None:
        observer_tag = world_right.inverse().compose(world_left.compose(own_left))
        predicted = unit(observer_tag.r.apply(tag_points) + observer_tag.p)
        categories.append((predicted - frame.cross_rl_rays).ravel())
    return balance(categories)


def camera_wall_residual(value: np.ndarray,
                         leftwall: list[tuple[np.ndarray, np.ndarray]],
                         rightwall: list[tuple[np.ndarray, np.ndarray]],
                         wall: Transform) -> np.ndarray:
    """Bearing residual for one camera using only fixed wall Tags."""
    camera = decode_transform(value)
    categories: list[np.ndarray] = []
    for observations, panel_transform in (
        (leftwall, None),
        (rightwall, wall),
    ):
        residual = []
        for points, rays in observations:
            world_points = points if panel_transform is None else transform_points(panel_transform, points)
            predicted = unit(camera.r.inv().apply(world_points - camera.p))
            residual.append((predicted - rays).ravel())
        categories.append(np.concatenate(residual) if residual else np.empty(0))
    return balance(categories)


def _ippe_world_camera_seeds(
    observations: list[tuple[np.ndarray, np.ndarray]],
    panel_transform: Transform | None,
) -> list[Transform]:
    if not observations:
        return []
    points = np.concatenate([item[0] for item in observations])
    if panel_transform is not None:
        points = transform_points(panel_transform, points)
    rays = np.concatenate([item[1] for item in observations])
    try:
        solutions = solve_bearing_ippe(points, rays)
    except (RuntimeError, ValueError, cv2.error):
        return []
    seeds = []
    for solution in solutions:
        # IPPE returns T_camera_world.  The optimizer stores T_world_camera.
        camera_world = Transform(
            np.asarray(solution["translation_tag_origin_in_panorama_m"]),
            Rotation.from_matrix(solution["rotation_tag_to_panorama"]),
        )
        seeds.append(camera_world.inverse())
    return seeds


def solve_camera_wall_candidates(
    initial: Transform,
    leftwall: list[tuple[np.ndarray, np.ndarray]],
    rightwall: list[tuple[np.ndarray, np.ndarray]],
    wall: Transform,
) -> list[tuple[float, Transform]]:
    """Return optimized wall-pose branches ordered by bearing fit.

    Each wall supplies both IPPE branches.  Every branch is then scored using
    all available wall corners, so a low-error mirror solution on one plane
    cannot silently become the room pose.
    """
    seeds = [initial]
    seeds.extend(_ippe_world_camera_seeds(leftwall, None))
    seeds.extend(_ippe_world_camera_seeds(rightwall, wall))
    fits = []
    for seed in seeds:
        fit = least_squares(
            camera_wall_residual,
            encode_transform(seed),
            args=(leftwall, rightwall, wall),
            loss="huber",
            f_scale=0.003,
            max_nfev=350,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
        residual = camera_wall_residual(fit.x, leftwall, rightwall, wall)
        candidate = decode_transform(fit.x)
        fits.append((float(np.mean(residual * residual)), candidate))

    world_points = []
    for points, _rays in leftwall:
        world_points.append(points)
    for points, _rays in rightwall:
        world_points.append(transform_points(wall, points))
    points = np.concatenate(world_points)
    singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    planar = singular[-1] <= max(singular[0], 1e-12) * 1e-6
    if not planar:
        return sorted(fits, key=lambda item: item[0])

    # Both planar IPPE branches can have indistinguishable reprojection error.
    # Reprojection remains the hard gate; the independent per-frame/temporal
    # prior is used only to choose among numerically equivalent branches.
    best_mse = min(item[0] for item in fits)
    # A relative 5% tolerance is numerically brittle when both planar fits
    # already have tiny bearing error: it can discard the physically correct
    # branch for a sub-pixel improvement.  Convert a small raw-ray angular
    # allowance to per-component MSE instead.  Beyond this bound visual fit
    # still wins; inside it the independent attitude/temporal prior disambiguates.
    # Large raw-fisheye wall Tags retain small edge-model bias even after
    # tangent rectification.  Keep all planar branches within 0.5 degrees and
    # let the independent temporal/IMU gate choose; 0.15 degrees prematurely
    # discarded the physical branch during fast hand motion.
    angular_equivalence_mse = math.radians(0.5) ** 2 / 3.0
    equivalent = [
        item for item in fits
        if item[0] <= best_mse + max(1e-10, angular_equivalence_mse)
    ]
    return sorted(
        equivalent,
        key=lambda item: (
            rotation_distance_deg(item[1].r, initial.r) / 30.0
            + np.linalg.norm(item[1].p - initial.p) / 0.20
        ),
    )


def solve_camera_wall_only(
    initial: Transform,
    leftwall: list[tuple[np.ndarray, np.ndarray]],
    rightwall: list[tuple[np.ndarray, np.ndarray]],
    wall: Transform,
) -> Transform:
    """Resolve arbitrary-plane ambiguity and return the preferred branch."""
    return solve_camera_wall_candidates(initial, leftwall, rightwall, wall)[0][1]


def solve_frame(frame: FrameData, initial: tuple[Transform, Transform], wall: Transform,
                own_left: Transform, own_right: Transform, tag_points: np.ndarray,
                include_cross: bool) -> tuple[Transform, Transform]:
    if not include_cross:
        return (
            solve_camera_wall_only(
                initial[0], frame.left_leftwall, frame.left_rightwall, wall),
            solve_camera_wall_only(
                initial[1], frame.right_leftwall, frame.right_rightwall, wall),
        )
    seeds = [initial]
    # The camera seeing both walls is normally the stronger anchor.  A direct
    # reciprocal Tag observation supplies an additional left-camera seed and
    # lets the optimizer escape a planar wall mirror basin.
    if include_cross and frame.cross_lr_pose is not None:
        left = initial[1].compose(own_right).compose(frame.cross_lr_pose.inverse())
        seeds.append((left, initial[1]))
    if include_cross and frame.cross_rl_pose is not None:
        left = initial[1].compose(frame.cross_rl_pose).compose(own_left.inverse())
        seeds.append((left, initial[1]))
    fits = []
    for seed in seeds:
        x0 = np.r_[encode_transform(seed[0]), encode_transform(seed[1])]
        fit = least_squares(
            frame_residual, x0,
            args=(frame, wall, own_left, own_right, tag_points, include_cross),
            loss="huber", f_scale=0.003, max_nfev=250,
            xtol=1e-10, ftol=1e-10, gtol=1e-10,
        )
        residual = frame_residual(
            fit.x, frame, wall, own_left, own_right, tag_points, include_cross)
        fits.append((float(np.mean(residual * residual)), fit.x))
    value = min(fits, key=lambda item: item[0])[1]
    return decode_transform(value, 0), decode_transform(value, 6)


def solve_frames(frames: list[FrameData], initial: dict[float, tuple[Transform, Transform]],
                 wall: Transform, own_left: Transform, own_right: Transform,
                 tag_points: np.ndarray, include_cross: bool, workers: int
                 ) -> dict[float, tuple[Transform, Transform]]:
    def solve(frame: FrameData):
        return frame.time_s, solve_frame(frame, initial[frame.time_s], wall, own_left, own_right,
                                         tag_points, include_cross)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(solve, frames))


def solve_frames_temporally(
    frames: list[FrameData],
    initial: dict[float, tuple[Transform, Transform]],
    wall: Transform,
    own_left: Transform,
    own_right: Transform,
    tag_points: np.ndarray,
    cross_enabled: set[float],
    max_prior_gap_s: float = 0.10,
) -> dict[float, tuple[Transform, Transform]]:
    """Solve in time order so a planar wall cannot flip independently.

    Cross-BaseTag observations are enabled only for explicitly supplied
    timestamps.  In particular, a holdout frame receives no held-out cross
    measurement; it only receives the previous wall-constrained camera pose as
    an IPPE branch prior.  A gap larger than ``max_prior_gap_s`` fails back to
    the independent input pose instead of bridging an unobserved interval.
    """
    result: dict[float, tuple[Transform, Transform]] = {}
    previous_time: float | None = None
    previous_pose: tuple[Transform, Transform] | None = None
    for frame in sorted(frames, key=lambda item: item.time_s):
        seed = initial[frame.time_s]
        if (previous_time is not None and previous_pose is not None
                and frame.time_s - previous_time <= max_prior_gap_s):
            seed = previous_pose
        solved = solve_frame(
            frame, seed, wall, own_left, own_right, tag_points,
            frame.time_s in cross_enabled,
        )
        result[frame.time_s] = solved
        previous_time = frame.time_s
        previous_pose = solved
    return result


def wall_support_score(frame: FrameData, side: int) -> tuple[int, int]:
    """Rank an independent wall solution without consulting the other gripper."""
    if side == 0:
        leftwall, rightwall = frame.left_leftwall, frame.left_rightwall
    else:
        leftwall, rightwall = frame.right_leftwall, frame.right_rightwall
    return (int(bool(leftwall) and bool(rightwall)), len(leftwall) + len(rightwall))


def anchored_cross_direction(frame: FrameData) -> str:
    """Return the only reciprocal factor allowed for this frame.

    `rl` means the independently wall-anchored right camera observes the weak
    left camera's Tag.  `lr` is the symmetric case.  The opposite observation
    remains an audit/holdout measurement and is never optimized in the same
    frame.
    """
    right_is_anchor = wall_support_score(frame, 1) >= wall_support_score(frame, 0)
    return "rl" if right_is_anchor else "lr"


def weak_camera_residual(
    value: np.ndarray,
    frame: FrameData,
    wall: Transform,
    anchor: Transform,
    weak_side: int,
    own_weak: Transform,
    tag_points: np.ndarray,
) -> np.ndarray:
    weak = decode_transform(value)
    if weak_side == 0:
        leftwall, rightwall = frame.left_leftwall, frame.left_rightwall
        cross_rays = frame.cross_rl_rays
        observer_tag = anchor.inverse().compose(weak.compose(own_weak))
    else:
        leftwall, rightwall = frame.right_leftwall, frame.right_rightwall
        cross_rays = frame.cross_lr_rays
        observer_tag = anchor.inverse().compose(weak.compose(own_weak))
    categories: list[np.ndarray] = []
    for observations, panel_transform in ((leftwall, None), (rightwall, wall)):
        residual = []
        for points, rays in observations:
            world_points = points if panel_transform is None else transform_points(panel_transform, points)
            residual.append((unit(weak.r.inv().apply(world_points - weak.p)) - rays).ravel())
        categories.append(np.concatenate(residual) if residual else np.empty(0))
    if cross_rays is not None:
        categories.append((unit(observer_tag.r.apply(tag_points) + observer_tag.p)
                           - cross_rays).ravel())
    return balance(categories)


def solve_frame_anchored(
    frame: FrameData,
    initial: tuple[Transform, Transform],
    wall: Transform,
    own_left: Transform,
    own_right: Transform,
    tag_points: np.ndarray,
) -> tuple[Transform, Transform]:
    """Solve one camera from walls, then the other without moving the anchor."""
    wall_left = solve_camera_wall_only(
        initial[0], frame.left_leftwall, frame.left_rightwall, wall)
    wall_right = solve_camera_wall_only(
        initial[1], frame.right_leftwall, frame.right_rightwall, wall)
    direction = anchored_cross_direction(frame)
    if direction == "rl":
        anchor, weak, weak_side, own_weak = wall_right, wall_left, 0, own_left
        pose = frame.cross_rl_pose
        if pose is None:
            return wall_left, wall_right
        seeds = [weak, anchor.compose(pose).compose(own_weak.inverse())]
    else:
        anchor, weak, weak_side, own_weak = wall_left, wall_right, 1, own_right
        pose = frame.cross_lr_pose
        if pose is None:
            return wall_left, wall_right
        seeds = [weak, anchor.compose(pose).compose(own_weak.inverse())]

    fits: list[tuple[float, Transform]] = []
    for seed in seeds:
        fit = least_squares(
            weak_camera_residual,
            encode_transform(seed),
            args=(frame, wall, anchor, weak_side, own_weak, tag_points),
            loss="huber", f_scale=0.003, max_nfev=250,
            xtol=1e-10, ftol=1e-10, gtol=1e-10,
        )
        residual = weak_camera_residual(
            fit.x, frame, wall, anchor, weak_side, own_weak, tag_points)
        fits.append((float(np.mean(residual * residual)), decode_transform(fit.x)))
    solved = min(fits, key=lambda item: item[0])[1]
    return (solved, anchor) if weak_side == 0 else (anchor, solved)


def solve_frames_anchored(
    frames: list[FrameData],
    initial: dict[float, tuple[Transform, Transform]],
    wall: Transform,
    own_left: Transform,
    own_right: Transform,
    tag_points: np.ndarray,
    workers: int,
) -> dict[float, tuple[Transform, Transform]]:
    def solve(frame: FrameData):
        return frame.time_s, solve_frame_anchored(
            frame, initial[frame.time_s], wall, own_left, own_right, tag_points)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(solve, frames))


def wall_residual(value: np.ndarray, frames: list[FrameData],
                  poses: dict[float, tuple[Transform, Transform]]) -> np.ndarray:
    wall = decode_transform(value)
    result = []
    for frame in frames:
        left_camera, right_camera = poses[frame.time_s]
        for camera, observations in ((left_camera, frame.left_rightwall),
                                     (right_camera, frame.right_rightwall)):
            for points, rays in observations:
                world_points = transform_points(wall, points)
                predicted = unit(camera.r.inv().apply(world_points - camera.p))
                result.append((predicted - rays).ravel())
    return np.concatenate(result)


def optimize_wall(initial: Transform, frames: list[FrameData],
                  poses: dict[float, tuple[Transform, Transform]]) -> Transform:
    fit = least_squares(
        wall_residual, encode_transform(initial), args=(frames, poses),
        loss="huber", f_scale=0.003, max_nfev=500,
        xtol=1e-12, ftol=1e-12, gtol=1e-12,
    )
    return decode_transform(fit.x)


def calibration_residual(value: np.ndarray, frames: list[FrameData],
                         poses: dict[float, tuple[Transform, Transform]],
                         tag_points: np.ndarray, prior_left: Transform,
                         prior_right: Transform) -> np.ndarray:
    wall = decode_transform(value, 0)
    own_left = decode_transform(value, 6)
    own_right = decode_transform(value, 12)
    result: list[np.ndarray] = []
    for frame in frames:
        wl, wr = poses[frame.time_s]
        for camera, observations in ((wl, frame.left_rightwall),
                                     (wr, frame.right_rightwall)):
            for points, rays in observations:
                world_points = transform_points(wall, points)
                result.append((unit(camera.r.inv().apply(world_points - camera.p)) - rays).ravel())
        if frame.cross_lr_rays is not None:
            observer_tag = wl.inverse().compose(wr.compose(own_right))
            result.append((unit(observer_tag.r.apply(tag_points) + observer_tag.p)
                           - frame.cross_lr_rays).ravel())
        if frame.cross_rl_rays is not None:
            observer_tag = wr.inverse().compose(wl.compose(own_left))
            result.append((unit(observer_tag.r.apply(tag_points) + observer_tag.p)
                           - frame.cross_rl_rays).ravel())
    # Weak metrology priors prevent unobservable camera/Tag gauge drift while
    # still allowing the reciprocal data to correct several millimetres/degrees.
    result.extend([
        (own_left.p - prior_left.p) * 0.05,
        (prior_left.r.inv() * own_left.r).as_rotvec() * 0.02,
        (own_right.p - prior_right.p) * 0.05,
        (prior_right.r.inv() * own_right.r).as_rotvec() * 0.02,
    ])
    return np.concatenate(result)


def optimize_calibration(wall: Transform, own_left: Transform, own_right: Transform,
                         frames: list[FrameData],
                         poses: dict[float, tuple[Transform, Transform]],
                         tag_points: np.ndarray, prior_left: Transform,
                         prior_right: Transform) -> tuple[Transform, Transform, Transform]:
    x0 = np.r_[encode_transform(wall), encode_transform(own_left), encode_transform(own_right)]
    fit = least_squares(
        calibration_residual, x0,
        args=(frames, poses, tag_points, prior_left, prior_right),
        loss="huber", f_scale=0.003, max_nfev=800,
        xtol=1e-11, ftol=1e-11, gtol=1e-11,
    )
    return decode_transform(fit.x, 0), decode_transform(fit.x, 6), decode_transform(fit.x, 12)


def summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "median": float("nan"),
                "p95": float("nan"), "min": float("nan"),
                "max": float("nan")}
    array = np.asarray(values)
    return {"count": len(values), "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95)),
            "min": float(array.min()), "max": float(array.max())}


def evaluate(frames: list[FrameData], poses: dict[float, tuple[Transform, Transform]],
             own_left: Transform, own_right: Transform) -> dict[str, Any]:
    position = {"left_tag3": [], "right_tag2": []}
    rotation = {"left_tag3": [], "right_tag2": []}
    for frame in frames:
        world_left, world_right = poses[frame.time_s]
        if frame.cross_lr_pose is not None:
            observed = world_right.inverse().compose(world_left.compose(frame.cross_lr_pose))
            position["right_tag2"].append(np.linalg.norm(observed.p - own_right.p) * 1000)
            rotation["right_tag2"].append(rotation_distance_deg(own_right.r, observed.r))
        if frame.cross_rl_pose is not None:
            observed = world_left.inverse().compose(world_right.compose(frame.cross_rl_pose))
            position["left_tag3"].append(np.linalg.norm(observed.p - own_left.p) * 1000)
            rotation["left_tag3"].append(rotation_distance_deg(own_left.r, observed.r))
    return {side: {"position_error_mm": summary(position[side]),
                   "rotation_error_deg": summary(rotation[side])}
            for side in position}


def metric_score(metrics: dict[str, Any]) -> float:
    score = 0.0
    for side in metrics:
        p = metrics[side]["position_error_mm"]; r = metrics[side]["rotation_error_deg"]
        if int(p["count"]) == 0 or int(r["count"]) == 0:
            return float("inf")
        score += p["median"] / 3 + p["p95"] / 5 + r["median"] / 2 + r["p95"] / 3
    return float(score)


def write_pose_csv(path: Path, frames: list[FrameData], poses: dict[float, tuple[Transform, Transform]], side: int) -> None:
    fields = ["frame", "timestamp", "camera_x_m", "camera_y_m", "camera_z_m",
              "qx", "qy", "qz", "qw", "parent_frame", "child_frame",
              "measurement_source", "quality_status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for frame in frames:
            transform = poses[frame.time_s][side]; q = transform.r.as_quat()
            writer.writerow({"frame": round(frame.time_s * 30), "timestamp": f"{frame.time_s:.9f}",
                             "camera_x_m": transform.p[0], "camera_y_m": transform.p[1], "camera_z_m": transform.p[2],
                             "qx": q[0], "qy": q[1], "qz": q[2], "qw": q[3],
                             "parent_frame": "tag_map", "child_frame": "fisheye1_camera_panorama_axes",
                             "measurement_source": "cached_joint_pose_graph", "quality_status": "valid"})


def payload(transform: Transform) -> dict[str, Any]:
    return {"translation_m": transform.p.tolist(), "quaternion_xyzw": transform.r.as_quat().tolist()}


def evaluate_anchored_cross_holdout(
    frames: list[FrameData],
    poses: dict[float, tuple[Transform, Transform]],
    own_left: Transform,
    own_right: Transform,
) -> dict[str, Any]:
    """Evaluate only the reciprocal direction excluded from each frame solve."""
    position: list[float] = []
    rotation: list[float] = []
    used_direction = {"lr": 0, "rl": 0}
    holdout_direction = {"lr": 0, "rl": 0}
    paired = 0
    for frame in frames:
        world_left, world_right = poses[frame.time_s]
        direction = anchored_cross_direction(frame)
        used_pose = frame.cross_rl_pose if direction == "rl" else frame.cross_lr_pose
        if used_pose is not None:
            used_direction[direction] += 1
        if direction == "rl" and frame.cross_lr_pose is not None:
            observed = world_right.inverse().compose(world_left.compose(frame.cross_lr_pose))
            position.append(float(np.linalg.norm(observed.p - own_right.p) * 1000))
            rotation.append(rotation_distance_deg(own_right.r, observed.r))
            holdout_direction["lr"] += 1
            paired += int(used_pose is not None)
        elif direction == "lr" and frame.cross_rl_pose is not None:
            observed = world_left.inverse().compose(world_right.compose(frame.cross_rl_pose))
            position.append(float(np.linalg.norm(observed.p - own_left.p) * 1000))
            rotation.append(rotation_distance_deg(own_left.r, observed.r))
            holdout_direction["rl"] += 1
            paired += int(used_pose is not None)
    return {
        "position_error_mm": summary(position),
        "rotation_error_deg": summary(rotation),
        "used_factor_frames": used_direction,
        "holdout_frames": holdout_direction,
        "paired_factor_and_holdout_frames": paired,
        "semantics": "opposite reciprocal direction; excluded from that frame optimization",
    }


def base_separation_audit(
    frames: list[FrameData],
    poses: dict[float, tuple[Transform, Transform]],
    own_left: Transform,
    own_right: Transform,
) -> dict[str, float | int]:
    # Hardware-confirmed T_base_tag: tag centre (26.25, 0, 19.6) mm in
    # base_link with aligned axes.  Convert camera->tag into camera->base.
    base_to_tag = Transform(np.asarray([0.02625, 0.0, 0.0196]), Rotation.identity())
    tag_to_base = base_to_tag.inverse()
    values = []
    for frame in frames:
        left, right = poses[frame.time_s]
        world_left_base = left.compose(own_left).compose(tag_to_base)
        world_right_base = right.compose(own_right).compose(tag_to_base)
        values.append(float(np.linalg.norm(world_left_base.p - world_right_base.p)))
    return summary(values)


def main_anchored_two_pass(
    args: argparse.Namespace,
    input_audit: dict[str, Any],
    left_points: dict[int, np.ndarray],
    right_points: dict[int, np.ndarray],
    tag_points: np.ndarray,
) -> int:
    """Fail-closed two-pass solution with an independent wall anchor."""
    wall = load_initial_wall_transform(args.initial_world_map)
    frames1, own_left, own_right, audit1 = build_frames(
        args, left_points, right_points, tag_points)
    initial1 = {frame.time_s: (frame.initial_left, frame.initial_right) for frame in frames1}
    poses1 = solve_frames_anchored(
        frames1, initial1, wall, own_left, own_right, tag_points, args.workers)
    pass1_left = args.output_dir / "pass1_left_pose.csv"
    pass1_right = args.output_dir / "pass1_right_pose.csv"
    write_pose_csv(pass1_left, frames1, poses1, 0)
    write_pose_csv(pass1_right, frames1, poses1, 1)

    # Rebuild raw-fisheye BaseTag selection against the first-pass rigid-chain
    # prediction.  This is what rejects a monitor Tag that happened to lie in
    # the coarse wall-only hemisphere, while allowing the real moving Tag to
    # re-enter after the weak camera branch has been corrected.
    args2 = copy.copy(args)
    args2.left_initial_pose = pass1_left
    args2.right_initial_pose = pass1_right
    frames2, own_left2, own_right2, audit2 = build_frames(
        args2, left_points, right_points, tag_points)
    initial2 = {frame.time_s: (frame.initial_left, frame.initial_right) for frame in frames2}
    poses2 = solve_frames_anchored(
        frames2, initial2, wall, own_left2, own_right2, tag_points, args.workers)
    write_pose_csv(args.output_dir / "left_pose.csv", frames2, poses2, 0)
    write_pose_csv(args.output_dir / "right_pose.csv", frames2, poses2, 1)

    holdout = evaluate_anchored_cross_holdout(
        frames2, poses2, own_left2, own_right2)
    separation = base_separation_audit(
        frames2, poses2, own_left2, own_right2)
    enough_holdout = int(holdout["position_error_mm"]["count"]) >= 10
    holdout_pass = bool(
        enough_holdout
        and holdout["position_error_mm"]["median"] <= 3.0
        and holdout["position_error_mm"]["p95"] <= 5.0
        and holdout["rotation_error_deg"]["median"] <= 2.0
        and holdout["rotation_error_deg"]["p95"] <= 3.0
    )
    physical_pass = bool(
        separation["min"] >= 0.050 and separation["max"] <= 0.800)
    passed = holdout_pass and physical_pass
    report = {
        "schema_version": "anchored-raw-fisheye-dual-gripper/1.0",
        "status": "HOLDOUT_PASS" if passed else "HOLDOUT_FAILED",
        "frame_count": len(frames2),
        "algorithm": "two-pass independent wall anchor + one-way reciprocal factor",
        "pass1_cross_selection": audit1["cross_basetag_selection"],
        "pass2_cross_selection": audit2["cross_basetag_selection"],
        "independent_reverse_holdout": holdout,
        "base_separation_m": separation,
        "gates": {
            "holdout_count_min": 10,
            "position_median_mm_max": 3.0,
            "position_p95_mm_max": 5.0,
            "rotation_median_deg_max": 2.0,
            "rotation_p95_deg_max": 3.0,
            "base_separation_m": [0.050, 0.800],
            "holdout_pass": holdout_pass,
            "physical_pass": physical_pass,
        },
        "own_basetag": {
            "left": {"transform": payload(own_left2), "audit": audit2["left"]},
            "right": {"transform": payload(own_right2), "audit": audit2["right"]},
        },
        "metric_input_audit": input_audit,
        "measurement_input": "raw_fisheye",
        "stitching_used": False,
        "synthetic_frames_used": False,
        "contact_constraint_used": False,
        "training_ready": False,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


def main() -> int:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    input_audit = {
        "left": raw_fisheye_cache_audit(args.left_cache),
        "right": raw_fisheye_cache_audit(args.right_cache),
    }
    holdout_blocks = {int(value) for value in args.holdout_blocks.split(",")}
    left_points = direct_map(args.left_panel_map); right_points = direct_map(args.right_panel_map)
    half = args.tag_size_m / 2
    tag_points = np.asarray([[-half, -half, 0], [half, -half, 0],
                             [half, half, 0], [-half, half, 0]], dtype=float)
    if args.anchored_two_pass:
        return main_anchored_two_pass(
            args, input_audit, left_points, right_points, tag_points)
    frames, own_left, own_right, own_audit = build_frames(args, left_points, right_points, tag_points)
    train = [frame for frame in frames if frame.block not in holdout_blocks]
    holdout = [frame for frame in frames if frame.block in holdout_blocks]
    initial_wall = load_initial_wall_transform(args.initial_world_map)
    initial_own_left, initial_own_right = own_left, own_right
    initial_poses = {frame.time_s: (frame.initial_left, frame.initial_right) for frame in frames}
    wall = initial_wall
    train_poses = {frame.time_s: initial_poses[frame.time_s] for frame in train}
    iterations = []
    for iteration in range(args.alternations):
        train_poses = solve_frames(train, train_poses, wall, own_left, own_right,
                                   tag_points, True, args.workers)
        updated, updated_left, updated_right = optimize_calibration(
            wall, own_left, own_right, train, train_poses, tag_points,
            initial_own_left, initial_own_right)
        iterations.append({"iteration": iteration + 1,
                           "wall_translation_change_mm": float(np.linalg.norm(updated.p - wall.p) * 1000),
                           "wall_rotation_change_deg": rotation_distance_deg(wall.r, updated.r),
                           "left_extrinsic_change_mm": float(np.linalg.norm(updated_left.p - own_left.p) * 1000),
                           "left_extrinsic_change_deg": rotation_distance_deg(own_left.r, updated_left.r),
                           "right_extrinsic_change_mm": float(np.linalg.norm(updated_right.p - own_right.p) * 1000),
                           "right_extrinsic_change_deg": rotation_distance_deg(own_right.r, updated_right.r),
                           "wall": payload(updated)})
        wall, own_left, own_right = updated, updated_left, updated_right
    hypotheses = []
    for name, candidate, candidate_left, candidate_right in (
        ("initial_single_camera", initial_wall, initial_own_left, initial_own_right),
        ("joint_pose_graph", wall, own_left, own_right),
    ):
        # Solve the complete time line, but expose reciprocal BaseTag factors
        # only on training blocks.  Holdout cross observations remain unseen
        # and are evaluated below.  Temporal propagation is used solely to
        # choose the physically continuous branch of planar wall PnP.
        training_times = {frame.time_s for frame in train}
        timeline_poses = solve_frames_temporally(
            frames, initial_poses, candidate, candidate_left, candidate_right,
            tag_points, training_times,
        )
        holdout_poses = {
            frame.time_s: timeline_poses[frame.time_s] for frame in holdout
        }
        metrics = evaluate(holdout, holdout_poses, candidate_left, candidate_right)
        hypotheses.append({"name": name, "wall": candidate, "metrics": metrics,
                           "own_left": candidate_left, "own_right": candidate_right,
                           "score": metric_score(metrics), "poses": timeline_poses})
    selected = min(hypotheses, key=lambda item: item["score"])
    selected_wall = selected["wall"]
    selected_own_left = selected["own_left"]
    selected_own_right = selected["own_right"]
    all_initial = initial_poses
    all_poses = solve_frames(frames, all_initial, selected_wall, selected_own_left, selected_own_right,
                             tag_points, True, args.workers)
    selected_metrics = selected["metrics"]
    passed = all(selected_metrics[side]["position_error_mm"]["median"] <= 3
                 and selected_metrics[side]["position_error_mm"]["p95"] <= 5
                 and selected_metrics[side]["rotation_error_deg"]["median"] <= 2
                 and selected_metrics[side]["rotation_error_deg"]["p95"] <= 3
                 for side in selected_metrics)
    write_pose_csv(args.output_dir / "left_pose.csv", frames, all_poses, 0)
    write_pose_csv(args.output_dir / "right_pose.csv", frames, all_poses, 1)
    report = {"schema_version": "cached-joint-dual-camera-pose-graph/1.0",
              "status": "HOLDOUT_PASS" if passed else "HOLDOUT_FAILED",
              "selected_model": selected["name"], "frame_count": len(frames),
              "train_frames": len(train), "holdout_frames": len(holdout),
              "own_basetag": {"left_tag3": {"transform": payload(selected_own_left), "audit": own_audit["left"]},
                              "right_tag2": {"transform": payload(selected_own_right), "audit": own_audit["right"]}},
              "initial_wall": payload(initial_wall), "selected_wall": payload(selected_wall),
              "alternations": iterations,
              "holdout_selection": [{"name": item["name"], "score": item["score"],
                                     "metrics": item["metrics"]} for item in hypotheses],
              "selected_holdout": selected_metrics,
              "all_data_diagnostic": evaluate(frames, all_poses, selected_own_left, selected_own_right),
              "metric_input_audit": input_audit,
              "measurement_input": "raw_fisheye",
              "stitching_used": False,
              "synthetic_frames_used": False,
              "video_decoded": False, "training_ready": False}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
