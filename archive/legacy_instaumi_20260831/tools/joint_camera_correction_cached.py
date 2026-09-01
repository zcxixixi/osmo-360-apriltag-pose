#!/usr/bin/env python3
"""Jointly fit fixed dual-camera pose corrections from cached Tag bearings.

The independently estimated world poses remain measurements. Two constant SE(3)
corrections are fitted together while fixed wall bearings, reciprocal BaseTags,
and directly observed own-BaseTag extrinsics share one objective. Model choice is
made only on blocked held-out frames; video is never decoded by this program.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from osmo360.calibration.calibrate_basetag_reciprocal import Transform, interpolate_pose, load_pose, rotation_distance_deg
from osmo360.calibration.estimate_gripper_extrinsic import solve_bearing_ippe
from osmo360.localization.world_frames import compile_world_tag_map


@dataclass
class FrameObservations:
    time_s: float
    block: int
    world_left: Transform
    world_right: Transform
    wall_left: list[tuple[np.ndarray, np.ndarray]]
    wall_right: list[tuple[np.ndarray, np.ndarray]]
    cross_lr_rays: np.ndarray | None
    cross_rl_rays: np.ndarray | None
    cross_lr_pose: Transform | None
    cross_rl_pose: Transform | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-cache", type=Path, required=True)
    parser.add_argument("--right-cache", type=Path, required=True)
    parser.add_argument("--left-pose", type=Path, required=True)
    parser.add_argument("--right-pose", type=Path, required=True)
    parser.add_argument("--tag-map", type=Path, required=True)
    parser.add_argument("--left-tag-id", type=int, default=3)
    parser.add_argument("--right-tag-id", type=int, default=2)
    parser.add_argument("--tag-size-m", type=float, default=0.020)
    parser.add_argument("--tag-corner-quarter-turns", type=int, choices=range(4), default=1)
    parser.add_argument("--start-common-s", type=float, required=True)
    parser.add_argument("--end-common-s", type=float, required=True)
    parser.add_argument("--sample-stride", type=int, default=3)
    parser.add_argument("--temporal-blocks", type=int, default=5)
    parser.add_argument("--holdout-blocks", default="1,3")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def unit(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float)
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def cache_by_frame(cache: np.lib.npyio.NpzFile) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    for index, frame in enumerate(cache["frame_index"]):
        result.setdefault(int(frame), []).append(index)
    return result


def largest_by_id(cache: np.lib.npyio.NpzFile, indices: list[int]) -> dict[int, int]:
    selected: dict[int, int] = {}
    for index in indices:
        tag_id = int(cache["tag_id"][index])
        if tag_id not in selected or cache["area_px2"][index] > cache["area_px2"][selected[tag_id]]:
            selected[tag_id] = index
    return selected


def own_tag_transform(cache: np.lib.npyio.NpzFile, tag_id: int, tag_points: np.ndarray,
                      quarter_turns: int) -> tuple[Transform, dict[str, Any]]:
    indices = np.flatnonzero(cache["tag_id"] == tag_id)
    if len(indices) < 3:
        raise RuntimeError(f"own Tag {tag_id} has only {len(indices)} cached observations")
    # A rigid own Tag occupies one stable image location.  A 360 camera can also
    # produce small duplicate detections close to the lens/seam.  Taking the
    # median of *all* detections mixes those components and can manufacture a
    # bearing which no real observation had.  Keep the large-area component;
    # the mounted tag is deliberately close to the camera and dominates area.
    areas = np.asarray(cache["area_px2"][indices], dtype=float)
    area_gate = float(np.percentile(areas, 75.0))
    selected = indices[areas >= area_gate]
    if len(selected) < 3:
        selected = indices[np.argsort(areas)[-min(len(indices), 3):]]
    rays = np.asarray(cache["rays_camera"][selected], dtype=float)
    median_rays = unit(np.median(rays, axis=0))
    median_rays = np.roll(median_rays, -quarter_turns, axis=0)
    candidates = solve_bearing_ippe(tag_points, median_rays)
    # Hardware/photo half-space validation used by estimate_gripper_extrinsic:
    # camera origin in Tag has x<=0 and -100mm<=z<=-40mm.
    valid = []
    for candidate in candidates:
        transform = Transform(
            np.asarray(candidate["translation_tag_origin_in_panorama_m"]),
            Rotation.from_matrix(candidate["rotation_tag_to_panorama"]),
        )
        camera_in_tag = transform.inverse().p
        if camera_in_tag[0] <= 0 and -0.100 <= camera_in_tag[2] <= -0.040:
            valid.append((candidate, transform, camera_in_tag))
    if not valid:
        raise RuntimeError(f"own Tag {tag_id} has no hardware-valid IPPE branch")
    candidate, transform, camera_in_tag = min(valid, key=lambda item: item[0]["angular_rmse_deg"])
    return transform, {
        "observation_count": int(len(indices)),
        "selected_observation_count": int(len(selected)),
        "area_gate_px2": area_gate,
        "selected_branch": int(candidate["branch"]),
        "angular_rmse_deg": float(candidate["angular_rmse_deg"]),
        "camera_origin_in_tag_m": camera_in_tag.tolist(),
    }


def detection_timeline(cache: np.lib.npyio.NpzFile,
                       index: dict[int, list[int]]) -> tuple[np.ndarray, np.ndarray]:
    frames = np.asarray(sorted(index), dtype=int)
    times = np.asarray([float(cache["common_time_s"][index[int(frame)][0]])
                        for frame in frames], dtype=float)
    return frames, times


def nearest_detection_frame(frames: np.ndarray, times: np.ndarray,
                            common_time: float,
                            max_delta_s: float = 0.020) -> int | None:
    if not len(times):
        return None
    candidate = int(np.searchsorted(times, common_time))
    choices = [i for i in (candidate - 1, candidate) if 0 <= i < len(times)]
    best = min(choices, key=lambda i: abs(float(times[i]) - common_time))
    if abs(float(times[best]) - common_time) > max_delta_s:
        return None
    return int(frames[best])


def transform_from_solution(solution: dict[str, Any]) -> Transform:
    return Transform(
        np.asarray(solution["translation_tag_origin_in_panorama_m"]),
        Rotation.from_matrix(solution["rotation_tag_to_panorama"]),
    )


def build_frames(args: argparse.Namespace, tag_points: np.ndarray,
                 map_corners: dict[int, np.ndarray]) -> tuple[list[FrameObservations], Transform, Transform, dict[str, Any]]:
    left_cache = np.load(args.left_cache)
    right_cache = np.load(args.right_cache)
    left_by_frame = cache_by_frame(left_cache)
    right_by_frame = cache_by_frame(right_cache)
    left_detection_frames, left_detection_times = detection_timeline(left_cache, left_by_frame)
    right_detection_frames, right_detection_times = detection_timeline(right_cache, right_by_frame)
    left_series = load_pose(args.left_pose)
    right_series = load_pose(args.right_pose)
    own_left, own_left_audit = own_tag_transform(
        left_cache, args.left_tag_id, tag_points, args.tag_corner_quarter_turns
    )
    own_right, own_right_audit = own_tag_transform(
        right_cache, args.right_tag_id, tag_points, args.tag_corner_quarter_turns
    )
    right_timeline = right_cache["timeline_common_time_s"]
    selected_times = right_timeline[
        (right_timeline >= args.start_common_s)
        & (right_timeline <= args.end_common_s)
    ][::args.sample_stride]
    preliminary = []
    for common_time in selected_times:
        left_frame = nearest_detection_frame(
            left_detection_frames, left_detection_times, float(common_time))
        right_frame = nearest_detection_frame(
            right_detection_frames, right_detection_times, float(common_time))
        if left_frame is None or right_frame is None:
            continue
        left_ids = largest_by_id(left_cache, left_by_frame.get(left_frame, []))
        right_ids = largest_by_id(right_cache, right_by_frame.get(right_frame, []))
        world_left = interpolate_pose(left_series, float(common_time))
        world_right = interpolate_pose(right_series, float(common_time))
        if world_left is None or world_right is None:
            continue
        wall_left = [
            (map_corners[tag_id], np.asarray(left_cache["rays_camera"][index], dtype=float))
            for tag_id, index in left_ids.items() if tag_id in map_corners
        ]
        wall_right = [
            (map_corners[tag_id], np.asarray(right_cache["rays_camera"][index], dtype=float))
            for tag_id, index in right_ids.items() if tag_id in map_corners
        ]
        if len(wall_left) < 2 or len(wall_right) < 2:
            continue
        cross_lr_rays = None
        cross_rl_rays = None
        cross_lr_pose = None
        cross_rl_pose = None
        if args.right_tag_id in left_ids:
            index = left_ids[args.right_tag_id]
            cross_lr_rays = np.roll(
                np.asarray(left_cache["rays_camera"][index], dtype=float),
                -args.tag_corner_quarter_turns, axis=0,
            )
            solutions = solve_bearing_ippe(tag_points, cross_lr_rays)
            cross_lr_pose = transform_from_solution(min(solutions, key=lambda item: item["angular_rmse_deg"]))
        if args.left_tag_id in right_ids:
            index = right_ids[args.left_tag_id]
            cross_rl_rays = np.roll(
                np.asarray(right_cache["rays_camera"][index], dtype=float),
                -args.tag_corner_quarter_turns, axis=0,
            )
            solutions = solve_bearing_ippe(tag_points, cross_rl_rays)
            cross_rl_pose = transform_from_solution(min(solutions, key=lambda item: item["angular_rmse_deg"]))
        if cross_lr_rays is None and cross_rl_rays is None:
            continue
        preliminary.append((float(common_time), world_left, world_right, wall_left, wall_right,
                            cross_lr_rays, cross_rl_rays, cross_lr_pose, cross_rl_pose))
    frames = []
    for rank, item in enumerate(preliminary):
        block = min(args.temporal_blocks - 1, rank * args.temporal_blocks // len(preliminary))
        frames.append(FrameObservations(item[0], block, *item[1:]))
    return frames, own_left, own_right, {
        "left": own_left_audit,
        "right": own_right_audit,
    }


def corrected(world_camera: Transform, correction: Transform) -> Transform:
    return world_camera.compose(correction)


def decode_corrections(parameters: np.ndarray) -> tuple[Transform, Transform]:
    return (
        Transform(parameters[:3], Rotation.from_rotvec(parameters[3:6])),
        Transform(parameters[6:9], Rotation.from_rotvec(parameters[9:12])),
    )


def category_residuals(parameters: np.ndarray, frames: list[FrameObservations],
                       own_left: Transform, own_right: Transform) -> dict[str, np.ndarray]:
    correction_left, correction_right = decode_corrections(parameters)
    categories: dict[str, list[np.ndarray]] = {
        "wall_left": [], "wall_right": [], "cross_lr": [], "cross_rl": [],
    }
    half = 0.010
    tag_points = np.asarray([
        [-half, -half, 0.0], [half, -half, 0.0],
        [half, half, 0.0], [-half, half, 0.0],
    ])
    for frame in frames:
        world_left = corrected(frame.world_left, correction_left)
        world_right = corrected(frame.world_right, correction_right)
        for points, rays in frame.wall_left:
            predicted = unit(world_left.r.inv().apply(points - world_left.p))
            categories["wall_left"].append((predicted - rays).ravel())
        for points, rays in frame.wall_right:
            predicted = unit(world_right.r.inv().apply(points - world_right.p))
            categories["wall_right"].append((predicted - rays).ravel())
        if frame.cross_lr_rays is not None:
            world_tag = world_right.compose(own_right)
            observer_tag = world_left.inverse().compose(world_tag)
            predicted = unit(observer_tag.r.apply(tag_points) + observer_tag.p)
            categories["cross_lr"].append((predicted - frame.cross_lr_rays).ravel())
        if frame.cross_rl_rays is not None:
            world_tag = world_left.compose(own_left)
            observer_tag = world_right.inverse().compose(world_tag)
            predicted = unit(observer_tag.r.apply(tag_points) + observer_tag.p)
            categories["cross_rl"].append((predicted - frame.cross_rl_rays).ravel())
    return {
        key: np.concatenate(value) if value else np.empty(0)
        for key, value in categories.items()
    }


def balanced_residual(parameters: np.ndarray, frames: list[FrameObservations],
                      own_left: Transform, own_right: Transform) -> np.ndarray:
    categories = category_residuals(parameters, frames, own_left, own_right)
    nonempty = [value for value in categories.values() if len(value)]
    target_count = max(len(value) for value in nonempty)
    return np.concatenate([
        value * np.sqrt(target_count / len(value)) for value in nonempty
    ])


def angular_summary(vector_residual: np.ndarray) -> dict[str, float | int]:
    if not len(vector_residual):
        return {"corner_count": 0, "median_deg": float("nan"), "p95_deg": float("nan")}
    error = np.degrees(np.linalg.norm(vector_residual.reshape(-1, 3), axis=1))
    return {
        "corner_count": int(len(error)),
        "median_deg": float(np.median(error)),
        "p95_deg": float(np.percentile(error, 95)),
    }


def closure_metrics(parameters: np.ndarray, frames: list[FrameObservations],
                    own_left: Transform, own_right: Transform) -> dict[str, Any]:
    correction_left, correction_right = decode_corrections(parameters)
    by_side = {"left_tag3": ([], []), "right_tag2": ([], [])}
    for frame in frames:
        world_left = corrected(frame.world_left, correction_left)
        world_right = corrected(frame.world_right, correction_right)
        if frame.cross_lr_pose is not None:
            observed = world_right.inverse().compose(world_left.compose(frame.cross_lr_pose))
            by_side["right_tag2"][0].append(np.linalg.norm(observed.p - own_right.p) * 1000)
            by_side["right_tag2"][1].append(rotation_distance_deg(own_right.r, observed.r))
        if frame.cross_rl_pose is not None:
            observed = world_left.inverse().compose(world_right.compose(frame.cross_rl_pose))
            by_side["left_tag3"][0].append(np.linalg.norm(observed.p - own_left.p) * 1000)
            by_side["left_tag3"][1].append(rotation_distance_deg(own_left.r, observed.r))
    result = {}
    for side, (position, rotation) in by_side.items():
        p = np.asarray(position); r = np.asarray(rotation)
        result[side] = {
            "count": int(len(p)),
            "position_error_mm": {
                "median": float(np.median(p)), "p95": float(np.percentile(p, 95)),
            },
            "rotation_error_deg": {
                "median": float(np.median(r)), "p95": float(np.percentile(r, 95)),
            },
        }
    categories = category_residuals(parameters, frames, own_left, own_right)
    result["bearing_residual"] = {
        key: angular_summary(value) for key, value in categories.items()
    }
    return result


def score(metrics: dict[str, Any]) -> float:
    value = 0.0
    for side in ("left_tag3", "right_tag2"):
        p = metrics[side]["position_error_mm"]
        r = metrics[side]["rotation_error_deg"]
        value += p["median"] / 3.0 + p["p95"] / 5.0
        value += r["median"] / 2.0 + r["p95"] / 3.0
    for side in ("wall_left", "wall_right"):
        value += metrics["bearing_residual"][side]["p95_deg"] / 0.5
    return float(value)


def transform_payload(transform: Transform) -> dict[str, list[float]]:
    return {
        "translation_m": transform.p.tolist(),
        "quaternion_xyzw": transform.r.as_quat().tolist(),
    }


def main() -> int:
    args = parse_args()
    holdout_blocks = {int(value) for value in args.holdout_blocks.split(",")}
    compiled = compile_world_tag_map(args.tag_map)
    map_corners = {int(tag["id"]): np.asarray(tag["corners_m"]) for tag in compiled["tags"]}
    half = args.tag_size_m / 2.0
    tag_points = np.asarray([
        [-half, -half, 0.0], [half, -half, 0.0],
        [half, half, 0.0], [-half, half, 0.0],
    ])
    frames, own_left, own_right, own_audit = build_frames(args, tag_points, map_corners)
    train = [frame for frame in frames if frame.block not in holdout_blocks]
    holdout = [frame for frame in frames if frame.block in holdout_blocks]
    identity = np.zeros(12)
    fit = least_squares(
        lambda value: balanced_residual(value, train, own_left, own_right),
        identity,
        loss="huber", f_scale=0.003, max_nfev=1000,
        xtol=1e-12, ftol=1e-12, gtol=1e-12,
    )
    hypotheses = []
    for name, parameters in (("independent_identity", identity), ("joint_fixed_correction", fit.x)):
        holdout_metrics = closure_metrics(parameters, holdout, own_left, own_right)
        hypotheses.append({
            "name": name,
            "parameters": parameters,
            "holdout": holdout_metrics,
            "score": score(holdout_metrics),
        })
    selected = min(hypotheses, key=lambda item: item["score"])
    # Refit the selected model family on all frames only after holdout selection.
    if selected["name"] == "joint_fixed_correction":
        final_fit = least_squares(
            lambda value: balanced_residual(value, frames, own_left, own_right),
            fit.x,
            loss="huber", f_scale=0.003, max_nfev=1000,
            xtol=1e-12, ftol=1e-12, gtol=1e-12,
        )
        final_parameters = final_fit.x
    else:
        final_parameters = identity
    correction_left, correction_right = decode_corrections(final_parameters)
    selected_holdout = selected["holdout"]
    passed = all(
        selected_holdout[side]["position_error_mm"]["median"] <= 3.0
        and selected_holdout[side]["position_error_mm"]["p95"] <= 5.0
        and selected_holdout[side]["rotation_error_deg"]["median"] <= 2.0
        and selected_holdout[side]["rotation_error_deg"]["p95"] <= 3.0
        for side in ("left_tag3", "right_tag2")
    )
    payload = {
        "schema_version": "joint-camera-correction-cached/1.0",
        "status": "HOLDOUT_PASS" if passed else "HOLDOUT_FAILED",
        "selected_model": selected["name"],
        "frame_count": len(frames),
        "train_frame_count": len(train),
        "holdout_frame_count": len(holdout),
        "temporal_blocks": args.temporal_blocks,
        "holdout_blocks": sorted(holdout_blocks),
        "own_basetag": {
            "left_tag3": {"transform": transform_payload(own_left), "audit": own_audit["left"]},
            "right_tag2": {"transform": transform_payload(own_right), "audit": own_audit["right"]},
        },
        "world_camera_corrections": {
            "left": transform_payload(correction_left),
            "right": transform_payload(correction_right),
            "composition": "T_world_camera_corrected = T_world_camera_independent @ correction",
        },
        "holdout_selection": [{
            "name": item["name"], "score": item["score"], "metrics": item["holdout"],
        } for item in hypotheses],
        "selected_holdout": selected_holdout,
        "all_data_diagnostic": closure_metrics(final_parameters, frames, own_left, own_right),
        "training_ready": False,
        "source": {
            "left_cache": str(args.left_cache.resolve()),
            "right_cache": str(args.right_cache.resolve()),
            "left_pose": str(args.left_pose.resolve()),
            "right_pose": str(args.right_pose.resolve()),
            "tag_map": str(args.tag_map.resolve()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
