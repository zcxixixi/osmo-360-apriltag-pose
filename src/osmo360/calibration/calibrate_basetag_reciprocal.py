#!/usr/bin/env python3
"""Calibrate fixed camera->BaseTag extrinsics from reciprocal observations.

The camera's own BaseTag is too close to the fisheye edge for a trustworthy
metric PnP extrinsic.  Instead, the opposite camera observes that BaseTag while
both cameras are independently localized by the fixed wall map:

    T_targetCamera_tag = inv(T_world_targetCamera)
                         @ T_world_observerCamera
                         @ T_observerCamera_tag

Multiple frames and both IPPE branches are evaluated in SE(3); only the
consistent branch/frame cluster is retained.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from osmo360.calibration.estimate_gripper_extrinsic import BODY_TO_PANORAMA_OPENCV, solve_bearing_ippe


@dataclass(frozen=True)
class Transform:
    p: np.ndarray
    r: Rotation

    def compose(self, other: "Transform") -> "Transform":
        return Transform(self.p + self.r.apply(other.p), self.r * other.r)

    def inverse(self) -> "Transform":
        inverse_rotation = self.r.inv()
        return Transform(-inverse_rotation.apply(self.p), inverse_rotation)


def parse_roi(text: str) -> tuple[float, float, float, float]:
    values = tuple(map(float, text.split(",")))
    if len(values) != 4:
        raise argparse.ArgumentTypeError("ROI must be xmin,ymin,xmax,ymax")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observer-name", required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--observer-video", type=Path, required=True)
    parser.add_argument("--observer-calibration", type=Path, required=True)
    parser.add_argument("--observer-pose", type=Path, required=True)
    parser.add_argument("--target-pose", type=Path, required=True)
    parser.add_argument("--camera-frame", default="panorama_camera",
                        help="parent frame of the calibrated camera-to-tag transform")
    parser.add_argument("--tag-id", type=int, required=True)
    parser.add_argument("--tag-size-m", type=float, default=0.020)
    parser.add_argument("--tag-corner-quarter-turns", type=int, choices=range(4), default=0,
                        help="rotate decoded canonical corners into the physical hardware Tag axes")
    parser.add_argument("--source-time-offset-s", type=float, default=0.0,
                        help="common time = observer video time - this offset")
    parser.add_argument("--stream", type=int, default=1)
    parser.add_argument("--source-width", type=int, default=3000)
    parser.add_argument("--source-height", type=int, default=3000)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--roi", type=parse_roi)
    parser.add_argument("--min-area-px2", type=float, default=1000.0)
    parser.add_argument("--max-position-cluster-m", type=float, default=0.025)
    parser.add_argument("--max-rotation-cluster-deg", type=float, default=12.0)
    parser.add_argument("--radial-model", choices=("stitch", "factory-polynomial"), default="stitch")
    parser.add_argument("--edge-rectification", action="store_true",
                        help="detect the reciprocal BaseTag in calibrated tangent views")
    parser.add_argument("--rectified-view-size", type=int, default=900)
    parser.add_argument("--panoforge-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_pose(path: Path) -> tuple[np.ndarray, np.ndarray, Rotation]:
    rows = [row for row in csv.DictReader(path.open(newline="", encoding="utf-8"))
            if row.get("camera_x_m") and row.get("qw")]
    times = np.asarray([float(row["timestamp"]) for row in rows])
    positions = np.asarray([[float(row[key]) for key in
                             ("camera_x_m", "camera_y_m", "camera_z_m")]
                            for row in rows])
    rotations = Rotation.from_quat([[float(row[key]) for key in
                                     ("qx", "qy", "qz", "qw")]
                                    for row in rows])
    return times, positions, rotations


def interpolate_pose(series: tuple[np.ndarray, np.ndarray, Rotation], time_s: float) -> Transform | None:
    times, positions, rotations = series
    if time_s < times[0] or time_s > times[-1]:
        return None
    position = np.asarray([np.interp(time_s, times, positions[:, axis]) for axis in range(3)])
    rotation = Slerp(times, rotations)([time_s])[0]
    return Transform(position, rotation)


def make_ray_converter(calibration_path: Path, panoforge_root: Path,
                       source_width: int, source_height: int, stream: int,
                       radial_model_name: str = "metric"):
    sys.path.insert(0, str(panoforge_root.resolve()))
    from app.core.maps import _quat_to_rot, _radial_model, scale_calibration_to_source

    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration = scale_calibration_to_source(calibration, source_width, source_height)
    lens = calibration["lenses"][stream]
    if radial_model_name == "factory-polynomial":
        from osmo360.localization.raw_fisheye_world_pose import metric_radial_model
        radial_model = metric_radial_model(lens)
    else:
        radial_model = _radial_model(lens)
    centre = np.asarray([lens["cx"], lens["cy"]], dtype=float)
    lens_rotation = _quat_to_rot(lens["extrinsic_quat"])

    def convert(corners: np.ndarray) -> np.ndarray:
        corners = np.asarray(corners, dtype=float).reshape(-1, 2)
        radius = np.linalg.norm(corners - centre, axis=1)
        low = np.zeros(len(corners)); high = np.full(len(corners), np.radians(96.0))
        for _ in range(80):
            middle = (low + high) / 2.0
            below = radial_model(middle) < radius
            low = np.where(below, middle, low)
            high = np.where(below, high, middle)
        theta = (low + high) / 2.0
        direction_lens = np.c_[
            np.sin(theta) * (corners[:, 0] - lens["cx"]) / radius,
            np.sin(theta) * (corners[:, 1] - lens["cy"]) / radius,
            np.cos(theta),
        ]
        direction_body = direction_lens @ lens_rotation
        rays = direction_body @ BODY_TO_PANORAMA_OPENCV.T
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)

    return convert


def detect_candidates(args: argparse.Namespace, observer_pose, target_pose) -> list[dict[str, Any]]:
    ray_converter = make_ray_converter(
        args.observer_calibration, args.panoforge_root,
        args.source_width, args.source_height, args.stream, args.radial_model,
    )
    rectified_maps = []
    if args.edge_rectification:
        from types import SimpleNamespace
        from osmo360.localization.raw_fisheye_world_pose import make_ray_converter as raw_make_ray_converter
        from osmo360.localization.raw_fisheye_world_pose import make_rectified_maps
        raw_args = SimpleNamespace(
            calibration=args.observer_calibration,
            panoforge_root=args.panoforge_root,
            source_width=args.source_width,
            source_height=args.source_height,
            stream=args.stream,
            edge_rectification=True,
            rectified_view_size=args.rectified_view_size,
            radial_model=args.radial_model,
        )
        _, scaled_calibration = raw_make_ray_converter(raw_args)
        rectified_maps = make_rectified_maps(raw_args, scaled_calibration)
    half = args.tag_size_m / 2.0
    tag_points = np.asarray([
        [-half, -half, 0.0], [half, -half, 0.0],
        [half, half, 0.0], [-half, half, 0.0],
    ])
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.adaptiveThreshWinSizeMax = 63
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    capture = cv2.VideoCapture(str(args.observer_video))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    end = total if args.end_frame is None else min(total, args.end_frame + 1)
    candidates: list[dict[str, Any]] = []
    frame = 0
    while frame < end:
        ok, image = capture.read()
        if not ok:
            break
        if frame >= args.start_frame and (frame - args.start_frame) % args.stride == 0:
            corners, ids, _ = detector.detectMarkers(image)
            detections = [] if ids is None else [
                (int(tag_id), quad.reshape(4, 2))
                for tag_id, quad in zip(ids.flatten(), corners)
                if int(tag_id) == args.tag_id
            ]
            if not detections and rectified_maps:
                from osmo360.localization.raw_fisheye_world_pose import detect_rectified_tags
                detections = [
                    (tag_id, quad) for tag_id, quad in
                    detect_rectified_tags(image, detector, rectified_maps)
                    if tag_id == args.tag_id
                ]
            if detections:
                for tag_id, quad in detections:
                    centre = quad.mean(axis=0)
                    area = abs(float(cv2.contourArea(quad.astype(np.float32))))
                    if area < args.min_area_px2:
                        continue
                    if args.roi is not None:
                        xmin, ymin, xmax, ymax = args.roi
                        if not (xmin <= centre[0] <= xmax and ymin <= centre[1] <= ymax):
                            continue
                    common_time = frame / fps - args.source_time_offset_s
                    world_observer = interpolate_pose(observer_pose, common_time)
                    world_target = interpolate_pose(target_pose, common_time)
                    if world_observer is None or world_target is None:
                        continue
                    rays = np.roll(
                        ray_converter(quad), -args.tag_corner_quarter_turns, axis=0
                    )
                    solutions = solve_bearing_ippe(tag_points, rays)
                    for solution in solutions:
                        observer_tag = Transform(
                            np.asarray(solution["translation_tag_origin_in_panorama_m"]),
                            Rotation.from_matrix(solution["rotation_tag_to_panorama"]),
                        )
                        target_tag = world_target.inverse().compose(
                            world_observer.compose(observer_tag)
                        )
                        candidates.append({
                            "frame": frame, "common_time_s": common_time,
                            "branch": int(solution["branch"]),
                            "angular_rmse_deg": float(solution["angular_rmse_deg"]),
                            "transform": target_tag,
                            "observer_tag": observer_tag,
                            "world_observer": world_observer,
                            "world_target": world_target,
                        })
        frame += 1
    capture.release()
    return candidates


def rotation_distance_deg(a: Rotation, b: Rotation) -> float:
    return float((a.inv() * b).magnitude() * 180.0 / np.pi)


def robust_cluster(candidates: list[dict[str, Any]], max_position_m: float,
                   max_rotation_deg: float) -> tuple[Transform, list[dict[str, Any]], dict[str, Any]]:
    if not candidates:
        raise RuntimeError("no reciprocal BaseTag observations")
    best = None
    best_key = None
    for seed in candidates:
        support = []
        for item in candidates:
            if item["frame"] == seed["frame"] and item["branch"] != seed["branch"]:
                continue
            dp = float(np.linalg.norm(item["transform"].p - seed["transform"].p))
            dr = rotation_distance_deg(seed["transform"].r, item["transform"].r)
            if dp <= max_position_m and dr <= max_rotation_deg:
                support.append(item)
        unique_frames = len({item["frame"] for item in support})
        residual = sum(
            np.linalg.norm(item["transform"].p - seed["transform"].p) / max_position_m
            + rotation_distance_deg(seed["transform"].r, item["transform"].r) / max_rotation_deg
            for item in support
        )
        key = (unique_frames, len(support), -residual, -seed["angular_rmse_deg"])
        if best_key is None or key > best_key:
            best_key, best = key, seed
    assert best is not None
    selected = []
    for frame in sorted({item["frame"] for item in candidates}):
        options = [item for item in candidates if item["frame"] == frame]
        options.sort(key=lambda item: (
            np.linalg.norm(item["transform"].p - best["transform"].p) / max_position_m
            + rotation_distance_deg(best["transform"].r, item["transform"].r) / max_rotation_deg
        ))
        item = options[0]
        if (np.linalg.norm(item["transform"].p - best["transform"].p) <= max_position_m
                and rotation_distance_deg(best["transform"].r, item["transform"].r) <= max_rotation_deg):
            selected.append(item)
    if len(selected) < 3:
        raise RuntimeError(f"only {len(selected)} consistent reciprocal observations")
    positions = np.asarray([item["transform"].p for item in selected])
    translation = np.median(positions, axis=0)
    weights = np.asarray([1.0 / max(item["angular_rmse_deg"], 0.03) for item in selected])
    rotation = Rotation.from_quat(
        np.asarray([item["transform"].r.as_quat() for item in selected])
    ).mean(weights=weights)
    calibrated = Transform(translation, rotation)
    position_errors = np.asarray([
        np.linalg.norm(item["transform"].p - translation) for item in selected
    ])
    rotation_errors = np.asarray([
        rotation_distance_deg(calibrated.r, item["transform"].r) for item in selected
    ])
    audit = {
        "candidate_count": len(candidates),
        "observed_frame_count": len({item["frame"] for item in candidates}),
        "inlier_count": len(selected),
        "inlier_frames": [item["frame"] for item in selected],
        "selected_branches": [item["branch"] for item in selected],
        "position_residual_mm": {
            "median": float(np.median(position_errors) * 1000),
            "p95": float(np.percentile(position_errors, 95) * 1000),
            "max": float(position_errors.max() * 1000),
        },
        "rotation_residual_deg": {
            "median": float(np.median(rotation_errors)),
            "p95": float(np.percentile(rotation_errors, 95)),
            "max": float(rotation_errors.max()),
        },
    }
    return calibrated, selected, audit


def main() -> int:
    args = parse_args()
    observer_pose = load_pose(args.observer_pose)
    target_pose = load_pose(args.target_pose)
    candidates = detect_candidates(args, observer_pose, target_pose)
    calibrated, selected, audit = robust_cluster(
        candidates, args.max_position_cluster_m, args.max_rotation_cluster_deg
    )
    closure_position = []
    closure_rotation = []
    for item in selected:
        observed_world = item["world_observer"].compose(item["observer_tag"])
        predicted_world = item["world_target"].compose(calibrated)
        closure_position.append(np.linalg.norm(observed_world.p - predicted_world.p))
        closure_rotation.append(rotation_distance_deg(observed_world.r, predicted_world.r))
    payload = {
        "schema_version": "reciprocal-basetag-calibration/1.0",
        "calibration_status": "PROVISIONAL_RECIPROCAL_CROSS_OBSERVATION",
        "observer": args.observer_name,
        "target": args.target_name,
        "tag_id": args.tag_id,
        "tag_size_m": args.tag_size_m,
        "tag_corner_quarter_turns": args.tag_corner_quarter_turns,
        "camera_to_basetag": {
            "parent_frame": args.camera_frame,
            "child_frame": f"{args.target_name}_mount_tag2",
            "translation_m": calibrated.p.tolist(),
            "quaternion_xyzw": calibrated.r.as_quat().tolist(),
            "corner_convention": "opencv_aruco_apriltag_canonical",
        },
        "audit": audit,
        "world_closure": {
            "position_error_mm": {
                "median": float(np.median(closure_position) * 1000),
                "p95": float(np.percentile(closure_position, 95) * 1000),
                "max": float(np.max(closure_position) * 1000),
            },
            "rotation_error_deg": {
                "median": float(np.median(closure_rotation)),
                "p95": float(np.percentile(closure_rotation, 95)),
                "max": float(np.max(closure_rotation)),
            },
        },
        "source": {
            "observer_video": str(args.observer_video.resolve()),
            "observer_pose": str(args.observer_pose.resolve()),
            "target_pose": str(args.target_pose.resolve()),
            "source_time_offset_s": args.source_time_offset_s,
            "roi": args.roi,
        },
        "training_ready": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
