#!/usr/bin/env python3
"""Estimate gripper joint/opening angles from calibrated fisheye yellow markers.

The detector uses the existing yellow marker features, but fixes three legacy
assumptions: BaseTag size is 20 mm (not 24 mm), T_base_tag comes from hardware
rather than a hard-coded [-1, 0] mm centre, and pixels are converted through the
factory fisheye bearing model instead of a planar Tag homography.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from estimate_gripper_extrinsic import BODY_TO_PANORAMA_OPENCV
from raw_fisheye_world_pose import make_ray_converter


SIDES = ("left", "right")

# The camera is rigidly mounted to the gripper base, so its own two yellow
# markers can only occupy these small, camera-specific parts of stream 1.  The
# bounds are normalized (x0, y0, x1, y1) and deliberately include the complete
# jaw travel while excluding the opposite gripper and most scene clutter.
DEFAULT_MARKER_ROI = {
    "left": (0.35, 0.54, 0.75, 0.72),
    "right": (0.38, 0.54, 0.63, 0.72),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--camera-mount-calibration", type=Path, required=True,
                        help="gripper-mount-calibration containing camera_to_base")
    parser.add_argument("--hardware-geometry", type=Path, required=True)
    parser.add_argument("--side", choices=SIDES, required=True)
    parser.add_argument("--panoforge-root", type=Path, required=True)
    parser.add_argument("--source-width", type=int, default=3840)
    parser.add_argument("--source-height", type=int, default=3840)
    parser.add_argument("--process-width", type=int, default=1500)
    parser.add_argument("--clock-intercept-s", type=float, default=0.0)
    parser.add_argument("--clock-slope", type=float, default=1.0)
    parser.add_argument(
        "--marker-roi", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
        help="normalized fixed ROI for this camera's own yellow markers",
    )
    parser.add_argument("--max-opening-deg", type=float, default=55.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def transform_from_dict(payload: dict) -> tuple[np.ndarray, Rotation]:
    return np.asarray(payload["translation_m"], dtype=float), Rotation.from_quat(
        payload["quaternion_xyzw"]
    )


def marker_geometry(hardware: dict, side: str) -> tuple[np.ndarray, np.ndarray]:
    joint_key = "joint1_origin_m" if side == "left" else "joint2_origin_m"
    joint = np.asarray(hardware["joints"][joint_key], dtype=float)
    xy = np.asarray(hardware["marker_points_m"][side], dtype=float)
    z = 0.0038 if side == "left" else 0.0042
    return joint, np.r_[xy, z]


def build_projector(args: argparse.Namespace, camera_base_p: np.ndarray,
                    camera_base_r: Rotation):
    sys.path.insert(0, str(args.panoforge_root.resolve()))
    from app.core.maps import _quat_to_rot, _radial_model, scale_calibration_to_source

    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    calibration = scale_calibration_to_source(
        calibration, args.source_width, args.source_height
    )
    lens = calibration["lenses"][1]
    radial = _radial_model(lens)
    lens_rotation = _quat_to_rot(lens["extrinsic_quat"])
    display_scale = args.process_width / args.source_width

    def project(points_base: np.ndarray) -> np.ndarray:
        points_camera = camera_base_r.apply(points_base) + camera_base_p
        rays = unit(points_camera)
        direction_body = rays @ BODY_TO_PANORAMA_OPENCV
        direction_lens = direction_body @ lens_rotation.T
        theta = np.arccos(np.clip(direction_lens[:, 2], -1.0, 1.0))
        radius = radial(theta)
        planar = np.linalg.norm(direction_lens[:, :2], axis=1)
        pixels = np.c_[
            lens["cx"] + radius * direction_lens[:, 0] / np.maximum(planar, 1e-12),
            lens["cy"] + radius * direction_lens[:, 1] / np.maximum(planar, 1e-12),
        ]
        return pixels * display_scale

    return project


def unit(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    return value / np.linalg.norm(value, axis=-1, keepdims=True)


def projected_curves(project, hardware: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    angles = np.linspace(-55.0, 55.0, 441)
    curves = {}
    for side in SIDES:
        joint, marker = marker_geometry(hardware, side)
        rotations = Rotation.from_euler("z", angles[:, None], degrees=True)
        points = joint + rotations.apply(np.broadcast_to(marker, (len(angles), 3)))
        curves[side] = project(points)
    return angles, curves


def marker_candidates(
    image: np.ndarray,
    curves: dict[str, np.ndarray],
    marker_roi: tuple[float, float, float, float] | None = None,
) -> list[np.ndarray]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([18, 70, 60]), np.array([48, 255, 255]))
    all_curve = np.vstack(list(curves.values()))
    low = np.maximum(np.floor(all_curve.min(axis=0) - 35).astype(int), 0)
    high = np.minimum(np.ceil(all_curve.max(axis=0) + 35).astype(int),
                      np.array([image.shape[1] - 1, image.shape[0] - 1]))
    roi = np.zeros_like(mask)
    roi[low[1]:high[1] + 1, low[0]:high[0] + 1] = 255
    if marker_roi is not None:
        x0, y0, x1, y1 = marker_roi
        if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
            raise ValueError(f"invalid normalized marker ROI {marker_roi}")
        fixed = np.zeros_like(mask)
        fx0, fx1 = round(x0 * image.shape[1]), round(x1 * image.shape[1])
        fy0, fy1 = round(y0 * image.shape[0]), round(y1 * image.shape[0])
        fixed[fy0:fy1, fx0:fx1] = 255
        roi = cv2.bitwise_and(roi, fixed)
    mask = cv2.bitwise_and(mask, roi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    result = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        x, y, width, height = cv2.boundingRect(contour)
        aspect = width / max(height, 1)
        fill = area / max(width * height, 1)
        if not (30.0 <= area <= 700.0 and 0.40 <= aspect <= 2.8 and fill >= 0.25):
            continue
        moments = cv2.moments(contour)
        if moments["m00"]:
            centre = np.asarray([
                moments["m10"] / moments["m00"],
                moments["m01"] / moments["m00"],
            ])
            # The real marker is a yellow island inside a black circular ring.
            # A yellow bag/background patch can lie on the projected kinematic
            # arc and previously produced a geometrically plausible false jaw.
            radius = max(width, height)
            patch_radius = int(np.ceil(1.8 * radius))
            px, py = np.round(centre).astype(int)
            ax0, ax1 = max(0, px - patch_radius), min(image.shape[1], px + patch_radius + 1)
            ay0, ay1 = max(0, py - patch_radius), min(image.shape[0], py + patch_radius + 1)
            yy, xx = np.mgrid[ay0:ay1, ax0:ax1]
            squared = (xx - centre[0]) ** 2 + (yy - centre[1]) ** 2
            annulus = (squared >= (0.65 * radius) ** 2) & (squared <= (1.55 * radius) ** 2)
            dark_fraction = float(np.mean(hsv[ay0:ay1, ax0:ax1, 2][annulus] < 145))
            if dark_fraction >= 0.28:
                result.append(centre)

    # One marker can touch the yellow printed jaw in image space, turning the
    # marker and jaw into one huge HSV contour.  Recover it with a small
    # yellow-disc / dark-annulus template inside the fixed kinematic ROI.  This
    # is deliberately only a candidate generator; paired-curve geometry below
    # still decides whether the point belongs to this gripper.
    ys, xs = np.where(roi > 0)
    if len(xs):
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        hsv_crop = hsv[y0:y1, x0:x1]
        yellow = cv2.inRange(
            hsv_crop, np.array([18, 70, 60]), np.array([48, 255, 255])
        ).astype(np.float32) / 255.0
        dark = (hsv_crop[..., 2] < 145).astype(np.float32)
        grid_y, grid_x = np.mgrid[-17:18, -17:18]
        squared = grid_x * grid_x + grid_y * grid_y
        inner_kernel = (squared <= 6 ** 2).astype(np.float32)
        ring_kernel = ((squared >= 9 ** 2) & (squared <= 17 ** 2)).astype(np.float32)
        inner_kernel /= inner_kernel.sum()
        ring_kernel /= ring_kernel.sum()
        inner_score = cv2.filter2D(yellow, -1, inner_kernel)
        ring_score = cv2.filter2D(dark, -1, ring_kernel)
        template_score = inner_score * ring_score
        local_max = cv2.dilate(template_score, np.ones((15, 15), np.uint8))
        peak_mask = (
            (template_score >= local_max - 1e-6) & (template_score >= 0.55)
        ).astype(np.uint8)
        count, labels, _, _ = cv2.connectedComponentsWithStats(peak_mask)
        for label in range(1, count):
            py, px = np.where(labels == label)
            best = int(np.argmax(template_score[py, px]))
            centre = np.asarray([px[best] + x0, py[best] + y0], dtype=float)
            if np.min(np.linalg.norm(all_curve - centre, axis=1)) > 42.0:
                continue
            if all(np.linalg.norm(existing - centre) >= 10.0 for existing in result):
                result.append(centre)
    return result


def curve_match(point: np.ndarray, curve: np.ndarray, angles: np.ndarray) -> tuple[float, float]:
    distance = np.linalg.norm(curve - point, axis=1)
    index = int(np.argmin(distance))
    return float(distance[index]), float(angles[index])


def choose_pair(candidates: list[np.ndarray], curves: dict[str, np.ndarray],
                angles: np.ndarray, previous: np.ndarray | None
                ) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    best = None
    for left_index, left_point in enumerate(candidates):
        left_distance, left_angle = curve_match(left_point, curves["left"], angles)
        if left_distance > 36.0:
            continue
        for right_index, right_point in enumerate(candidates):
            if right_index == left_index or np.linalg.norm(left_point - right_point) < 18.0:
                continue
            right_distance, right_angle = curve_match(right_point, curves["right"], angles)
            if right_distance > 36.0:
                continue
            # The two equal gears impose opposite joint rotations.  This is the
            # decisive ownership test when the other gripper enters the same
            # image ROI: two unrelated yellow dots do not satisfy this closure.
            if abs(left_angle + right_angle) > 6.0:
                continue
            cost = left_distance + right_distance + 0.45 * abs(left_angle + right_angle)
            pair_angles = np.asarray([left_angle, right_angle])
            if previous is not None:
                cost += 0.35 * np.sum(np.abs(pair_angles - previous))
            if best is None or cost < best[0]:
                best = (cost, np.asarray([left_point, right_point]), pair_angles)
    if best is None:
        return None, None, 0.0
    confidence = float(np.exp(-best[0] / 24.0))
    return best[1], best[2], confidence


def ray_joint_solution(points_process: np.ndarray, args: argparse.Namespace,
                       ray_converter, camera_base_p: np.ndarray,
                       camera_base_r: Rotation, hardware: dict
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_source = points_process * (args.source_width / args.process_width)
    rays_camera = ray_converter(points_source)
    base_camera_r = camera_base_r.inv()
    camera_origin_base = -base_camera_r.apply(camera_base_p)
    directions_base = base_camera_r.apply(rays_camera)
    joints = []
    radius_errors_mm = []
    intersections = []
    for side, direction in zip(SIDES, directions_base):
        origin, marker = marker_geometry(hardware, side)
        plane_z = origin[2] + marker[2]
        distance = (plane_z - camera_origin_base[2]) / direction[2]
        point = camera_origin_base + distance * direction
        vector = point[:2] - origin[:2]
        angle = np.degrees(
            np.arctan2(vector[1], vector[0]) - np.arctan2(marker[1], marker[0])
        )
        angle = (angle + 180.0) % 360.0 - 180.0
        joints.append(angle)
        radius_errors_mm.append((np.linalg.norm(vector) - np.linalg.norm(marker[:2])) * 1000.0)
        intersections.append(point)
    return np.asarray(joints), np.asarray(radius_errors_mm), np.asarray(intersections)


def interpolate_smooth(raw: np.ndarray, window: int = 7) -> np.ndarray:
    result = np.empty_like(raw)
    index = np.arange(len(raw))
    for side in range(raw.shape[1]):
        values = raw[:, side]
        valid = np.isfinite(values)
        if valid.sum() < 2:
            raise RuntimeError(f"only {valid.sum()} valid marker observations for joint {side + 1}")
        unwrapped = np.unwrap(np.radians(values[valid]))
        filled = np.interp(index, index[valid], unwrapped)
        kernel = np.ones(window) / window
        padded = np.pad(filled, (window // 2, window // 2), mode="edge")
        result[:, side] = np.degrees(np.convolve(padded, kernel, mode="valid"))
    return result


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hardware = json.loads(args.hardware_geometry.read_text(encoding="utf-8"))
    if abs(float(hardware["frames"]["basetag"]["tag_outer_size_m"]) - 0.020) > 1e-12:
        raise ValueError("hardware BaseTag must be 20 mm")
    mount = json.loads(args.camera_mount_calibration.read_text(encoding="utf-8"))
    camera_base_p, camera_base_r = transform_from_dict(mount["camera_to_base"])
    project = build_projector(args, camera_base_p, camera_base_r)
    angles, curves = projected_curves(project, hardware)
    ray_converter, _ = make_ray_converter(SimpleNamespace(
        calibration=args.calibration,
        panoforge_root=args.panoforge_root,
        source_width=args.source_width,
        source_height=args.source_height,
        stream=1,
        radial_model="stitch",
    ))
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    process_height = round(args.process_width * args.source_height / args.source_width)
    raw_joints = []
    radius_errors = []
    image_points = []
    confidences = []
    previous = None
    previous_solution = None
    previous_solution_frame = None
    marker_roi = tuple(args.marker_roi or DEFAULT_MARKER_ROI[args.side])
    frame_index = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        image = cv2.resize(image, (args.process_width, process_height),
                           interpolation=cv2.INTER_AREA)
        candidates = marker_candidates(image, curves, marker_roi)
        pair, curve_angles, confidence = choose_pair(candidates, curves, angles, previous)
        joints = np.full(2, np.nan)
        radii = np.full(2, np.nan)
        if pair is not None:
            solved, radii, _ = ray_joint_solution(
                pair, args, ray_converter, camera_base_p, camera_base_r, hardware
            )
            opening_candidate = abs(
                np.degrees(np.arctan2(hardware["marker_points_m"]["left"][1],
                                      hardware["marker_points_m"]["left"][0])
                           - np.arctan2(hardware["marker_points_m"]["right"][1],
                                       hardware["marker_points_m"]["right"][0]))
                + solved[0] - solved[1]
            )
            continuous = True
            if previous_solution is not None and previous_solution_frame is not None:
                gap = frame_index - previous_solution_frame
                maximum_step = min(28.0, 8.0 + 4.0 * max(gap - 1, 0))
                continuous = bool(np.max(np.abs(solved - previous_solution)) <= maximum_step)
            if (np.all(np.abs(radii) <= 10.0) and np.all(np.abs(solved) <= 55.0)
                    and 0.0 <= opening_candidate <= args.max_opening_deg and continuous):
                joints = solved
                previous = curve_angles
                previous_solution = solved
                previous_solution_frame = frame_index
                confidence *= float(np.exp(-0.5 * np.max(np.abs(radii) / 8.0) ** 2))
            else:
                pair = None
                confidence = 0.0
        raw_joints.append(joints)
        radius_errors.append(radii)
        image_points.append(pair if pair is not None else np.full((2, 2), np.nan))
        confidences.append(confidence)
        frame_index += 1
    capture.release()
    raw_joints = np.asarray(raw_joints)
    radius_errors = np.asarray(radius_errors)
    image_points = np.asarray(image_points)
    confidences = np.asarray(confidences)
    measured = np.all(np.isfinite(raw_joints), axis=1)
    joints = interpolate_smooth(raw_joints)
    left_marker = np.asarray(hardware["marker_points_m"]["left"])
    right_marker = np.asarray(hardware["marker_points_m"]["right"])
    neutral = abs(np.degrees(
        np.arctan2(left_marker[1], left_marker[0])
        - np.arctan2(right_marker[1], right_marker[0])
    ))
    opening = np.abs((neutral + joints[:, 0] - joints[:, 1] + 180.0) % 360.0 - 180.0)
    local_times = np.arange(len(joints)) / fps
    common_times = (local_times - args.clock_intercept_s) / args.clock_slope
    csv_path = args.output_dir / "gripper_opening.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "local_time_s", "common_time_s", "opening_angle_deg",
                         "joint1_deg", "joint2_deg", "measured", "confidence",
                         "left_x", "left_y", "right_x", "right_y",
                         "left_radius_error_mm", "right_radius_error_mm"])
        for index in range(len(joints)):
            writer.writerow([
                index, f"{local_times[index]:.9f}", f"{common_times[index]:.9f}",
                f"{opening[index]:.6f}", f"{joints[index, 0]:.6f}",
                f"{joints[index, 1]:.6f}", int(measured[index]),
                f"{confidences[index]:.6f}", *image_points[index].reshape(-1),
                *radius_errors[index],
            ])
    np.savez_compressed(
        args.output_dir / "yellow_marker_observations.npz",
        frame_index=np.arange(len(joints), dtype=np.int32),
        local_time_s=local_times,
        common_time_s=common_times,
        image_points_px=image_points.astype(np.float32),
        raw_joint_deg=raw_joints.astype(np.float32),
        filtered_joint_deg=joints.astype(np.float32),
        opening_angle_deg=opening.astype(np.float32),
        measured=measured,
        confidence=confidences.astype(np.float32),
        radius_error_mm=radius_errors.astype(np.float32),
    )
    finite_radius = np.abs(radius_errors[measured])
    steps = np.abs(np.diff(opening))
    summary = {
        "schema_version": "calibrated-fisheye-gripper-opening/1.0",
        "side": args.side,
        "video": str(args.video.resolve()),
        "frames": len(joints),
        "fps": fps,
        "measured_ratio": float(measured.mean()),
        "opening_angle_deg": {
            "min": float(opening.min()), "max": float(opening.max()),
            "median": float(np.median(opening)),
        },
        "frame_step_deg": {
            "median": float(np.median(steps)), "p95": float(np.percentile(steps, 95)),
        },
        "radius_error_abs_mm": {
            "median": float(np.median(finite_radius)),
            "p95": float(np.percentile(finite_radius, 95)),
            "gate": 10.0,
        },
        "neutral_marker_direction_angle_deg": float(neutral),
        "marker_roi_normalized": list(marker_roi),
        "max_opening_deg": float(args.max_opening_deg),
        "method": "yellow marker contours -> calibrated fisheye bearings -> base motion-plane intersection -> CAD joint inverse",
        "legacy_bug_fixes": [
            "20 mm BaseTag instead of hard-coded 24 mm",
            "hardware T_base_tag instead of hard-coded [-1,0] mm Tag centre",
            "factory fisheye bearing intersection instead of planar homography",
        ],
        "measurement_semantics": "measured is direct two-marker geometry; unmeasured frames are explicitly interpolated",
        "training_ready": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
