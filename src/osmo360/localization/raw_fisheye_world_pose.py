#!/usr/bin/env python3
"""Estimate metric world poses directly from one raw Insta360 fisheye stream.

This bypasses equirectangular stitching and its near-field parallax. Pixels are
converted to calibrated unit-bearing rays, then a robust world->camera bearing
bundle fit estimates the optical-centre pose against the fixed AprilTag map.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from osmo360.calibration.estimate_gripper_extrinsic import BODY_TO_PANORAMA_OPENCV
from osmo360.localization.world_frames import compile_world_tag_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path, help="extracted raw fisheye stream MP4")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--tag-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--panoforge-root", type=Path, required=True)
    parser.add_argument("--stream", type=int, default=1)
    parser.add_argument("--source-width", type=int, default=3000)
    parser.add_argument("--source-height", type=int, default=3000)
    parser.add_argument("--common-duration-s", type=float, required=True)
    parser.add_argument("--source-time-offset-s", type=float, default=0.0,
                        help="raw local time = common time + this offset")
    parser.add_argument("--sample-fps", type=float, default=10.0)
    parser.add_argument("--initial-pose", type=Path,
                        help="initialization only; never accepted as a measurement")
    parser.add_argument("--min-tags", type=int, default=2)
    parser.add_argument("--max-angular-rmse-deg", type=float, default=1.5)
    parser.add_argument("--radial-model", choices=("stitch", "factory-polynomial"), default="stitch",
                        help="stitch is the hardware-validated LUT-scaled model; factory-polynomial is diagnostic only")
    parser.add_argument("--edge-rectification", action="store_true",
                        help="decode strongly curved edge Tags in calibrated tangent views")
    parser.add_argument("--rectified-view-size", type=int, default=900)
    parser.add_argument("--rectification-radial-model", choices=("metric", "stitch"), default="stitch",
                        help="projection used only to decode tangent views; raw returned corners still use --radial-model")
    return parser.parse_args()


def load_initial(path: Path | None):
    if path is None:
        return None
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


def interpolate_initial(series, time_s: float):
    if series is None:
        return None
    times, positions, rotations = series
    if time_s < times[0] or time_s > times[-1]:
        return None
    position = np.asarray([np.interp(time_s, times, positions[:, axis]) for axis in range(3)])
    rotation = Slerp(times, rotations)([time_s])[0]
    return position, rotation


def metric_radial_model(lens: dict[str, Any]):
    """Factory fisheye polynomial without PanoForge's panorama seam scale."""
    fx = float(lens["fx"])
    k1, k2, k3, k4 = map(float, lens.get("dist", [0, 0, 0, 0]))
    theta0 = np.radians(85.0)

    def polynomial(theta):
        t2 = theta * theta
        return theta * (1 + k1 * t2 + k2 * t2**2 + k3 * t2**3 + k4 * t2**4)

    g0 = polynomial(theta0)
    derivative0 = (1 + 3 * k1 * theta0**2 + 5 * k2 * theta0**4
                   + 7 * k3 * theta0**6 + 9 * k4 * theta0**8)

    def model(theta):
        theta = np.asarray(theta)
        return fx * np.where(
            theta <= theta0,
            polynomial(theta),
            g0 + derivative0 * (theta - theta0),
        )

    return model


def make_x5_offset_ray_converter(
    offset: str,
    *,
    stream: int,
    source_width: int,
    source_height: int,
):
    """Convert X5 raw pixels to unit rays from its embedded lens-offset record."""
    fields = offset.strip().split("_")
    if len(fields) != 16 or fields[0] not in {"m2", "n2"}:
        raise ValueError("invalid Insta360 X5 dual-lens offset record")
    values = np.asarray([float(value) for value in fields[1:]], dtype=float)
    if stream not in (0, 1):
        raise ValueError("X5 stream must be 0 or 1")
    stacked_height, calibration_width = values[12:14]
    if not math.isclose(stacked_height, 2 * calibration_width, rel_tol=0.01):
        raise ValueError("X5 offset record is not a stacked dual-fisheye calibration")
    start = stream * 6
    centre_x, stacked_centre_y, radius, tilt_x, tilt_y, half_fov_deg = (
        values[start:start + 6]
    )
    centre_y = stacked_centre_y - stream * calibration_width
    scale_x = source_width / calibration_width
    scale_y = source_height / calibration_width
    centre = np.asarray([centre_x * scale_x, centre_y * scale_y], dtype=float)
    scaled_radius = radius * math.sqrt(scale_x * scale_y)
    lens_to_rig = Rotation.from_euler("xy", [tilt_x, tilt_y], degrees=True)
    if stream == 1:
        lens_to_rig = Rotation.from_euler("y", 180.0, degrees=True) * lens_to_rig
    half_fov = math.radians(half_fov_deg)

    def convert(pixels: np.ndarray) -> np.ndarray:
        pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
        delta = pixels - centre
        radius_px = np.linalg.norm(delta, axis=1)
        theta = radius_px / scaled_radius * half_fov
        planar = np.divide(
            delta,
            radius_px[:, None],
            out=np.zeros_like(delta),
            where=radius_px[:, None] > 1e-9,
        )
        rays_lens = np.column_stack(
            (
                np.sin(theta) * planar[:, 0],
                np.sin(theta) * planar[:, 1],
                np.cos(theta),
            )
        )
        rays = lens_to_rig.apply(rays_lens)
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)

    metadata = {
        "camera_model": "insta360-x5",
        "offset": offset,
        "stream": stream,
        "source_size": [source_width, source_height],
        "centre_px": centre.tolist(),
        "radius_px": scaled_radius,
        "half_fov_deg": half_fov_deg,
        "tilt_deg": [tilt_x, tilt_y],
        "ray_frame": "x5_dual_fisheye_rig_stream0",
    }
    return convert, metadata


def make_x5_rectified_maps(
    offset: str,
    *,
    stream: int,
    source_width: int,
    source_height: int,
    view_size: int = 960,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Map overlapping tangent views into one X5 raw fisheye lens image."""
    _, metadata = make_x5_offset_ray_converter(
        offset,
        stream=stream,
        source_width=source_width,
        source_height=source_height,
    )
    centre = np.asarray(metadata["centre_px"], dtype=float)
    radius_px = float(metadata["radius_px"])
    half_fov = math.radians(float(metadata["half_fov_deg"]))
    fov = math.radians(85.0)
    coordinate = (
        (np.arange(view_size) + 0.5 - view_size / 2.0)
        / (view_size / 2.0)
        * np.tan(fov / 2.0)
    )
    xx, yy = np.meshgrid(coordinate, coordinate)
    base = np.stack([xx, yy, np.ones_like(xx)], axis=-1)
    base /= np.linalg.norm(base, axis=-1, keepdims=True)
    centres = [
        (0, 0),
        (-55, 0),
        (55, 0),
        (0, -55),
        (0, 55),
        (-45, -45),
        (-45, 45),
        (45, -45),
        (45, 45),
        (-80, 0),
        (80, 0),
    ]
    maps = []
    for yaw, pitch in centres:
        rays = Rotation.from_euler("yx", [yaw, pitch], degrees=True).apply(
            base.reshape(-1, 3)
        ).reshape(base.shape)
        theta = np.arccos(np.clip(rays[..., 2], -1.0, 1.0))
        planar_radius = np.linalg.norm(rays[..., :2], axis=-1)
        image_radius = theta / half_fov * radius_px
        valid = theta <= half_fov
        xmap = (
            centre[0]
            + image_radius * rays[..., 0] / np.maximum(planar_radius, 1e-9)
        ).astype(np.float32)
        ymap = (
            centre[1]
            + image_radius * rays[..., 1] / np.maximum(planar_radius, 1e-9)
        ).astype(np.float32)
        xmap[~valid] = -1
        ymap[~valid] = -1
        maps.append((xmap, ymap))
    return maps


def make_ray_converter(args: argparse.Namespace):
    sys.path.insert(0, str(args.panoforge_root.resolve()))
    from app.core.maps import _quat_to_rot, _radial_model, scale_calibration_to_source

    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    calibration = scale_calibration_to_source(
        calibration, args.source_width, args.source_height
    )
    lens = calibration["lenses"][args.stream]
    radial_model = (metric_radial_model(lens)
                    if getattr(args, "radial_model", "stitch") == "factory-polynomial"
                    else _radial_model(lens))
    centre = np.asarray([lens["cx"], lens["cy"]], dtype=float)
    lens_rotation = _quat_to_rot(lens["extrinsic_quat"])

    def convert(pixels: np.ndarray) -> np.ndarray:
        pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
        radius = np.linalg.norm(pixels - centre, axis=1)
        low = np.zeros(len(pixels)); high = np.full(len(pixels), np.radians(96.0))
        for _ in range(80):
            middle = (low + high) / 2.0
            below = radial_model(middle) < radius
            low = np.where(below, middle, low)
            high = np.where(below, high, middle)
        theta = (low + high) / 2.0
        direction_lens = np.c_[
            np.sin(theta) * (pixels[:, 0] - lens["cx"]) / radius,
            np.sin(theta) * (pixels[:, 1] - lens["cy"]) / radius,
            np.cos(theta),
        ]
        direction_body = direction_lens @ lens_rotation
        rays = direction_body @ BODY_TO_PANORAMA_OPENCV.T
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)

    return convert, calibration


def make_rectified_maps(args: argparse.Namespace, calibration: dict[str, Any]):
    """Map overlapping tangent views back into one raw fisheye image."""
    if not args.edge_rectification:
        return []
    sys.path.insert(0, str(args.panoforge_root.resolve()))
    from app.core.maps import _radial_model

    lens = calibration["lenses"][args.stream]
    radial_model = (metric_radial_model(lens)
                    if getattr(args, "rectification_radial_model", "stitch") == "metric"
                    else _radial_model(lens))
    size = args.rectified_view_size
    fov = np.radians(85.0)
    coordinate = ((np.arange(size) + 0.5 - size / 2.0) / (size / 2.0)
                  * np.tan(fov / 2.0))
    xx, yy = np.meshgrid(coordinate, coordinate)
    base = np.stack([xx, yy, np.ones_like(xx)], axis=-1)
    base /= np.linalg.norm(base, axis=-1, keepdims=True)
    centres = [
        (0, 0), (-55, 0), (55, 0), (0, -55), (0, 55),
        (-45, -45), (-45, 45), (45, -45), (45, 45),
        (-80, 0), (80, 0),
    ]
    maps = []
    for yaw, pitch in centres:
        centre_rotation = Rotation.from_euler("yx", [yaw, pitch], degrees=True)
        rays = centre_rotation.apply(base.reshape(-1, 3)).reshape(base.shape)
        theta = np.arccos(np.clip(rays[..., 2], -1.0, 1.0))
        planar_radius = np.linalg.norm(rays[..., :2], axis=-1)
        image_radius = radial_model(theta)
        xmap = (lens["cx"] + image_radius * rays[..., 0]
                / np.maximum(planar_radius, 1e-9)).astype(np.float32)
        ymap = (lens["cy"] + image_radius * rays[..., 1]
                / np.maximum(planar_radius, 1e-9)).astype(np.float32)
        maps.append((xmap, ymap))
    return maps


def bilinear_map_points(mapping: np.ndarray, points: np.ndarray) -> np.ndarray:
    height, width = mapping.shape
    x = np.clip(points[:, 0], 0, width - 1.001)
    y = np.clip(points[:, 1], 0, height - 1.001)
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, width - 1); y1 = np.minimum(y0 + 1, height - 1)
    wx = x - x0; wy = y - y0
    return ((1 - wx) * (1 - wy) * mapping[y0, x0]
            + wx * (1 - wy) * mapping[y0, x1]
            + (1 - wx) * wy * mapping[y1, x0]
            + wx * wy * mapping[y1, x1])


def detect_rectified_tags(image: np.ndarray, detector, rectified_maps):
    """Return the largest tangent-view detection per ID in raw pixel coordinates."""
    selected = {}
    for xmap, ymap in rectified_maps:
        view = cv2.remap(image, xmap, ymap, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)
        quads, ids, _ = detector.detectMarkers(view)
        if ids is None:
            continue
        for tag_id, quad in zip(ids.flatten(), quads):
            quad = quad.reshape(4, 2)
            area = abs(float(cv2.contourArea(quad.astype(np.float32))))
            raw_quad = np.column_stack([
                bilinear_map_points(xmap, quad),
                bilinear_map_points(ymap, quad),
            ]).astype(np.float32)
            if int(tag_id) not in selected or area > selected[int(tag_id)][0]:
                selected[int(tag_id)] = (area, raw_quad)
    return [(tag_id, value[1]) for tag_id, value in selected.items()]


def solve_pose(world_points: np.ndarray, rays: np.ndarray,
               initial: tuple[np.ndarray, Rotation], *, regularize: bool = False
               ) -> tuple[np.ndarray, Rotation, np.ndarray]:
    initial_position, initial_rotation = initial

    def bearing_residual(parameters: np.ndarray) -> np.ndarray:
        position = parameters[:3]
        rotation_world_camera = Rotation.from_rotvec(parameters[3:])
        predicted = rotation_world_camera.inv().apply(world_points - position)
        predicted /= np.linalg.norm(predicted, axis=1, keepdims=True)
        return (predicted - rays).ravel()

    def residual(parameters: np.ndarray) -> np.ndarray:
        bearing = bearing_residual(parameters)
        if not regularize:
            return bearing
        rotation_world_camera = Rotation.from_rotvec(parameters[3:])
        # Edge views are often a single physical plane and therefore retain a
        # planar mirror branch. A weak independent-pose prior chooses a branch;
        # it is deliberately far weaker than the corner-bearing observations.
        position_prior = (parameters[:3] - initial_position) * 0.20
        rotation_prior = (initial_rotation.inv() * rotation_world_camera).as_rotvec() * 0.05
        return np.r_[bearing, position_prior, rotation_prior]

    x0 = np.r_[initial_position, initial_rotation.as_rotvec()]
    fit = least_squares(
        residual, x0, loss="huber", f_scale=0.003, max_nfev=4000,
        xtol=1e-12, ftol=1e-12, gtol=1e-12,
    )
    position = fit.x[:3]
    rotation = Rotation.from_rotvec(fit.x[3:])
    vector_error = bearing_residual(fit.x).reshape(-1, 3)
    angular_error = np.degrees(np.arccos(np.clip(
        1.0 - np.sum(vector_error * vector_error, axis=1) / 2.0, -1.0, 1.0
    )))
    return position, rotation, angular_error


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    compiled = compile_world_tag_map(args.tag_map)
    map_corners = {
        int(tag["id"]): np.asarray(tag["corners_m"], dtype=float)
        for tag in compiled["tags"]
    }
    convert_rays, calibration = make_ray_converter(args)
    rectified_maps = make_rectified_maps(args, calibration)
    initial_series = load_initial(args.initial_pose)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.adaptiveThreshWinSizeMax = 63
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    capture = cv2.VideoCapture(str(args.video))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    common_times = np.arange(
        int(math.floor(args.common_duration_s * args.sample_fps)) + 1,
        dtype=float,
    ) / args.sample_fps
    raw_frames = np.rint(
        (common_times + args.source_time_offset_s) * source_fps
    ).astype(int)
    valid_targets = {
        int(raw_frame): (index, float(common_time))
        for index, (raw_frame, common_time) in enumerate(zip(raw_frames, common_times))
        if 0 <= raw_frame < total_frames
    }
    rows: list[dict[str, Any]] = []
    frame_index = 0
    previous = None
    while True:
        ok, image = capture.read()
        if not ok:
            break
        target = valid_targets.get(frame_index)
        if target is None:
            frame_index += 1
            continue
        common_index, common_time = target
        quads, ids, _ = detector.detectMarkers(image)
        detections = [] if ids is None else [
            (int(tag_id), quad.reshape(4, 2))
            for tag_id, quad in zip(ids.flatten(), quads)
        ]
        direct_map_ids = {tag_id for tag_id, _ in detections if tag_id in map_corners}
        edge_rectified = False
        if rectified_maps and len(direct_map_ids) < max(args.min_tags, 2):
            rectified = detect_rectified_tags(image, detector, rectified_maps)
            if rectified:
                edge_rectified = True
                by_id = {tag_id: quad for tag_id, quad in detections}
                # Tangent-view corners are preferable at the fisheye boundary:
                # the direct decoder fits straight edges to visibly curved Tags.
                by_id.update({tag_id: quad for tag_id, quad in rectified})
                detections = list(by_id.items())
        world_points = []
        pixels = []
        used_ids = []
        for tag_id, quad in detections:
            if tag_id not in map_corners:
                continue
            world_points.extend(map_corners[tag_id])
            pixels.extend(quad.reshape(4, 2))
            used_ids.append(tag_id)
        initial = interpolate_initial(initial_series, common_time) or previous
        output_frame = int(round(common_time * source_fps))
        if len(set(used_ids)) < args.min_tags or initial is None:
            rows.append({
                "frame": output_frame, "timestamp": f"{common_time:.6f}",
                "parent_frame": compiled.get("world_frame", "tag_map"),
                "child_frame": "fisheye1_camera_panorama_axes",
                "tag_map_sha256": compiled["tag_map_sha256"],
                "measurement_source": "raw_fisheye_failed", "quality_status": "invalid",
                "detected_tag_count": len(set(used_ids)),
                "detected_ids": " ".join(map(str, sorted(set(used_ids)))),
                "edge_rectified": str(edge_rectified).lower(),
            })
            frame_index += 1
            continue
        position, rotation, angular_errors = solve_pose(
            np.asarray(world_points), convert_rays(np.asarray(pixels)), initial,
            regularize=edge_rectified,
        )
        rmse = float(np.sqrt(np.mean(angular_errors ** 2)))
        valid = rmse <= args.max_angular_rmse_deg
        if valid:
            previous = (position, rotation)
        quaternion = rotation.as_quat()
        rows.append({
            "frame": output_frame, "timestamp": f"{common_time:.6f}",
            "camera_x_m": f"{position[0]:.9f}" if valid else "",
            "camera_y_m": f"{position[1]:.9f}" if valid else "",
            "camera_z_m": f"{position[2]:.9f}" if valid else "",
            "qx": f"{quaternion[0]:.12f}" if valid else "",
            "qy": f"{quaternion[1]:.12f}" if valid else "",
            "qz": f"{quaternion[2]:.12f}" if valid else "",
            "qw": f"{quaternion[3]:.12f}" if valid else "",
            "parent_frame": compiled.get("world_frame", "tag_map"),
            "child_frame": "fisheye1_camera_panorama_axes",
            "tag_map_sha256": compiled["tag_map_sha256"],
            "detected_tag_count": len(set(used_ids)),
            "inlier_count": len(world_points),
            "reprojection_rmse_px": "",
            "angular_rmse_deg": f"{rmse:.6f}",
            "detected_ids": " ".join(map(str, sorted(set(used_ids)))),
            "measurement_source": ("raw_fisheye_unit_bearing_edge_rectified"
                                   if edge_rectified else "raw_fisheye_unit_bearing"),
            "quality_status": "valid" if valid else "angular_rmse_rejected",
            "edge_rectified": str(edge_rectified).lower(),
        })
        frame_index += 1
    capture.release()
    fields = [
        "frame", "timestamp", "camera_x_m", "camera_y_m", "camera_z_m",
        "qx", "qy", "qz", "qw", "parent_frame", "child_frame",
        "tag_map_sha256", "detected_tag_count", "inlier_count",
        "reprojection_rmse_px", "angular_rmse_deg", "detected_ids",
        "measurement_source", "quality_status", "edge_rectified",
    ]
    with (args.output_dir / "pose.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    valid_rows = [row for row in rows if row.get("quality_status") == "valid"]
    summary = {
        "schema_version": "raw-fisheye-world-pose/1.0",
        "input": str(args.video.resolve()),
        "camera_serial": calibration.get("serial"),
        "stream": args.stream,
        "reference_origin": "raw fisheye optical centre; axes rotated into panorama OpenCV convention",
        "common_frames": len(rows),
        "valid_frames": len(valid_rows),
        "valid_ratio": len(valid_rows) / len(rows) if rows else 0.0,
        "angular_rmse_deg": {
            "median": float(np.median([float(row["angular_rmse_deg"]) for row in valid_rows]))
            if valid_rows else None,
            "p95": float(np.percentile([float(row["angular_rmse_deg"]) for row in valid_rows], 95))
            if valid_rows else None,
        },
        "tag_map_sha256": compiled["tag_map_sha256"],
        "measurement_grade_camera_model": True,
        "stitching_used": False,
        "radial_model": args.radial_model,
        "panorama_seam_lut_scale_used": args.radial_model == "stitch",
        "radial_model_hardware_distance_validated": args.radial_model == "stitch",
        "edge_rectification_used": args.edge_rectification,
        "edge_rectified_valid_frames": sum(
            row.get("edge_rectified") == "true" for row in valid_rows
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
