#!/usr/bin/env python3
"""Estimate the rigid panorama-camera -> gripper-base transform from Tag ID 2.

The four tag corners are measured in raw Osmo stream 1. Factory fisheye
calibration converts those pixels to panorama-frame bearing rays; a bearing
PnP fit then recovers the physical 24 mm tag pose. CAD's tag-to-base transform
finishes the camera-to-gripper chain used by the trajectory renderer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--fixed-tag-corners", type=Path, required=True)
    parser.add_argument("--panoforge-root", type=Path, required=True)
    parser.add_argument("--source-width", type=int, default=3000)
    parser.add_argument("--source-height", type=int, default=3000)
    parser.add_argument("--corner-width", type=int, default=1500)
    parser.add_argument("--stream", type=int, default=1)
    parser.add_argument("--tag-size-m", type=float, default=0.024)
    parser.add_argument("--tag-center-base-x-m", type=float, default=-0.001)
    parser.add_argument("--tag-plane-base-z-m", type=float, default=0.004)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.panoforge_root))
    from app.core.maps import _quat_to_rot, _radial_model, scale_calibration_to_source

    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    calibration = scale_calibration_to_source(
        calibration, args.source_width, args.source_height
    )
    lens = calibration["lenses"][args.stream]
    corners = np.asarray(
        json.loads(args.fixed_tag_corners.read_text(encoding="utf-8")),
        dtype=np.float64,
    )
    corners *= args.source_width / args.corner_width
    centre = np.array([lens["cx"], lens["cy"]], dtype=np.float64)
    radius = np.linalg.norm(corners - centre, axis=1)
    radial_model = _radial_model(lens)
    low = np.zeros(4)
    high = np.full(4, np.radians(96.0))
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
    # Calibration quaternion maps panorama body -> lens. Panorama pose code
    # uses OpenCV axes: X right, Y down, Z forward.
    direction_body = direction_lens @ _quat_to_rot(lens["extrinsic_quat"])
    body_to_panorama = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )
    rays = direction_body @ body_to_panorama.T
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)

    half = args.tag_size_m / 2.0
    tag_points = np.array(
        [[-half, -half, 0.0], [half, -half, 0.0],
         [half, half, 0.0], [-half, half, 0.0]],
        dtype=np.float64,
    )
    normalized_pixels = rays[:, :2] / rays[:, 2:3]
    _, rvec, tvec = cv2.solvePnP(
        tag_points, normalized_pixels, np.eye(3), None,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    def residual(parameters: np.ndarray) -> np.ndarray:
        rotation = cv2.Rodrigues(parameters[:3])[0]
        predicted = tag_points @ rotation.T + parameters[3:]
        predicted /= np.linalg.norm(predicted, axis=1, keepdims=True)
        return (predicted - rays).ravel()

    fit = least_squares(
        residual, np.r_[rvec.ravel(), tvec.ravel()],
        max_nfev=10000, xtol=1e-14, ftol=1e-14, gtol=1e-14,
    )
    rotation_camera_tag = cv2.Rodrigues(fit.x[:3])[0]
    translation_camera_tag = fit.x[3:]

    # CAD convention supplied with the angle solver:
    # base_x = -tag_y - 1 mm; base_y = -tag_x. Tag z is chosen so the 3-D
    # mapping remains a proper rotation and the tag lies 4 mm above base zero.
    rotation_gripper_tag = np.array(
        [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]
    )
    translation_gripper_tag = np.array(
        [args.tag_center_base_x_m, 0.0, args.tag_plane_base_z_m]
    )
    transform_camera_tag = np.eye(4)
    transform_camera_tag[:3, :3] = rotation_camera_tag
    transform_camera_tag[:3, 3] = translation_camera_tag
    transform_gripper_tag = np.eye(4)
    transform_gripper_tag[:3, :3] = rotation_gripper_tag
    transform_gripper_tag[:3, 3] = translation_gripper_tag
    transform_camera_gripper = transform_camera_tag @ np.linalg.inv(transform_gripper_tag)

    angular_errors = np.degrees(
        np.arccos(np.clip(1.0 - np.sum(residual(fit.x).reshape(-1, 3) ** 2, axis=1) / 2.0, -1.0, 1.0))
    )
    output = {
        "rotation_gripper_to_camera": transform_camera_gripper[:3, :3].tolist(),
        "translation_gripper_origin_in_camera_m": transform_camera_gripper[:3, 3].tolist(),
        "tag_pose_in_camera": {
            "rotation_tag_to_camera": rotation_camera_tag.tolist(),
            "translation_tag_origin_in_camera_m": translation_camera_tag.tolist(),
        },
        "fit": {
            "bearing_rmse": float(np.sqrt(np.mean(residual(fit.x) ** 2))),
            "angular_error_deg": angular_errors.tolist(),
            "angular_rmse_deg": float(np.sqrt(np.mean(angular_errors ** 2))),
        },
        "source": {
            "factory_calibration": str(args.calibration.resolve()),
            "fixed_tag_corners": str(args.fixed_tag_corners.resolve()),
            "stream": args.stream,
            "tag_size_m": args.tag_size_m,
        },
        "accuracy_note": (
            "Rigid transform uses factory lens intrinsics/rotation and a manually fixed tag quad; "
            "lens-centre translation is unavailable, so this is a calibrated visualization transform, not metrology truth."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
