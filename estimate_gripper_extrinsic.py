#!/usr/bin/env python3
"""Estimate the rigid panorama-camera -> gripper-base transform from Tag ID 2.

The four tag corners are measured in raw Osmo stream 1. Factory fisheye
calibration converts those pixels to panorama-frame bearing rays; a bearing
PnP fit then recovers the physical 24 mm tag pose. CAD's tag-to-base transform
finishes the camera-to-gripper chain used by the trajectory renderer.

The tag is close to 90 degrees from the equirectangular forward direction.  It
is therefore invalid to divide the panorama rays by their panorama ``z``
component and feed them directly to a pinhole PnP solver: some corners can be
behind that arbitrary virtual camera.  We first construct a tangent pinhole
view centred on the tag, run IPPE there, and rotate every solution back into
the panorama frame.  This also keeps the planar two-solution ambiguity
explicit instead of silently converging to whichever branch ITERATIVE finds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from coordinate_frames import DJI_BODY_TO_PANORAMA_OPENCV

# Backwards-compatible public name. The value itself is defined once and is
# shared with the visual/IMU solver so mount calibration cannot silently use a
# different panorama frame.
BODY_TO_PANORAMA_OPENCV = DJI_BODY_TO_PANORAMA_OPENCV


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
    parser.add_argument(
        "--tcp-definition",
        type=Path,
        default=Path(__file__).resolve().parent / "config/gripper_tcp_cad.json",
        help="explicit base_link->TCP CAD definition",
    )
    parser.add_argument(
        "--tag-corner-quarter-turns",
        type=int,
        default=0,
        choices=range(4),
        help=(
            "cyclic orientation of this physical mount's printed Tag relative "
            "to the CAD tag frame; calibrated once per mount revision"
        ),
    )
    parser.add_argument(
        "--ippe-branch",
        type=int,
        default=0,
        choices=(0, 1),
        help="explicit planar pose branch selected during mount calibration",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def tangent_view_basis(rays_panorama: np.ndarray) -> np.ndarray:
    """Return a right-handed tangent-view basis expressed in panorama axes."""
    rays = np.asarray(rays_panorama, dtype=np.float64).reshape(-1, 3)
    forward = rays.mean(axis=0)
    forward /= np.linalg.norm(forward)
    panorama_down = np.array([0.0, 1.0, 0.0])
    down = panorama_down - forward * float(panorama_down @ forward)
    if np.linalg.norm(down) < 1e-6:
        fallback = np.array([1.0, 0.0, 0.0])
        down = fallback - forward * float(fallback @ forward)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    basis = np.column_stack((right, down, forward))
    if np.linalg.det(basis) < 0.999999:
        raise ValueError("failed to construct a proper tangent-view rotation")
    return basis


def solve_bearing_ippe(
    tag_points: np.ndarray,
    rays_panorama: np.ndarray,
) -> list[dict[str, np.ndarray | float]]:
    """Solve both planar Tag poses from unit rays in panorama coordinates."""
    rays = np.asarray(rays_panorama, dtype=np.float64).reshape(-1, 3)
    basis = tangent_view_basis(rays)
    rays_view = rays @ basis
    if np.min(rays_view[:, 2]) <= 0.5:
        raise ValueError("tag rays do not form a valid local tangent view")
    normalized_pixels = rays_view[:, :2] / rays_view[:, 2:3]
    ok, rvecs, tvecs, _errors = cv2.solvePnPGeneric(
        np.asarray(tag_points, dtype=np.float64),
        normalized_pixels,
        np.eye(3),
        None,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not ok or not rvecs:
        raise RuntimeError("IPPE could not solve the mount Tag pose")

    results: list[dict[str, np.ndarray | float]] = []
    for branch, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        rotation_view_tag = cv2.Rodrigues(rvec)[0]
        rotation_panorama_tag = basis @ rotation_view_tag
        translation_panorama_tag = basis @ np.asarray(tvec).reshape(3)

        def residual(parameters: np.ndarray) -> np.ndarray:
            rotation = Rotation.from_rotvec(parameters[:3]).as_matrix()
            predicted = tag_points @ rotation.T + parameters[3:]
            predicted /= np.linalg.norm(predicted, axis=1, keepdims=True)
            return (predicted - rays).ravel()

        initial = np.r_[
            Rotation.from_matrix(rotation_panorama_tag).as_rotvec(),
            translation_panorama_tag,
        ]
        fit = least_squares(
            residual,
            initial,
            max_nfev=10000,
            xtol=1e-14,
            ftol=1e-14,
            gtol=1e-14,
        )
        rotation = Rotation.from_rotvec(fit.x[:3]).as_matrix()
        angular_errors = np.degrees(
            np.arccos(
                np.clip(
                    1.0
                    - np.sum(residual(fit.x).reshape(-1, 3) ** 2, axis=1) / 2.0,
                    -1.0,
                    1.0,
                )
            )
        )
        results.append(
            {
                "branch": float(branch),
                "rotation_tag_to_panorama": rotation,
                "translation_tag_origin_in_panorama_m": fit.x[3:].copy(),
                "bearing_rmse": float(np.sqrt(np.mean(residual(fit.x) ** 2))),
                "angular_error_deg": angular_errors,
                "angular_rmse_deg": float(np.sqrt(np.mean(angular_errors**2))),
                "local_minimum_forward_component": float(np.min(rays_view[:, 2])),
            }
        )
    return results


def compose_camera_base_tcp(
    transform_camera_base: np.ndarray,
    tcp_translation_in_base_m: np.ndarray,
    tcp_quaternion_xyzw: np.ndarray,
) -> np.ndarray:
    """Compose ``T_camera_tcp = T_camera_base @ T_base_tcp`` explicitly."""
    camera_base = np.asarray(transform_camera_base, dtype=np.float64).reshape(4, 4)
    translation = np.asarray(tcp_translation_in_base_m, dtype=np.float64).reshape(3)
    quaternion = np.asarray(tcp_quaternion_xyzw, dtype=np.float64).reshape(4)
    transform_base_tcp = np.eye(4)
    transform_base_tcp[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    transform_base_tcp[:3, 3] = translation
    return camera_base @ transform_base_tcp


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
    rays = direction_body @ BODY_TO_PANORAMA_OPENCV.T
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    rays = np.roll(rays, -args.tag_corner_quarter_turns, axis=0)

    half = args.tag_size_m / 2.0
    tag_points = np.array(
        [[-half, -half, 0.0], [half, -half, 0.0],
         [half, half, 0.0], [-half, half, 0.0]],
        dtype=np.float64,
    )
    candidates = solve_bearing_ippe(tag_points, rays)
    if args.ippe_branch >= len(candidates):
        raise RuntimeError(
            f"requested IPPE branch {args.ippe_branch}, only {len(candidates)} available"
        )
    selected = candidates[args.ippe_branch]
    rotation_camera_tag = np.asarray(
        selected["rotation_tag_to_panorama"], dtype=np.float64
    )
    translation_camera_tag = np.asarray(
        selected["translation_tag_origin_in_panorama_m"], dtype=np.float64
    )

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
    transform_camera_base = transform_camera_tag @ np.linalg.inv(transform_gripper_tag)
    tcp_definition = json.loads(args.tcp_definition.read_text(encoding="utf-8"))
    tcp_in_base = tcp_definition["tcp_in_base"]
    transform_camera_tcp = compose_camera_base_tcp(
        transform_camera_base,
        np.asarray(tcp_in_base["translation_m"], dtype=np.float64),
        np.asarray(tcp_in_base["quaternion_xyzw"], dtype=np.float64),
    )

    def rigid_payload(parent: str, child: str, transform: np.ndarray) -> dict:
        return {
            "parent_frame": parent,
            "child_frame": child,
            "translation_m": transform[:3, 3].tolist(),
            "quaternion_xyzw": Rotation.from_matrix(
                transform[:3, :3]
            ).as_quat().tolist(),
        }

    output = {
        "schema_version": "gripper-mount-calibration/2.0",
        "camera_to_base": rigid_payload(
            "panorama_camera", "base_link", transform_camera_base
        ),
        "base_to_tcp": rigid_payload(
            "base_link", "gripper_tcp",
            np.linalg.inv(transform_camera_base) @ transform_camera_tcp,
        ),
        "camera_to_tcp": rigid_payload(
            "panorama_camera", "gripper_tcp", transform_camera_tcp
        ),
        # Legacy aliases describe the base origin, never the TCP. New code must
        # consume the explicit framed transforms above.
        "rotation_gripper_to_camera": transform_camera_base[:3, :3].tolist(),
        "translation_gripper_origin_in_camera_m": transform_camera_base[:3, 3].tolist(),
        "legacy_fields_status": "DEPRECATED_BASE_ORIGIN_NOT_TCP",
        "tag_pose_in_camera": {
            "rotation_tag_to_camera": rotation_camera_tag.tolist(),
            "translation_tag_origin_in_camera_m": translation_camera_tag.tolist(),
        },
        "fit": {
            "bearing_rmse": selected["bearing_rmse"],
            "angular_error_deg": np.asarray(
                selected["angular_error_deg"], dtype=float
            ).tolist(),
            "angular_rmse_deg": selected["angular_rmse_deg"],
            "local_minimum_forward_component": selected[
                "local_minimum_forward_component"
            ],
        },
        "pose_branch": {
            "solver": "tangent-view IPPE + unit-bearing refinement",
            "selected": args.ippe_branch,
            "tag_corner_quarter_turns": args.tag_corner_quarter_turns,
            "candidate_angular_rmse_deg": [
                candidate["angular_rmse_deg"] for candidate in candidates
            ],
        },
        "source": {
            "factory_calibration": str(args.calibration.resolve()),
            "fixed_tag_corners": str(args.fixed_tag_corners.resolve()),
            "tcp_definition": str(args.tcp_definition.resolve()),
            "stream": args.stream,
            "tag_size_m": args.tag_size_m,
        },
        "accuracy_note": (
            "Rigid transform uses factory lens intrinsics/rotation and a manually fixed tag quad; "
            "lens-centre translation is unavailable, so this is a calibrated visualization transform, not metrology truth. "
            "Tag corner orientation and IPPE branch are immutable mount-revision calibration fields."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
