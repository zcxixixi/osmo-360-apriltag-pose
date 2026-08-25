from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from estimate_gripper_extrinsic import (
    BODY_TO_PANORAMA_OPENCV,
    compose_camera_base_tcp,
    compose_camera_tag_to_base,
    select_ippe_candidate,
    solve_bearing_ippe,
    tangent_view_basis,
)


def test_body_to_panorama_matches_visual_solver_axis_conversion():
    assert np.allclose(BODY_TO_PANORAMA_OPENCV @ BODY_TO_PANORAMA_OPENCV.T, np.eye(3))
    assert np.isclose(np.linalg.det(BODY_TO_PANORAMA_OPENCV), 1.0)
    # PanoForge's +90 degree longitude offset puts body +X at panorama centre.
    assert np.allclose(BODY_TO_PANORAMA_OPENCV @ [1, 0, 0], [0, 0, 1])
    assert np.allclose(BODY_TO_PANORAMA_OPENCV @ [0, 1, 0], [-1, 0, 0])
    assert np.allclose(BODY_TO_PANORAMA_OPENCV @ [0, 0, 1], [0, -1, 0])


def test_camera_base_tcp_chain_does_not_confuse_base_origin_with_tcp():
    camera_base = np.eye(4)
    camera_base[:3, :3] = Rotation.from_euler(
        "xyz", [12.0, -23.0, 41.0], degrees=True
    ).as_matrix()
    camera_base[:3, 3] = [0.04, -0.08, 0.11]
    base_tcp_translation = np.array([0.1356, 0.0, 0.0101])
    base_tcp_quaternion = Rotation.from_euler(
        "z", 7.0, degrees=True
    ).as_quat()
    camera_tcp = compose_camera_base_tcp(
        camera_base, base_tcp_translation, base_tcp_quaternion
    )
    np.testing.assert_allclose(
        camera_tcp[:3, 3],
        camera_base[:3, 3] + camera_base[:3, :3] @ base_tcp_translation,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        camera_tcp[:3, :3],
        camera_base[:3, :3] @ Rotation.from_quat(base_tcp_quaternion).as_matrix(),
        atol=1e-12,
    )


def test_tangent_view_handles_tag_crossing_panorama_forward_horizon():
    rays = np.asarray(
        [
            [0.98, -0.10, -0.17],
            [0.99, -0.08, 0.08],
            [0.97, 0.14, 0.13],
            [0.96, 0.16, -0.20],
        ],
        dtype=float,
    )
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    # Direct x/z projection is invalid because the arbitrary panorama-forward
    # plane has corners on both sides. The tangent view must put all in front.
    assert np.min(rays[:, 2]) < 0 < np.max(rays[:, 2])
    basis = tangent_view_basis(rays)
    local = rays @ basis
    assert np.min(local[:, 2]) > 0.9
    assert np.isclose(np.linalg.det(basis), 1.0)


def test_bearing_ippe_recovers_metric_pose_near_panorama_horizon():
    half = 0.012
    tag = np.asarray(
        [[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]],
        dtype=float,
    )
    expected_rotation = Rotation.from_euler("xyz", [18, -23, 31], degrees=True).as_matrix()
    expected_translation = np.asarray([0.085, -0.006, 0.004], dtype=float)
    points = tag @ expected_rotation.T + expected_translation
    rays = points / np.linalg.norm(points, axis=1, keepdims=True)
    assert np.min(rays[:, 2]) < 0 < np.max(rays[:, 2])

    candidates = solve_bearing_ippe(tag, rays)
    errors = []
    for candidate in candidates:
        rotation_error = Rotation.from_matrix(
            expected_rotation.T @ candidate["rotation_tag_to_panorama"]
        ).magnitude()
        translation_error = np.linalg.norm(
            expected_translation - candidate["translation_tag_origin_in_panorama_m"]
        )
        errors.append((rotation_error, translation_error))
    assert min(error[0] for error in errors) < np.deg2rad(0.05)
    assert min(error[1] for error in errors) < 1e-4


def test_camera_base_uses_authoritative_hardware_base_to_tag():
    camera_tag_rotation = Rotation.from_euler(
        "xyz", [17.0, -31.0, 8.0], degrees=True
    )
    camera_tag_translation = np.array([-0.032, 0.059, 0.007])
    base_tag_rotation = Rotation.from_euler(
        "xyz", [3.0, 5.0, -7.0], degrees=True
    )
    base_tag_translation = np.array([0.02625, 0.0, 0.0196])
    base_to_tag = {
        "parent_frame": "base_link",
        "child_frame": "basetag",
        "translation_m": base_tag_translation.tolist(),
        "quaternion_xyzw": base_tag_rotation.as_quat().tolist(),
    }
    actual = compose_camera_tag_to_base(
        camera_tag_rotation.as_matrix(), camera_tag_translation, base_to_tag
    )
    camera_tag = np.eye(4)
    camera_tag[:3, :3] = camera_tag_rotation.as_matrix()
    camera_tag[:3, 3] = camera_tag_translation
    expected_base_tag = np.eye(4)
    expected_base_tag[:3, :3] = base_tag_rotation.as_matrix()
    expected_base_tag[:3, 3] = base_tag_translation
    np.testing.assert_allclose(actual, camera_tag @ np.linalg.inv(expected_base_tag))


def test_physical_mount_constraints_select_camera_behind_tag():
    candidates = [
        {
            "branch": 0.0,
            "raw_camera_origin_in_tag_m": np.array([-0.010, 0.012, -0.067]),
            "raw_angular_rmse_deg": 0.30,
            "angular_rmse_deg": 0.20,
        },
        {
            "branch": 1.0,
            "raw_camera_origin_in_tag_m": np.array([0.010, -0.012, -0.067]),
            "raw_angular_rmse_deg": 0.25,
            "angular_rmse_deg": 0.18,
        },
    ]
    selected = select_ippe_candidate(
        candidates,
        {
            "camera_origin_in_tag_m": {
                "x_max": 0.0,
                "z_min": -0.10,
                "z_max": -0.04,
            }
        },
    )
    assert int(selected["branch"]) == 0


def test_explicit_wrong_ippe_branch_is_rejected_by_hardware_constraints():
    candidates = [
        {
            "branch": 1.0,
            "raw_camera_origin_in_tag_m": np.array([0.010, 0.0, -0.067]),
            "raw_angular_rmse_deg": 0.1,
            "angular_rmse_deg": 0.1,
        }
    ]
    try:
        select_ippe_candidate(
            candidates,
            {"camera_origin_in_tag_m": {"x_max": 0.0}},
            requested_branch="1",
        )
    except ValueError as error:
        assert "hardware camera-in-tag constraints" in str(error)
    else:
        raise AssertionError("physically impossible explicit branch was accepted")
