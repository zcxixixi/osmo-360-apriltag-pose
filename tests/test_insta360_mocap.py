import os
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from tools.evaluate_insta360_mocap import (
    MotiveData,
    PoseSeries,
    calibrate_extrinsics,
    estimate_time_offset,
    parse_motive,
    pose_matrices,
    trajectory_errors,
)
from tools.osmo_360_offline import View, load_tag_map, perspective_intrinsics, solve_view


ROOT = Path(__file__).resolve().parents[1]


def test_non_contiguous_x5_tag_map_geometry():
    tag_map = load_tag_map(ROOT / "mocap-evaluation/config/insta360_x5_tag_map.json")
    assert tag_map.expected_ids == [128, 129, 130, 131]
    np.testing.assert_allclose(tag_map.center(130), [-0.33, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(tag_map.center(131), [-0.11, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(tag_map.center(129), [0.11, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(tag_map.center(128), [0.33, 0.0, 0.0], atol=1e-7)
    for tag_id in tag_map.expected_ids:
        corners = tag_map.corners(tag_id)
        edges = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
        np.testing.assert_allclose(edges, 0.2, atol=1e-7)


def test_real_motive_parser_counts_and_quarantines_identity_swap():
    configured = os.environ.get("MOTIVE_REAL_CSV")
    if not configured:
        return
    path = Path(configured)
    motive = parse_motive(path)
    assert len(motive.frames) == 7320
    assert motive.metadata["raw_valid_frames"] == 6840
    assert motive.quarantined_ranges == [(6374, 6825)]
    assert motive.valid.sum() == 6419
    assert motive.metadata["longest_raw_missing_run_frames"] == 4


def test_non_contiguous_tag_map_pnp_recovers_camera_pose():
    tag_map = load_tag_map(ROOT / "mocap-evaluation/config/insta360_x5_tag_map.json")
    view = View("synthetic", 0.0, 0.0, 90.0)
    camera_matrix = perspective_intrinsics(1200, view.fov)
    rvec = np.zeros(3)
    tvec = np.asarray([0.04, -0.03, 1.25])
    detections = []
    for tag_id in (129, 130, 128, 131):
        corners = tag_map.corners(tag_id).astype(np.float32)
        projected, _ = cv2.projectPoints(corners, rvec, tvec, camera_matrix, None)
        pixels = projected.reshape(-1, 2)
        detections.append({
            "id": tag_id, "object_corners": corners, "corners_px": pixels,
            "object_center": tag_map.center(tag_id).astype(np.float32),
            "center_px": pixels.mean(axis=0), "area_px2": 10000.0,
        })
    pose = solve_view(detections, view, 1200, 2, 0.5, "corners", "ippe")
    assert pose is not None
    np.testing.assert_allclose(pose.xyz, [-0.04, 0.03, -1.25], atol=1e-5)


def test_time_offset_uses_linear_and_angular_motion():
    def samples(times):
        positions = np.column_stack((
            0.35 * np.sin(1.3 * times) + 0.08 * np.sin(3.7 * times),
            0.22 * np.cos(0.9 * times), 0.18 * np.sin(1.9 * times),
        ))
        rotations = Rotation.from_euler("xyz", np.column_stack((
            0.5 * np.sin(1.1 * times), 0.35 * np.cos(1.7 * times),
            0.45 * np.sin(0.8 * times) + 0.2 * np.sin(2.6 * times),
        )))
        return positions, rotations

    offset = -0.37
    mocap_times = np.arange(0.0, 24.0, 1.0 / 120.0)
    mocap_positions, mocap_rotations = samples(mocap_times)
    motive = MotiveData(
        mocap_times, mocap_positions, mocap_rotations.as_quat(),
        np.ones(len(mocap_times), dtype=bool), np.arange(len(mocap_times)), [], {},
    )
    video_times = np.arange(1.0, 22.0, 1.0 / 20.0)
    visual_positions, visual_rotations = samples(video_times + offset)
    visual = PoseSeries(video_times, visual_positions, visual_rotations, np.arange(len(video_times)))
    recovered, correlation, uncertainty, components, _curve = estimate_time_offset(
        visual, motive, initial_offset=-0.4, search_radius=0.2
    )
    assert abs(recovered - offset) <= 0.003
    assert correlation > 0.99
    assert components["linear_correlation"] > 0.99
    assert components["angular_correlation"] > 0.99
    assert uncertainty <= 0.0201


def test_hand_eye_calibration_recovers_held_constant_transforms():
    rng = np.random.default_rng(7)
    count = 80
    times = np.arange(count) / 20.0
    body_rotations = Rotation.from_rotvec(rng.normal(scale=0.45, size=(count, 3)))
    body_positions = rng.normal(scale=0.35, size=(count, 3))
    world_body = pose_matrices(PoseSeries(times, body_positions, body_rotations, np.arange(count)))

    body_camera = np.eye(4)
    body_camera[:3, :3] = Rotation.from_euler("xyz", [18, -9, 33], degrees=True).as_matrix()
    body_camera[:3, 3] = [0.045, -0.021, 0.083]
    world_tag = np.eye(4)
    world_tag[:3, :3] = Rotation.from_euler("xyz", [-12, 7, 41], degrees=True).as_matrix()
    world_tag[:3, 3] = [0.6, -0.4, 1.2]
    tag_camera = np.asarray([np.linalg.inv(world_tag) @ item @ body_camera for item in world_body])

    recovered_body_camera, recovered_world_tag, info = calibrate_extrinsics(world_body, tag_camera)
    position, orientation, _truth, _estimate = trajectory_errors(
        world_body, tag_camera, recovered_body_camera, recovered_world_tag
    )
    assert info["success"]
    assert np.sqrt(np.mean(position**2)) < 1e-6
    assert np.sqrt(np.mean(orientation**2)) < 1e-4
