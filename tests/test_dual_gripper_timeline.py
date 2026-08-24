import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from export_dual_gripper_timeline import (
    continuous_quaternions, load_extrinsic, rebase, rebase_shared_world, resample_imu_attitude,
    smooth_quaternions,
)


def test_load_extrinsic_requires_explicit_camera_to_tcp_frames(tmp_path):
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({
        "translation_gripper_origin_in_camera_m": [0, 0, 0],
        "rotation_gripper_to_camera": np.eye(3).tolist(),
    }))
    with pytest.raises(ValueError, match="deprecated camera->base"):
        load_extrinsic(legacy)

    current = tmp_path / "current.json"
    current.write_text(json.dumps({"camera_to_tcp": {
        "parent_frame": "panorama_camera", "child_frame": "gripper_tcp",
        "translation_m": [0.1, 0.2, 0.3],
        "quaternion_xyzw": [0, 0, 0, 1],
    }}))
    translation, rotation = load_extrinsic(current)
    np.testing.assert_allclose(translation, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(rotation.as_matrix(), np.eye(3))


def test_rebase_exact_start_and_preserves_local_motion():
    positions = np.asarray([[1.0, 2.0, 3.0], [1.2, 2.0, 3.0]])
    rotations = Rotation.from_rotvec(np.radians([[0.0, 0.0, 30.0], [0.0, 0.0, 45.0]]))
    target = {"translation_m": [-0.23, -0.01, -0.009], "rotation_rpy_deg": [10.0, 20.0, 40.0]}
    rebased_positions, rebased_rotations = rebase(positions, rotations, target)
    target_rotation = Rotation.from_euler("xyz", target["rotation_rpy_deg"], degrees=True)
    assert np.allclose(rebased_positions[0], target["translation_m"])
    assert (target_rotation.inv() * rebased_rotations[0]).magnitude() < 1e-12
    measured_local = rotations[0].inv().apply(positions[1] - positions[0])
    rebased_local = rebased_rotations[0].inv().apply(rebased_positions[1] - rebased_positions[0])
    assert np.allclose(measured_local, rebased_local)


def test_quaternion_smoothing_keeps_unit_norm_and_sign_continuity():
    angles = np.radians([0, 5, 10, 15, 20])
    quaternions = Rotation.from_rotvec(np.column_stack((np.zeros(5), np.zeros(5), angles))).as_quat()
    quaternions[2:] *= -1
    result = smooth_quaternions(quaternions, radius=2)
    assert np.allclose(np.linalg.norm(result, axis=1), 1.0)
    assert np.all(np.sum(continuous_quaternions(result)[:-1] * continuous_quaternions(result)[1:], axis=1) > 0)


def test_shared_world_rebase_keeps_equal_translation_directions_equal():
    positions_a = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]])
    positions_b = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.1]])
    rotations_a = Rotation.from_euler("z", [[0.0], [0.0]], degrees=True)
    rotations_b = Rotation.from_euler("z", [[180.0], [180.0]], degrees=True)
    target_a = {"translation_m": [-0.2, 0.0, 0.0], "rotation_rpy_deg": [0.0, 0.0, 0.0]}
    target_b = {"translation_m": [0.2, 0.0, 0.0], "rotation_rpy_deg": [0.0, 0.0, 180.0]}
    pa, _, pb, _ = rebase_shared_world(
        positions_a, rotations_a, positions_b, rotations_b, target_a, target_b
    )
    assert np.allclose(pa[1] - pa[0], pb[1] - pb[0])


def test_shared_world_maps_grid_positive_y_to_animation_up():
    positions = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.1, 0.0]])
    rotations = Rotation.from_euler("z", [[0.0], [0.0]], degrees=True)
    target = {"translation_m": [0.0, 0.0, 0.0], "rotation_rpy_deg": [0.0, 0.0, 0.0]}
    left, _, right, _ = rebase_shared_world(
        positions, rotations, positions, rotations, target, target
    )
    assert left[1, 2] > left[0, 2]
    assert right[1, 2] > right[0, 2]


def test_imu_level_changes_heading_without_changing_confirmed_tilt():
    times = np.linspace(0.0, 2.0, 61)
    # Deliberately include large IMU pitch/roll which must not stand the model up.
    imu = Rotation.from_euler("ZYX", np.column_stack((
        np.linspace(0, 80, len(times)),
        70 * np.sin(np.linspace(0, 2 * np.pi, len(times))),
        30 * np.sin(np.linspace(0, 4 * np.pi, len(times))),
    )), degrees=True)
    target = {"translation_m": [0, 0, 0], "rotation_rpy_deg": [-100, 14, 42]}
    result = resample_imu_attitude(times, imu, times, target, "imu-level")
    target_rotation = Rotation.from_euler("xyz", target["rotation_rpy_deg"], degrees=True)
    target_normal = target_rotation.apply([0, 0, 1])
    for rotation in result:
        normal = rotation.apply([0, 0, 1])
        assert np.isclose(normal[2], target_normal[2])
    assert np.degrees((result[0].inv() * result[-1]).magnitude()) > 50
