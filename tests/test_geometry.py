import cv2
import numpy as np
import pytest

from osmo_360_offline import (
    _BODY_TO_PANORAMA,
    View,
    pose_view_to_panorama,
    propagate_view_with_imu,
    quaternion_to_rotation,
    view_to_panorama_rotation,
)
from osmo_apriltag_demo import Grid


def test_kalibr_grid_is_column_major():
    grid = Grid(rows=6, cols=6, tag_size=0.088, spacing_ratio=0.30)
    assert grid.center(1)[0] == pytest.approx(grid.center(0)[0])
    assert grid.center(1)[1] < grid.center(0)[1]
    assert grid.center(6)[0] > grid.center(0)[0]
    assert grid.center(6)[1] == pytest.approx(grid.center(0)[1])
    assert grid.center(35) is not None
    assert grid.center(36) is None


@pytest.mark.parametrize("yaw,pitch", [(0, 0), (45, 0), (-90, 0), (0, 45), (135, -30)])
def test_view_rotation_is_proper(yaw, pitch):
    r = view_to_panorama_rotation(yaw, pitch)
    np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1.0)


def test_forward_ray_matches_view_heading():
    np.testing.assert_allclose(
        view_to_panorama_rotation(90, 0) @ [0, 0, 1], [1, 0, 0], atol=1e-12
    )


def test_imu_view_propagation_preserves_world_facing_basis():
    previous = np.array([1.0, 0.0, 0.0, 0.0])
    angle = np.radians(37.0)
    current = np.array([np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0])
    view = View("tracked", 42.0, -18.0, 85.0, 23.0)
    propagated = propagate_view_with_imu(view, previous, current)
    previous_world_basis = (
        quaternion_to_rotation(previous)
        @ _BODY_TO_PANORAMA.T
        @ view_to_panorama_rotation(view.yaw, view.pitch, view.roll)
    )
    current_world_basis = (
        quaternion_to_rotation(current)
        @ _BODY_TO_PANORAMA.T
        @ view_to_panorama_rotation(
            propagated.yaw, propagated.pitch, propagated.roll
        )
    )
    np.testing.assert_allclose(current_world_basis, previous_world_basis, atol=1e-9)
    np.testing.assert_allclose(
        view_to_panorama_rotation(0, 90) @ [0, 0, 1], [0, -1, 0], atol=1e-12
    )


def test_same_camera_center_from_different_perspective_views():
    camera_xyz_board = np.array([0.22, -0.14, 1.35])
    pano_to_board, _ = cv2.Rodrigues(np.array([0.08, -0.20, 0.04]))
    board_to_pano = pano_to_board.T
    t_pano = -board_to_pano @ camera_xyz_board.reshape(3, 1)
    recovered = []
    for view in (View("front", 0, 0), View("right", 70, -10), View("left", -55, 20)):
        v2p = view_to_panorama_rotation(view.yaw, view.pitch)
        board_to_view = v2p.T @ board_to_pano
        t_view = v2p.T @ t_pano
        rvec, _ = cv2.Rodrigues(board_to_view)
        xyz, rotation = pose_view_to_panorama(rvec, t_view, view)
        recovered.append(xyz)
        np.testing.assert_allclose(rotation, pano_to_board, atol=1e-9)
    for xyz in recovered:
        np.testing.assert_allclose(xyz, camera_xyz_board, atol=1e-9)
