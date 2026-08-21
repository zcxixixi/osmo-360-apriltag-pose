import numpy as np

from osmo_apriltag_demo import Grid
from osmo_360_offline import Pose, View, choose_scout_base


def test_wall_panel_row_major_ids_advance_left_to_right():
    grid = Grid(8, 8, 0.084, 24 / 84, 0, "row-major")
    center_0 = grid.center(0)
    center_1 = grid.center(1)
    center_8 = grid.center(8)
    assert center_0 is not None and center_1 is not None and center_8 is not None
    np.testing.assert_allclose(center_1 - center_0, [0.108, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(center_8 - center_0, [0.0, -0.108, 0.0], atol=1e-6)


def test_kalibr_column_major_remains_default():
    grid = Grid(8, 8, 0.084, 24 / 84)
    center_0 = grid.center(0)
    center_1 = grid.center(1)
    assert center_0 is not None and center_1 is not None
    np.testing.assert_allclose(center_1 - center_0, [0.0, -0.108, 0.0], atol=1e-6)


def test_scout_prefers_valid_pose_over_more_unusable_tags():
    weak_but_valid = Pose(
        np.zeros(3), np.eye(3), (0.0, 0.0, 0.0), 7, 1.64, "h-090", [],
    )
    valid_view = View("h-090", -90.0, 0.0)
    invalid_view = View("h-060", -60.0, 0.0)
    selected = choose_scout_base([
        (invalid_view, [{}] * 33, None),
        (valid_view, [{}] * 7, weak_but_valid),
    ])
    assert selected[0] == valid_view
    assert selected[2] is weak_but_valid
