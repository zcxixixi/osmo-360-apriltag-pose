import numpy as np
from scipy.spatial.transform import Rotation

from calibrate_basetag_reciprocal import Transform
from render_single_gripper_motion_demo import (
    CadOpeningModel,
    CameraTrack,
    compose_base_track,
)


def test_camera_to_base_is_composed_in_parent_child_direction():
    camera = CameraTrack(
        time_s=np.array([0.0]),
        position_m=np.array([[1.0, 2.0, 3.0]]),
        rotation=Rotation.identity(1),
        fit_error=np.array([0.1]),
        fit_error_name="angular_rmse_deg",
    )
    camera_base = Transform(np.array([0.1, -0.2, 0.3]), Rotation.identity())

    positions, rotations = compose_base_track(camera, camera_base)

    np.testing.assert_allclose(positions, [[1.1, 1.8, 3.3]])
    np.testing.assert_allclose(rotations.as_quat(), [[0.0, 0.0, 0.0, 1.0]])


def test_cad_opening_width_increases_from_closed_state():
    left = np.array([[0.1, -0.01, 0.0], [0.12, -0.01, 0.0]])
    right = np.array([[0.1, 0.01, 0.0], [0.12, 0.01, 0.0]])
    model = CadOpeningModel(
        left_vertices=left,
        right_vertices=right,
        joint1_origin_m=np.array([0.0, 0.01, 0.0]),
        joint2_origin_m=np.array([0.0, -0.01, 0.0]),
        closed_joint_rotation_deg=0.0,
    )

    assert model.width_m(10.0) > model.width_m(0.0)
    assert model.joint_angles(10.0) == (5.0, -5.0)
