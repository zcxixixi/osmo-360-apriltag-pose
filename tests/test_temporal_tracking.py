import cv2
import numpy as np

from tools.osmo_360_offline import track_view_detections
from osmo360.visualization.render_trajectory_overlay_video import rpy_to_rotation


def test_optical_flow_tracks_tag_corners_under_small_translation():
    rng = np.random.default_rng(7)
    previous = rng.integers(0, 256, (180, 240, 3), dtype=np.uint8)
    translation = np.array([3.0, 2.0], dtype=np.float32)
    current = cv2.warpAffine(
        previous,
        np.array([[1.0, 0.0, translation[0]], [0.0, 1.0, translation[1]]]),
        (previous.shape[1], previous.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    corners = np.array(
        [[70.0, 60.0], [110.0, 60.0], [110.0, 100.0], [70.0, 100.0]],
        dtype=np.float32,
    )
    detections = [
        {
            "id": 3,
            "corners_px": corners,
            "center_px": corners.mean(axis=0),
            "object_center": np.zeros(3, dtype=np.float32),
            "object_corners": np.zeros((4, 3), dtype=np.float32),
            "area_px2": abs(float(cv2.contourArea(corners))),
        }
    ]

    tracked = track_view_detections(previous, current, detections)

    assert len(tracked) == 1
    measured_translation = tracked[0]["corners_px"] - corners
    np.testing.assert_allclose(
        measured_translation,
        np.broadcast_to(translation, measured_translation.shape),
        atol=0.15,
    )


def test_rpy_rotation_draws_camera_axes_in_board_coordinates():
    rotation = rpy_to_rotation(np.array([0.0, 0.0, 90.0]))

    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(rotation @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)
