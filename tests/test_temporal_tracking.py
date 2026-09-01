import cv2
import numpy as np

from tools.osmo_360_offline import track_view_detections
from osmo360.pipeline.temporal_apriltag import (
    grayscale_scout_and_refine,
    redetect_rois,
    track_quads_forward_backward,
)
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


def test_raw_grayscale_flow_tracks_all_four_corners_with_audited_status():
    rng = np.random.default_rng(19)
    previous = rng.integers(0, 256, (180, 240), dtype=np.uint8)
    translation = np.asarray([2.5, -1.5], dtype=np.float32)
    current = cv2.warpAffine(
        previous,
        np.asarray([[1.0, 0.0, translation[0]], [0.0, 1.0, translation[1]]]),
        (previous.shape[1], previous.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    quad = np.asarray(
        [[70.0, 60.0], [110.0, 60.0], [110.0, 100.0], [70.0, 100.0]],
        dtype=np.float32,
    )

    tracked, audit = track_quads_forward_backward(previous, current, {3: quad})

    assert audit.attempted_tags == 1
    assert audit.accepted_tags == 1
    np.testing.assert_allclose(
        tracked[3] - quad,
        np.broadcast_to(translation, quad.shape),
        atol=0.2,
    )


def test_low_resolution_gray_scout_refines_only_near_full_resolution_roi():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    marker = cv2.aruco.generateImageMarker(dictionary, 203, 160)
    gray = np.full((480, 640), 255, dtype=np.uint8)
    gray[140:300, 240:400] = marker
    expected = np.asarray(
        [[240, 140], [399, 140], [399, 299], [240, 299]], dtype=np.float32
    )

    scouted = grayscale_scout_and_refine(gray, detector, scale=0.5)
    local = redetect_rois(gray, detector, {203: expected + 4})

    assert set(scouted) == {203}
    assert set(local) == {203}
    np.testing.assert_allclose(scouted[203], expected, atol=1.0)
    np.testing.assert_allclose(local[203], expected, atol=1.0)


def test_rpy_rotation_draws_camera_axes_in_board_coordinates():
    rotation = rpy_to_rotation(np.array([0.0, 0.0, 90.0]))

    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(rotation @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)
