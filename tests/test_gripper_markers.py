from __future__ import annotations

import cv2
import numpy as np
import pytest

from osmo360.gripper_markers import (
    detect_bgr_gripper_markers,
    detect_yuv420_gripper_markers,
    detect_yuv420_gripper_triads,
    included_angle_deg,
)


def test_fixed_roi_yuv420_detector_finds_two_physical_marker_triads() -> None:
    luma = np.full((1920, 1920), 35, dtype=np.uint8)
    chroma_u = np.full((960, 960), 128, dtype=np.uint8)
    chroma_v = np.full((960, 960), 128, dtype=np.uint8)
    left_expected = np.asarray([(800, 1200), (750, 1350), (700, 1500)])
    right_expected = np.asarray([(1120, 1200), (1170, 1350), (1220, 1500)])
    for x, y in np.vstack((left_expected, right_expected)):
        cv2.circle(luma, (int(x), int(y)), 18, 165, -1)
        cv2.circle(chroma_u, (int(x // 2), int(y // 2)), 9, 60, -1)
        cv2.circle(chroma_v, (int(x // 2), int(y // 2)), 9, 135, -1)

    left, right = detect_yuv420_gripper_triads(luma, chroma_u, chroma_v)

    assert left is not None and right is not None
    assert np.allclose(left, left_expected, atol=3)
    assert np.allclose(right, right_expected, atol=3)
    assert 30 < included_angle_deg(left, right) < 40


def test_dual_colour_detector_finds_black_pair_on_yellow_gripper() -> None:
    image = np.full((1920, 1920, 3), 25, dtype=np.uint8)
    yellow = (0, 220, 245)
    cv2.rectangle(image, (780, 1160), (930, 1480), yellow, -1)
    cv2.rectangle(image, (990, 1160), (1140, 1480), yellow, -1)
    cv2.circle(image, (870, 1260), 9, (10, 10, 10), -1)
    cv2.circle(image, (1050, 1260), 9, (10, 10, 10), -1)

    bgr_markers = detect_bgr_gripper_markers(image)
    i420 = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420)
    flat_chroma = i420[1920:].reshape(-1)
    plane_size = 960 * 960
    yuv_markers = detect_yuv420_gripper_markers(
        i420[:1920],
        flat_chroma[:plane_size].reshape(960, 960),
        flat_chroma[plane_size:].reshape(960, 960),
        full_range=False,
    )

    for markers in (bgr_markers, yuv_markers):
        assert markers.black_left is not None
        assert markers.black_right is not None
        assert np.allclose(markers.black_left, (870, 1260), atol=4)
        assert np.allclose(markers.black_right, (1050, 1260), atol=4)
        assert markers.black_pair_gap_px == pytest.approx(180, abs=5)
