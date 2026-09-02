from __future__ import annotations

import cv2
import numpy as np

from osmo360.gripper_markers import (
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
