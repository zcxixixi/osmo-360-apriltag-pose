"""Canonical body-to-panorama coordinate bridge used by Insta360 geometry."""

from __future__ import annotations

import numpy as np


BODY_TO_PANORAMA_OPENCV = np.asarray(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


def validate_coordinate_bridge() -> None:
    bridge = BODY_TO_PANORAMA_OPENCV
    if not np.allclose(bridge.T @ bridge, np.eye(3), atol=1e-12):
        raise RuntimeError("body-to-panorama bridge is not orthonormal")
    if not np.isclose(np.linalg.det(bridge), 1.0, atol=1e-12):
        raise RuntimeError("body-to-panorama bridge must be a proper rotation")


validate_coordinate_bridge()
