"""Canonical coordinate-frame bridges shared by pose and mount calibration.

PanoForge generates the equirectangular panorama from DJI body directions
``X=right, Y=forward, Z=up`` and applies a +90 degree longitude offset. The
centre pixel therefore looks along body ``+X``. The local perspective views
used by :mod:`osmo_360_offline` follow OpenCV axes ``x=right, y=down,
z=forward``. Reading the same projection one pixel to the right and upward
gives the remaining two basis directions::

    DJI body +X -> panorama +Z  (forward)
    DJI body +Y -> panorama -X  (left)
    DJI body +Z -> panorama -Y  (up)

This is a proper SO(3) basis change (determinant +1). Keep it in one module:
using its negative transpose reflects the mount and makes a physically
horizontal gripper appear vertical.
"""

from __future__ import annotations

import numpy as np


DJI_BODY_TO_PANORAMA_OPENCV = np.array(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)

# The four-MP4 tracker solves poses in a stream-0-centred OpenCV frame:
# x=image-right, y=image-down, z=the back-lens optical direction.  Published
# hand-camera poses use FLU, with +X deliberately bound to that back direction.
# Columns are the hand-FLU basis vectors expressed in the internal source frame.
X5_STREAM0_OPENCV_FROM_HAND_CAMERA_FLU = np.array(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


def validate_coordinate_bridges() -> None:
    """Fail fast if the canonical bridge is edited into a reflection."""
    bridge = DJI_BODY_TO_PANORAMA_OPENCV
    if not np.allclose(bridge.T @ bridge, np.eye(3), atol=1e-12):
        raise RuntimeError("DJI body to panorama bridge is not orthonormal")
    if not np.isclose(np.linalg.det(bridge), 1.0, atol=1e-12):
        raise RuntimeError("DJI body to panorama bridge must be a proper rotation")
    hand_bridge = X5_STREAM0_OPENCV_FROM_HAND_CAMERA_FLU
    if not np.allclose(hand_bridge.T @ hand_bridge, np.eye(3), atol=1e-12):
        raise RuntimeError("X5 stream-0 to hand-FLU bridge is not orthonormal")
    if not np.isclose(np.linalg.det(hand_bridge), 1.0, atol=1e-12):
        raise RuntimeError("X5 stream-0 to hand-FLU bridge must be a proper rotation")


validate_coordinate_bridges()
