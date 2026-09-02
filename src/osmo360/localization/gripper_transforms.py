from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from osmo360.calibration.calibrate_basetag_reciprocal import Transform


def camera_to_base(hardware: dict, role: str) -> Transform:
    robot = hardware["robots"][role]
    tcp = robot["camera_to_eef_reference"]
    camera_tcp = Transform(
        np.asarray(tcp["translation_m"], dtype=float),
        Rotation.from_quat(tcp["quaternion_xyzw"]),
    )
    base_tcp = Transform(
        np.asarray(hardware["eef_reference"]["base_to_tcp_translation_m"], dtype=float),
        Rotation.identity(),
    )
    return camera_tcp.compose(base_tcp.inverse())
