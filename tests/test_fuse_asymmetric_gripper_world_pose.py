from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fuse_asymmetric_gripper_world_pose import resample_bounded_base


def test_bounded_base_resampling_keeps_measured_rotation_and_rejects_long_gaps() -> None:
    times = np.asarray([0.0, 0.1, 1.0, 1.1])
    positions = np.column_stack([times, np.zeros((len(times), 2))])
    rotations = Rotation.from_euler("z", (times * 40.0)[:, None], degrees=True)
    query = np.asarray([-0.1, 0.05, 0.5, 1.05, 1.2])

    sampled_position, sampled_rotation, trusted = resample_bounded_base(
        times, positions, rotations, query, maximum_gap_s=0.25,
    )

    assert trusted.tolist() == [False, True, False, True, False]
    np.testing.assert_allclose(sampled_position[:, 0], [0.0, 0.05, 0.5, 1.05, 1.1])
    np.testing.assert_allclose(
        sampled_rotation.as_euler("zyx", degrees=True)[:, 0],
        [0.0, 2.0, 20.0, 42.0, 44.0],
        atol=1e-8,
    )


def test_bounded_base_resampling_rejects_bad_timestamps() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        resample_bounded_base(
            np.asarray([0.0, 0.0]),
            np.zeros((2, 3)),
            Rotation.identity(2),
            np.asarray([0.0]),
            maximum_gap_s=0.25,
        )
