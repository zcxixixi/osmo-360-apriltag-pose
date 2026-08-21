import uuid
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from dual_camera_alignment_audit import (
    Trajectory,
    at_most,
    estimate_spatial_alignment,
    estimate_time_offset,
    sample_trajectory,
    uuid4_text,
)


def trajectory(times, positions, rotations):
    count = len(times)
    return Trajectory(
        times, positions, rotations, np.arange(count),
        np.full(count, "MEASURED", dtype=object), np.ones(count, dtype=bool),
    )


def motion(times):
    positions = np.column_stack((
        0.5 * np.sin(0.8 * times) + 0.07 * np.sin(3.1 * times),
        0.3 * np.cos(1.1 * times),
        0.2 * np.sin(1.7 * times),
    ))
    rotations = Rotation.from_euler("xyz", np.column_stack((
        0.4 * np.sin(0.7 * times),
        0.3 * np.cos(1.3 * times),
        0.5 * np.sin(0.9 * times) + 0.1 * np.sin(2.8 * times),
    )))
    return positions, rotations


def test_uuid4_is_generated_and_validated():
    generated = uuid.UUID(uuid4_text())
    assert generated.version == 4
    supplied = str(uuid.uuid4())
    assert uuid4_text(supplied) == supplied


def test_twenty_ms_uncertainty_boundary_is_numerically_stable():
    uncertainty = 0.020000000000000018
    assert at_most(uncertainty, 0.020)


def test_time_and_right_to_left_coordinate_alignment():
    left_times = np.arange(0.5, 16.0, 1 / 50)
    right_times = np.arange(0.0, 16.5, 1 / 60)
    offset = 0.37
    left_positions, left_rotations = motion(left_times)
    physical_right_positions, physical_right_rotations = motion(right_times - offset)
    right_to_left = Rotation.from_euler("xyz", [13, -8, 27], degrees=True)
    translation = np.asarray([0.42, -0.18, 0.09])
    right_positions = right_to_left.inv().apply(physical_right_positions - translation)
    right_rotations = right_to_left.inv() * physical_right_rotations
    left = trajectory(left_times, left_positions, left_rotations)
    right = trajectory(right_times, right_positions, right_rotations)

    recovered_offset, correlation, uncertainty, _components, _curve = estimate_time_offset(
        left, right, initial_offset_s=0.3, search_radius_s=0.3
    )
    assert abs(recovered_offset - offset) <= 0.003
    assert correlation > 0.99
    assert uncertainty <= 0.021

    sampled = sample_trajectory(right, left.times + recovered_offset, 0.05)
    recovered_rotation, recovered_translation, inliers = estimate_spatial_alignment(
        left.positions, left.rotations, sampled.positions, sampled.rotations, sampled.valid
    )
    assert inliers.sum() > 700
    np.testing.assert_allclose(recovered_rotation.as_matrix(), right_to_left.as_matrix(), atol=2e-3)
    np.testing.assert_allclose(recovered_translation, translation, atol=2e-3)
