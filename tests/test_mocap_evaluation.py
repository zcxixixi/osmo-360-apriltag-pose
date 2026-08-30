import numpy as np
from scipy.spatial.transform import Rotation

from tools.evaluate_mocap_ground_truth import Trajectory, rigid_alignment, _relative_errors


def test_rigid_alignment_recovers_known_transform():
    source = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=float)
    rotation = Rotation.from_euler("z", 37, degrees=True).as_matrix()
    translation = np.array([0.4, -0.2, 1.3])
    target = (rotation @ source.T).T + translation
    recovered_rotation, recovered_translation = rigid_alignment(source, target)
    np.testing.assert_allclose(recovered_rotation, rotation, atol=1e-12)
    np.testing.assert_allclose(recovered_translation, translation, atol=1e-12)


def test_identical_trajectories_have_zero_rpe():
    times = np.arange(6, dtype=float)
    positions = np.column_stack((times, times**2 / 10, np.zeros_like(times)))
    rotations = Rotation.from_euler("z", (times * 3)[:, None], degrees=True)
    trajectory = Trajectory(times, positions, rotations)
    translation, orientation = _relative_errors(trajectory, trajectory, 1)
    np.testing.assert_allclose(translation, 0, atol=1e-12)
    np.testing.assert_allclose(orientation, 0, atol=1e-12)
