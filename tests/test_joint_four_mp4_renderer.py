from __future__ import annotations

import numpy as np

from tools.render_joint_four_mp4_trajectory import Projector


def test_flu_front_above_projector_is_fixed_in_front_and_above_tag_plane():
    points = np.asarray([
        [0.0, -0.4, -0.2],
        [0.0, 0.8, 0.3],
        [0.8, -0.2, -0.2],
        [0.8, 0.6, 0.2],
        [0.22, 0.0, 0.0],
        [0.0, 0.22, 0.0],
        [0.0, 0.0, 0.22],
    ])
    projector = Projector(
        points,
        (1000, 72),
        (880, 565),
        preset="flu-front-above",
        focus=np.zeros(3),
    )

    assert np.allclose(projector.eye, [1.55, 0.0, 0.85])
    assert np.allclose(projector.target, [0.28, 0.0, 0.0])
    for point in points:
        x, y = projector(point)
        assert 1000 <= x <= 1880
        assert 72 <= y <= 637


def test_tag_map_front_above_keeps_native_wall_and_physical_up():
    points = np.asarray([
        [-0.2, -0.2, 0.0],
        [0.8, 0.2, 0.0],
        [-0.1, -0.2, -0.75],
        [0.6, 0.2, -0.35],
        [0.22, 0.0, 0.0],
        [0.0, 0.22, 0.0],
        [0.0, 0.0, 0.22],
    ])
    projector = Projector(
        points,
        (1000, 72),
        (880, 565),
        preset="tag-map-front-above",
        focus=np.zeros(3),
    )

    assert np.allclose(projector.eye, [0.0, -0.85, -1.55])
    assert np.allclose(projector.target, [0.0, 0.0, -0.28])
    for point in points:
        x, y = projector(point)
        assert 1000 <= x <= 1880
        assert 72 <= y <= 637
