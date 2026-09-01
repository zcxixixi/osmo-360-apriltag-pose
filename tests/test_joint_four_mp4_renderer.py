from __future__ import annotations

import numpy as np

from tools.render_joint_four_mp4_trajectory import Projector, Track, parse_args


def test_review_renderer_defaults_to_accepted_tag_map_view(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["render", "/dataset", "/tracking", "/output.mp4"],
    )

    assert parse_args().view_preset == "tag-map-front-above"


def _track_row(time_s: float, x: float, *, state: str = "MEASURED") -> dict[str, str]:
    return {
        "timestamp_s": str(time_s),
        "left_quality_status": "valid" if state == "MEASURED" else "interpolated",
        "left_pose_state": state,
        "left_camera_x_m": str(x),
        "left_camera_y_m": "0",
        "left_camera_z_m": "0",
        "left_qx": "0",
        "left_qy": "0",
        "left_qz": "0",
        "left_qw": "1",
    }


def test_track_shows_explicit_untrusted_pose_in_long_measurement_gap():
    track = Track([
        _track_row(0.0, 0.0),
        _track_row(0.2, 0.2, state="INTERPOLATED_UNTRUSTED"),
        _track_row(0.4, 0.4),
    ], "left")

    assert track.sample(0.0) is not None
    assert track.sample(0.2) is not None
    assert track.sample(0.2).state == "INTERPOLATED_UNTRUSTED"
    assert track.sample(0.4) is not None
    assert track.segment_slices() == [slice(0, 3)]


def test_track_samples_bounded_interpolation():
    track = Track([
        _track_row(0.0, 0.0),
        _track_row(0.2, 0.2, state="INTERPOLATED"),
        _track_row(0.4, 0.4),
    ], "left")

    sample = track.sample(0.1)

    assert sample is not None
    assert sample.position[0] == 0.1
    assert len(track.segment_slices()) == 1


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
