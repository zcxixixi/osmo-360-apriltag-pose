from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibrate_wall_pair_transform import PoseSample, load_tag_map, robust_panel_transform


def _write_square_map(path: Path, *, size_m: float, edge_m: float | None = None) -> None:
    edge = size_m if edge_m is None else edge_m
    payload = {
        "schema_version": "test-tag-map/1.0",
        "units": "m",
        "tag_outer_size_m": size_m,
        "tags": [
            {
                "id": 7,
                "corners_m": [
                    [0.0, 0.0, 0.0],
                    [edge, 0.0, 0.0],
                    [edge, edge, 0.0],
                    [0.0, edge, 0.0],
                ],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_wall_calibration_accepts_declared_non_200mm_tag_size(tmp_path: Path) -> None:
    path = tmp_path / "map.json"
    _write_square_map(path, size_m=0.15)

    tag_map = load_tag_map(path)

    assert tag_map.tag_outer_size_m == pytest.approx(0.15)
    assert set(tag_map.corners_by_id) == {7}


def test_wall_calibration_can_enforce_an_expected_tag_size(tmp_path: Path) -> None:
    path = tmp_path / "map.json"
    _write_square_map(path, size_m=0.15)

    with pytest.raises(ValueError, match="expected 0.2"):
        load_tag_map(path, expected_tag_size_m=0.2)


def test_wall_calibration_rejects_geometry_that_disagrees_with_declared_size(
    tmp_path: Path,
) -> None:
    path = tmp_path / "map.json"
    _write_square_map(path, size_m=0.15, edge_m=0.16)

    with pytest.raises(ValueError, match="not a 0.15 m square"):
        load_tag_map(path)


def _pose_pair_for_transform(
    frame: int, transform: Rotation, translation: np.ndarray
) -> tuple[PoseSample, PoseSample]:
    secondary_orientation = Rotation.from_euler(
        "xyz", [frame * 0.7, -frame * 0.3, frame * 0.5], degrees=True
    )
    secondary_position = np.asarray(
        [0.2 + frame * 0.004, -0.1 + frame * 0.002, 0.8 + frame * 0.001]
    )
    primary = PoseSample(
        frame=frame,
        timestamp_s=frame / 20.0,
        position_m=transform.apply(secondary_position) + translation,
        orientation=transform * secondary_orientation,
        rmse_px=0.8,
        detected_ids=(134, 135),
        measurement_source="direct",
    )
    secondary = PoseSample(
        frame=frame,
        timestamp_s=frame / 20.0,
        position_m=secondary_position,
        orientation=secondary_orientation,
        rmse_px=0.8,
        detected_ids=(128, 129),
        measurement_source="direct",
    )
    return primary, secondary


def test_physical_wall_angle_gate_rejects_larger_wrong_planar_cluster() -> None:
    correct = Rotation.from_euler("y", 90.0, degrees=True)
    wrong = Rotation.from_euler("y", 70.0, degrees=True)
    correct_translation = np.asarray([0.15, -0.02, 0.03])
    wrong_translation = np.asarray([0.31, 0.08, -0.04])
    pairs = [
        _pose_pair_for_transform(frame, wrong, wrong_translation) for frame in range(25)
    ] + [
        _pose_pair_for_transform(frame, correct, correct_translation)
        for frame in range(25, 45)
    ]

    rotation, translation, keep, audit = robust_panel_transform(
        [pair[0] for pair in pairs],
        [pair[1] for pair in pairs],
        minimum_inliers=20,
        expected_wall_plane_angle_deg=90.0,
        wall_plane_angle_tolerance_deg=5.0,
    )

    assert np.degrees((correct.inv() * rotation).magnitude()) < 1e-8
    assert translation == pytest.approx(correct_translation, abs=1e-10)
    assert keep.tolist() == [False] * 25 + [True] * 20
    assert audit["physical_geometry_gate"]["candidate_frames_rejected"] == 25
    assert audit["physical_geometry_gate"]["selected_wall_plane_angle_deg"] == pytest.approx(90.0)


def test_physical_wall_angle_gate_fails_closed_without_enough_support() -> None:
    wrong = Rotation.from_euler("y", 70.0, degrees=True)
    pairs = [
        _pose_pair_for_transform(frame, wrong, np.zeros(3)) for frame in range(30)
    ]

    with pytest.raises(ValueError, match="physical wall-angle gate"):
        robust_panel_transform(
            [pair[0] for pair in pairs],
            [pair[1] for pair in pairs],
            minimum_inliers=20,
            expected_wall_plane_angle_deg=90.0,
            wall_plane_angle_tolerance_deg=5.0,
        )
