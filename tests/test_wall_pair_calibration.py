from __future__ import annotations

import json
from pathlib import Path

import pytest

from calibrate_wall_pair_transform import load_tag_map


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
