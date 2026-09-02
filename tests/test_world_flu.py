from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from osmo360.datasets.world_flu import (
    WORLD_FLU_FRAME,
    derive_world_flu_transform,
    transform_trajectory_rows,
    transform_world_map,
)


def _tag(tag_id: int, panel: str, center_x: float) -> dict:
    return {
        "id": tag_id,
        "panel": panel,
        "corners_m": [
            [center_x - 0.1, -0.1, 0.0],
            [center_x + 0.1, -0.1, 0.0],
            [center_x + 0.1, 0.1, 0.0],
            [center_x - 0.1, 0.1, 0.0],
        ],
    }


def _world_map() -> dict:
    return {
        "schema_version": "world-apriltag-map/1.0",
        "map_id": "test-map",
        "world_frame": "session_grid_A",
        "physical_up_vector": [0.0, -1.0, 0.0],
        "panel_transform": {"source": "test"},
        "tags": [_tag(200, "grid_A", 0.0), _tag(210, "grid_B", 1.0)],
    }


def test_world_flu_uses_grid_midpoint_back_left_up_axes() -> None:
    transform = derive_world_flu_transform(_world_map())

    assert np.allclose(transform.origin_source_m, [0.5, 0.0, 0.0])
    points_source = np.asarray(
        [
            [0.5, 0.0, 1.0],  # behind the grids
            [0.4, 0.0, 0.0],  # left while looking along +X
            [0.5, -1.0, 0.0],  # physical up
        ]
    )
    assert np.allclose(
        transform.transform_points(points_source),
        [[1.0, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 1.0]],
    )
    assert np.isclose(np.linalg.det(transform.rotation_target_from_source), 1.0)


def test_world_flu_reframes_position_orientation_and_map() -> None:
    source_map = _world_map()
    transform = derive_world_flu_transform(source_map)
    rows = [
        {
            "world_frame": "session_grid_A",
            "map_id": "test-map",
            "left_camera_x_m": "0.5",
            "left_camera_y_m": "0",
            "left_camera_z_m": "-1",
            "left_qx": "0",
            "left_qy": "0",
            "left_qz": "0",
            "left_qw": "1",
            "right_camera_x_m": "",
            "right_camera_y_m": "",
            "right_camera_z_m": "",
            "right_qx": "",
            "right_qy": "",
            "right_qz": "",
            "right_qw": "",
        }
    ]

    reframed = transform_trajectory_rows(rows, transform)[0]
    target_map = transform_world_map(source_map, transform)

    assert reframed["world_frame"] == WORLD_FLU_FRAME
    assert [float(reframed[f"left_camera_{axis}_m"]) for axis in "xyz"] == [
        -1.0,
        0.0,
        0.0,
    ]
    expected = Rotation.from_matrix(transform.rotation_target_from_source)
    actual = Rotation.from_quat([float(reframed[f"left_q{axis}"]) for axis in "xyzw"])
    assert np.allclose((expected.inv() * actual).as_rotvec(), 0.0, atol=1e-10)
    assert target_map["world_frame"] == WORLD_FLU_FRAME
    assert target_map["physical_up_vector"] == [0.0, 0.0, 1.0]
    centers = {
        panel: np.mean(
            [
                corner
                for tag in target_map["tags"]
                if tag["panel"] == panel
                for corner in tag["corners_m"]
            ],
            axis=0,
        )
        for panel in ("grid_A", "grid_B")
    }
    assert np.allclose(0.5 * (centers["grid_A"] + centers["grid_B"]), 0.0)
