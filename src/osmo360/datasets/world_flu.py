"""Re-express native AprilGrid-map trajectories in the requested world FLU frame."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from osmo360.pipeline.manifest import ManifestError


WORLD_FLU_FRAME = "world_flu_aprilgrid_midpoint"
WORLD_FLU_REVISION = "aprilgrid-midpoint-flu-back-x-v1"


def _unit(value: np.ndarray, *, field: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if vector.shape != (3,) or not np.isfinite(vector).all() or norm < 1e-9:
        raise ManifestError(f"{field} must be a finite non-zero 3-vector")
    return vector / norm


@dataclass(frozen=True)
class WorldFluTransform:
    source_frame: str
    source_map_id: str
    target_frame: str
    target_map_id: str
    origin_source_m: np.ndarray
    rotation_target_from_source: np.ndarray
    panel_centers_source_m: dict[str, np.ndarray]
    panel_normals_source: dict[str, np.ndarray]

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        return (
            self.rotation_target_from_source
            @ (values - self.origin_source_m).T
        ).T

    def transform_rotation(self, rotation_source_from_child: Rotation) -> Rotation:
        return (
            Rotation.from_matrix(self.rotation_target_from_source)
            * rotation_source_from_child
        )

    def metadata(self) -> dict[str, Any]:
        rotation = Rotation.from_matrix(self.rotation_target_from_source)
        return {
            "revision": WORLD_FLU_REVISION,
            "source_frame": self.source_frame,
            "target_frame": self.target_frame,
            "origin_definition": "midpoint_of_grid_A_and_grid_B_geometric_centers",
            "axis_definition": {
                "x_positive": "average_AprilGrid_corner_winding_normal_toward_grid_back",
                "y_positive": "left_when_looking_along_positive_x",
                "z_positive": "physical_up",
            },
            "origin_source_m": self.origin_source_m.tolist(),
            "rotation_target_from_source_matrix": (
                self.rotation_target_from_source.tolist()
            ),
            "rotation_target_from_source_quaternion_xyzw": rotation.as_quat().tolist(),
            "panel_centers_source_m": {
                name: value.tolist()
                for name, value in self.panel_centers_source_m.items()
            },
            "panel_normals_source": {
                name: value.tolist()
                for name, value in self.panel_normals_source.items()
            },
        }


def derive_world_flu_transform(world_map: dict[str, Any]) -> WorldFluTransform:
    tags = world_map.get("tags")
    if not isinstance(tags, list) or not tags:
        raise ManifestError("world map must contain AprilGrid tags")
    panel_points: dict[str, list[np.ndarray]] = {"grid_A": [], "grid_B": []}
    panel_normals: dict[str, list[np.ndarray]] = {"grid_A": [], "grid_B": []}
    for tag in tags:
        panel = str(tag.get("panel", ""))
        if panel not in panel_points:
            continue
        corners = np.asarray(tag.get("corners_m"), dtype=np.float64)
        if corners.shape != (4, 3) or not np.isfinite(corners).all():
            raise ManifestError(f"{panel} contains invalid AprilTag corners")
        normal = _unit(
            np.cross(corners[1] - corners[0], corners[3] - corners[0]),
            field=f"{panel} tag normal",
        )
        panel_points[panel].append(corners)
        panel_normals[panel].append(normal)
    if any(not panel_points[name] for name in panel_points):
        raise ManifestError("world map must contain both grid_A and grid_B")

    centers = {
        name: np.concatenate(values, axis=0).mean(axis=0)
        for name, values in panel_points.items()
    }
    normals: dict[str, np.ndarray] = {}
    reference: np.ndarray | None = None
    for name in ("grid_A", "grid_B"):
        values = panel_normals[name]
        if reference is not None:
            values = [
                value if np.dot(value, reference) >= 0 else -value
                for value in values
            ]
        normal = _unit(np.mean(values, axis=0), field=f"{name} average normal")
        normals[name] = normal
        if reference is None:
            reference = normal
    if np.dot(normals["grid_A"], normals["grid_B"]) < 0:
        normals["grid_B"] = -normals["grid_B"]

    # The map's TL,TR,BR,BL corner winding points through the printed panel to
    # its rear. That requested direction is world-FLU +X.
    x_axis_source = _unit(
        normals["grid_A"] + normals["grid_B"],
        field="joint AprilGrid back normal",
    )
    physical_up = _unit(
        np.asarray(world_map.get("physical_up_vector"), dtype=np.float64),
        field="physical_up_vector",
    )
    z_axis_source = physical_up - np.dot(physical_up, x_axis_source) * x_axis_source
    z_axis_source = _unit(
        z_axis_source, field="physical up projected onto grid plane"
    )
    y_axis_source = _unit(
        np.cross(z_axis_source, x_axis_source), field="world FLU left axis"
    )
    z_axis_source = _unit(
        np.cross(x_axis_source, y_axis_source),
        field="world FLU orthogonal up axis",
    )
    rotation_target_from_source = np.stack(
        (x_axis_source, y_axis_source, z_axis_source), axis=0
    )
    if not math.isclose(
        float(np.linalg.det(rotation_target_from_source)), 1.0, abs_tol=1e-9
    ):
        raise ManifestError("derived world FLU rotation is not right-handed")
    origin = 0.5 * (centers["grid_A"] + centers["grid_B"])
    source_map_id = str(world_map.get("map_id", "source-map"))
    return WorldFluTransform(
        source_frame=str(world_map.get("world_frame", "source_world")),
        source_map_id=source_map_id,
        target_frame=WORLD_FLU_FRAME,
        target_map_id=f"{source_map_id}-{WORLD_FLU_REVISION}",
        origin_source_m=origin,
        rotation_target_from_source=rotation_target_from_source,
        panel_centers_source_m=centers,
        panel_normals_source=normals,
    )


def transform_trajectory_rows(
    rows: list[dict[str, Any]], transform: WorldFluTransform
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source_row in rows:
        row = dict(source_row)
        row["world_frame"] = transform.target_frame
        row["map_id"] = transform.target_map_id
        for side in ("left", "right"):
            position_keys = [f"{side}_camera_{axis}_m" for axis in "xyz"]
            quaternion_keys = [f"{side}_q{axis}" for axis in "xyzw"]
            position_values = [str(row.get(key, "")) for key in position_keys]
            quaternion_values = [str(row.get(key, "")) for key in quaternion_keys]
            if not any(position_values) and not any(quaternion_values):
                continue
            if not all(position_values) or not all(quaternion_values):
                raise ManifestError(f"{side} pose is incomplete and cannot be reframed")
            position = np.asarray(list(map(float, position_values)), dtype=np.float64)
            quaternion = np.asarray(list(map(float, quaternion_values)), dtype=np.float64)
            if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
                raise ManifestError(f"{side} pose contains non-finite values")
            target_position = transform.transform_points(position[None])[0]
            target_rotation = transform.transform_rotation(
                Rotation.from_quat(quaternion)
            )
            for key, value in zip(position_keys, target_position, strict=True):
                row[key] = f"{float(value):.9f}"
            for key, value in zip(
                quaternion_keys, target_rotation.as_quat(), strict=True
            ):
                row[key] = f"{float(value):.12f}"
        result.append(row)
    return result


def transform_world_map(
    world_map: dict[str, Any], transform: WorldFluTransform
) -> dict[str, Any]:
    result = copy.deepcopy(world_map)
    result["map_id"] = transform.target_map_id
    result["world_frame"] = transform.target_frame
    result["physical_up_vector"] = [0.0, 0.0, 1.0]
    result["source_panel_transform"] = result.pop("panel_transform", None)
    result["world_flu_transform"] = transform.metadata()
    for tag in result["tags"]:
        corners = np.asarray(tag["corners_m"], dtype=np.float64)
        tag["corners_m"] = transform.transform_points(corners).tolist()
    return result
