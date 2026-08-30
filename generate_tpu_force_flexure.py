#!/usr/bin/env python3
"""Generate an adhesive TPU fingertip flexure for visual relative-force sensing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "assets/gripper_v52_new_r1/force_flexure_tpu_r1"


def box(extents: tuple[float, float, float], center: tuple[float, float, float]) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return mesh


def beam_between(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    width_y_mm: float,
    thickness_mm: float,
) -> trimesh.Trimesh:
    start_vector = np.asarray(start, dtype=float)
    end_vector = np.asarray(end, dtype=float)
    direction = end_vector - start_vector
    length = float(np.linalg.norm(direction))
    transform = trimesh.geometry.align_vectors([1.0, 0.0, 0.0], direction / length)
    mesh = trimesh.creation.box(extents=(length, width_y_mm, thickness_mm))
    mesh.apply_transform(transform)
    mesh.apply_translation((start_vector + end_vector) / 2.0)
    return mesh



def voxel_union(mesh: trimesh.Trimesh, pitch_mm: float = 0.25) -> trimesh.Trimesh:
    points = mesh.voxelized(pitch_mm).fill().points
    origin = points.min(axis=0)
    indices = {
        tuple(np.round((point - origin) / pitch_mm).astype(int))
        for point in points
    }
    faces_by_direction = {
        (1, 0, 0): ((1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)),
        (-1, 0, 0): ((-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1)),
        (0, 1, 0): ((-1, 1, -1), (-1, 1, 1), (1, 1, 1), (1, 1, -1)),
        (0, -1, 0): ((-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)),
        (0, 0, 1): ((-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)),
        (0, 0, -1): ((-1, -1, -1), (-1, 1, -1), (1, 1, -1), (1, -1, -1)),
    }
    vertices: list[list[float]] = []
    vertex_ids: dict[tuple[int, int, int], int] = {}
    faces: list[list[int]] = []

    def vertex_id(index: tuple[int, int, int], signs: tuple[int, int, int]) -> int:
        key = tuple(2 * index[axis] + signs[axis] for axis in range(3))
        if key not in vertex_ids:
            vertex_ids[key] = len(vertices)
            vertices.append(
                [
                    float(origin[axis] + key[axis] * pitch_mm / 2.0)
                    for axis in range(3)
                ]
            )
        return vertex_ids[key]

    for index in indices:
        for direction, corners in faces_by_direction.items():
            neighbour = tuple(index[axis] + direction[axis] for axis in range(3))
            if neighbour in indices:
                continue
            quad = [vertex_id(index, corner) for corner in corners]
            faces.append([quad[0], quad[1], quad[2]])
            faces.append([quad[0], quad[2], quad[3]])
    result = trimesh.Trimesh(
        vertices=np.asarray(vertices),
        faces=np.asarray(faces),
        process=True,
    )
    result.metadata["units"] = "mm"
    return result

def make_flexure() -> trimesh.Trimesh:
    marker_boss = trimesh.creation.cylinder(radius=3.0, height=0.8, sections=48)
    marker_boss.apply_transform(
        trimesh.transformations.rotation_matrix(np.pi / 2.0, (1.0, 0.0, 0.0))
    )
    marker_boss.apply_translation((14.0, 0.4, 6.0))
    parts = [
        box((28.0, 12.0, 1.6), (14.0, 6.0, 0.8)),
        box((18.0, 12.0, 2.0), (14.0, 6.0, 8.0)),
        box((3.0, 3.0, 2.0), (5.0, 6.0, 2.6)),
        box((3.0, 3.0, 2.0), (23.0, 6.0, 2.6)),
        marker_boss,
    ]
    for y in (2.5, 9.5):
        parts.append(beam_between((3.0, y, 1.35), (8.0, y, 7.0), 2.2, 1.2))
        parts.append(beam_between((25.0, y, 1.35), (20.0, y, 7.0), 2.2, 1.2))
    return voxel_union(trimesh.util.concatenate(parts))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_preview(mesh: trimesh.Trimesh, path: Path) -> None:
    fig = plt.figure(figsize=(12, 8), facecolor="#071018")
    views = (
        (24, -55, "Installation perspective"),
        (0, -90, "Camera-facing side / compression"),
        (90, -90, "Top view / 28 x 12 mm"),
        (0, 0, "End view"),
    )
    faces = mesh.faces
    marker = trimesh.creation.cylinder(radius=3.0, height=0.85, sections=48)
    marker.apply_transform(
        trimesh.transformations.rotation_matrix(np.pi / 2.0, (1.0, 0.0, 0.0))
    )
    marker.apply_translation((14.0, 0.225, 6.0))
    for index, (elevation, azimuth, title) in enumerate(views, 1):
        axis = fig.add_subplot(2, 2, index, projection="3d", facecolor="#0b1721")
        collection = Poly3DCollection(
            mesh.vertices[faces],
            facecolor="#e4a92f",
            edgecolor="none",
            linewidth=0.0,
        )
        axis.add_collection3d(collection)
        marker_collection = Poly3DCollection(
            marker.vertices[marker.faces],
            facecolor="#111820",
            edgecolor="#111820",
            linewidth=0.05,
        )
        axis.add_collection3d(marker_collection)
        padding = np.asarray([0.7, 0.7, 0.7])
        axis.set_xlim(*(mesh.bounds[:, 0] + np.asarray([-padding[0], padding[0]])))
        axis.set_ylim(*(mesh.bounds[:, 1] + np.asarray([-padding[1], padding[1]])))
        axis.set_zlim(*(mesh.bounds[:, 2] + np.asarray([-padding[2], padding[2]])))
        axis.set_box_aspect(mesh.extents.copy())
        axis.set_proj_type("ortho")
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_title(title, color="white", pad=8)
        axis.set_axis_off()
    fig.suptitle(
        "TPU fingertip flexure r1 / camera-facing moving marker boss",
        color="white",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.02,
        "28 x 12 x 9.0 mm  |  compression travel 3.4 mm hard stop  |  TPU 85A-95A",
        color="#b8c7d1",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    single = make_flexure()
    pair_left = single.copy()
    pair_right = single.copy()
    pair_right.apply_translation((0.0, 18.0, 0.0))
    pair = trimesh.util.concatenate((pair_left, pair_right))
    single_path = output / "TPU_force_flexure_single_r1.STL"
    pair_path = output / "TPU_force_flexure_pair_r1.STL"
    single.export(single_path)
    pair.export(pair_path)
    preview_path = output / "TPU_force_flexure_preview_r1.png"
    render_preview(single, preview_path)
    report = {
        "schema_version": "tpu-force-flexure-build/1.0",
        "units": "mm",
        "single": {
            "path": str(single_path),
            "sha256": sha256(single_path),
            "bounds_mm": single.bounds.tolist(),
            "extents_mm": single.extents.tolist(),
            "watertight_components": all(component.is_watertight for component in single.split()),
            "component_count": len(single.split()),
            "volume_mm3": float(single.volume),
        },
        "pair": {
            "path": str(pair_path),
            "sha256": sha256(pair_path),
            "bounds_mm": pair.bounds.tolist(),
            "extents_mm": pair.extents.tolist(),
            "watertight_components": all(component.is_watertight for component in pair.split()),
            "component_count": len(pair.split()),
            "volume_mm3": float(pair.volume),
        },
        "print": {
            "material": "TPU 85A-95A",
            "orientation": "camera-facing side on print bed",
            "layer_height_mm": 0.2,
            "perimeters": 3,
            "infill_percent": 100,
            "supports": "none for first trial; inspect bridge quality",
        },
        "mechanical": {
            "adhesive_base_mm": [28.0, 12.0, 1.6],
            "contact_shoe_mm": [18.0, 12.0, 2.0],
            "initial_height_mm": 9.0,
            "hard_stop_travel_mm": 3.4,
            "target_visual_travel_mm": [3.0, 5.0],
            "marker_boss_diameter_mm": 6.0,
            "marker_boss_face": "camera-facing side; clear of the contact surface",
        },
    }
    (output / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
