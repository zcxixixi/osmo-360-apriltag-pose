#!/usr/bin/env python3
"""Typed rigid transforms and composable AprilTag world maps.

The project previously passed anonymous matrices between stages.  This module
keeps the frame direction explicit: ``T_parent_child`` maps child-frame points
into the parent frame.
"""

from __future__ import annotations

import hashlib
import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


def canonical_sha256(payload: dict[str, Any]) -> str:
    clean = {
        key: value for key, value in payload.items()
        if key not in {"tag_map_sha256", "source_path"}
    }
    encoded = json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class RigidTransform:
    parent_frame: str
    child_frame: str
    translation_m: np.ndarray
    rotation: Rotation

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RigidTransform":
        parent = str(payload.get("parent_frame", "")).strip()
        child = str(payload.get("child_frame", "")).strip()
        translation = np.asarray(payload.get("translation_m"), dtype=float)
        quaternion = np.asarray(payload.get("quaternion_xyzw"), dtype=float)
        if not parent or not child or parent == child:
            raise ValueError("transform needs distinct non-empty parent_frame and child_frame")
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError("translation_m must contain three finite metres")
        if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
            raise ValueError("quaternion_xyzw must contain four finite values")
        norm = float(np.linalg.norm(quaternion))
        if abs(norm - 1.0) > 1e-3:
            raise ValueError(f"quaternion_xyzw is not unit length: {norm:.6f}")
        rotation = Rotation.from_quat(quaternion / norm)
        matrix = rotation.as_matrix()
        if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-8):
            raise ValueError("rotation is not orthonormal")
        if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-8):
            raise ValueError("rotation determinant must be +1")
        return cls(parent, child, translation, rotation)

    def apply_points(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=float)
        return self.rotation.apply(values) + self.translation_m

    def inverse(self) -> "RigidTransform":
        inverse_rotation = self.rotation.inv()
        return RigidTransform(
            self.child_frame,
            self.parent_frame,
            -inverse_rotation.apply(self.translation_m),
            inverse_rotation,
        )

    def compose(self, child: "RigidTransform") -> "RigidTransform":
        if self.child_frame != child.parent_frame:
            raise ValueError(
                f"cannot compose T_{self.parent_frame}_{self.child_frame} with "
                f"T_{child.parent_frame}_{child.child_frame}"
            )
        return RigidTransform(
            self.parent_frame,
            child.child_frame,
            self.translation_m + self.rotation.apply(child.translation_m),
            self.rotation * child.rotation,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_frame": self.parent_frame,
            "child_frame": self.child_frame,
            "translation_m": self.translation_m.tolist(),
            "quaternion_xyzw": self.rotation.as_quat().tolist(),
        }


def _load_direct_map(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("tags"), list) or not payload["tags"]:
        raise ValueError(f"tag map has no tags: {path}")
    return payload


def compile_world_tag_map(path: Path) -> dict[str, Any]:
    """Return one explicit per-ID map, expanding panel references if needed."""
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    excluded_ids = set(map(int, payload.get("excluded_tag_ids", [])))
    if isinstance(payload.get("tags"), list):
        compiled = dict(payload)
        compiled["tags"] = [
            tag for tag in payload["tags"] if int(tag["id"]) not in excluded_ids
        ]
    else:
        panels = payload.get("panels")
        if not isinstance(panels, list) or not panels:
            raise ValueError("world tag map needs tags or panels")
        tags: list[dict[str, Any]] = []
        seen: set[int] = set()
        for panel in panels:
            transform = RigidTransform.from_dict(panel["T_world_map"])
            if transform.parent_frame != payload.get("world_frame"):
                raise ValueError("panel transform parent must equal world_frame")
            source = _load_direct_map((path.parent / panel["tag_map"]).resolve())
            allowed = set(map(int, panel.get("expected_ids", [])))
            for tag in source["tags"]:
                tag_id = int(tag["id"])
                if allowed and tag_id not in allowed:
                    continue
                if tag_id in excluded_ids:
                    continue
                if tag_id in seen:
                    raise ValueError(f"duplicate world tag id {tag_id}")
                corners = transform.apply_points(np.asarray(tag["corners_m"], dtype=float))
                tags.append({
                    "id": tag_id,
                    "corners_m": corners.tolist(),
                    "panel": str(panel["name"]),
                })
                seen.add(tag_id)
        expected = set(map(int, payload.get("expected_ids", []))) - excluded_ids
        if expected and seen != expected:
            raise ValueError(f"world map IDs {sorted(seen)} do not match expected {sorted(expected)}")
        compiled = {
            key: value for key, value in payload.items()
            if key not in {"panels", "tag_map_sha256"}
        }
        compiled["panels"] = panels
        compiled["tags"] = tags
    ids = [int(tag["id"]) for tag in compiled["tags"]]
    if len(ids) != len(set(ids)):
        raise ValueError("world tag IDs must be globally unique")
    compiled["source_path"] = str(path)
    compiled["tag_map_sha256"] = canonical_sha256(compiled)
    declared = payload.get("tag_map_sha256")
    if declared and declared != compiled["tag_map_sha256"]:
        raise ValueError("declared tag_map_sha256 does not match compiled map")
    return compiled


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile and validate a framed world AprilTag map")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    compiled = compile_world_tag_map(args.input)
    text = json.dumps(compiled, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(json.dumps({
        "map_id": compiled.get("map_id"),
        "world_frame": compiled.get("world_frame"),
        "calibration_status": compiled.get("calibration_status"),
        "tag_map_sha256": compiled["tag_map_sha256"],
        "ids": sorted(int(tag["id"]) for tag in compiled["tags"]),
        "output": str(args.output.resolve()) if args.output else None,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
