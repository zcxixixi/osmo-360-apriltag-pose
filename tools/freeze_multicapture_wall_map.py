#!/usr/bin/env python3
"""Freeze a two-panel world map from independent raw-fisheye calibrations.

Each input is a ``tools.calibrate_wall_pair_transform`` report.  Sources with too
few geometric inliers are excluded before estimation.  The fixed SE(3) is
accepted only when leave-one-source-out prediction passes explicit translation
and rotation gates and at least two capture IDs are represented.  Scale is
always one; Sim(3) is not permitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-world-map", type=Path, required=True)
    parser.add_argument("--source", nargs=3, action="append", metavar=("CAPTURE", "CAMERA", "REPORT"), required=True)
    parser.add_argument("--minimum-inliers", type=int, default=50)
    parser.add_argument("--maximum-holdout-translation-mm", type=float, default=20.0)
    parser.add_argument("--maximum-holdout-rotation-deg", type=float, default=2.5)
    parser.add_argument("--output-map", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    return parser.parse_args()


def weighted_transform(items: list[dict]) -> tuple[np.ndarray, Rotation]:
    weights = np.asarray([item["inliers"] for item in items], dtype=float)
    weights /= weights.sum()
    positions = np.asarray([item["p"] for item in items])
    position = np.sum(positions * weights[:, None], axis=0)
    matrices = np.asarray([item["r"].as_matrix() for item in items])
    chord = np.sum(matrices * weights[:, None, None], axis=0)
    u, _, vt = np.linalg.svd(chord)
    matrix = u @ vt
    if np.linalg.det(matrix) < 0:
        u[:, -1] *= -1
        matrix = u @ vt
    return position, Rotation.from_matrix(matrix)


def transform_error(reference: tuple[np.ndarray, Rotation], item: dict) -> tuple[float, float]:
    p, r = reference
    return (
        float(np.linalg.norm(p - item["p"]) * 1000.0),
        float(np.degrees((r.inv() * item["r"]).magnitude())),
    )


def main() -> int:
    args = parse_args()
    sources = []
    excluded = []
    for capture, camera, value in args.source:
        path = Path(value).resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        count = int(payload["frames"]["inlier"])
        transform = payload["T_primary_secondary"]
        record = {
            "capture_id": capture, "camera": camera, "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "inliers": count,
            "p": np.asarray(transform["translation_m"], dtype=float),
            "r": Rotation.from_quat(transform["quaternion_xyzw"]),
        }
        (sources if count >= args.minimum_inliers else excluded).append(record)
    if len(sources) < 3 or len({item["capture_id"] for item in sources}) < 2:
        raise ValueError("need >=3 qualified sources spanning >=2 captures")

    selected = weighted_transform(sources)
    holdouts = []
    for index, item in enumerate(sources):
        training = [other for other_index, other in enumerate(sources) if other_index != index]
        dp, dr = transform_error(weighted_transform(training), item)
        holdouts.append({
            "capture_id": item["capture_id"], "camera": item["camera"],
            "translation_error_mm": dp, "rotation_error_deg": dr,
        })
    translation_pass = max(item["translation_error_mm"] for item in holdouts) <= args.maximum_holdout_translation_mm
    rotation_pass = max(item["rotation_error_deg"] for item in holdouts) <= args.maximum_holdout_rotation_deg
    passed = translation_pass and rotation_pass

    audit = {
        "schema_version": "multicapture-wall-map-freeze/1.0",
        "status": "VERIFIED" if passed else "HOLDOUT_FAILED",
        "scale": 1.0,
        "sim3_used": False,
        "minimum_inliers": args.minimum_inliers,
        "qualified_sources": [{
            key: value for key, value in item.items() if key not in {"p", "r"}
        } for item in sources],
        "excluded_sources": [{
            key: value for key, value in item.items() if key not in {"p", "r"}
        } for item in excluded],
        "leave_one_source_out": holdouts,
        "gates": {
            "maximum_translation_mm": args.maximum_holdout_translation_mm,
            "maximum_rotation_deg": args.maximum_holdout_rotation_deg,
            "translation_pass": translation_pass,
            "rotation_pass": rotation_pass,
        },
        "T_leftwall_rightwall": {
            "parent_frame": "tag_map", "child_frame": "right_wall_tag_map",
            "translation_m": selected[0].tolist(),
            "quaternion_xyzw": selected[1].as_quat().tolist(),
        },
        "training_ready": passed,
    }
    args.output_audit.parent.mkdir(parents=True, exist_ok=True)
    args.output_audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    world = json.loads(args.template_world_map.read_text(encoding="utf-8"))
    world["map_id"] = "room-corner-10tag-200mm-multicapture-v1"
    world["calibration_status"] = "VERIFIED" if passed else "HOLDOUT_FAILED"
    world["training_ready"] = passed
    world["calibration_audit"] = str(args.output_audit.resolve())
    world["panels"][1]["T_world_map"] = audit["T_leftwall_rightwall"]
    args.output_map.parent.mkdir(parents=True, exist_ok=True)
    args.output_map.write_text(json.dumps(world, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
