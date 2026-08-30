#!/usr/bin/env python3
"""Compose a reciprocal camera->BaseTag calibration into camera->base_link."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("basetag_calibration", type=Path)
    parser.add_argument("--hardware-geometry", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--camera-frame", default="fisheye1_camera_panorama_axes")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.basetag_calibration.read_text(encoding="utf-8"))
    hardware = json.loads(args.hardware_geometry.read_text(encoding="utf-8"))
    camera_tag = source["camera_to_basetag"]
    p_camera_tag = np.asarray(camera_tag["translation_m"], dtype=float)
    r_camera_tag = Rotation.from_quat(camera_tag["quaternion_xyzw"])
    base_tag = hardware["base_to_tag"]
    p_base_tag = np.asarray(base_tag["translation_m"], dtype=float)
    r_base_tag = Rotation.from_quat(base_tag["quaternion_xyzw"])
    r_tag_base = r_base_tag.inv()
    p_tag_base = -r_tag_base.apply(p_base_tag)
    p_camera_base = p_camera_tag + r_camera_tag.apply(p_tag_base)
    r_camera_base = r_camera_tag * r_tag_base
    output = {
        "schema_version": "camera-to-base-calibration/1.0",
        "calibration_status": source.get("calibration_status", "PROVISIONAL"),
        "side": args.side,
        "camera_to_base": {
            "parent_frame": args.camera_frame,
            "child_frame": f"{args.side}_base_link",
            "translation_m": p_camera_base.tolist(),
            "quaternion_xyzw": r_camera_base.as_quat().tolist(),
        },
        "composition": "T_camera_base = T_camera_basetag @ inverse(T_base_basetag)",
        "base_to_tag": base_tag,
        "reciprocal_audit": source.get("audit"),
        "world_closure": source.get("world_closure"),
        "source": {
            "basetag_calibration": str(args.basetag_calibration.resolve()),
            "hardware_geometry": str(args.hardware_geometry.resolve()),
        },
        "training_ready": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
