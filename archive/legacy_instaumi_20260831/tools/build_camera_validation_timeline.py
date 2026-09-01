#!/usr/bin/env python3
"""Build an interactive camera-only 3D timeline from a validated pose CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from osmo360.localization.world_frames import compile_world_tag_map

CAMERA_VISUAL_MODELS = {
    "dji-osmo-360": {
        "label": "DJI OSMO 360",
        "body_size_m": [0.075, 0.052, 0.040],
        "geometry_status": "legacy display-only model",
    },
    "insta360-x5": {
        "label": "INSTA360 X5",
        "body_size_m": [0.0382, 0.046, 0.1245],
        "geometry_status": "official body dimensions; optical-centre offset unverified",
        "source": "https://onlinemanual.insta360.com/x5/en-us/specs/hardware",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-csv", type=Path, required=True)
    parser.add_argument("--tag-map", type=Path, required=True)
    parser.add_argument("--capture-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pose-label", default="SCREEN TAGS")
    parser.add_argument(
        "--camera-hardware-model", choices=tuple(CAMERA_VISUAL_MODELS),
        default="dji-osmo-360",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = [
        row for row in csv.DictReader(args.pose_csv.open(newline="", encoding="utf-8"))
        if row.get("camera_x_m") and row.get("qw")
    ]
    if len(rows) < 2:
        raise RuntimeError("pose CSV needs at least two valid frames")
    times = np.asarray([float(row["timestamp"]) for row in rows])
    positions = np.asarray([
        [float(row[key]) for key in ("camera_x_m", "camera_y_m", "camera_z_m")]
        for row in rows
    ])
    rotations = Rotation.from_quat([
        [float(row[key]) for key in ("qx", "qy", "qz", "qw")]
        for row in rows
    ])
    fps = float(1.0 / np.median(np.diff(times)))
    compiled = compile_world_tag_map(args.tag_map)
    frames = []
    for index, (time_s, position, rotation) in enumerate(zip(times, positions, rotations)):
        quaternion = rotation.as_quat().tolist()
        euler = rotation.as_euler("xyz", degrees=True).tolist()
        left = {
            "p": position.tolist(), "q": quaternion,
            "source_p": position.tolist(), "source_q": quaternion,
            "euler_deg": euler, "opening": 0.0, "joints": [0.0, 0.0],
            "pose_state": args.pose_label, "angle_state": "N/A", "visible": True,
        }
        right = {
            "p": position.tolist(), "q": quaternion,
            "source_p": position.tolist(), "source_q": quaternion,
            "euler_deg": [0.0, 0.0, 0.0], "opening": 0.0,
            "joints": [0.0, 0.0], "pose_state": "HIDDEN",
            "angle_state": "HIDDEN", "visible": False,
        }
        frames.append({"t": float(time_s), "source_index": int(rows[index]["frame"]), "left": left, "right": right})
    anchors = []
    for tag in compiled["tags"]:
        tag_id = int(tag["id"])
        anchors.append({
            "id": tag_id,
            "side": "left" if tag_id == 200 else "right",
            "corners": tag["corners_m"],
            "color": "#37c8ff" if tag_id == 200 else "#58df91",
        })
    timeline = {
        "schema_version": "camera-world-diagnostic/v1",
        "audit_mode": False,
        "render_mode": "camera_world_diagnostic",
        "default_view": "human_corner",
        "view_roll_deg": 0.0,
        "operator_eye_elevation_factor": 0.10,
        "operator_tag_look_fraction": 0.35,
        "capture_pair_id": args.capture_id,
        "camera_hardware_model": args.camera_hardware_model,
        "camera_visual_model": CAMERA_VISUAL_MODELS[args.camera_hardware_model],
        "layout_calibration_id": compiled.get("map_id", "two-screen-map"),
        "reference_frame": compiled.get("world_frame", "tag_map"),
        "fps": fps,
        "duration_s": float(times[-1]),
        "source_frames": len(frames),
        "training_frames": 0,
        "training_episodes": 0,
        "training_ready": False,
        "sync": {"offset_s": 0.0, "correlation": 1.0},
        "side_mapping": {"left": "camera", "right": "hidden"},
        "coordinate_status": {
            "mode": "world",
            "frame_id": compiled.get("world_frame", "tag_map"),
            "calibration_status": compiled.get("calibration_status", "DIAGNOSTIC"),
            "frame_convention": {
                "up_vector": compiled.get("physical_up_vector", [0.0, -1.0, 0.0]),
                "units": "m",
            },
        },
        "eef_reference": {"type": "camera"},
        "attitude": {"mode": "two-screen AprilTag diagnostic", "level_constraint": False},
        "bounds_m": {},
        "tag_anchors": anchors,
        "validation_events": [],
        "segment_untrusted_tracks": False,
        "frames": frames,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(timeline, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
