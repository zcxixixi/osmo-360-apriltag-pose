import csv
import json
import sys

from tools import build_camera_validation_timeline as builder


def test_x5_camera_model_is_serialized_into_timeline(tmp_path, monkeypatch):
    pose = tmp_path / "pose.csv"
    with pose.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "frame", "timestamp", "camera_x_m", "camera_y_m", "camera_z_m",
            "qx", "qy", "qz", "qw",
        ])
        writer.writeheader()
        writer.writerows([
            {"frame": 0, "timestamp": 0.0, "camera_x_m": 0, "camera_y_m": 0,
             "camera_z_m": 1, "qx": 0, "qy": 0, "qz": 0, "qw": 1},
            {"frame": 1, "timestamp": 0.1, "camera_x_m": 0.1, "camera_y_m": 0,
             "camera_z_m": 1, "qx": 0, "qy": 0, "qz": 0, "qw": 1},
        ])
    tag_map = tmp_path / "map.json"
    tag_map.write_text(json.dumps({
        "world_frame": "grid_A_board",
        "tags": [{"id": 200, "corners_m": [
            [-0.06, -0.06, 0], [0.06, -0.06, 0],
            [0.06, 0.06, 0], [-0.06, 0.06, 0],
        ]}],
    }))
    output = tmp_path / "timeline.json"
    monkeypatch.setattr(sys, "argv", [
        "build_camera_validation_timeline.py",
        "--pose-csv", str(pose), "--tag-map", str(tag_map),
        "--capture-id", "x5-test", "--output", str(output),
        "--camera-hardware-model", "insta360-x5",
    ])

    assert builder.main() == 0
    timeline = json.loads(output.read_text())
    assert timeline["camera_hardware_model"] == "insta360-x5"
    assert timeline["camera_visual_model"]["label"] == "INSTA360 X5"
    assert timeline["camera_visual_model"]["body_size_m"] == [0.0382, 0.046, 0.1245]
