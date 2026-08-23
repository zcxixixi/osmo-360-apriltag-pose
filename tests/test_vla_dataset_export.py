import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import zarr

from vla_dataset_export import _rot6d, build_episode, load_pose_csv, resample_pose


def write_pose(path: Path, duration: float = 4.0, fps: int = 20) -> None:
    fields = [
        "frame", "timestamp", "camera_x_m", "camera_y_m", "camera_z_m",
        "roll_deg", "pitch_deg", "yaw_deg", "measurement_source", "quality_status",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for frame in range(int(duration * fps) + 1):
            time = frame / fps
            writer.writerow({
                "frame": frame, "timestamp": time, "camera_x_m": time * 0.1,
                "camera_y_m": 0, "camera_z_m": 1, "roll_deg": 0,
                "pitch_deg": 0, "yaw_deg": time * 10,
                "measurement_source": "direct", "quality_status": "valid",
            })


def write_gripper(path: Path, duration: float = 4.0, fps: int = 20) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time_s", "opening_angle_deg", "measured"])
        writer.writeheader()
        for frame in range(int(duration * fps) + 1):
            writer.writerow({"time_s": frame / fps, "opening_angle_deg": frame / 2, "measured": 1})


def test_pose_resampling_and_rotation6d(tmp_path: Path) -> None:
    pose = tmp_path / "pose.csv"; write_pose(pose)
    series = load_pose_csv(pose)
    position, rotation, direct, tracked = resample_pose(series, np.array([0.5, 1.5]))
    assert np.allclose(position[:, 0], [0.05, 0.15])
    assert np.all(direct)
    assert np.all(tracked)
    six = _rot6d(rotation)
    assert six.shape == (2, 6)
    assert np.allclose(np.linalg.norm(six[:, :3], axis=1), 1)


def test_verified_episode_exports_canonical_arrays(tmp_path: Path) -> None:
    pose = tmp_path / "pose.csv"; angle = tmp_path / "angle.csv"
    write_pose(pose); write_gripper(angle)
    hardware = {
        "calibration_status": "verified",
        "robots": {name: {
            "camera_to_tcp": {"translation_m": [0, 0, 0], "quaternion_xyzw": [0, 0, 0, 1]},
            "camera_to_tcp_verified": True,
            "gripper_width_calibration": {
                "type": "linear", "closed_angle_deg": 0, "closed_width_m": 0,
                "open_angle_deg": 40, "open_width_m": 0.08,
            },
            "gripper_width_verified": True,
        } for name in ("left", "right")},
    }
    (tmp_path / "hardware.json").write_text(json.dumps(hardware), encoding="utf-8")
    spec = {
        "task": {"instruction": "pick up the object", "success": True},
        "start_s": 0, "end_s": 4, "frequency_hz": 20,
        "hardware_config": "hardware.json", "sync": {"uncertainty_s": 0.01},
        "robots": [
            {"name": name, "trajectory_csv": "pose.csv", "gripper_csv": "angle.csv"}
            for name in ("left", "right")
        ],
    }
    spec_path = tmp_path / "episode.json"; spec_path.write_text(json.dumps(spec), encoding="utf-8")
    output = tmp_path / "output"
    metadata = build_episode(spec_path, output, skip_rgb=True)
    assert not metadata["training_ready"]  # RGB is an explicit readiness gate.
    arrays = np.load(output / "episode_arrays.npz")
    assert arrays["action"].shape == (80, 20)
    assert np.all(np.isfinite(arrays["action"]))
    assert arrays["robot0_eef_pos"][-1, 0] > 0.3


def test_pending_hardware_stays_draft(tmp_path: Path) -> None:
    pose = tmp_path / "pose.csv"; angle = tmp_path / "angle.csv"
    write_pose(pose); write_gripper(angle)
    (tmp_path / "hardware.json").write_text(json.dumps({"robots": {}}), encoding="utf-8")
    spec = {
        "task": {"instruction": "demo"}, "start_s": 0, "end_s": 4, "frequency_hz": 20,
        "hardware_config": "hardware.json", "sync": {"uncertainty_s": 0.01},
        "robots": [{"name": "left", "trajectory_csv": "pose.csv", "gripper_csv": "angle.csv"}],
    }
    path = tmp_path / "episode.json"; path.write_text(json.dumps(spec), encoding="utf-8")
    metadata = build_episode(path, tmp_path / "out", skip_rgb=True)
    assert metadata["status"] == "DRAFT_HARDWARE_OR_QUALITY_PENDING"
    report = json.loads((tmp_path / "out" / "quality_report.json").read_text())
    assert "left.camera_to_tcp_verified" in report["failed"]


def test_ready_episode_writes_umi_zarr(tmp_path: Path, monkeypatch) -> None:
    pose = tmp_path / "pose.csv"; angle = tmp_path / "angle.csv"
    write_pose(pose); write_gripper(angle)
    hardware = {"robots": {"left": {
        "camera_to_tcp": {"translation_m": [0, 0, 0], "quaternion_xyzw": [0, 0, 0, 1]},
        "camera_to_tcp_verified": True,
        "gripper_width_calibration": {
            "type": "linear", "closed_angle_deg": 0, "closed_width_m": 0,
            "open_angle_deg": 40, "open_width_m": 0.08,
        },
        "gripper_width_verified": True,
    }}}
    (tmp_path / "hardware.json").write_text(json.dumps(hardware), encoding="utf-8")
    video = tmp_path / "video.mp4"; video.touch()
    spec = {
        "task": {"instruction": "close the gripper"}, "start_s": 0, "end_s": 4,
        "frequency_hz": 20, "hardware_config": "hardware.json",
        "sync": {"uncertainty_s": 0.01},
        "robots": [{
            "name": "left", "trajectory_csv": "pose.csv", "gripper_csv": "angle.csv",
            "video": "video.mp4", "observation": {"size": [8, 8]},
        }],
    }
    path = tmp_path / "episode.json"; path.write_text(json.dumps(spec), encoding="utf-8")
    monkeypatch.setattr(
        "vla_dataset_export.extract_rgb",
        lambda video, query_time, output_hw, view: np.zeros((len(query_time), *output_hw, 3), dtype=np.uint8),
    )
    output = tmp_path / "ready"
    metadata = build_episode(path, output)
    assert metadata["training_ready"]
    store = zarr.ZipStore(str(output / "dataset.zarr.zip"), mode="r")
    root = zarr.open_group(store=store, mode="r")
    assert root["data/camera0_rgb"].shape == (80, 8, 8, 3)
    assert root["data/robot0_eef_pos"].shape == (80, 3)
    assert root["meta/episode_ends"][:].tolist() == [80]
    store.close()
