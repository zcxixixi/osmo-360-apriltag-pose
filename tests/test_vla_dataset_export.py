import csv
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import zarr

from vla_dataset_export import _rot6d, build_episode, load_pose_csv, resample_pose
from world_frames import compile_world_tag_map


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


def _write_serial_bound_episode(tmp_path: Path, calibration_serial: str) -> Path:
    pose = tmp_path / "pose.csv"; angle = tmp_path / "angle.csv"
    write_pose(pose); write_gripper(angle)
    expected_serial = "95SXN-LEFT"
    (tmp_path / "camera_calibration.json").write_text(
        json.dumps({"serial": calibration_serial, "lenses": []}), encoding="utf-8"
    )
    hardware = {"robots": {"left": {
        "camera_serial": expected_serial,
        "source_view": "view2-left",
        "mount_revision": "mount-rev-20260824",
        "camera_to_tcp": {
            "translation_m": [0, 0, 0], "quaternion_xyzw": [0, 0, 0, 1],
        },
        "camera_to_tcp_verified": True,
        "gripper_width_calibration": {
            "type": "linear", "closed_angle_deg": 0, "closed_width_m": 0,
            "open_angle_deg": 40, "open_width_m": 0.08,
        },
        "gripper_width_verified": True,
    }}}
    (tmp_path / "hardware.json").write_text(json.dumps(hardware), encoding="utf-8")
    spec = {
        "task": {"instruction": "serial binding audit"},
        "start_s": 0, "end_s": 4, "frequency_hz": 20,
        "hardware_config": "hardware.json", "sync": {"uncertainty_s": 0.01},
        "robots": [{
            "name": "left", "trajectory_csv": "pose.csv", "gripper_csv": "angle.csv",
            "camera_serial": expected_serial,
            "camera_calibration": "camera_calibration.json",
        }],
    }
    spec_path = tmp_path / "episode.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    return spec_path


def test_camera_serial_binding_gate_accepts_matching_calibration(tmp_path: Path) -> None:
    spec_path = _write_serial_bound_episode(tmp_path, calibration_serial="95SXN-LEFT")
    output = tmp_path / "out"
    metadata = build_episode(spec_path, output, skip_rgb=True)
    report = json.loads((output / "quality_report.json").read_text())
    by_name = {check["name"]: check for check in report["checks"]}

    assert by_name["left.camera_serial_matches_hardware"]["pass"]
    assert by_name["left.camera_serial_matches_calibration"]["pass"]
    assert by_name["left.hardware_camera_serial_matches_calibration"]["pass"]
    assert by_name["left.camera_mount_revision_declared"]["pass"]
    assert not report["critical_failures"]
    assert metadata["robots"][0]["camera_serial"] == "95SXN-LEFT"


def test_camera_serial_binding_gate_rejects_swapped_calibration(tmp_path: Path) -> None:
    spec_path = _write_serial_bound_episode(tmp_path, calibration_serial="95SXN-RIGHT")
    output = tmp_path / "out"
    metadata = build_episode(spec_path, output, skip_rgb=True)
    report = json.loads((output / "quality_report.json").read_text())
    by_name = {check["name"]: check for check in report["checks"]}

    assert not by_name["left.camera_serial_matches_calibration"]["pass"]
    assert not by_name["left.hardware_camera_serial_matches_calibration"]["pass"]
    assert "left.camera_serial_matches_calibration" in report["critical_failures"]
    assert not metadata["training_ready"]
    assert report["umi_export"] == "blocked: invalid or uncalibrated common coordinate frame"


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


def test_dual_robot_world_mode_preserves_shared_positions(tmp_path: Path) -> None:
    tag_map = {
        "schema_version": "world-apriltag-map/1.0",
        "map_id": "synthetic-frozen",
        "calibration_status": "FROZEN",
        "world_frame": "room_world",
        "tags": [
            {"id": 128, "corners_m": [[0, 0, 0], [.2, 0, 0], [.2, .2, 0], [0, .2, 0]]},
            {"id": 134, "corners_m": [[0, 0, .4], [.2, 0, .4], [.2, .2, .4], [0, .2, .4]]},
        ],
    }
    map_path = tmp_path / "world.json"
    map_path.write_text(json.dumps(tag_map), encoding="utf-8")
    map_hash = compile_world_tag_map(map_path)["tag_map_sha256"]

    fields = [
        "frame", "timestamp", "camera_x_m", "camera_y_m", "camera_z_m",
        "qx", "qy", "qz", "qw", "measurement_source", "quality_status",
        "reprojection_rmse_px", "detected_ids", "parent_frame", "child_frame",
        "tag_map_sha256",
    ]
    for name, x0 in (("left", 0.2), ("right", 0.8)):
        with (tmp_path / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
            for frame in range(81):
                writer.writerow({
                    "frame": frame, "timestamp": frame / 20,
                    "camera_x_m": x0 + frame * .001, "camera_y_m": .3,
                    "camera_z_m": .2, "qx": 0, "qy": 0, "qz": 0, "qw": 1,
                    "measurement_source": "direct", "quality_status": "valid",
                    "reprojection_rmse_px": .4, "detected_ids": "128 134",
                    "parent_frame": "room_world", "child_frame": "panorama_camera",
                    "tag_map_sha256": map_hash,
                })
    angle = tmp_path / "angle.csv"; write_gripper(angle)
    hardware = {"robots": {name: {
        "camera_to_tcp": {
            "parent_frame": "panorama_camera", "child_frame": f"{name}_tcp",
            "translation_m": [0, 0, 0], "quaternion_xyzw": [0, 0, 0, 1],
        },
        "camera_to_tcp_verified": True,
        "gripper_width_calibration": {
            "type": "linear", "closed_angle_deg": 0, "closed_width_m": 0,
            "open_angle_deg": 40, "open_width_m": .08,
        },
        "gripper_width_verified": True,
    } for name in ("left", "right")}}
    (tmp_path / "hardware.json").write_text(json.dumps(hardware), encoding="utf-8")
    spec = {
        "task": {"instruction": "move object"}, "start_s": 0, "end_s": 4,
        "frequency_hz": 20, "hardware_config": "hardware.json",
        "coordinate_frame": {"mode": "world", "frame_id": "room_world", "tag_map": "world.json"},
        "workspace": {"type": "tabletop"}, "sync": {"uncertainty_s": .005},
        "robots": [
            {"name": name, "trajectory_csv": f"{name}.csv", "gripper_csv": "angle.csv"}
            for name in ("left", "right")
        ],
    }
    spec_path = tmp_path / "episode.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    output = tmp_path / "world-output"
    metadata = build_episode(spec_path, output, skip_rgb=True)
    arrays = np.load(output / "episode_arrays.npz")
    assert metadata["coordinate_frame"]["mode"] == "world"
    assert arrays["robot0_eef_pos"][0, 0] == pytest.approx(.2, abs=.01)
    assert arrays["robot1_eef_pos"][0, 0] == pytest.approx(.8, abs=.01)
    np.testing.assert_allclose(arrays["robot0_eef_delta_from_start_pos"][0], 0, atol=1e-6)
    report = json.loads((output / "quality_report.json").read_text())
    assert not report["critical_failures"]
