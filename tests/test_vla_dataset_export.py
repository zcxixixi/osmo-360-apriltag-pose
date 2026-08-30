import csv
import json
from pathlib import Path

import numpy as np
import pytest
import cv2
from scipy.spatial.transform import Rotation
import zarr

from osmo360.datasets.vla_dataset_export import (
    _compose_hardware_camera_to_tcp, _hardware_role, _rot6d, apply_camera_to_tcp,
    audit_multicamera_pair, build_episode, extract_rgb, load_gripper, load_pose_csv,
    resample_pose,
)
from osmo360.localization.world_frames import compile_world_tag_map


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


def test_gripper_loader_accepts_audio_aligned_common_time(tmp_path: Path) -> None:
    path = tmp_path / "opening.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["common_time_s", "opening_angle_deg", "measured"]
        )
        writer.writeheader()
        writer.writerows([
            {"common_time_s": 3.0, "opening_angle_deg": 4.0, "measured": 1},
            {"common_time_s": 4.0, "opening_angle_deg": 14.0, "measured": 0},
        ])
    angle, _, measured = load_gripper(path, np.asarray([3.0, 3.5, 4.0]), None)
    np.testing.assert_allclose(angle, [4.0, 9.0, 14.0])
    assert measured.tolist() == [True, False, False]


def test_semantic_hand_can_bind_the_opposite_serialized_hardware_slot() -> None:
    assert _hardware_role({"name": "left", "hardware_role": "right"}, "left") == "right"
    assert _hardware_role({"name": "right"}, "right") == "right"


def test_raw_fisheye_observation_is_resized_without_equirectangular_projection(
    tmp_path: Path, monkeypatch,
) -> None:
    bgr = np.zeros((6, 8, 3), dtype=np.uint8)
    bgr[..., 0] = np.arange(8, dtype=np.uint8)[None, :] * 20
    bgr[..., 1] = np.arange(6, dtype=np.uint8)[:, None] * 30
    bgr[..., 2] = 111

    class FakeCapture:
        def isOpened(self): return True
        def get(self, _): return 30.0
        def set(self, *_): return True
        def read(self): return True, bgr.copy()
        def release(self): pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: FakeCapture())
    result = extract_rgb(
        tmp_path / "stream1.mp4", np.asarray([0.0]), (3, 4),
        {"projection": "raw_resize"},
    )
    expected = cv2.resize(
        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (4, 3), interpolation=cv2.INTER_AREA,
    )
    np.testing.assert_array_equal(result[0], expected)


def test_mount_tag2_can_be_the_eef_reference_without_tcp_lever() -> None:
    position = np.asarray([[1.0, 2.0, 3.0]])
    rotation = Rotation.identity(1)
    transformed, transformed_rotation = apply_camera_to_tcp(position, rotation, {
        "parent_frame": "panorama_camera", "child_frame": "left_mount_tag2",
        "translation_m": [0.04, 0.07, 0.0],
        "quaternion_xyzw": [0, 0, 0, 1],
    })
    np.testing.assert_allclose(transformed, [[1.04, 2.07, 3.0]])
    np.testing.assert_allclose(transformed_rotation.as_matrix(), np.eye(3)[None])


def test_mount_tag3_can_be_the_eef_reference_after_id_migration() -> None:
    position = np.asarray([[1.0, 2.0, 3.0]])
    transformed, _ = apply_camera_to_tcp(position, Rotation.identity(1), {
        "parent_frame": "panorama_camera", "child_frame": "left_mount_tag3",
        "translation_m": [0.01, 0.02, 0.03],
        "quaternion_xyzw": [0, 0, 0, 1],
    })
    np.testing.assert_allclose(transformed, [[1.01, 2.02, 3.03]])


def test_pair_integrity_rejects_sequential_repetitions(tmp_path: Path) -> None:
    for name, creation in (("left", "2026-08-24T11:11:12Z"),
                           ("right", "2026-08-24T11:10:38Z")):
        (tmp_path / f"{name}.json").write_text(json.dumps({
            "creation_time_utc": creation, "duration_s": 21.0,
        }), encoding="utf-8")
    spec = {
        "start_s": .5, "end_s": 20.5,
        "sync": {
            "method": "audio_cross_correlation_diagnostic",
            "correlation": .235, "uncertainty_s": .05,
            "creation_time_clock_aligned": True,
        },
    }
    robots = [
        {"name": "left", "source_info": "left.json"},
        {"name": "right", "source_info": "right.json"},
    ]
    audit = audit_multicamera_pair(tmp_path / "episode.json", spec, robots)
    assert not audit["valid"]
    assert audit["status"] == "INVALID_SEQUENTIAL_CAPTURE"
    assert audit["creation_time_overlap_s"] == 0.0


def test_pair_integrity_accepts_strong_audio_sync_with_unaligned_clocks(tmp_path: Path) -> None:
    for name, creation in (("left", "2026-08-24T11:11:12Z"),
                           ("right", "2026-08-24T11:10:38Z")):
        (tmp_path / f"{name}.json").write_text(json.dumps({
            "creation_time_utc": creation, "duration_s": 21.0,
        }), encoding="utf-8")
    spec = {
        "start_s": 0.0, "end_s": 20.0,
        "sync": {
            "method": "audio_cross_correlation",
            "correlation": .91, "uncertainty_s": .008,
            "creation_time_clock_aligned": False,
        },
    }
    robots = [
        {"name": "left", "source_info": "left.json"},
        {"name": "right", "source_info": "right.json"},
    ]
    audit = audit_multicamera_pair(tmp_path / "episode.json", spec, robots)
    assert audit["valid"]
    assert audit["evidence"]["audio"]


def test_hardware_chain_composes_once_and_rejects_conflicting_tcp_alias() -> None:
    robot = {
        "camera_to_base": {
            "parent_frame": "panorama_camera", "child_frame": "base_link",
            "translation_m": [1, 0, 0], "quaternion_xyzw": [0, 0, 0, 1],
        },
        "base_to_tcp": {
            "parent_frame": "base_link", "child_frame": "left_tcp",
            "translation_m": [.1356, 0, .0101], "quaternion_xyzw": [0, 0, 0, 1],
        },
    }
    composed = _compose_hardware_camera_to_tcp(robot)
    np.testing.assert_allclose(composed["translation_m"], [1.1356, 0, .0101])
    robot["camera_to_tcp"] = {
        "parent_frame": "panorama_camera", "child_frame": "left_tcp",
        "translation_m": [1, 0, 0], "quaternion_xyzw": [0, 0, 0, 1],
    }
    with pytest.raises(ValueError, match="conflicts"):
        _compose_hardware_camera_to_tcp(robot)


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


def test_raw_fisheye_unit_bearing_is_a_direct_trusted_measurement(tmp_path: Path) -> None:
    pose = tmp_path / "raw_pose.csv"
    pose.write_text(
        "timestamp,camera_x_m,camera_y_m,camera_z_m,qx,qy,qz,qw,measurement_source,quality_status\n"
        "0,0,0,1,0,0,0,1,raw_fisheye_unit_bearing,valid\n"
        "1,0.1,0,1,0,0,0,1,raw_fisheye_unit_bearing_edge_rectified,valid\n",
        encoding="utf-8",
    )
    series = load_pose_csv(pose)
    assert series.direct.tolist() == [True, True]
    assert series.tracked.tolist() == [True, True]


def test_verified_episode_exports_canonical_arrays(tmp_path: Path) -> None:
    pose = tmp_path / "pose.csv"; angle = tmp_path / "angle.csv"
    write_pose(pose); write_gripper(angle)
    hardware = {
        "calibration_status": "verified",
        "robots": {name: {
            "camera_to_tcp": {
                "parent_frame": "panorama_camera", "child_frame": f"{name}_tcp",
                "translation_m": [0, 0, 0], "quaternion_xyzw": [0, 0, 0, 1],
            },
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
            "parent_frame": "panorama_camera", "child_frame": "left_tcp",
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
        "camera_to_tcp": {
            "parent_frame": "panorama_camera", "child_frame": "left_tcp",
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
        "osmo360.datasets.vla_dataset_export.extract_rgb",
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
        "frame_convention": {"up_axis": "+Z", "up_vector": [0, 0, 1]},
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
        "base_tag_id": 2 if name == "left" else 3,
        "camera_to_base": {
            "parent_frame": "panorama_camera", "child_frame": "base_link",
            "translation_m": [0, 0, 0], "quaternion_xyzw": [0, 0, 0, 1],
        },
        "base_to_tcp": {
            "parent_frame": "base_link", "child_frame": f"{name}_tcp",
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
        "workspace": {
            "type": "tabletop", "up_vector": [0, 0, 1],
            "table_plane_status": "CALIBRATED", "table_plane_offset_m": 0.0,
        }, "sync": {"uncertainty_s": .005},
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
    by_name = {item["name"]: item for item in report["checks"]}
    assert by_name["hardware.base_tag_ids_disjoint_from_world_map"]["pass"]
    assert not report["critical_failures"]
