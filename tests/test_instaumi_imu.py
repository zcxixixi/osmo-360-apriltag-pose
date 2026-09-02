from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from osmo360.localization.cached_a3_bootstrap import write_joint_pose_csv
from osmo360.localization.coordinate_frames import (
    X5_STREAM0_OPENCV_FROM_HAND_CAMERA_FLU,
)
from osmo360.localization.instaumi_imu import (
    IMU_ROTATION_BASELINE_PATH,
    ImuSeries,
    calibrate_instaumi_imu_from_visual,
    load_instaumi_imu,
)


def _transform(rotation: np.ndarray) -> list[list[float]]:
    value = np.eye(4)
    value[:3, :3] = rotation
    return value.tolist()


def _h5(tmp_path: Path, *, per_side: bool) -> Path:
    path = tmp_path / "dataset.h5"
    calibration = {
        "extrinsics": {
            "transform_convention": "T_target_source",
            "T_rig_camera_left": _transform(np.eye(3)),
            "T_rig_camera_right": _transform(np.eye(3)),
            "T_rig_imu_left": _transform(X5_STREAM0_OPENCV_FROM_HAND_CAMERA_FLU),
            "T_rig_imu_right": _transform(X5_STREAM0_OPENCV_FROM_HAND_CAMERA_FLU),
        },
        "imu_calibration": {
            "gyroscope_bias_rad_s": [0, 0, 0],
            "gyroscope_scale": [1, 1, 1],
        },
    }
    string = h5py.string_dtype(encoding="utf-8")
    timestamps = np.arange(0, 0.401, 0.01)
    angular_velocity = np.tile([0.0, 0.0, np.pi], (len(timestamps), 1))
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "/metadata/dataset.json",
            data=json.dumps({"time": {"reference": "dataset_start"}}),
            dtype=string,
        )
        handle.create_dataset(
            "/calib/calibration_full.json",
            data=json.dumps(calibration),
            dtype=string,
        )
        if per_side:
            for side in ("left", "right"):
                base = f"/sensor/imu/{side}"
                handle.create_dataset(
                    f"{base}/timestamp_ns", data=np.rint(timestamps * 1e9).astype(np.int64)
                )
                handle.create_dataset(f"{base}/angular_velocity", data=angular_velocity)
                handle.create_dataset(f"{base}/valid", data=np.ones(len(timestamps), bool))
        else:
            handle.create_dataset("/sensor/imu/timestamp_ns", data=np.empty(0, np.int64))
            handle.create_dataset("/sensor/imu/angular_velocity", data=np.empty((0, 3)))
            handle.create_dataset("/sensor/imu/valid", data=np.empty(0, bool))
    return path


def _visual_self_calibration_fixture(
    tmp_path: Path,
) -> tuple[Path, dict[str, list[dict[str, str | int]]], dict[str, float]]:
    path = tmp_path / "self-calibration.h5"
    raw_time = np.arange(-1.0, 11.001, 0.001)

    def raw_gyro(time_s: np.ndarray) -> np.ndarray:
        return np.column_stack((
            0.8 * np.sin(1.3 * time_s) + 0.4 * np.cos(2.1 * time_s),
            0.7 * np.cos(0.9 * time_s) + 0.3 * np.sin(2.7 * time_s),
            0.6 * np.sin(1.7 * time_s) + 0.2 * np.cos(0.5 * time_s),
        ))

    angular_velocity = raw_gyro(raw_time)
    true_rotation = Rotation.from_euler("xyz", [2.0, -90.0, 1.0], degrees=True)
    bias_hand = np.asarray([0.01, -0.02, 0.005])
    offsets = {"left": -0.30, "right": 0.012}
    rows: dict[str, list[dict[str, str | int]]] = {}
    video_time = np.arange(0.0, 10.001, 1 / 30)
    for side, offset_s in offsets.items():
        rotation = Rotation.identity()
        side_rows = []
        for frame, time_s in enumerate(video_time):
            quaternion = rotation.as_quat()
            side_rows.append({
                "frame": frame,
                "timestamp": f"{time_s:.12f}",
                "qx": f"{quaternion[0]:.15f}",
                "qy": f"{quaternion[1]:.15f}",
                "qz": f"{quaternion[2]:.15f}",
                "qw": f"{quaternion[3]:.15f}",
                "quality_status": "valid",
                "inlier_tag_count": 6,
            })
            if frame + 1 < len(video_time):
                midpoint = 0.5 * (time_s + video_time[frame + 1]) + offset_s
                omega_hand = true_rotation.apply(raw_gyro(np.asarray([midpoint]))[0])
                omega_hand += bias_hand
                rotation = rotation * Rotation.from_rotvec(
                    omega_hand * (video_time[frame + 1] - time_s)
                )
        rows[side] = side_rows
    string = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "/metadata/dataset.json",
            data=json.dumps({"time": {"reference": "dataset_start"}}),
            dtype=string,
        )
        for side in ("left", "right"):
            base = f"/sensor/imu/{side}"
            handle.create_dataset(
                f"{base}/timestamp_ns",
                data=np.rint(raw_time * 1e9).astype(np.int64),
            )
            handle.create_dataset(f"{base}/angular_velocity", data=angular_velocity)
            handle.create_dataset(
                f"{base}/valid", data=np.ones(len(raw_time), dtype=bool)
            )
    return path, rows, offsets


def _row(frame: int, time_s: float, x: float | None, yaw_deg: float = 0) -> dict[str, str | int]:
    valid = x is not None
    quaternion = Rotation.from_euler("z", yaw_deg, degrees=True).as_quat()
    return {
        "frame": frame,
        "timestamp": f"{time_s:.6f}",
        "camera_x_m": "" if x is None else str(x),
        "camera_y_m": "" if x is None else "0",
        "camera_z_m": "" if x is None else "0",
        "qx": "" if x is None else str(quaternion[0]),
        "qy": "" if x is None else str(quaternion[1]),
        "qz": "" if x is None else str(quaternion[2]),
        "qw": "" if x is None else str(quaternion[3]),
        "quality_status": "valid" if valid else "angular_rmse_rejected",
        "angular_rmse_deg": "0.2" if valid else "2.5",
        "detected_tag_count": 4,
        "inlier_tag_count": 4 if valid else 0,
        "measurement_source": "cached_raw_fisheye_bearing_direct",
    }


def test_calibrated_gyro_bridge_shapes_rotation_but_keeps_visual_position(tmp_path: Path):
    bundle = load_instaumi_imu(_h5(tmp_path, per_side=True))
    assert bundle.audit["status"] == "AVAILABLE"
    assert set(bundle.streams) == {"left", "right"}

    left = [_row(0, 0.0, 0, 0), _row(1, 0.2, None), _row(2, 0.4, 2, 80)]
    right = [_row(0, 0.0, 5), _row(1, 0.2, 6), _row(2, 0.4, 7)]
    output = tmp_path / "joint.csv"
    summary = write_joint_pose_csv(
        output,
        left,
        right,
        map_id="shared-map",
        imu_streams=bundle.streams,
        imu_audit=bundle.audit,
    )
    rows = list(csv.DictReader(output.open(newline="", encoding="utf-8")))

    recovered = Rotation.from_quat([
        float(rows[1][f"left_q{axis}"]) for axis in ("x", "y", "z", "w")
    ])
    assert rows[1]["left_pose_state"] == "IMU_ASSISTED_UNTRUSTED"
    assert rows[1]["left_quality_status"] == "imu_assisted_untrusted"
    assert float(rows[1]["left_camera_x_m"]) == pytest.approx(1.0)
    assert recovered.as_euler("xyz", degrees=True)[2] == pytest.approx(40.0, abs=0.05)
    assert summary["joint_valid_ratio"] == 2 / 3
    assert summary["imu_assistance"]["assisted_side_frames"] == {
        "left": 1,
        "right": 0,
    }
    assert summary["imu_assistance"]["translation_source"] == (
        "visual_endpoint_anchored_timestamp_aligned_accelerometer_bridge"
        "_when_available"
    )


def test_gyro_prediction_propagates_visual_orientation(tmp_path: Path):
    bundle = load_instaumi_imu(_h5(tmp_path, per_side=True))

    prediction = bundle.streams["left"].predict_orientation(
        0.0,
        Rotation.identity(),
        0.2,
    )

    assert prediction.rotation.as_euler("xyz", degrees=True)[2] == pytest.approx(
        36.0,
        abs=0.05,
    )
    assert prediction.maximum_sample_gap_s == pytest.approx(0.01)


def test_short_visual_gap_uses_trusted_gyro_bridge(tmp_path: Path):
    bundle = load_instaumi_imu(_h5(tmp_path, per_side=True))
    left = [_row(0, 0.0, 0, 0), _row(1, 0.1, None), _row(2, 0.2, 2, 40)]
    right = [_row(0, 0.0, 5), _row(1, 0.1, 6), _row(2, 0.2, 7)]
    output = tmp_path / "joint.csv"

    summary = write_joint_pose_csv(
        output,
        left,
        right,
        map_id="shared-map",
        imu_streams=bundle.streams,
        imu_audit=bundle.audit,
    )
    rows = list(csv.DictReader(output.open(newline="", encoding="utf-8")))

    assert rows[1]["left_pose_state"] == "IMU_ASSISTED"
    assert rows[1]["left_quality_status"] == "imu_assisted"
    assert rows[1]["joint_valid"] == "true"
    assert summary["joint_valid_ratio"] == 1.0
    assert summary["imu_assistance"]["trusted_assisted_side_frames"] == {
        "left": 1,
        "right": 0,
    }


def test_empty_shared_imu_is_audited_and_never_assigned_to_both_hands(tmp_path: Path):
    bundle = load_instaumi_imu(_h5(tmp_path, per_side=False))

    assert bundle.streams == {}
    assert bundle.audit["status"] == "UNAVAILABLE_NO_SAMPLES"
    assert bundle.audit["shared_stream_used"] is False
    assert bundle.audit["singular_shared_imu_sample_count"] == 0


def test_visual_self_calibration_recovers_per_side_imu_time_and_axes(
    tmp_path: Path,
) -> None:
    path, rows, offsets = _visual_self_calibration_fixture(tmp_path)

    bundle = calibrate_instaumi_imu_from_visual(path, rows)

    assert bundle.audit["status"] == "VISUAL_SELF_CALIBRATED_PASS"
    assert set(bundle.streams) == {"left", "right"}
    assert bundle.audit["cross_side_extrinsic_disagreement_deg"] < 1.0
    for side in ("left", "right"):
        audit = bundle.audit["sides"][side]
        assert audit["time_offset_s_raw_imu_minus_visual"] == pytest.approx(
            offsets[side], abs=0.003
        )
        assert audit["holdout_rotation_error_deg"]["p95"] < 0.1


def test_identity_h5_extrinsics_use_serial_bound_rotation_baseline_and_timestamps(
    tmp_path: Path,
) -> None:
    path = _h5(tmp_path, per_side=True)
    with h5py.File(path, "r+") as handle:
        metadata = {
            "time": {"reference": "dataset_start"},
            "devices": {
                "left": {"serial_number": "IAHEA2606KMDGP"},
                "right": {"serial_number": "IAHEA2606KMURQ"},
            },
        }
        calibration = {
            "extrinsics": {
                "transform_convention": "T_target_source",
                "T_rig_camera_left": np.eye(4).tolist(),
                "T_rig_camera_right": np.eye(4).tolist(),
                "T_rig_imu_left": np.eye(4).tolist(),
                "T_rig_imu_right": np.eye(4).tolist(),
            },
            "imu_calibration": {
                side: {
                    "gyroscope_bias_rad_s": None,
                    "gyroscope_scale": None,
                    "accelerometer_bias_m_s2": None,
                    "accelerometer_scale": None,
                }
                for side in ("left", "right")
            },
        }
        handle["/metadata/dataset.json"][()] = json.dumps(metadata)
        handle["/calib/calibration_full.json"][()] = json.dumps(calibration)

    bundle = load_instaumi_imu(path)

    baseline = json.loads(IMU_ROTATION_BASELINE_PATH.read_text(encoding="utf-8"))
    assert bundle.audit["status"] == "AVAILABLE"
    assert bundle.audit["time_alignment"] == {
        "method": "linear_interpolation_by_h5_timestamp_ns",
        "reference": "dataset_start",
        "fixed_time_offset_s": 0.0,
        "frame_index_alignment_used": False,
    }
    for side, serial in (
        ("left", "IAHEA2606KMDGP"),
        ("right", "IAHEA2606KMURQ"),
    ):
        audit = bundle.audit["sides"][side]
        expected_rotation = np.asarray(
            baseline["devices"][serial]
            ["T_hand_camera_flu_back_x_imu_rotation_only"]
        )[:3, :3]
        assert audit["orientation_calibration_source"] == (
            "serial_bound_visual_gyro_baseline"
        )
        assert audit["placeholder_identity_extrinsics_replaced"] is True
        assert audit["gyroscope_bias_source"] == "safe_default"
        assert np.allclose(
            bundle.streams[side].timestamp_s,
            np.arange(0, 0.401, 0.01),
            atol=1e-12,
            rtol=0,
        )
        assert bundle.streams[side].angular_velocity_hand_rad_s[0] == pytest.approx(
            expected_rotation @ np.asarray([0.0, 0.0, np.pi])
        )


def test_explicit_non_identity_calibration_full_extrinsics_override_baseline(
    tmp_path: Path,
) -> None:
    path = _h5(tmp_path, per_side=True)
    with h5py.File(path, "r+") as handle:
        raw = handle["/metadata/dataset.json"][()]
        metadata = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
        metadata["devices"] = {
            "left": {"serial_number": "IAHEA2606KMDGP"},
            "right": {"serial_number": "IAHEA2606KMURQ"},
        }
        handle["/metadata/dataset.json"][()] = json.dumps(metadata)

    bundle = load_instaumi_imu(path)

    for side in ("left", "right"):
        assert bundle.audit["sides"][side]["orientation_calibration_source"] == (
            "calibration_full_explicit_extrinsics"
        )
        assert bundle.audit["sides"][side][
            "placeholder_identity_extrinsics_replaced"
        ] is False
        assert bundle.streams[side].angular_velocity_hand_rad_s[0] == pytest.approx(
            [0.0, 0.0, np.pi]
        )


def test_accelerometer_bridge_keeps_visual_endpoints_and_changes_gap_shape() -> None:
    timestamps = np.arange(0.0, 1.001, 0.01)
    acceleration = np.zeros((len(timestamps), 3))
    acceleration[:, 0] = 4.0 * np.sin(2 * np.pi * timestamps)
    stream = ImuSeries(
        side="left",
        timestamp_s=timestamps,
        angular_velocity_hand_rad_s=np.zeros((len(timestamps), 3)),
        calibration_sha256="test",
        dataset_path="test.h5",
        linear_acceleration_hand_m_s2=acceleration,
    )

    bridge = stream.bridge_positions(
        0.0,
        np.asarray([0.0, 0.0, 0.0]),
        Rotation.identity(),
        1.0,
        np.asarray([1.0, 0.0, 0.0]),
        Rotation.identity(),
        np.asarray([0.0, 0.25, 1.0]),
    )

    assert bridge.positions_m[0] == pytest.approx([0.0, 0.0, 0.0])
    assert bridge.positions_m[-1] == pytest.approx([1.0, 0.0, 0.0])
    assert bridge.positions_m[1, 0] != pytest.approx(0.5, abs=1e-3)
    assert bridge.maximum_deviation_from_linear_m <= 0.15


def test_joint_gap_uses_accelerometer_translation_with_visual_scale_anchors(
    tmp_path: Path,
) -> None:
    timestamps = np.arange(0.0, 1.001, 0.01)
    acceleration = np.zeros((len(timestamps), 3))
    acceleration[:, 0] = 4.0 * np.sin(2 * np.pi * timestamps)
    stream = ImuSeries(
        side="left",
        timestamp_s=timestamps,
        angular_velocity_hand_rad_s=np.zeros((len(timestamps), 3)),
        calibration_sha256="test",
        dataset_path="test.h5",
        linear_acceleration_hand_m_s2=acceleration,
    )
    left = [_row(0, 0.0, 0.0), _row(1, 0.25, None), _row(2, 1.0, 1.0)]
    right = [_row(0, 0.0, 2.0), _row(1, 0.25, 2.25), _row(2, 1.0, 3.0)]
    output = tmp_path / "joint.csv"

    summary = write_joint_pose_csv(
        output,
        left,
        right,
        map_id="shared-map",
        imu_streams={"left": stream},
        imu_audit={"status": "AVAILABLE"},
    )
    rows = list(csv.DictReader(output.open(newline="", encoding="utf-8")))

    assert rows[1]["left_pose_state"] == "IMU_ASSISTED_UNTRUSTED"
    assert float(rows[1]["left_camera_x_m"]) != pytest.approx(0.25, abs=1e-3)
    assert "accelerometer_translation" in rows[1]["left_measurement_source"]
    assert summary["imu_assistance"]["accelerometer_assisted_side_frames"] == {
        "left": 1,
        "right": 0,
    }
