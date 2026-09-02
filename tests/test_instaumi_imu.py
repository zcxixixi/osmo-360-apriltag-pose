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
from osmo360.localization.instaumi_imu import load_instaumi_imu


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
        "visual_endpoint_interpolation"
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
