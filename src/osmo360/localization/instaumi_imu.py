"""Fail-closed InstaUMI IMU loading and bounded gyro orientation bridging."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from osmo360.localization.coordinate_frames import (
    X5_STREAM0_OPENCV_FROM_HAND_CAMERA_FLU,
)


MAXIMUM_IMU_SAMPLE_GAP_S = 0.05


class ImuAssistanceUnavailable(RuntimeError):
    """Raised when a requested visual gap is not safely covered by the IMU."""


@dataclass(frozen=True)
class GyroBridge:
    rotations: Rotation
    maximum_sample_gap_s: float
    endpoint_closure_deg: float


@dataclass(frozen=True)
class ImuSeries:
    side: str
    timestamp_s: np.ndarray
    angular_velocity_hand_rad_s: np.ndarray
    calibration_sha256: str
    dataset_path: str

    def bridge_orientations(
        self,
        start_s: float,
        start_rotation: Rotation,
        end_s: float,
        end_rotation: Rotation,
        query_s: np.ndarray,
        *,
        maximum_sample_gap_s: float = MAXIMUM_IMU_SAMPLE_GAP_S,
    ) -> GyroBridge:
        """Gyro-propagate between visual anchors and close exactly at the end.

        Gyro increments provide the short-gap rotation shape.  A geodesic
        endpoint correction distributes accumulated bias error so both trusted
        visual anchors remain exact.  Translation is intentionally outside this
        method: the caller retains bounded visual endpoint interpolation.
        """
        query = np.asarray(query_s, dtype=np.float64).reshape(-1)
        if not np.isfinite([start_s, end_s, maximum_sample_gap_s]).all():
            raise ValueError("IMU bridge bounds must be finite")
        if end_s <= start_s or maximum_sample_gap_s <= 0:
            raise ValueError("invalid IMU bridge interval")
        if np.any(~np.isfinite(query)) or np.any(query < start_s) or np.any(query > end_s):
            raise ValueError("IMU bridge queries must lie inside the visual anchors")
        timestamps = self.timestamp_s
        if timestamps[0] > start_s or timestamps[-1] < end_s:
            raise ImuAssistanceUnavailable("imu_does_not_cover_visual_gap")
        lower = max(0, int(np.searchsorted(timestamps, start_s, side="right")) - 1)
        upper = min(len(timestamps), int(np.searchsorted(timestamps, end_s, side="left")) + 1)
        covered_times = timestamps[lower:upper]
        if len(covered_times) < 2:
            raise ImuAssistanceUnavailable("insufficient_imu_samples")
        largest_gap = float(np.max(np.diff(covered_times)))
        if largest_gap > maximum_sample_gap_s + 1e-12:
            raise ImuAssistanceUnavailable(
                f"imu_sample_gap_exceeds_limit:{largest_gap:.9f}"
            )

        inside = timestamps[(timestamps > start_s) & (timestamps < end_s)]
        nodes = np.unique(np.concatenate((
            np.asarray([start_s, end_s]), inside, query,
        )))
        omega = np.column_stack([
            np.interp(nodes, timestamps, self.angular_velocity_hand_rad_s[:, axis])
            for axis in range(3)
        ])
        propagated: list[Rotation] = [start_rotation]
        for index in range(1, len(nodes)):
            delta_s = float(nodes[index] - nodes[index - 1])
            midpoint_omega = 0.5 * (omega[index - 1] + omega[index])
            propagated.append(
                propagated[-1] * Rotation.from_rotvec(midpoint_omega * delta_s)
            )
        predicted_end = propagated[-1]
        endpoint_closure = predicted_end.inv() * end_rotation
        closure_vector = endpoint_closure.as_rotvec()
        duration_s = end_s - start_s
        corrected = [
            rotation * Rotation.from_rotvec(
                closure_vector * ((float(time_s) - start_s) / duration_s)
            )
            for time_s, rotation in zip(nodes, propagated)
        ]
        by_time = {float(time_s): rotation for time_s, rotation in zip(nodes, corrected)}
        result = Rotation.from_quat(np.asarray([
            by_time[float(time_s)].as_quat() for time_s in query
        ]))
        return GyroBridge(
            rotations=result,
            maximum_sample_gap_s=largest_gap,
            endpoint_closure_deg=float(np.degrees(endpoint_closure.magnitude())),
        )


@dataclass(frozen=True)
class ImuBundle:
    streams: dict[str, ImuSeries]
    audit: dict[str, Any]


def _json_dataset(handle: h5py.File, key: str) -> tuple[str, dict[str, Any]]:
    if key not in handle:
        raise ImuAssistanceUnavailable(f"missing_calibration:{key}")
    raw = handle[key][()]
    text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    return text, json.loads(text)


def _rotation(extrinsics: dict[str, Any], key: str) -> np.ndarray:
    if key not in extrinsics:
        raise ImuAssistanceUnavailable(f"missing_extrinsic:{key}")
    transform = np.asarray(extrinsics[key], dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ImuAssistanceUnavailable(f"invalid_extrinsic:{key}")
    rotation = transform[:3, :3]
    if (
        not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
    ):
        raise ImuAssistanceUnavailable(f"non_rigid_extrinsic:{key}")
    return rotation


def _calibration_vector(
    calibration: dict[str, Any], side: str, key: str, default: list[float]
) -> np.ndarray:
    imu_calibration = calibration.get("imu_calibration", {})
    side_calibration = imu_calibration.get(side, imu_calibration)
    value = np.asarray(side_calibration.get(key, default), dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ImuAssistanceUnavailable(f"invalid_imu_calibration:{side}:{key}")
    return value


def load_instaumi_imu(path: Path) -> ImuBundle:
    """Load only independently bound left/right IMUs from an InstaUMI H5.

    A singular ``/sensor/imu`` stream is intentionally not assigned to two
    independently moving hand cameras.  Future datasets must use
    ``/sensor/imu/{left,right}`` (or ``/sensor/{left,right}/imu``) together
    with ``T_rig_imu_left`` and ``T_rig_imu_right``.
    """
    source = path.resolve(strict=True)
    streams: dict[str, ImuSeries] = {}
    side_audit: dict[str, Any] = {}
    with h5py.File(source, "r") as handle:
        calibration_text, calibration = _json_dataset(
            handle, "/calib/calibration_full.json"
        )
        calibration_sha256 = hashlib.sha256(calibration_text.encode()).hexdigest()
        extrinsics = calibration.get("extrinsics", {})
        if extrinsics.get("transform_convention") != "T_target_source":
            convention_status = "UNAVAILABLE_TRANSFORM_CONVENTION"
        else:
            convention_status = "OK"
        for side in ("left", "right"):
            bases = (f"/sensor/imu/{side}", f"/sensor/{side}/imu")
            base = next((candidate for candidate in bases if candidate in handle), None)
            if base is None:
                side_audit[side] = {"status": "UNAVAILABLE_PER_SIDE_STREAM_MISSING"}
                continue
            required = ("timestamp_ns", "angular_velocity", "valid")
            missing = [name for name in required if f"{base}/{name}" not in handle]
            if missing:
                side_audit[side] = {
                    "status": "UNAVAILABLE_DATASET_MISSING",
                    "missing": missing,
                    "dataset_path": base,
                }
                continue
            timestamp_s = np.asarray(handle[f"{base}/timestamp_ns"], dtype=np.float64) / 1e9
            angular_velocity = np.asarray(
                handle[f"{base}/angular_velocity"], dtype=np.float64
            )
            valid = np.asarray(handle[f"{base}/valid"], dtype=bool)
            if (
                angular_velocity.shape != (len(timestamp_s), 3)
                or valid.shape != (len(timestamp_s),)
            ):
                side_audit[side] = {
                    "status": "UNAVAILABLE_SHAPE_MISMATCH",
                    "dataset_path": base,
                }
                continue
            finite = np.isfinite(timestamp_s) & np.isfinite(angular_velocity).all(axis=1)
            keep = valid & finite
            timestamp_s = timestamp_s[keep]
            angular_velocity = angular_velocity[keep]
            if len(timestamp_s) < 2:
                side_audit[side] = {
                    "status": "UNAVAILABLE_NO_SAMPLES",
                    "sample_count": int(len(timestamp_s)),
                    "dataset_path": base,
                }
                continue
            if np.any(np.diff(timestamp_s) <= 0):
                side_audit[side] = {
                    "status": "UNAVAILABLE_NON_MONOTONIC_TIMESTAMPS",
                    "dataset_path": base,
                }
                continue
            if convention_status != "OK":
                side_audit[side] = {"status": convention_status, "dataset_path": base}
                continue
            try:
                rig_from_camera = _rotation(extrinsics, f"T_rig_camera_{side}")
                rig_from_imu = _rotation(extrinsics, f"T_rig_imu_{side}")
                bias = _calibration_vector(
                    calibration, side, "gyroscope_bias_rad_s", [0, 0, 0]
                )
                scale = _calibration_vector(
                    calibration, side, "gyroscope_scale", [1, 1, 1]
                )
                if np.any(scale <= 0):
                    raise ImuAssistanceUnavailable(
                        f"invalid_imu_calibration:{side}:gyroscope_scale"
                    )
            except ImuAssistanceUnavailable as exc:
                side_audit[side] = {
                    "status": "UNAVAILABLE_CALIBRATION_INVALID",
                    "reason": str(exc),
                    "dataset_path": base,
                }
                continue
            camera_from_imu = rig_from_camera.T @ rig_from_imu
            hand_from_imu = (
                X5_STREAM0_OPENCV_FROM_HAND_CAMERA_FLU.T @ camera_from_imu
            )
            corrected = (angular_velocity - bias) * scale
            angular_velocity_hand = (hand_from_imu @ corrected.T).T
            stream = ImuSeries(
                side=side,
                timestamp_s=timestamp_s,
                angular_velocity_hand_rad_s=angular_velocity_hand,
                calibration_sha256=calibration_sha256,
                dataset_path=base,
            )
            streams[side] = stream
            side_audit[side] = {
                "status": "AVAILABLE",
                "sample_count": int(len(timestamp_s)),
                "maximum_sample_gap_s": float(np.max(np.diff(timestamp_s))),
                "dataset_path": base,
                "output_child_frame": "hand_camera_flu_back_x",
            }

        singular_count = 0
        singular_key = "/sensor/imu/angular_velocity"
        if singular_key in handle:
            singular_count = int(handle[singular_key].shape[0])
    if len(streams) == 2:
        status = "AVAILABLE"
    elif streams:
        status = "PARTIAL_PER_SIDE_IMU"
    elif singular_count == 0:
        status = "UNAVAILABLE_NO_SAMPLES"
    else:
        status = "UNAVAILABLE_AMBIGUOUS_SHARED_IMU"
    return ImuBundle(
        streams=streams,
        audit={
            "status": status,
            "h5": str(source),
            "calibration_sha256": calibration_sha256,
            "singular_shared_imu_sample_count": singular_count,
            "shared_stream_used": False,
            "translation_source": "visual_endpoint_interpolation",
            "orientation_source": "calibrated_gyro_bridge_when_available",
            "maximum_allowed_imu_sample_gap_s": MAXIMUM_IMU_SAMPLE_GAP_S,
            "sides": side_audit,
        },
    )
