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
from osmo360.paths import ROOT


MAXIMUM_IMU_SAMPLE_GAP_S = 0.05
VISUAL_IMU_SELF_CALIBRATION_REVISION = "visual-gyro-hand-flu-r1"
MINIMUM_VISUAL_IMU_CALIBRATION_PAIRS = 200
MINIMUM_VISUAL_IMU_SPEED_CORRELATION = 0.70
MAXIMUM_VISUAL_IMU_HOLDOUT_MEDIAN_DEG = 0.50
MAXIMUM_VISUAL_IMU_HOLDOUT_P95_DEG = 2.0
MAXIMUM_CROSS_SIDE_EXTRINSIC_DISAGREEMENT_DEG = 10.0
MAXIMUM_GYRO_ENDPOINT_CLOSURE_DEG = 20.0
MAXIMUM_ACCELEROMETER_NORM_M_S2 = 50.0
MAXIMUM_ACCELEROMETER_BRIDGE_DEVIATION_M = 0.15
IMU_ROTATION_BASELINE_PATH = (
    ROOT / "config/imu_revisions/x5_kmdgp_kmurq_visual_gyro_20260902_r1.json"
)
IMU_ROTATION_BASELINE_SCHEMA = "instaumi-imu-rotation-baseline/1.0"
IDENTITY_EXTRINSIC_ATOL = 1e-8


class ImuAssistanceUnavailable(RuntimeError):
    """Raised when a requested visual gap is not safely covered by the IMU."""


@dataclass(frozen=True)
class GyroBridge:
    rotations: Rotation
    maximum_sample_gap_s: float
    endpoint_closure_deg: float


@dataclass(frozen=True)
class GyroPrediction:
    rotation: Rotation
    maximum_sample_gap_s: float


@dataclass(frozen=True)
class AccelerometerBridge:
    positions_m: np.ndarray
    maximum_sample_gap_s: float
    maximum_deviation_from_linear_m: float


@dataclass(frozen=True)
class ImuSeries:
    side: str
    timestamp_s: np.ndarray
    angular_velocity_hand_rad_s: np.ndarray
    calibration_sha256: str
    dataset_path: str
    linear_acceleration_hand_m_s2: np.ndarray | None = None

    def predict_orientation(
        self,
        start_s: float,
        start_rotation: Rotation,
        end_s: float,
        *,
        maximum_sample_gap_s: float = MAXIMUM_IMU_SAMPLE_GAP_S,
    ) -> GyroPrediction:
        """Propagate a visual attitude to ``end_s`` using calibrated gyro data."""
        if not np.isfinite([start_s, end_s, maximum_sample_gap_s]).all():
            raise ValueError("IMU prediction bounds must be finite")
        if end_s <= start_s or maximum_sample_gap_s <= 0:
            raise ValueError("invalid IMU prediction interval")
        timestamps = self.timestamp_s
        if timestamps[0] > start_s or timestamps[-1] < end_s:
            raise ImuAssistanceUnavailable("imu_does_not_cover_visual_gap")
        lower = max(0, int(np.searchsorted(timestamps, start_s, side="right")) - 1)
        upper = min(
            len(timestamps),
            int(np.searchsorted(timestamps, end_s, side="left")) + 1,
        )
        covered_times = timestamps[lower:upper]
        if len(covered_times) < 2:
            raise ImuAssistanceUnavailable("insufficient_imu_samples")
        largest_gap = float(np.max(np.diff(covered_times)))
        if largest_gap > maximum_sample_gap_s + 1e-12:
            raise ImuAssistanceUnavailable(
                f"imu_sample_gap_exceeds_limit:{largest_gap:.9f}"
            )
        inside = timestamps[(timestamps > start_s) & (timestamps < end_s)]
        nodes = np.unique(np.concatenate((np.asarray([start_s, end_s]), inside)))
        omega = np.column_stack([
            np.interp(nodes, timestamps, self.angular_velocity_hand_rad_s[:, axis])
            for axis in range(3)
        ])
        propagated = start_rotation
        for index in range(1, len(nodes)):
            delta_s = float(nodes[index] - nodes[index - 1])
            midpoint_omega = 0.5 * (omega[index - 1] + omega[index])
            propagated = propagated * Rotation.from_rotvec(midpoint_omega * delta_s)
        return GyroPrediction(
            rotation=propagated,
            maximum_sample_gap_s=largest_gap,
        )

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
        endpoint_closure_deg = float(np.degrees(endpoint_closure.magnitude()))
        if endpoint_closure_deg > MAXIMUM_GYRO_ENDPOINT_CLOSURE_DEG:
            raise ImuAssistanceUnavailable(
                "gyro_endpoint_closure_exceeds_limit:"
                f"{endpoint_closure_deg:.9f}"
            )
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
            endpoint_closure_deg=endpoint_closure_deg,
        )

    def bridge_positions(
        self,
        start_s: float,
        start_position_m: np.ndarray,
        start_rotation: Rotation,
        end_s: float,
        end_position_m: np.ndarray,
        end_rotation: Rotation,
        query_s: np.ndarray,
        *,
        maximum_sample_gap_s: float = MAXIMUM_IMU_SAMPLE_GAP_S,
        maximum_deviation_m: float = MAXIMUM_ACCELEROMETER_BRIDGE_DEVIATION_M,
    ) -> AccelerometerBridge:
        """Shape an internal visual gap with timestamp-aligned acceleration.

        The two visual positions remain exact metric anchors.  Mean world-frame
        specific force is removed over the interval, which cancels gravity and
        constant bias without pretending that this capture supplied a full
        accelerometer calibration.  A final endpoint closure prevents inertial
        drift from changing visual scale.
        """
        if self.linear_acceleration_hand_m_s2 is None:
            raise ImuAssistanceUnavailable("accelerometer_stream_unavailable")
        query = np.asarray(query_s, dtype=np.float64).reshape(-1)
        start_position = np.asarray(start_position_m, dtype=np.float64).reshape(3)
        end_position = np.asarray(end_position_m, dtype=np.float64).reshape(3)
        if not np.isfinite([start_s, end_s, maximum_sample_gap_s, maximum_deviation_m]).all():
            raise ValueError("accelerometer bridge bounds must be finite")
        if end_s <= start_s or maximum_sample_gap_s <= 0 or maximum_deviation_m <= 0:
            raise ValueError("invalid accelerometer bridge interval")
        if (
            np.any(~np.isfinite(query))
            or np.any(query < start_s)
            or np.any(query > end_s)
            or not np.isfinite(start_position).all()
            or not np.isfinite(end_position).all()
        ):
            raise ValueError("accelerometer bridge inputs are invalid")
        timestamps = self.timestamp_s
        if timestamps[0] > start_s or timestamps[-1] < end_s:
            raise ImuAssistanceUnavailable("imu_does_not_cover_visual_gap")
        lower = max(0, int(np.searchsorted(timestamps, start_s, side="right")) - 1)
        upper = min(
            len(timestamps), int(np.searchsorted(timestamps, end_s, side="left")) + 1
        )
        covered_times = timestamps[lower:upper]
        if len(covered_times) < 2:
            raise ImuAssistanceUnavailable("insufficient_accelerometer_samples")
        largest_gap = float(np.max(np.diff(covered_times)))
        if largest_gap > maximum_sample_gap_s + 1e-12:
            raise ImuAssistanceUnavailable(
                f"imu_sample_gap_exceeds_limit:{largest_gap:.9f}"
            )
        inside = timestamps[(timestamps > start_s) & (timestamps < end_s)]
        nodes = np.unique(np.concatenate((np.asarray([start_s, end_s]), inside, query)))
        acceleration_hand = np.column_stack([
            np.interp(
                nodes,
                timestamps,
                self.linear_acceleration_hand_m_s2[:, axis],
            )
            for axis in range(3)
        ])
        if (
            not np.isfinite(acceleration_hand).all()
            or float(np.max(np.linalg.norm(acceleration_hand, axis=1)))
            > MAXIMUM_ACCELEROMETER_NORM_M_S2
        ):
            raise ImuAssistanceUnavailable("accelerometer_norm_exceeds_limit")
        rotations = self.bridge_orientations(
            start_s,
            start_rotation,
            end_s,
            end_rotation,
            nodes,
            maximum_sample_gap_s=maximum_sample_gap_s,
        ).rotations
        acceleration_world = rotations.apply(acceleration_hand)
        duration_s = end_s - start_s
        interval_s = np.diff(nodes)
        weighted_mean = np.sum(
            0.5
            * (acceleration_world[:-1] + acceleration_world[1:])
            * interval_s[:, None],
            axis=0,
        ) / duration_s
        acceleration_world -= weighted_mean
        positions = np.empty((len(nodes), 3), dtype=np.float64)
        positions[0] = start_position
        velocity = (end_position - start_position) / duration_s
        for index, delta_s in enumerate(interval_s, start=1):
            midpoint_acceleration = 0.5 * (
                acceleration_world[index - 1] + acceleration_world[index]
            )
            positions[index] = (
                positions[index - 1]
                + velocity * delta_s
                + 0.5 * midpoint_acceleration * delta_s**2
            )
            velocity = velocity + midpoint_acceleration * delta_s
        phase = ((nodes - start_s) / duration_s)[:, None]
        positions += phase * (end_position - positions[-1])
        linear = start_position + phase * (end_position - start_position)
        maximum_deviation = float(np.max(np.linalg.norm(positions - linear, axis=1)))
        if maximum_deviation > maximum_deviation_m + 1e-12:
            raise ImuAssistanceUnavailable(
                "accelerometer_bridge_deviation_exceeds_limit:"
                f"{maximum_deviation:.9f}"
            )
        by_time = {float(time_s): position for time_s, position in zip(nodes, positions)}
        result = np.asarray([by_time[float(time_s)] for time_s in query])
        return AccelerometerBridge(
            positions_m=result,
            maximum_sample_gap_s=largest_gap,
            maximum_deviation_from_linear_m=maximum_deviation,
        )


@dataclass(frozen=True)
class ImuBundle:
    streams: dict[str, ImuSeries]
    audit: dict[str, Any]


def _visual_angular_velocity(
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    samples: list[tuple[float, float, np.ndarray, int]] = []
    valid = [
        row
        for row in rows
        if row.get("quality_status") == "valid"
        and int(row.get("inlier_tag_count", 0)) >= 4
    ]
    valid.sort(key=lambda row: int(row["frame"]))
    for first, second in zip(valid[:-1], valid[1:]):
        first_frame = int(first["frame"])
        second_frame = int(second["frame"])
        if second_frame != first_frame + 1:
            continue
        start_s = float(first["timestamp"])
        end_s = float(second["timestamp"])
        delta_s = end_s - start_s
        if not 0.020 <= delta_s <= 0.050:
            continue
        start_rotation = Rotation.from_quat(
            [float(first[f"q{axis}"]) for axis in "xyzw"]
        )
        end_rotation = Rotation.from_quat(
            [float(second[f"q{axis}"]) for axis in "xyzw"]
        )
        angular_velocity = (
            start_rotation.inv() * end_rotation
        ).as_rotvec() / delta_s
        samples.append((
            0.5 * (start_s + end_s),
            delta_s,
            angular_velocity,
            first_frame,
        ))
    if len(samples) < MINIMUM_VISUAL_IMU_CALIBRATION_PAIRS:
        raise ImuAssistanceUnavailable("insufficient_visual_imu_calibration_pairs")
    return (
        np.asarray([sample[0] for sample in samples], dtype=np.float64),
        np.asarray([sample[1] for sample in samples], dtype=np.float64),
        np.asarray([sample[2] for sample in samples], dtype=np.float64),
        np.asarray([sample[3] for sample in samples], dtype=np.int64),
    )


def _interpolate_gyro(
    timestamps_s: np.ndarray,
    angular_velocity: np.ndarray,
    query_s: np.ndarray,
) -> np.ndarray:
    return np.column_stack([
        np.interp(
            query_s,
            timestamps_s,
            angular_velocity[:, axis],
            left=np.nan,
            right=np.nan,
        )
        for axis in range(3)
    ])


def _fit_visual_gyro_calibration(
    side: str,
    timestamp_s: np.ndarray,
    angular_velocity_imu: np.ndarray,
    linear_acceleration_imu: np.ndarray | None,
    rows: list[dict[str, Any]],
) -> tuple[ImuSeries, dict[str, Any], Rotation]:
    mid_s, delta_s, visual_omega, frame = _visual_angular_velocity(rows)
    visual_speed = np.linalg.norm(visual_omega, axis=1)
    raw_speed = np.linalg.norm(angular_velocity_imu, axis=1)
    correlations: list[tuple[float, float]] = []
    for offset_s in np.arange(-1.0, 1.0001, 0.010):
        candidate = np.interp(
            mid_s + offset_s,
            timestamp_s,
            raw_speed,
            left=np.nan,
            right=np.nan,
        )
        finite = np.isfinite(candidate)
        if np.count_nonzero(finite) < MINIMUM_VISUAL_IMU_CALIBRATION_PAIRS:
            continue
        correlation = float(np.corrcoef(visual_speed[finite], candidate[finite])[0, 1])
        if np.isfinite(correlation):
            correlations.append((correlation, float(offset_s)))
    if not correlations:
        raise ImuAssistanceUnavailable("visual_imu_time_offset_search_has_no_overlap")
    speed_correlation, coarse_offset_s = max(correlations)
    if speed_correlation < MINIMUM_VISUAL_IMU_SPEED_CORRELATION:
        raise ImuAssistanceUnavailable(
            f"visual_imu_speed_correlation_below_limit:{speed_correlation:.6f}"
        )

    best: tuple[float, dict[str, Any]] | None = None
    for offset_s in np.arange(
        coarse_offset_s - 0.020,
        coarse_offset_s + 0.0201,
        0.001,
    ):
        gyro = _interpolate_gyro(
            timestamp_s,
            angular_velocity_imu,
            mid_s + offset_s,
        )
        finite = np.isfinite(gyro).all(axis=1)
        selected = (
            finite
            & (visual_speed >= 0.15)
            & (visual_speed <= 10.0)
            & (np.linalg.norm(gyro, axis=1) >= 0.15)
            & (np.linalg.norm(gyro, axis=1) <= 10.0)
        )
        indices = np.flatnonzero(selected)
        training = indices[frame[indices] % 5 != 0]
        holdout = indices[frame[indices] % 5 == 0]
        if (
            len(training) < MINIMUM_VISUAL_IMU_CALIBRATION_PAIRS
            or len(holdout) < 40
        ):
            continue
        rotation, _ = Rotation.align_vectors(
            visual_omega[training], gyro[training]
        )
        bias_hand = np.median(
            visual_omega[training] - rotation.apply(gyro[training]), axis=0
        )
        for _ in range(4):
            rotation, _ = Rotation.align_vectors(
                visual_omega[training] - bias_hand,
                gyro[training],
            )
            bias_hand = np.median(
                visual_omega[training] - rotation.apply(gyro[training]), axis=0
            )
        predicted = rotation.apply(gyro[holdout]) + bias_hand
        error_deg = np.degrees(
            np.linalg.norm(visual_omega[holdout] - predicted, axis=1)
            * delta_s[holdout]
        )
        metrics = {
            "offset_s": float(offset_s),
            "rotation": rotation,
            "bias_hand_rad_s": bias_hand,
            "training": training,
            "holdout": holdout,
            "error_deg": error_deg,
        }
        score = float(np.median(error_deg))
        if best is None or score < best[0]:
            best = (score, metrics)
    if best is None:
        raise ImuAssistanceUnavailable("insufficient_excited_visual_imu_pairs")
    metrics = best[1]
    rotation = metrics["rotation"]
    training = metrics["training"]
    error_deg = metrics["error_deg"]
    singular_values = np.linalg.svd(
        visual_omega[training] - np.mean(visual_omega[training], axis=0),
        compute_uv=False,
    )
    excitation_ratio = float(singular_values[-1] / singular_values[0])
    holdout_median_deg = float(np.median(error_deg))
    holdout_p95_deg = float(np.percentile(error_deg, 95))
    if excitation_ratio < 0.10:
        raise ImuAssistanceUnavailable(
            f"visual_imu_three_axis_excitation_below_limit:{excitation_ratio:.6f}"
        )
    if (
        holdout_median_deg > MAXIMUM_VISUAL_IMU_HOLDOUT_MEDIAN_DEG
        or holdout_p95_deg > MAXIMUM_VISUAL_IMU_HOLDOUT_P95_DEG
    ):
        raise ImuAssistanceUnavailable(
            "visual_imu_holdout_residual_exceeds_limit:"
            f"median={holdout_median_deg:.6f},p95={holdout_p95_deg:.6f}"
        )
    bias_hand = metrics["bias_hand_rad_s"]
    corrected = rotation.apply(angular_velocity_imu) + bias_hand
    linear_acceleration_hand = (
        None
        if linear_acceleration_imu is None
        else rotation.apply(linear_acceleration_imu)
    )
    adjusted_timestamp_s = timestamp_s - metrics["offset_s"]
    calibration_payload = {
        "revision": VISUAL_IMU_SELF_CALIBRATION_REVISION,
        "side": side,
        "time_offset_s_raw_imu_minus_visual": metrics["offset_s"],
        "rotation_hand_from_imu_quaternion_xyzw": rotation.as_quat().tolist(),
        "bias_hand_rad_s": bias_hand.tolist(),
    }
    calibration_sha256 = hashlib.sha256(
        json.dumps(calibration_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    stream = ImuSeries(
        side=side,
        timestamp_s=adjusted_timestamp_s,
        angular_velocity_hand_rad_s=corrected,
        calibration_sha256=calibration_sha256,
        dataset_path=f"/sensor/imu/{side}",
        linear_acceleration_hand_m_s2=linear_acceleration_hand,
    )
    audit = {
        "status": "VISUAL_SELF_CALIBRATED_PASS",
        **calibration_payload,
        "calibration_sha256": calibration_sha256,
        "speed_norm_correlation": speed_correlation,
        "training_pair_count": int(len(training)),
        "holdout_pair_count": int(len(metrics["holdout"])),
        "three_axis_excitation_ratio": excitation_ratio,
        "holdout_rotation_error_deg": {
            "median": holdout_median_deg,
            "p95": holdout_p95_deg,
            "max": float(np.max(error_deg)),
        },
        "accelerometer_status": (
            "AVAILABLE_ROTATION_ONLY_ZERO_BIAS_SCALE"
            if linear_acceleration_hand is not None
            else "UNAVAILABLE"
        ),
    }
    return stream, audit, rotation


def calibrate_instaumi_imu_from_visual(
    path: Path,
    visual_rows: dict[str, list[dict[str, Any]]],
) -> ImuBundle:
    """Self-calibrate per-side gyro timing, axes, and bias from visual rotation."""
    source = path.resolve(strict=True)
    streams: dict[str, ImuSeries] = {}
    sides: dict[str, Any] = {}
    rotations: dict[str, Rotation] = {}
    with h5py.File(source, "r") as handle:
        metadata_text, metadata = _json_dataset(handle, "/metadata/dataset.json")
        if metadata.get("time", {}).get("reference") != "dataset_start":
            raise ImuAssistanceUnavailable("imu_visual_clock_is_not_dataset_start")
        for side in ("left", "right"):
            base = f"/sensor/imu/{side}"
            required = ("timestamp_ns", "angular_velocity", "valid")
            if any(f"{base}/{name}" not in handle for name in required):
                sides[side] = {"status": "UNAVAILABLE_PER_SIDE_STREAM_MISSING"}
                continue
            timestamp_s = np.asarray(
                handle[f"{base}/timestamp_ns"], dtype=np.float64
            ) / 1e9
            angular_velocity = np.asarray(
                handle[f"{base}/angular_velocity"], dtype=np.float64
            )
            acceleration_key = f"{base}/linear_acceleration"
            linear_acceleration = (
                np.asarray(handle[acceleration_key], dtype=np.float64)
                if acceleration_key in handle
                else None
            )
            valid = np.asarray(handle[f"{base}/valid"], dtype=bool)
            finite = np.isfinite(timestamp_s) & np.isfinite(angular_velocity).all(axis=1)
            if linear_acceleration is not None:
                if linear_acceleration.shape != angular_velocity.shape:
                    sides[side] = {"status": "UNAVAILABLE_RAW_IMU_INVALID"}
                    continue
                finite &= np.isfinite(linear_acceleration).all(axis=1)
            keep = valid & finite
            timestamp_s = timestamp_s[keep]
            angular_velocity = angular_velocity[keep]
            if linear_acceleration is not None:
                linear_acceleration = linear_acceleration[keep]
            if (
                len(timestamp_s) < 2
                or angular_velocity.shape != (len(timestamp_s), 3)
                or np.any(np.diff(timestamp_s) <= 0)
            ):
                sides[side] = {"status": "UNAVAILABLE_RAW_IMU_INVALID"}
                continue
            try:
                stream, audit, rotation = _fit_visual_gyro_calibration(
                    side,
                    timestamp_s,
                    angular_velocity,
                    linear_acceleration,
                    visual_rows[side],
                )
            except (ImuAssistanceUnavailable, KeyError, ValueError) as exc:
                sides[side] = {
                    "status": "UNAVAILABLE_VISUAL_SELF_CALIBRATION_FAILED",
                    "reason": str(exc),
                }
                continue
            streams[side] = stream
            sides[side] = audit
            rotations[side] = rotation
    cross_side_disagreement_deg = None
    if len(rotations) == 2:
        cross_side_disagreement_deg = float(np.degrees(
            (rotations["left"].inv() * rotations["right"]).magnitude()
        ))
        if cross_side_disagreement_deg > MAXIMUM_CROSS_SIDE_EXTRINSIC_DISAGREEMENT_DEG:
            raise ImuAssistanceUnavailable(
                "visual_imu_cross_side_extrinsic_disagreement_exceeds_limit:"
                f"{cross_side_disagreement_deg:.6f}"
            )
    status = (
        "VISUAL_SELF_CALIBRATED_PASS"
        if len(streams) == 2
        else "UNAVAILABLE_VISUAL_SELF_CALIBRATION_FAILED"
    )
    return ImuBundle(
        streams=streams if len(streams) == 2 else {},
        audit={
            "status": status,
            "h5": str(source),
            "metadata_sha256": hashlib.sha256(metadata_text.encode()).hexdigest(),
            "revision": VISUAL_IMU_SELF_CALIBRATION_REVISION,
            "translation_source": "visual_endpoint_interpolation",
            "orientation_source": "capture_local_visual_self_calibrated_gyro",
            "accelerometer_translation_integration_used": False,
            "cross_side_extrinsic_disagreement_deg": cross_side_disagreement_deg,
            "maximum_cross_side_extrinsic_disagreement_deg": (
                MAXIMUM_CROSS_SIDE_EXTRINSIC_DISAGREEMENT_DEG
            ),
            "sides": sides,
        },
    )


def _json_dataset(handle: h5py.File, key: str) -> tuple[str, dict[str, Any]]:
    if key not in handle:
        raise ImuAssistanceUnavailable(f"missing_calibration:{key}")
    raw = handle[key][()]
    text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    return text, json.loads(text)


def _transform(extrinsics: dict[str, Any], key: str) -> np.ndarray:
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
    return transform


def _optional_transform(extrinsics: dict[str, Any], key: str) -> np.ndarray | None:
    if key not in extrinsics or extrinsics[key] is None:
        return None
    return _transform(extrinsics, key)


def _load_rotation_baseline() -> tuple[str, dict[str, Any]]:
    text = IMU_ROTATION_BASELINE_PATH.read_text(encoding="utf-8")
    payload = json.loads(text)
    if payload.get("schema_version") != IMU_ROTATION_BASELINE_SCHEMA:
        raise ImuAssistanceUnavailable("invalid_imu_rotation_baseline_schema")
    if payload.get("transform_convention") != "T_target_source":
        raise ImuAssistanceUnavailable("invalid_imu_rotation_baseline_convention")
    if payload.get("target_frame") != "hand_camera_flu_back_x":
        raise ImuAssistanceUnavailable("invalid_imu_rotation_baseline_target_frame")
    timestamp_policy = payload.get("timestamp_policy", {})
    if (
        timestamp_policy.get("reference") != "dataset_start"
        or timestamp_policy.get("alignment") != "interpolate_by_h5_timestamp_ns"
        or float(timestamp_policy.get("fixed_time_offset_s", float("nan"))) != 0.0
    ):
        raise ImuAssistanceUnavailable("invalid_imu_rotation_baseline_timestamp_policy")
    return text, payload


def _resolve_hand_from_imu(
    calibration: dict[str, Any],
    metadata: dict[str, Any],
    baseline: dict[str, Any],
    *,
    side: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    extrinsics = calibration.get("extrinsics", {})
    if not isinstance(extrinsics, dict):
        raise ImuAssistanceUnavailable("invalid_extrinsics_object")
    camera_key = f"T_rig_camera_{side}"
    imu_key = f"T_rig_imu_{side}"
    rig_from_camera = _optional_transform(extrinsics, camera_key)
    rig_from_imu = _optional_transform(extrinsics, imu_key)
    camera_is_explicit = (
        rig_from_camera is not None
        and not np.allclose(
            rig_from_camera, np.eye(4), atol=IDENTITY_EXTRINSIC_ATOL, rtol=0
        )
    )
    imu_is_explicit = (
        rig_from_imu is not None
        and not np.allclose(
            rig_from_imu, np.eye(4), atol=IDENTITY_EXTRINSIC_ATOL, rtol=0
        )
    )
    if camera_is_explicit or imu_is_explicit:
        if rig_from_camera is None or rig_from_imu is None:
            raise ImuAssistanceUnavailable(
                f"incomplete_explicit_camera_imu_extrinsics:{side}"
            )
        if extrinsics.get("transform_convention") != "T_target_source":
            raise ImuAssistanceUnavailable("unsupported_extrinsic_transform_convention")
        camera_from_imu = rig_from_camera[:3, :3].T @ rig_from_imu[:3, :3]
        hand_from_imu = (
            X5_STREAM0_OPENCV_FROM_HAND_CAMERA_FLU.T @ camera_from_imu
        )
        return hand_from_imu, {
            "orientation_calibration_source": "calibration_full_explicit_extrinsics",
            "camera_extrinsic_key": camera_key,
            "imu_extrinsic_key": imu_key,
            "placeholder_identity_extrinsics_replaced": False,
        }

    serial = str(metadata.get("devices", {}).get(side, {}).get("serial_number", ""))
    record = baseline.get("devices", {}).get(serial)
    if not serial or not isinstance(record, dict):
        raise ImuAssistanceUnavailable(
            f"no_serial_bound_imu_rotation_baseline:{side}:{serial or 'missing'}"
        )
    if record.get("rig_side") != side:
        raise ImuAssistanceUnavailable(
            f"imu_rotation_baseline_side_mismatch:{side}:{serial}"
        )
    baseline_transform = _transform(
        {"baseline": record.get("T_hand_camera_flu_back_x_imu_rotation_only")},
        "baseline",
    )
    if not np.allclose(
        baseline_transform[:3, 3], np.zeros(3), atol=1e-12, rtol=0
    ):
        raise ImuAssistanceUnavailable(
            f"imu_rotation_baseline_contains_unobservable_translation:{serial}"
        )
    return baseline_transform[:3, :3], {
        "orientation_calibration_source": "serial_bound_visual_gyro_baseline",
        "baseline_revision": baseline["revision"],
        "camera_serial": serial,
        "placeholder_identity_extrinsics_replaced": (
            rig_from_camera is not None and rig_from_imu is not None
        ),
        "calibration_full_placeholder": (
            "identity" if rig_from_camera is not None and rig_from_imu is not None
            else "missing_or_null"
        ),
    }


def _calibration_vector(
    calibration: dict[str, Any], side: str, key: str, default: list[float]
) -> tuple[np.ndarray, str]:
    imu_calibration = calibration.get("imu_calibration", {})
    side_calibration = imu_calibration.get(side, imu_calibration)
    raw = side_calibration.get(key) if isinstance(side_calibration, dict) else None
    source = "calibration_full" if raw is not None else "safe_default"
    value = np.asarray(default if raw is None else raw, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ImuAssistanceUnavailable(f"invalid_imu_calibration:{side}:{key}")
    return value, source


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
    baseline_text, baseline = _load_rotation_baseline()
    baseline_sha256 = hashlib.sha256(baseline_text.encode()).hexdigest()
    with h5py.File(source, "r") as handle:
        metadata_text, metadata = _json_dataset(handle, "/metadata/dataset.json")
        if metadata.get("time", {}).get("reference") != "dataset_start":
            raise ImuAssistanceUnavailable("imu_timestamp_reference_is_not_dataset_start")
        calibration_text, calibration = _json_dataset(
            handle, "/calib/calibration_full.json"
        )
        calibration_sha256 = hashlib.sha256(calibration_text.encode()).hexdigest()
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
            acceleration_key = f"{base}/linear_acceleration"
            linear_acceleration = (
                np.asarray(handle[acceleration_key], dtype=np.float64)
                if acceleration_key in handle
                else None
            )
            valid = np.asarray(handle[f"{base}/valid"], dtype=bool)
            if (
                angular_velocity.shape != (len(timestamp_s), 3)
                or valid.shape != (len(timestamp_s),)
                or (
                    linear_acceleration is not None
                    and linear_acceleration.shape != (len(timestamp_s), 3)
                )
            ):
                side_audit[side] = {
                    "status": "UNAVAILABLE_SHAPE_MISMATCH",
                    "dataset_path": base,
                }
                continue
            finite = np.isfinite(timestamp_s) & np.isfinite(angular_velocity).all(axis=1)
            if linear_acceleration is not None:
                finite &= np.isfinite(linear_acceleration).all(axis=1)
            keep = valid & finite
            timestamp_s = timestamp_s[keep]
            angular_velocity = angular_velocity[keep]
            if linear_acceleration is not None:
                linear_acceleration = linear_acceleration[keep]
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
            try:
                hand_from_imu, orientation_audit = _resolve_hand_from_imu(
                    calibration,
                    metadata,
                    baseline,
                    side=side,
                )
                bias, bias_source = _calibration_vector(
                    calibration, side, "gyroscope_bias_rad_s", [0, 0, 0]
                )
                scale, scale_source = _calibration_vector(
                    calibration, side, "gyroscope_scale", [1, 1, 1]
                )
                accelerometer_bias, accelerometer_bias_source = _calibration_vector(
                    calibration, side, "accelerometer_bias_m_s2", [0, 0, 0]
                )
                accelerometer_scale, accelerometer_scale_source = _calibration_vector(
                    calibration, side, "accelerometer_scale", [1, 1, 1]
                )
                if np.any(scale <= 0):
                    raise ImuAssistanceUnavailable(
                        f"invalid_imu_calibration:{side}:gyroscope_scale"
                    )
                if np.any(accelerometer_scale <= 0):
                    raise ImuAssistanceUnavailable(
                        f"invalid_imu_calibration:{side}:accelerometer_scale"
                    )
            except ImuAssistanceUnavailable as exc:
                side_audit[side] = {
                    "status": "UNAVAILABLE_CALIBRATION_INVALID",
                    "reason": str(exc),
                    "dataset_path": base,
                }
                continue
            corrected = (angular_velocity - bias) * scale
            angular_velocity_hand = (hand_from_imu @ corrected.T).T
            linear_acceleration_hand = None
            if linear_acceleration is not None:
                corrected_acceleration = (
                    linear_acceleration - accelerometer_bias
                ) * accelerometer_scale
                linear_acceleration_hand = (
                    hand_from_imu @ corrected_acceleration.T
                ).T
            effective_calibration = {
                "h5_calibration_sha256": calibration_sha256,
                "baseline_sha256": (
                    baseline_sha256
                    if orientation_audit["orientation_calibration_source"]
                    == "serial_bound_visual_gyro_baseline"
                    else None
                ),
                "side": side,
                "orientation": orientation_audit,
                "gyroscope_bias_rad_s": bias.tolist(),
                "gyroscope_bias_source": bias_source,
                "gyroscope_scale": scale.tolist(),
                "gyroscope_scale_source": scale_source,
                "accelerometer_bias_m_s2": accelerometer_bias.tolist(),
                "accelerometer_bias_source": accelerometer_bias_source,
                "accelerometer_scale": accelerometer_scale.tolist(),
                "accelerometer_scale_source": accelerometer_scale_source,
                "time_alignment": "h5_timestamp_ns",
                "fixed_time_offset_s": 0.0,
            }
            effective_calibration_sha256 = hashlib.sha256(
                json.dumps(effective_calibration, sort_keys=True).encode("utf-8")
            ).hexdigest()
            stream = ImuSeries(
                side=side,
                timestamp_s=timestamp_s,
                angular_velocity_hand_rad_s=angular_velocity_hand,
                calibration_sha256=effective_calibration_sha256,
                dataset_path=base,
                linear_acceleration_hand_m_s2=linear_acceleration_hand,
            )
            streams[side] = stream
            side_audit[side] = {
                "status": "AVAILABLE",
                "sample_count": int(len(timestamp_s)),
                "maximum_sample_gap_s": float(np.max(np.diff(timestamp_s))),
                "dataset_path": base,
                "output_child_frame": "hand_camera_flu_back_x",
                "time_alignment": {
                    "method": "linear_interpolation_by_h5_timestamp_ns",
                    "timestamp_dataset": f"{base}/timestamp_ns",
                    "reference": "dataset_start",
                    "fixed_time_offset_s": 0.0,
                },
                "gyroscope_bias_source": bias_source,
                "gyroscope_scale_source": scale_source,
                "accelerometer_status": (
                    "AVAILABLE" if linear_acceleration_hand is not None else "UNAVAILABLE"
                ),
                "accelerometer_bias_source": accelerometer_bias_source,
                "accelerometer_scale_source": accelerometer_scale_source,
                "effective_calibration_sha256": effective_calibration_sha256,
                **orientation_audit,
            }

        singular_count = 0
        singular_key = "/sensor/imu/angular_velocity"
        if singular_key in handle:
            singular_count = int(handle[singular_key].shape[0])
    if len(streams) == 2:
        status = "AVAILABLE"
    elif streams:
        status = "PARTIAL_PER_SIDE_IMU"
    elif any(
        item.get("status") == "UNAVAILABLE_CALIBRATION_INVALID"
        for item in side_audit.values()
    ):
        status = "UNAVAILABLE_CALIBRATION_INVALID"
    elif singular_count == 0:
        status = "UNAVAILABLE_NO_SAMPLES"
    else:
        status = "UNAVAILABLE_AMBIGUOUS_SHARED_IMU"
    return ImuBundle(
        streams=streams,
        audit={
            "status": status,
            "h5": str(source),
            "metadata_sha256": hashlib.sha256(metadata_text.encode()).hexdigest(),
            "calibration_sha256": calibration_sha256,
            "imu_rotation_baseline_path": str(IMU_ROTATION_BASELINE_PATH),
            "imu_rotation_baseline_sha256": baseline_sha256,
            "singular_shared_imu_sample_count": singular_count,
            "shared_stream_used": False,
            "translation_source": (
                "visual_endpoint_anchored_timestamp_aligned_accelerometer_bridge"
                "_when_available"
            ),
            "orientation_source": "calibrated_gyro_bridge_when_available",
            "time_alignment": {
                "method": "linear_interpolation_by_h5_timestamp_ns",
                "reference": "dataset_start",
                "fixed_time_offset_s": 0.0,
                "frame_index_alignment_used": False,
            },
            "maximum_allowed_imu_sample_gap_s": MAXIMUM_IMU_SAMPLE_GAP_S,
            "maximum_allowed_gyro_endpoint_closure_deg": (
                MAXIMUM_GYRO_ENDPOINT_CLOSURE_DEG
            ),
            "sides": side_audit,
        },
    )
