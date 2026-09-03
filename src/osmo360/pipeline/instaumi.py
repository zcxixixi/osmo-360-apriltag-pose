"""Adapter for the HDF5-backed InstaUMI four-fisheye dataset format."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from .manifest import ManifestError, ROOT, validate_path_component


SCHEMA_VERSION = "instaumi"
SUPPORTED_METADATA_SCHEMA = "1.0.0"
FACTORY_OFFSETS = ROOT / "config/devices/x5_factory_lens_offsets.json"


def is_instaumi_dataset(root: Path) -> bool:
    video = root / "video"
    return (root / "dataset.h5").is_file() and all(
        (video / name).is_file()
        for name in (
            "Left_back.mp4",
            "Left_forward.mp4",
            "Right_back.mp4",
            "Right_forward.mp4",
        )
    )


def _json_scalar(handle: h5py.File, key: str) -> dict[str, Any]:
    if key not in handle:
        raise ManifestError(f"InstaUMI H5 is missing {key}")
    value = handle[key][()]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError as error:
        raise ManifestError(f"InstaUMI H5 contains invalid JSON at {key}: {error}") from error
    if not isinstance(payload, dict):
        raise ManifestError(f"InstaUMI H5 {key} must contain a JSON object")
    return payload


def _factory_offsets() -> dict[str, Any]:
    try:
        payload = json.loads(FACTORY_OFFSETS.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot load serial-bound X5 factory offsets: {error}") from error
    if payload.get("schema_version") != "x5-factory-lens-offsets/1.0":
        raise ManifestError("invalid serial-bound X5 factory offset registry")
    return payload["devices"]


def _valid_x5_offset(value: Any) -> bool:
    fields = str(value).split("_")
    if len(fields) != 16 or fields[0] not in {"m2", "n2"}:
        return False
    try:
        numbers = np.asarray([float(item) for item in fields[1:]], dtype=float)
    except ValueError:
        return False
    return bool(np.isfinite(numbers).all())


def _factory_record(
    calibration: dict[str, Any],
    registry: dict[str, Any],
    *,
    side: str,
    serial: str,
) -> dict[str, Any]:
    registered = registry.get(serial)
    camera = calibration.get("cameras", {}).get(side, {})
    factory_lens = camera.get("factory_lens") if isinstance(camera, dict) else None
    embedded_offset = (
        factory_lens.get("x5_offset")
        if isinstance(factory_lens, dict)
        and factory_lens.get("source") == "embedded_insv.x5_offset"
        and factory_lens.get("stream") == 0
        else None
    )
    if embedded_offset is not None:
        if not _valid_x5_offset(embedded_offset):
            raise ManifestError(f"InstaUMI camera {serial!r} has an invalid H5 X5 factory lens record")
        if registered is not None and registered.get("x5_offset") != embedded_offset:
            raise ManifestError(
                f"InstaUMI camera {serial!r} H5 and registry X5 factory lens records disagree"
            )
        return {
            "x5_offset": str(embedded_offset),
            "source": f"dataset.h5:/calib/calibration_full.json/cameras/{side}/factory_lens",
        }
    if registered is not None:
        return registered
    raise ManifestError(
        f"InstaUMI camera {serial!r} has no serial-bound X5 factory lens record"
    )


def _rear_calibration(
    calibration: dict[str, Any], side: str, x5_offset: str
) -> dict[str, Any] | None:
    camera = calibration.get("cameras", {}).get(side, {})
    intrinsics = camera.get("intrinsics") or {}
    distortion = camera.get("distortion") or {}
    intrinsic_values = [intrinsics.get(name) for name in ("fx", "fy", "cx", "cy")]
    if all(value is None for value in intrinsic_values):
        return None
    image_size = np.asarray(camera.get("image_size"), dtype=int)
    if image_size.shape != (2,) or np.any(image_size <= 0):
        raise ManifestError(f"InstaUMI H5 {side} rear calibration has invalid image_size")
    try:
        values = np.asarray([
            intrinsics[name] for name in ("fx", "fy", "cx", "cy")
        ], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ManifestError(f"InstaUMI H5 {side} rear intrinsics are incomplete") from error
    coefficients = np.asarray(distortion.get("coefficients"), dtype=float)
    if not np.isfinite(values).all() or values[0] <= 0 or values[1] <= 0:
        raise ManifestError(f"InstaUMI H5 {side} rear intrinsics are invalid")
    if distortion.get("model") != "kannala_brandt" or coefficients.shape != (4,):
        raise ManifestError(f"InstaUMI H5 {side} must use four-coefficient Kannala-Brandt")
    transform = np.asarray(
        calibration.get("extrinsics", {}).get(f"T_rig_camera_{side}"), dtype=float
    )
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ManifestError(f"InstaUMI H5 {side} T_rig_camera is invalid")

    fields = x5_offset.split("_")
    offset = np.asarray([float(value) for value in fields[1:]], dtype=float)
    calibration_width = float(offset[13])
    centre_x, centre_y, radius, tilt_x, tilt_y, half_fov_deg = offset[:6]
    expected = np.asarray([
        radius / math.radians(half_fov_deg) / calibration_width,
        radius / math.radians(half_fov_deg) / calibration_width,
        centre_x / calibration_width,
        centre_y / calibration_width,
    ])
    normalized = values / np.asarray([
        image_size[0], image_size[1], image_size[0], image_size[1]
    ])
    expected_rotation = Rotation.from_euler(
        "xy", [tilt_x, tilt_y], degrees=True
    ).as_matrix()
    compatible = bool(
        np.allclose(normalized, expected, atol=2e-6, rtol=2e-6)
        and np.allclose(coefficients, 0.0, atol=1e-12)
        and np.allclose(transform[:3, :3], expected_rotation, atol=2e-6, rtol=2e-6)
        and np.allclose(transform[:3, 3], 0.0, atol=1e-12)
    )
    return {
        "image_size": image_size.tolist(),
        "projection_model": camera.get("projection_model"),
        "intrinsics": {name: float(intrinsics[name]) for name in ("fx", "fy", "cx", "cy")},
        "distortion": {
            "model": distortion["model"],
            "coefficients": coefficients.tolist(),
        },
        "T_rig_camera": transform.tolist(),
        "factory_offset_compatible": compatible,
    }


def _timeline(
    handle: h5py.File,
    side: str,
    *,
    expected_first_timestamp_ns: int = 0,
) -> dict[str, Any]:
    base = f"/sensor/camera/{side}"
    required = ("timestamp_ns", "source_timestamp_ns", "frame_index", "valid")
    missing = [name for name in required if f"{base}/{name}" not in handle]
    if missing:
        raise ManifestError(f"InstaUMI H5 {side} timeline is missing {missing}")
    timestamp = np.asarray(handle[f"{base}/timestamp_ns"], dtype=np.int64)
    source_timestamp = np.asarray(handle[f"{base}/source_timestamp_ns"], dtype=np.int64)
    frame_index = np.asarray(handle[f"{base}/frame_index"], dtype=np.int64)
    valid = np.asarray(handle[f"{base}/valid"], dtype=np.bool_)
    count = len(timestamp)
    if count == 0 or any(len(value) != count for value in (source_timestamp, frame_index, valid)):
        raise ManifestError(f"InstaUMI H5 {side} timeline arrays have invalid lengths")
    if not np.array_equal(frame_index, np.arange(count)):
        raise ManifestError(f"InstaUMI H5 {side} frame_index must be contiguous from zero")
    if not bool(np.all(valid)):
        raise ManifestError(f"InstaUMI H5 {side} contains invalid video frames")
    if timestamp[0] != expected_first_timestamp_ns or np.any(np.diff(timestamp) <= 0):
        raise ManifestError(
            f"InstaUMI H5 {side} aligned timestamps must start at its declared "
            "first-frame offset and increase"
        )
    return {
        "frame_count": count,
        "first_timestamp_ns": int(timestamp[0]),
        "last_timestamp_ns": int(timestamp[-1]),
        "first_source_timestamp_ns": int(source_timestamp[0]),
        "last_source_timestamp_ns": int(source_timestamp[-1]),
    }


def load_instaumi_config(root: Path) -> dict[str, Any]:
    """Translate an InstaUMI H5 + video directory into the four-MP4 descriptor."""
    h5_path = root / "dataset.h5"
    try:
        with h5py.File(h5_path, "r") as handle:
            if handle.attrs.get("schema_name", handle.attrs.get("schema")) != SCHEMA_VERSION:
                raise ManifestError("dataset.h5 is not an InstaUMI dataset")
            metadata = _json_scalar(handle, "/metadata/dataset.json")
            calibration_raw = handle["/calib/calibration_full.json"][()]
            if isinstance(calibration_raw, bytes):
                calibration_raw = calibration_raw.decode("utf-8")
            calibration = json.loads(str(calibration_raw))
            first_offsets = metadata.get("time", {}).get(
                "first_frame_time_offset_ns", {}
            )
            left_timeline = _timeline(
                handle,
                "left",
                expected_first_timestamp_ns=int(first_offsets.get("left", 0)),
            )
            right_timeline = _timeline(
                handle,
                "right",
                expected_first_timestamp_ns=int(first_offsets.get("right", 0)),
            )
    except OSError as error:
        raise ManifestError(f"cannot read InstaUMI H5: {error}") from error

    if metadata.get("schema_version") != SUPPORTED_METADATA_SCHEMA:
        raise ManifestError(
            f"unsupported InstaUMI metadata schema: {metadata.get('schema_version')!r}"
        )
    if left_timeline["frame_count"] != right_timeline["frame_count"]:
        raise ManifestError("InstaUMI left/right aligned timelines have different lengths")
    time = metadata.get("time", {})
    if time.get("reference") != "dataset_start":
        raise ManifestError("InstaUMI video timestamps must use dataset_start as reference")
    offsets = _factory_offsets()
    calibration_sha256 = hashlib.sha256(
        str(calibration_raw).encode("utf-8")
    ).hexdigest()
    h5_sha256 = hashlib.sha256(h5_path.read_bytes()).hexdigest()
    cameras: dict[str, Any] = {}
    for side, title, timeline in (
        ("left", "Left", left_timeline),
        ("right", "Right", right_timeline),
    ):
        device = metadata.get("devices", {}).get(side, {})
        serial = str(device.get("serial_number", ""))
        factory = _factory_record(
            calibration,
            offsets,
            side=side,
            serial=serial,
        )
        expected_count = metadata.get("video", {}).get(side, {}).get("frame_count")
        if expected_count is not None and int(expected_count) != timeline["frame_count"]:
            raise ManifestError(f"InstaUMI H5 {side} video/timeline frame counts disagree")
        rear_calibration = _rear_calibration(
            calibration, side, factory["x5_offset"]
        )
        cameras[side] = {
            "serial": serial,
            "x5_offset": factory["x5_offset"],
            # The H5 rear-lens preview declares source_stream=0 and matches *_back.
            "lenses": [f"video/{title}_back.mp4", f"video/{title}_forward.mp4"],
            "timeline_h5": "dataset.h5",
            "timeline_camera": side,
            "timeline": timeline,
            "lens_mapping": {
                "stream_0": "back/rear",
                "stream_1": "forward/front",
                "evidence": "H5 rear preview source_stream=0 and frame-equivalent *_back video",
            },
            "factory_offset_source": factory["source"],
            "timeline_h5_sha256": h5_sha256,
            **(
                {
                    "rear_calibration": rear_calibration,
                    "rear_calibration_source": "dataset.h5:/calib/calibration_full.json",
                    "rear_calibration_sha256": calibration_sha256,
                }
                if rear_calibration is not None
                else {}
            ),
        }

    time_calibration = calibration.get("time_calibration", {})
    source_offset_ns = int(time_calibration.get("right_left_offset_ns", 0))
    created = metadata.get("created_at_utc")
    return {
        "schema_version": "dual-x5-four-mp4-input/1.0",
        "input_format": "instaumi-four-fisheye-mp4-hdf5/1.0",
        "pair_id": validate_path_component(
            metadata["dataset_id"] if "dataset_id" in metadata else root.name,
            field="InstaUMI metadata.dataset_id",
        ),
        "recorded_at": created,
        "cameras": cameras,
        # MP4s are already clipped onto the common dataset timeline.  The source
        # offset is provenance only and must not be applied a second time.
        "sync": {
            "method": "instaumi_h5_aligned_timeline",
            "offset_s": 0.0,
            "source_right_left_offset_s": source_offset_ns / 1e9,
            "uncertainty_s": float(time_calibration.get("uncertainty_ns", 0)) / 1e9,
            "source_method": time_calibration.get("method"),
        },
        "auto_tracking": {
            "enabled": True,
            "mode": "shared-a3-self-calibrated-bearing-pnp",
            "panel_a_map": str(ROOT / "config/a3_aprilgrid_A_200_205_120mm.json"),
            "panel_b_map": str(ROOT / "config/a3_aprilgrid_B_210_215_120mm.json"),
        },
        "processing": {
            "profile": "fast-cpu",
            "cache_workers": 4,
            "threads_per_worker": 4,
            "maximum_concurrent_jobs": 1,
            "job_slot_timeout_s": 3600.0,
            "trajectory_observation_fps": 30.0,
            "decode_fps": 30.0,
            "native_grayscale_decode": True,
            "optical_flow_scale": 0.5,
            "forward_backward_check_hz": 5.0,
            "optical_flow_window_size": 21,
            "optical_flow_max_level": 3,
            "optical_flow_max_iterations": 15,
            "rectified_recovery_hz": 0.5,
            "rectified_view_size": 720,
            "global_scout_hz": 2.0,
            "global_scout_scale": 0.35,
            # Refresh tracked corners before a partial hand occlusion can drag
            # LK features away from the printed Tag edges. This remains a
            # small predicted ROI decode rather than a full-frame search.
            "local_redetection_hz": 5.0,
            "maximum_track_age_s": 0.5,
            "cache_chunk_duration_s": 120.0,
        },
        "instaumi": {
            "h5_path": "dataset.h5",
            "metadata_schema_version": metadata["schema_version"],
            "created_at_utc": created,
            "timeline_frame_count": left_timeline["frame_count"],
            "calibration_intrinsics_complete": all(
                (
                    calibration.get("cameras", {}).get(side, {}).get("intrinsics")
                    or {}
                ).get(name)
                is not None
                for side in ("left", "right")
                for name in ("fx", "fy", "cx", "cy")
            ),
            "rear_intrinsics_factory_compatible": all(
                cameras[side].get("rear_calibration", {}).get(
                    "factory_offset_compatible", False
                )
                for side in ("left", "right")
            ),
            "calibration_sha256": calibration_sha256,
            "h5_sha256": h5_sha256,
            "extrinsics_status": "placeholder_identity"
            if calibration.get("extrinsics", {}).get("T_right_left")
            == [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            else "provided",
        },
    }
