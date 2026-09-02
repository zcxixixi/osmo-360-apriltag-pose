from __future__ import annotations

import hashlib
import math
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .manifest import ManifestError
from .insta360_telemetry import ImuSamples


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def common_window(left_duration_s: float, right_duration_s: float, offset_s: float) -> tuple[float, float, float]:
    """Return left start, right start and duration on the audio-aligned timeline.

    ``offset_s`` follows ``right_time = left_time + offset``.
    """
    left_start = max(0.0, -offset_s)
    right_start = max(0.0, offset_s)
    duration = min(left_duration_s - left_start, right_duration_s - right_start)
    if duration <= 0:
        raise ManifestError("left/right videos have no common aligned interval")
    return left_start, right_start, duration


def export_aligned_video(
    ffmpeg: Path, source: Path, output: Path, *, start_s: float, duration_s: float,
    log: Path, stream: int = 0, output_size: int,
) -> None:
    """Create one aligned HEVC lens video without decoding unchanged 1920² tracks."""
    if output.is_file() and output.stat().st_size:
        return
    if output_size not in (1024, 1920):
        raise ValueError("aligned video output_size must be 1024 or 1920")
    output.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".partial.mp4")
    temporary.unlink(missing_ok=True)
    command = [
        str(ffmpeg), "-y", "-ss", f"{start_s:.9f}", "-i", str(source),
        "-map", f"0:v:{stream}", "-t", f"{duration_s:.9f}", "-an",
    ]
    if output_size == 1920:
        command += ["-c:v", "copy", "-avoid_negative_ts", "make_zero"]
    else:
        command += [
            "-vf", "scale=1024:1024:flags=lanczos", "-c:v", "libx265",
            "-preset", "ultrafast", "-crf", "24", "-pix_fmt", "yuv420p",
            "-tag:v", "hvc1", "-bf", "0", "-g", "30",
        ]
    command += ["-movflags", "+faststart", str(temporary)]
    process = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write_text(process.stdout, encoding="utf-8")
    if process.returncode or not temporary.is_file() or not temporary.stat().st_size:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"aligned video export failed: {source}; log={log}")
    os.replace(temporary, output)


def probe_mp4(ffprobe: Path, path: Path) -> dict[str, Any]:
    process = subprocess.run([
        str(ffprobe), "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,time_base,nb_frames,codec_name,pix_fmt,bit_rate,duration",
        "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    stream = json.loads(process.stdout)["streams"][0]
    fps_num, fps_den = (int(value) for value in stream["avg_frame_rate"].split("/"))
    tb_num, tb_den = (int(value) for value in stream["time_base"].split("/"))
    return {
        "width": int(stream["width"]), "height": int(stream["height"]),
        "frame_rate_num": fps_num, "frame_rate_den": fps_den,
        "time_base_num": tb_num, "time_base_den": tb_den,
        "frame_count": int(stream.get("nb_frames") or round(float(stream["duration"]) * fps_num / fps_den)),
        "duration_ns": round(float(stream["duration"]) * 1_000_000_000),
        "codec": (
            "H.265" if stream.get("codec_name") == "hevc"
            else stream.get("codec_name", "")
        ),
        "pixel_format": stream.get("pix_fmt", ""),
        "bitrate_bps": int(stream.get("bit_rate") or 0), "sha256": sha256(path),
    }


def _packet_records(
    ffprobe: Path,
    path: Path,
    *,
    stream: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    process = subprocess.run([
        str(ffprobe),
        "-v",
        "error",
        "-select_streams",
        f"v:{stream}",
        "-show_packets",
        "-show_entries",
        "packet=pts,pts_time,flags",
        "-of",
        "csv=p=0",
        str(path),
    ], check=True, capture_output=True, text=True)
    packets: list[tuple[int, int, int]] = []
    for line in process.stdout.splitlines():
        fields = line.split(",")
        if len(fields) < 3 or fields[0] == "N/A" or fields[1] == "N/A":
            raise ManifestError(f"video has a packet without PTS: {path}")
        packets.append((
            int(fields[0]),
            round(float(fields[1]) * 1_000_000_000),
            1 if "K" in fields[2] else 0,
        ))
    if not packets:
        raise ManifestError(f"video has no timestamped packets: {path}")

    # ffprobe reports packets in decode order. Sort the complete packet record
    # into presentation order without separating keyframe state from its PTS.
    packets.sort(key=lambda packet: packet[0])
    pts = np.asarray([packet[0] for packet in packets], dtype=np.int64)
    presentation = np.asarray([packet[1] for packet in packets], dtype=np.int64)
    keyframes = np.asarray([packet[2] for packet in packets], dtype=np.uint8)
    if np.any(np.diff(pts) <= 0) or np.any(np.diff(presentation) <= 0):
        raise ManifestError(f"video has duplicate or invalid PTS: {path}")
    return pts, presentation, keyframes


def packet_timeline(
    ffprobe: Path,
    path: Path,
    source_start_s: float,
    *,
    source_path: Path,
    source_stream: int = 0,
) -> dict[str, np.ndarray]:
    pts, presentation, keyframes = _packet_records(ffprobe, path, stream=0)
    _, source_presentation, _ = _packet_records(
        ffprobe,
        source_path,
        stream=source_stream,
    )

    aligned_presentation = presentation - presentation[0]
    source_start_ns = round(source_start_s * 1_000_000_000)
    targets = aligned_presentation + source_start_ns
    upper = np.searchsorted(source_presentation, targets, side="left")
    upper = np.clip(upper, 0, len(source_presentation) - 1)
    lower = np.clip(upper - 1, 0, len(source_presentation) - 1)
    use_upper = (
        np.abs(source_presentation[upper] - targets)
        < np.abs(source_presentation[lower] - targets)
    )
    source_indices = np.where(use_upper, upper, lower)
    source_timestamp_ns = source_presentation[source_indices]
    if np.any(np.diff(source_timestamp_ns) <= 0):
        raise ManifestError(
            f"aligned video frames do not map one-to-one to source packets: {source_path}"
        )
    source_interval_ns = int(np.median(np.diff(source_presentation)))
    if np.any(np.abs(source_timestamp_ns - targets) > source_interval_ns):
        raise ManifestError(
            f"aligned video timestamps do not match source packets: {source_path}"
        )

    # The aligned time comes from the selected original INSV packet, not from
    # synthetic PTS generated by cropping or frame-rate conversion.
    timestamp_ns = source_timestamp_ns - source_start_ns
    return {
        "frame_index": np.arange(len(timestamp_ns), dtype=np.uint64),
        "timestamp_ns": timestamp_ns,
        "source_timestamp_ns": source_timestamp_ns,
        "pts": pts,
        "keyframe": keyframes,
        "valid": np.ones(len(timestamp_ns), dtype=np.uint8),
    }


def _x5_rear_calibration(record: dict[str, Any]) -> dict[str, Any]:
    offset = record.get("x5_offset")
    if not offset:
        raise ManifestError("X5 embedded factory lens metadata is missing")
    fields = str(offset).strip().split("_")
    if len(fields) != 16 or fields[0] not in {"m2", "n2"}:
        raise ManifestError("invalid embedded Insta360 X5 lens calibration")
    values = np.asarray([float(value) for value in fields[1:]], dtype=float)
    stacked_height, calibration_width = values[12:14]
    if not math.isclose(stacked_height, 2 * calibration_width, rel_tol=0.01):
        raise ManifestError("X5 calibration is not a stacked dual-fisheye record")

    return {
        "image_size": [1024, 1024],
        "projection_model": "insta360_x5_factory_fisheye",
        "intrinsics": None,
        "distortion": {
            "model": "insta360_x5_factory",
            "coefficients": None,
        },
        "factory_lens": {
            "source": "embedded_insv.x5_offset",
            "stream": 0,
            "x5_offset": str(offset),
            "calibration_width": int(calibration_width),
            "status": "factory_model_preserved_uninterpreted",
        },
    }


def _constant_sensor_offset_ns(
    timestamp_ns: np.ndarray,
    source_timestamp_ns: np.ndarray,
    dataset_start_source_timestamp_ns: int,
    *,
    sensor: str,
) -> int:
    if (
        timestamp_ns.ndim != 1
        or source_timestamp_ns.shape != timestamp_ns.shape
        or len(timestamp_ns) == 0
    ):
        raise ManifestError(f"{sensor} timestamps must be non-empty matching vectors")
    delta = np.subtract(timestamp_ns, source_timestamp_ns)
    if not np.all(delta == delta[0]):
        raise ManifestError(f"{sensor} timestamps cannot be represented by one clock offset")
    return dataset_start_source_timestamp_ns + int(delta[0])


def _imu_calibration(samples: ImuSamples) -> dict[str, Any]:
    config = samples.provenance.get("gyro_config")
    if not isinstance(config, dict):
        raise ManifestError("X5 IMU gyro_config is missing")

    def positive_config_value(name: str) -> float:
        value = config.get(name)
        if isinstance(value, bool):
            raise ManifestError(
                f"X5 IMU gyro_config.{name} must be a finite positive number"
            )
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise ManifestError(
                f"X5 IMU gyro_config.{name} must be a finite positive number"
            ) from error
        if not math.isfinite(number) or number <= 0:
            raise ManifestError(
                f"X5 IMU gyro_config.{name} must be a finite positive number"
            )
        return number

    gyroscope_range_deg_s = positive_config_value("gyro_range")
    accelerometer_range_g = positive_config_value("acc_range")
    # Embedded gyro_cfg_info records full-scale limits in deg/s and g.
    return {
        "gyroscope_bias_rad_s": None,
        "accelerometer_bias_m_s2": None,
        "gyroscope_scale": None,
        "accelerometer_scale": None,
        "gyroscope_range_rad_s": math.radians(gyroscope_range_deg_s),
        "accelerometer_range_m_s2": accelerometer_range_g * 9.80665,
        "status": "range_only",
    }


def _calibration(
    sync: dict[str, Any],
    source_records: dict[str, dict[str, Any]],
    sensor_offsets_ns: dict[str, int],
    imu: dict[str, ImuSamples],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "coordinate_system": {
            "rig_frame": None,
            "left_rig_frame": "rig_left",
            "right_rig_frame": "rig_right",
            "left_imu_frame": "imu_left",
            "right_imu_frame": "imu_right",
            "left_camera_frame": "camera_left",
            "right_camera_frame": "camera_right",
            "handedness": "right",
            "camera_axes": {
                "x": "right",
                "y": "down",
                "z": "forward",
            },
            "rig_axes": {
                "x": "forward",
                "y": "left",
                "z": "up",
            },
            "translation_unit": "m",
            "rotation_unit": "rad",
        },
        "cameras": {
            side: _x5_rear_calibration(record)
            for side, record in source_records.items()
        },
        "extrinsics": {
            "matrix_convention": "row_major",
            "transform_convention": "T_target_source",
            "status": "unavailable",
            "reason": (
                "No measured camera-to-IMU or gripper mounting transforms; "
                "left and right grippers move independently."
            ),
            "T_rig_camera_left": None,
            "T_rig_camera_right": None,
            "T_rig_imu_left": None,
            "T_rig_imu_right": None,
            "T_right_left": None,
        },
        "time_calibration": {
            "reference": "dataset_start",
            **sensor_offsets_ns,
            "right_left_offset_ns": round(
                -float(sync["offset_s"]) * 1_000_000_000
            ),
            "uncertainty_ns": round(
                float(sync["uncertainty_s"]) * 1_000_000_000
            ),
            "method": "audio_cross_correlation",
        },
        "imu_calibration": {
            "left": _imu_calibration(imu["left"]),
            "right": _imu_calibration(imu["right"]),
        },
    }


def write_dataset_h5(
    output: Path, *, dataset_id: str, left_video: Path, right_video: Path,
    left_source: Path, right_source: Path, left_start_s: float, right_start_s: float,
    sync: dict[str, Any], ffprobe: Path, source_records: dict[str, dict[str, Any]],
    imu: dict[str, ImuSamples],
) -> dict[str, Any]:
    created = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    videos = {
        "left": probe_mp4(ffprobe, left_video),
        "right": probe_mp4(ffprobe, right_video),
    }
    timelines = {
        "left": packet_timeline(
            ffprobe,
            left_video,
            left_start_s,
            source_path=left_source,
        ),
        "right": packet_timeline(
            ffprobe,
            right_video,
            right_start_s,
            source_path=right_source,
        ),
    }
    for side in ("left", "right"):
        if videos[side]["width"] != 1024 or videos[side]["height"] != 1024:
            raise ManifestError(f"{side} aligned video is not 1024x1024")
        if len(timelines[side]["frame_index"]) != videos[side]["frame_count"]:
            videos[side]["frame_count"] = len(timelines[side]["frame_index"])
    if set(imu) != {"left", "right"}:
        raise ManifestError("InstaUMI requires separate left and right X5 IMU streams")
    if len(imu["left"].timestamp_ns) == 0:
        raise ManifestError("InstaUMI left IMU stream is empty")
    dataset_start_source_timestamp_ns = int(
        imu["left"].source_timestamp_ns[0] - imu["left"].timestamp_ns[0]
    )
    sensor_offsets_ns = {}
    for side, timeline in timelines.items():
        sensor_offsets_ns[f"{side}_camera_offset_ns"] = _constant_sensor_offset_ns(
            timeline["timestamp_ns"],
            timeline["source_timestamp_ns"],
            dataset_start_source_timestamp_ns,
            sensor=f"{side} camera",
        )
    for side, samples in imu.items():
        sensor_offsets_ns[f"{side}_imu_offset_ns"] = _constant_sensor_offset_ns(
            samples.timestamp_ns,
            samples.source_timestamp_ns,
            dataset_start_source_timestamp_ns,
            sensor=f"{side} imu",
        )
    metadata = {
        "schema_version": "1.0.0",
        "dataset_id": dataset_id,
        "created_at_utc": created,
        "description": "Audio-aligned dual Insta360 rear-lens videos",
        "time": {
            "unit": "ns",
            "reference": "dataset_start",
            "source_clock": "per_sensor_native",
            "dataset_start_source_timestamp_ns": dataset_start_source_timestamp_ns,
            "dataset_start_utc_ns": None,
            "first_frame_time_offset_ns": {
                side: int(timelines[side]["timestamp_ns"][0])
                for side in ("left", "right")
            },
            "source_timestamp_provenance": {
                "camera": "original_insv_video_packet_pts",
                "imu": "embedded_insv_imu_monotonic_clock",
            },
        },
        "devices": {
            side: {
                "manufacturer": "Insta360",
                "model": "X5",
                "serial_number": source_records[side]["serial"],
                "firmware_version": imu[side].provenance.get(
                    "firmware_version", ""
                ),
                "active_sensor": "rear",
                "rig_position": side,
            }
            for side in ("left", "right")
        },
        "video": {
            side: {
                "path": f"video/{side.title()}.mp4",
                **videos[side],
            }
            for side in ("left", "right")
        },
        "capture": {
            "mode": "normal_video",
            "panorama_mode": False,
            "source_lens": "rear",
            "source_resolution": [
                2 * int(source_records["left"]["lens_size"][0]),
                int(source_records["left"]["lens_size"][1]),
            ],
            "stored_resolution": [1024, 1024],
            "frame_rate": (
                videos["left"]["frame_rate_num"]
                / videos["left"]["frame_rate_den"]
            ),
            "encoding": "H.265",
            "bitrate_mode": "standard",
            "color_mode": "standard",
            "hdr_enabled": False,
            "i_log_enabled": False,
            "low_light_stabilization": False,
            "sharpness": "",
            "white_balance": {
                "mode": "auto",
                "temperature_k": None,
            },
        },
        "exposure": {
            "mode": "auto",
            "ev": 0,
            "iso": None,
            "iso_upper_limit": None,
            "shutter_s": None,
            "left_samples": {
                "timestamp_ns": [],
                "exposure_time_ns": [],
                "iso": [],
            },
            "right_samples": {
                "timestamp_ns": [],
                "exposure_time_ns": [],
                "iso": [],
            },
        },
        "lens_and_stitching": {
            "left": {
                "lens": "rear",
                "lens_guard": "",
                "field_of_view": "",
                "crop": {
                    "output_width": 1024,
                    "output_height": 1024,
                    "parameters": {
                        "source_stream": 0,
                        "resize": "lanczos",
                    },
                },
            },
            "right": {
                "lens": "rear",
                "lens_guard": "",
                "field_of_view": "",
                "crop": {
                    "output_width": 1024,
                    "output_height": 1024,
                    "parameters": {
                        "source_stream": 0,
                        "resize": "lanczos",
                    },
                },
            },
            "stitching": {
                "enabled": False,
                "mode": "rear_lens_only",
                "vendor": "Insta360",
                "profile": "",
                "parameters": {},
            },
            "stabilization": {
                "enabled": False,
                "mode": "",
            },
        },
        "imu": {
            "source": "Insta360",
            "left_sample_count": int(len(imu["left"].timestamp_ns)),
            "right_sample_count": int(len(imu["right"].timestamp_ns)),
            "gyroscope_unit": "rad/s",
            "accelerometer_unit": "m/s^2",
            "axis_order": ["x", "y", "z"],
            "coordinate_frames": {
                "left": "imu_left",
                "right": "imu_right",
            },
            "calibration_provenance": {
                side: {
                    "range_source": "embedded_insv.gyro_cfg_info",
                    "gyro_config": imu[side].provenance.get("gyro_config"),
                    "bias_scale": "unavailable",
                }
                for side in ("left", "right")
            },
        },
        "speaker": {
            "present": False,
            "sample_rate_hz": 48000,
            "channels": 1,
            "sample_format": "s16le",
            "sample_count": 0,
        },
        "source": {
            "left_original_insv": left_source.name,
            "right_original_insv": right_source.name,
            "left_original_sha256": source_records["left"].get("sha256", ""),
            "right_original_sha256": source_records["right"].get("sha256", ""),
            "insta360_sdk_version": "CameraSDK 2.1.1.1",
            "media_sdk_version": "MediaSDK 3.1.1.0",
            "conversion_software": (
                "instaumi-x5-pipeline + telemetry-parser/gyro2bb"
            ),
            "conversion_version": "1.1.0",
        },
    }
    calibration = _calibration(sync, source_records, sensor_offsets_ns, imu)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial.h5")
    temporary.unlink(missing_ok=True)
    string_type = h5py.string_dtype("utf-8")
    with h5py.File(temporary, "w") as handle:
        handle.attrs.update({
            "schema_name": "instaumi",
            "schema_version": "1.0.0",
            "dataset_id": dataset_id,
            "time_unit": "ns",
            "time_reference": "dataset_start",
            "created_at_utc": created,
        })
        handle.create_dataset(
            "calib/calibration_full.json",
            data=json.dumps(calibration, ensure_ascii=False, indent=2),
            dtype=string_type,
        )
        handle.create_dataset(
            "metadata/dataset.json",
            data=json.dumps(metadata, ensure_ascii=False, indent=2),
            dtype=string_type,
        )
        camera_group = handle.require_group("sensor/camera")
        for side in ("left", "right"):
            group = camera_group.create_group(side)
            group.create_dataset(
                "video_path",
                data=f"video/{side.title()}.mp4",
                dtype=string_type,
            )
            for name, values in timelines[side].items():
                group.create_dataset(
                    name,
                    data=values,
                    compression="gzip",
                    shuffle=True,
                    chunks=True,
                )
        imu_root = handle.require_group("sensor/imu")
        for side in ("left", "right"):
            imu_group = imu_root.create_group(side)
            samples = imu[side]
            imu_group.create_dataset(
                "timestamp_ns",
                data=samples.timestamp_ns,
                compression="gzip",
                shuffle=True,
                chunks=True,
            )
            imu_group.create_dataset(
                "source_timestamp_ns",
                data=samples.source_timestamp_ns,
                compression="gzip",
                shuffle=True,
                chunks=True,
            )
            angular = imu_group.create_dataset(
                "angular_velocity",
                data=samples.angular_velocity,
                compression="gzip",
                shuffle=True,
                chunks=True,
            )
            linear = imu_group.create_dataset(
                "linear_acceleration",
                data=samples.linear_acceleration,
                compression="gzip",
                shuffle=True,
                chunks=True,
            )
            imu_group.create_dataset(
                "valid",
                data=samples.valid,
                compression="gzip",
                shuffle=True,
                chunks=True,
            )
            angular.attrs.update({
                "unit": "rad/s",
                "axis_order": "x,y,z",
                "frame": f"imu_{side}",
            })
            linear.attrs.update({
                "unit": "m/s^2",
                "axis_order": "x,y,z",
                "frame": f"imu_{side}",
            })
        speaker = handle.require_group("sensor/speaker")
        speaker.create_dataset("timestamp_ns", shape=(0,), dtype="i8")
        speaker.create_dataset("samples", shape=(0, 1), dtype="i2")
        speaker.create_dataset("valid", shape=(0,), dtype="u1")
    os.replace(temporary, output)
    return metadata
