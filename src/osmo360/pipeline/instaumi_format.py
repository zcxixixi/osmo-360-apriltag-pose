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


def packet_timeline(ffprobe: Path, path: Path, source_start_s: float) -> dict[str, np.ndarray]:
    process = subprocess.run([
        str(ffprobe), "-v", "error", "-select_streams", "v:0", "-show_packets",
        "-show_entries", "packet=pts,pts_time,flags", "-of", "csv=p=0", str(path),
    ], check=True, capture_output=True, text=True)
    pts: list[int] = []; timestamps: list[int] = []; keyframes: list[int] = []
    for line in process.stdout.splitlines():
        fields = line.split(",")
        if len(fields) < 3 or fields[0] == "N/A" or fields[1] == "N/A":
            continue
        pts.append(int(fields[0]))
        timestamps.append(round(float(fields[1]) * 1_000_000_000))
        keyframes.append(1 if "K" in fields[2] else 0)
    presentation = np.asarray(timestamps, dtype=np.int64)
    if not len(presentation):
        raise ManifestError(f"aligned video has no timestamped packets: {path}")
    aligned = presentation - presentation[0]
    return {
        "frame_index": np.arange(len(aligned), dtype=np.uint64),
        "timestamp_ns": aligned,
        "source_timestamp_ns": (
            presentation + round(source_start_s * 1_000_000_000)
        ),
        "pts": np.asarray(pts, dtype=np.int64),
        "keyframe": np.asarray(keyframes, dtype=np.uint8),
        "valid": np.ones(len(aligned), dtype=np.uint8),
    }


def _x5_rear_calibration(record: dict[str, Any]) -> tuple[dict[str, Any], list[list[float]]]:
    identity = np.eye(4, dtype=float)
    offset = record.get("x5_offset")
    if not offset:
        camera = {
            "image_size": [1024, 1024],
            "projection_model": "fisheye",
            "intrinsics": {"fx": None, "fy": None, "cx": None, "cy": None},
            "distortion": {"model": "kannala_brandt", "coefficients": []},
        }
        return camera, identity.tolist()

    fields = str(offset).strip().split("_")
    if len(fields) != 16 or fields[0] not in {"m2", "n2"}:
        raise ManifestError("invalid embedded Insta360 X5 lens calibration")
    values = np.asarray([float(value) for value in fields[1:]], dtype=float)
    stacked_height, calibration_width = values[12:14]
    if not math.isclose(stacked_height, 2 * calibration_width, rel_tol=0.01):
        raise ManifestError("X5 calibration is not a stacked dual-fisheye record")

    centre_x, centre_y, radius, tilt_x, tilt_y, half_fov_deg = values[:6]
    output_scale = 1024.0 / calibration_width
    half_fov_rad = math.radians(half_fov_deg)
    focal_length = radius * output_scale / half_fov_rad
    camera = {
        "image_size": [1024, 1024],
        "projection_model": "fisheye",
        "intrinsics": {
            "fx": focal_length,
            "fy": focal_length,
            "cx": centre_x * output_scale,
            "cy": centre_y * output_scale,
        },
        "distortion": {
            "model": "kannala_brandt",
            "coefficients": [0.0, 0.0, 0.0, 0.0],
        },
    }

    x = math.radians(tilt_x)
    y = math.radians(tilt_y)
    rotation_x = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(x), -math.sin(x)],
        [0.0, math.sin(x), math.cos(x)],
    ])
    rotation_y = np.asarray([
        [math.cos(y), 0.0, math.sin(y)],
        [0.0, 1.0, 0.0],
        [-math.sin(y), 0.0, math.cos(y)],
    ])
    identity[:3, :3] = rotation_y @ rotation_x
    return camera, identity.tolist()


def _calibration(
    sync: dict[str, Any], source_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    identity = np.eye(4, dtype=float).tolist()
    left_camera, left_extrinsic = _x5_rear_calibration(source_records["left"])
    right_camera, right_extrinsic = _x5_rear_calibration(source_records["right"])
    return {
        "schema_version": "1.0.0",
        "coordinate_system": {"rig_frame": "rig", "left_imu_frame": "imu_left", "right_imu_frame": "imu_right", "left_camera_frame": "camera_left", "right_camera_frame": "camera_right", "handedness": "right", "camera_axes": {"x": "right", "y": "down", "z": "forward"}, "rig_axes": {"x": "forward", "y": "left", "z": "up"}, "translation_unit": "m", "rotation_unit": "rad"},
        "cameras": {"left": left_camera, "right": right_camera},
        "extrinsics": {"matrix_convention": "row_major", "transform_convention": "T_target_source", "T_rig_camera_left": left_extrinsic, "T_rig_camera_right": right_extrinsic, "T_rig_imu_left": identity, "T_rig_imu_right": identity, "T_right_left": identity},
        "time_calibration": {"reference": "dataset_start", "left_camera_offset_ns": 0, "right_camera_offset_ns": round(-float(sync["offset_s"]) * 1_000_000_000), "left_imu_offset_ns": 0, "right_imu_offset_ns": round(-float(sync["offset_s"]) * 1_000_000_000), "right_left_offset_ns": round(-float(sync["offset_s"]) * 1_000_000_000), "uncertainty_ns": round(float(sync["uncertainty_s"]) * 1_000_000_000), "method": "audio_cross_correlation"},
        "imu_calibration": {"left": {"gyroscope_bias_rad_s": [0, 0, 0], "accelerometer_bias_m_s2": [0, 0, 0], "gyroscope_scale": [1, 1, 1], "accelerometer_scale": [1, 1, 1], "gyroscope_range_rad_s": None, "accelerometer_range_m_s2": None}, "right": {"gyroscope_bias_rad_s": [0, 0, 0], "accelerometer_bias_m_s2": [0, 0, 0], "gyroscope_scale": [1, 1, 1], "accelerometer_scale": [1, 1, 1], "gyroscope_range_rad_s": None, "accelerometer_range_m_s2": None}},
    }


def write_dataset_h5(
    output: Path, *, dataset_id: str, left_video: Path, right_video: Path,
    left_source: Path, right_source: Path, left_start_s: float, right_start_s: float,
    sync: dict[str, Any], ffprobe: Path, source_records: dict[str, dict[str, Any]],
    imu: dict[str, ImuSamples],
) -> dict[str, Any]:
    created = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    videos = {"left": probe_mp4(ffprobe, left_video), "right": probe_mp4(ffprobe, right_video)}
    timelines = {
        "left": packet_timeline(ffprobe, left_video, left_start_s),
        "right": packet_timeline(ffprobe, right_video, right_start_s),
    }
    for side in ("left", "right"):
        if videos[side]["width"] != 1024 or videos[side]["height"] != 1024:
            raise ManifestError(f"{side} aligned video is not 1024x1024")
        if len(timelines[side]["frame_index"]) != videos[side]["frame_count"]:
            videos[side]["frame_count"] = len(timelines[side]["frame_index"])
    if set(imu) != {"left", "right"}:
        raise ManifestError("InstaUMI requires separate left and right X5 IMU streams")
    metadata = {
        "schema_version": "1.0.0", "dataset_id": dataset_id, "created_at_utc": created,
        "description": "Audio-aligned dual Insta360 rear-lens videos",
        "time": {"unit": "ns", "reference": "dataset_start", "source_clock": "camera_monotonic", "dataset_start_source_timestamp_ns": int(imu["left"].source_timestamp_ns[0] - imu["left"].timestamp_ns[0]), "dataset_start_utc_ns": None, "first_frame_time_offset_ns": {"left": 0, "right": 0}},
        "devices": {side: {"manufacturer": "Insta360", "model": "X5", "serial_number": source_records[side]["serial"], "firmware_version": imu[side].provenance.get("firmware_version", ""), "active_sensor": "rear", "rig_position": side} for side in ("left", "right")},
        "video": {side: {"path": f"video/{side.title()}.mp4", **videos[side]} for side in ("left", "right")},
        "capture": {"mode": "normal_video", "panorama_mode": False, "source_lens": "rear", "source_resolution": [2 * int(source_records["left"]["lens_size"][0]), int(source_records["left"]["lens_size"][1])], "stored_resolution": [1024, 1024], "frame_rate": videos["left"]["frame_rate_num"] / videos["left"]["frame_rate_den"], "encoding": "H.265", "bitrate_mode": "standard", "color_mode": "standard", "hdr_enabled": False, "i_log_enabled": False, "low_light_stabilization": False, "sharpness": "", "white_balance": {"mode": "auto", "temperature_k": None}},
        "exposure": {"mode": "auto", "ev": 0, "iso": None, "iso_upper_limit": None, "shutter_s": None, "left_samples": {"timestamp_ns": [], "exposure_time_ns": [], "iso": []}, "right_samples": {"timestamp_ns": [], "exposure_time_ns": [], "iso": []}},
        "lens_and_stitching": {"left": {"lens": "rear", "lens_guard": "", "field_of_view": "", "crop": {"output_width": 1024, "output_height": 1024, "parameters": {"source_stream": 0, "resize": "lanczos"}}}, "right": {"lens": "rear", "lens_guard": "", "field_of_view": "", "crop": {"output_width": 1024, "output_height": 1024, "parameters": {"source_stream": 0, "resize": "lanczos"}}}, "stitching": {"enabled": False, "mode": "rear_lens_only", "vendor": "Insta360", "profile": "", "parameters": {}}, "stabilization": {"enabled": False, "mode": ""}},
        "imu": {"source": "Insta360", "left_sample_count": int(len(imu["left"].timestamp_ns)), "right_sample_count": int(len(imu["right"].timestamp_ns)), "gyroscope_unit": "rad/s", "accelerometer_unit": "m/s^2", "axis_order": ["x", "y", "z"], "coordinate_frames": {"left": "imu_left", "right": "imu_right"}},
        "speaker": {"present": False, "sample_rate_hz": 48000, "channels": 1, "sample_format": "s16le", "sample_count": 0},
        "source": {"left_original_insv": left_source.name, "right_original_insv": right_source.name, "left_original_sha256": source_records["left"].get("sha256", ""), "right_original_sha256": source_records["right"].get("sha256", ""), "insta360_sdk_version": "CameraSDK 2.1.1.1", "media_sdk_version": "MediaSDK 3.1.1.0", "conversion_software": "osmo360 + telemetry-parser/gyro2bb", "conversion_version": "1.1.0"},
    }
    calibration = _calibration(sync, source_records)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial.h5"); temporary.unlink(missing_ok=True)
    string_type = h5py.string_dtype("utf-8")
    with h5py.File(temporary, "w") as handle:
        handle.attrs.update({"schema_name": "instaumi", "schema_version": "1.0.0", "dataset_id": dataset_id, "time_unit": "ns", "time_reference": "dataset_start", "created_at_utc": created})
        handle.create_dataset("calib/calibration_full.json", data=json.dumps(calibration, ensure_ascii=False), dtype=string_type)
        handle.create_dataset("metadata/dataset.json", data=json.dumps(metadata, ensure_ascii=False), dtype=string_type)
        camera_group = handle.require_group("sensor/camera")
        for side in ("left", "right"):
            group = camera_group.create_group(side)
            group.create_dataset("video_path", data=f"video/{side.title()}.mp4", dtype=string_type)
            for name, values in timelines[side].items():
                group.create_dataset(name, data=values, compression="gzip", shuffle=True, chunks=True)
        imu_root = handle.require_group("sensor/imu")
        for side in ("left", "right"):
            imu_group = imu_root.create_group(side)
            samples = imu[side]
            imu_group.create_dataset("timestamp_ns", data=samples.timestamp_ns, compression="gzip", shuffle=True, chunks=True)
            imu_group.create_dataset("source_timestamp_ns", data=samples.source_timestamp_ns, compression="gzip", shuffle=True, chunks=True)
            angular = imu_group.create_dataset("angular_velocity", data=samples.angular_velocity, compression="gzip", shuffle=True, chunks=True)
            linear = imu_group.create_dataset("linear_acceleration", data=samples.linear_acceleration, compression="gzip", shuffle=True, chunks=True)
            imu_group.create_dataset("valid", data=samples.valid, compression="gzip", shuffle=True, chunks=True)
            angular.attrs.update({"unit": "rad/s", "axis_order": "x,y,z", "frame": f"imu_{side}"})
            linear.attrs.update({"unit": "m/s^2", "axis_order": "x,y,z", "frame": f"imu_{side}"})
        speaker = handle.require_group("sensor/speaker")
        speaker.create_dataset("timestamp_ns", shape=(0,), dtype="i8")
        speaker.create_dataset("samples", shape=(0, 1), dtype="i2")
        speaker.create_dataset("valid", shape=(0,), dtype="u1")
    os.replace(temporary, output)
    return metadata
