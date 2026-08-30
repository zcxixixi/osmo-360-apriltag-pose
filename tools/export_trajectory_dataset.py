#!/usr/bin/env python3
"""Package a stitched panorama and AprilTag 6DoF results as a dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from render_trajectory_overlay_video import kalman_rts_filter, load_and_filter, sample_pose


FORMAT_VERSION = "camera-trajectory-dataset/1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export timestamped 6DoF trajectory dataset")
    parser.add_argument("video", type=Path, help="factory-stitched 2:1 panorama")
    parser.add_argument("pose_csv", type=Path)
    parser.add_argument("summary_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source-raw", type=Path)
    parser.add_argument("--camera-family", required=True)
    parser.add_argument("--sensor-metadata-dir", type=Path)
    parser.add_argument("--detections-jsonl", type=Path)
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--prediction-max-age", type=float, default=0.25)
    parser.add_argument("--median-window", type=int, default=5)
    parser.add_argument("--kalman-measurement-noise", type=float, default=0.04)
    parser.add_argument("--kalman-accel-noise", type=float, default=0.8)
    parser.add_argument("--kalman-angle-noise", type=float, default=2.0)
    parser.add_argument("--kalman-angular-accel-noise", type=float, default=30.0)
    parser.add_argument("--extract-frames", action="store_true")
    parser.add_argument("--frame-width", type=int, default=1920)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    return parser.parse_args()


def file_identity(path: Path | None) -> dict | None:
    if path is None or not path.is_file():
        return None
    stat = path.stat()
    return {
        "path": str(path.resolve()), "name": path.name,
        "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
    }


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def finite(value: str | None) -> float:
    try:
        result = float(value or "nan")
    except ValueError:
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def relative_pose(pose: np.ndarray, origin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    position = pose[:3] - origin[:3]
    world_rotation = Rotation.from_euler("xyz", pose[3:], degrees=True)
    origin_rotation = Rotation.from_euler("xyz", origin[3:], degrees=True)
    relative = origin_rotation.inv() * world_rotation
    return position, relative


def extract_selected_frames(video: Path, frame_numbers: set[int], output: Path,
                            width: int, quality: int) -> int:
    output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video}")
    written = 0
    frame_no = 0
    while frame_numbers:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_no in frame_numbers:
            height = max(2, int(round(frame.shape[0] * width / frame.shape[1])))
            resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            destination = output / f"{frame_no:06d}.jpg"
            if not cv2.imwrite(str(destination), resized, [cv2.IMWRITE_JPEG_QUALITY, quality]):
                raise RuntimeError(f"failed to write {destination}")
            frame_numbers.remove(frame_no)
            written += 1
            if written % 250 == 0:
                print(f"DATASET_FRAMES {written}", flush=True)
        frame_no += 1
    capture.release()
    return written


def main() -> int:
    args = parse_args()
    video = args.video.resolve()
    pose_csv = args.pose_csv.resolve()
    summary_json = args.summary_json.resolve()
    output = args.output_dir.resolve()
    for required in (video, pose_csv, summary_json):
        if not required.is_file():
            raise SystemExit(f"missing dataset input: {required}")
    if args.prediction_max_age < 0 or args.frame_width < 160 or not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("invalid dataset export parameters")

    media_dir = output / "media"
    annotations_dir = output / "annotations"
    calibration_dir = output / "calibration"
    sensor_dir = output / "sensor"
    reports_dir = output / "reports"
    previews_dir = output / "previews"
    for directory in (media_dir, annotations_dir, calibration_dir, sensor_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)
    dataset_video = media_dir / "panorama.mp4"
    link_or_copy(video, dataset_video)
    shutil.copy2(pose_csv, annotations_dir / "pose_direct.csv")
    shutil.copy2(summary_json, reports_dir / "visual_summary.json")
    if args.detections_jsonl and args.detections_jsonl.is_file():
        shutil.copy2(args.detections_jsonl, annotations_dir / "detections.jsonl")
    if args.preview and args.preview.is_file():
        previews_dir.mkdir(parents=True, exist_ok=True)
        link_or_copy(args.preview.resolve(), previews_dir / "trajectory_overlay.mp4")

    copied_sensor_files: list[str] = []
    if args.sensor_metadata_dir and args.sensor_metadata_dir.is_dir():
        for source in sorted(args.sensor_metadata_dir.iterdir()):
            if not source.is_file():
                continue
            if "calibration" in source.name or source.name == "source_info.json":
                target_dir = calibration_dir
            elif source.suffix.lower() in {".csv", ".json"}:
                target_dir = sensor_dir
            else:
                continue
            shutil.copy2(source, target_dir / source.name)
            copied_sensor_files.append(str((target_dir / source.name).relative_to(output)))

    with pose_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("pose CSV is empty")
    try:
        filtered = load_and_filter(pose_csv, smooth=0.0, median_window=args.median_window)
        smoothed = kalman_rts_filter(
            filtered,
            args.kalman_measurement_noise,
            args.kalman_accel_noise,
            args.kalman_angle_noise,
            args.kalman_angular_accel_noise,
        )
        origin = smoothed[0, 1:].copy()
    except ValueError:
        smoothed = None
        origin = None

    output_fields = [
        "frame", "timestamp_s", "state", "direct_measurement", "temporally_recovered",
        "detected_tag_count", "detected_ids", "measurement_source", "quality_status",
        "reprojection_rmse_px",
        "measured_x_m", "measured_y_m", "measured_z_m",
        "measured_roll_deg", "measured_pitch_deg", "measured_yaw_deg",
        "optimized_x_m", "optimized_y_m", "optimized_z_m",
        "optimized_qx", "optimized_qy", "optimized_qz", "optimized_qw",
        "relative_x_m", "relative_y_m", "relative_z_m",
        "relative_qx", "relative_qy", "relative_qz", "relative_qw",
        "image_path",
    ]
    trajectory_path = annotations_dir / "trajectory_6dof.csv"
    states: Counter[str] = Counter()
    frame_numbers: set[int] = set()
    with trajectory_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        for row in rows:
            timestamp = finite(row.get("timestamp"))
            frame = int(row["frame"])
            visual_valid = row.get("quality_status") == "valid" and bool(row.get("camera_x_m"))
            source = row.get("measurement_source", "")
            direct = visual_valid and source in {"", "direct"}
            if smoothed is None:
                pose, sampled_state = None, "SEARCHING"
            else:
                pose, sampled_state, _opacity = sample_pose(
                    smoothed, timestamp, args.prediction_max_age
                )
            if direct:
                state = "MEASURED"
            elif visual_valid and source == "optical_flow":
                state = "TRACKED_FLOW"
            elif sampled_state == "TRACKED":
                state = "RECOVERED"
            elif sampled_state in {"PREDICTED", "ENTERING"}:
                state = "PREDICTED"
            else:
                state = "LOST"
                pose = None
            states[state] += 1
            frame_numbers.add(frame)
            result = dict.fromkeys(output_fields, "")
            result.update(
                frame=frame,
                timestamp_s=f"{timestamp:.6f}",
                state=state,
                direct_measurement=int(direct),
                temporally_recovered=int(state == "RECOVERED"),
                detected_tag_count=row.get("detected_tag_count", ""),
                detected_ids=row.get("detected_ids", ""),
                measurement_source=row.get("measurement_source", ""),
                quality_status=row.get("quality_status", ""),
                reprojection_rmse_px=row.get("reprojection_rmse_px", ""),
                measured_x_m=row.get("camera_x_m", ""),
                measured_y_m=row.get("camera_y_m", ""),
                measured_z_m=row.get("camera_z_m", ""),
                measured_roll_deg=row.get("roll_deg", ""),
                measured_pitch_deg=row.get("pitch_deg", ""),
                measured_yaw_deg=row.get("yaw_deg", ""),
                image_path=f"frames/{frame:06d}.jpg" if args.extract_frames else "",
            )
            if pose is not None and np.isfinite(pose).all():
                quaternion = Rotation.from_euler("xyz", pose[3:], degrees=True).as_quat()
                assert origin is not None
                relative_position, relative_rotation = relative_pose(pose, origin)
                relative_quaternion = relative_rotation.as_quat()
                result.update(
                    optimized_x_m=f"{pose[0]:.8f}", optimized_y_m=f"{pose[1]:.8f}",
                    optimized_z_m=f"{pose[2]:.8f}",
                    optimized_qx=f"{quaternion[0]:.9f}", optimized_qy=f"{quaternion[1]:.9f}",
                    optimized_qz=f"{quaternion[2]:.9f}", optimized_qw=f"{quaternion[3]:.9f}",
                    relative_x_m=f"{relative_position[0]:.8f}",
                    relative_y_m=f"{relative_position[1]:.8f}",
                    relative_z_m=f"{relative_position[2]:.8f}",
                    relative_qx=f"{relative_quaternion[0]:.9f}",
                    relative_qy=f"{relative_quaternion[1]:.9f}",
                    relative_qz=f"{relative_quaternion[2]:.9f}",
                    relative_qw=f"{relative_quaternion[3]:.9f}",
                )
            writer.writerow(result)

    extracted = 0
    if args.extract_frames:
        extracted = extract_selected_frames(
            video, frame_numbers, output / "frames", args.frame_width, args.jpeg_quality
        )

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    metadata = {
        "format": FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "camera_family": args.camera_family,
        "raw_source": file_identity(args.source_raw.resolve() if args.source_raw else None),
        "panorama_source": file_identity(video),
        "coordinate_frames": {
            "measured": "AprilTag board/map frame",
            "optimized": "AprilTag board/map frame, Kalman + RTS",
            "relative": "first optimized pose (position and orientation are zero/identity)",
        },
        "labels": {
            "MEASURED": "direct decoded AprilTag PnP measurement",
            "TRACKED_FLOW": "PnP from forward/backward checked temporal corner tracking",
            "RECOVERED": "interpolated only across a short bounded visual gap",
            "PREDICTED": "short constant-velocity prediction",
            "LOST": "no publishable pose",
        },
        "counts": {"rows": len(rows), "states": dict(states), "extracted_frames": extracted},
        "filter": {
            "name": "constant-velocity Kalman + fixed-interval RTS",
            "median_window": args.median_window,
            "prediction_max_age_s": args.prediction_max_age,
            "predicted_or_recovered_are_not_direct_measurements": True,
        },
        "visual_summary": summary,
        "files": {
            "video": "media/panorama.mp4",
            "trajectory": "annotations/trajectory_6dof.csv",
            "direct_pose": "annotations/pose_direct.csv",
            "sensor_metadata": copied_sensor_files,
            "preview": "previews/trajectory_overlay.mp4" if args.preview and args.preview.is_file() else None,
        },
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
