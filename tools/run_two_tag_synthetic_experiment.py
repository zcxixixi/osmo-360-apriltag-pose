#!/usr/bin/env python3
"""Render two A3 AprilTags into a calibrated Osmo fisheye and run the locator."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from osmo360.localization.coordinate_frames import DJI_BODY_TO_PANORAMA_OPENCV
from tools._root import ROOT
PANO_ROOT = ROOT.parent / "panoforge-test"
CALIBRATION = Path("/home/cenxi/Videos/umi-captures/20260827/single-0063-smoke100-v1/metadata/calibration.json")
FRAME_SIZE = 1920
FPS = 30.0
FRAME_COUNT = 90
TAG_SIZE_M = 0.240
TAG_IDS = (200, 201)


def tag_corners(center: tuple[float, float, float], size: float = TAG_SIZE_M) -> np.ndarray:
    x, y, z = center
    half = size / 2
    return np.asarray([
        [x - half, y - half, z], [x + half, y - half, z],
        [x + half, y + half, z], [x - half, y + half, z],
    ], dtype=float)


def camera_pose(time_s: float) -> tuple[np.ndarray, Rotation]:
    phase = 2 * np.pi * time_s / ((FRAME_COUNT - 1) / FPS)
    position = np.asarray([0.050 * np.sin(phase), 0.030 * np.cos(phase), 0.025 * np.sin(2 * phase)])
    rotation = Rotation.from_euler(
        "xyz",
        [2.5 * np.sin(phase), 4.0 * np.cos(phase), 2.0 * np.sin(2 * phase)],
        degrees=True,
    )
    return position, rotation


def load_projection():
    sys.path.insert(0, str(PANO_ROOT))
    from app.core.maps import _quat_to_rot, _radial_model, scale_calibration_to_source

    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    calibration = scale_calibration_to_source(calibration, FRAME_SIZE, FRAME_SIZE)
    lens = calibration["lenses"][1]
    lens_rotation = _quat_to_rot(lens["extrinsic_quat"])
    radial = _radial_model(lens)
    centre = np.asarray([lens["cx"], lens["cy"]], dtype=float)

    def project(rays: np.ndarray) -> np.ndarray:
        rays = np.asarray(rays, dtype=float)
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        body = rays @ DJI_BODY_TO_PANORAMA_OPENCV
        lens_directions = body @ lens_rotation.T
        theta = np.arccos(np.clip(lens_directions[:, 2], -1.0, 1.0))
        rho = np.maximum(np.linalg.norm(lens_directions[:, :2], axis=1), 1e-12)
        radius = radial(theta)
        return np.c_[
            lens["cx"] + radius * lens_directions[:, 0] / rho,
            lens["cy"] + radius * lens_directions[:, 1] / rho,
        ]

    def pixels_to_rays(pixels: np.ndarray) -> np.ndarray:
        pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
        delta = pixels - centre
        radius = np.linalg.norm(delta, axis=1)
        low = np.zeros(len(pixels))
        high = np.full(len(pixels), np.radians(96.0))
        for _ in range(80):
            middle = (low + high) / 2
            below = radial(middle) < radius
            low = np.where(below, middle, low)
            high = np.where(below, high, middle)
        theta = (low + high) / 2
        safe_radius = np.maximum(radius, 1e-12)
        direction_lens = np.c_[
            np.sin(theta) * delta[:, 0] / safe_radius,
            np.sin(theta) * delta[:, 1] / safe_radius,
            np.cos(theta),
        ]
        direction_body = direction_lens @ lens_rotation
        rays = direction_body @ DJI_BODY_TO_PANORAMA_OPENCV.T
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)

    return project, pixels_to_rays


def render_tag(
    image: np.ndarray,
    marker: np.ndarray,
    center_world: np.ndarray,
    camera_position: np.ndarray,
    camera_rotation: Rotation,
    project,
    pixels_to_rays,
) -> None:
    board_size = TAG_SIZE_M * 1.18
    board_world = tag_corners(tuple(center_world), board_size)
    board_px = project(camera_rotation.inv().apply(board_world - camera_position))
    x0 = max(0, int(np.floor(board_px[:, 0].min())) - 4)
    x1 = min(image.shape[1] - 1, int(np.ceil(board_px[:, 0].max())) + 4)
    y0 = max(0, int(np.floor(board_px[:, 1].min())) - 4)
    y1 = min(image.shape[0] - 1, int(np.ceil(board_px[:, 1].max())) + 4)
    if x0 >= x1 or y0 >= y1:
        return
    grid_x, grid_y = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
    pixels = np.c_[grid_x.ravel(), grid_y.ravel()]
    rays_world = camera_rotation.apply(pixels_to_rays(pixels))
    scale = (center_world[2] - camera_position[2]) / np.maximum(rays_world[:, 2], 1e-12)
    hits = camera_position + rays_world * scale[:, None]
    local_x = hits[:, 0] - center_world[0]
    local_y = hits[:, 1] - center_world[1]
    valid_forward = scale > 0
    board_mask = valid_forward & (np.abs(local_x) <= board_size / 2) & (np.abs(local_y) <= board_size / 2)
    tag_mask = valid_forward & (np.abs(local_x) <= TAG_SIZE_M / 2) & (np.abs(local_y) <= TAG_SIZE_M / 2)
    patch_image = image[y0:y1 + 1, x0:x1 + 1].copy()
    patch = patch_image.reshape(-1, 3)
    patch[board_mask] = 248
    u = np.clip((local_x[tag_mask] / TAG_SIZE_M + .5) * (marker.shape[1] - 1), 0, marker.shape[1] - 1).astype(int)
    v = np.clip((local_y[tag_mask] / TAG_SIZE_M + .5) * (marker.shape[0] - 1), 0, marker.shape[0] - 1).astype(int)
    values = marker[v, u]
    patch[tag_mask] = values[:, None]
    image[y0:y1 + 1, x0:x1 + 1] = patch_image


def render_inputs(root: Path, tags_dir: Path) -> tuple[Path, Path, Path]:
    synthetic = root / "synthetic"
    synthetic.mkdir()
    world_tags = {
        200: tag_corners((-0.62, -0.16, 1.20)),
        201: tag_corners((-0.32, 0.14, 1.30)),
    }
    tag_map = {
        "schema_version": "world-apriltag-map/1.0",
        "map_id": "synthetic-two-a3-tags-20260828-v1",
        "calibration_status": "SYNTHETIC_GROUND_TRUTH",
        "world_frame": "tag_map",
        "world_origin": "synthetic camera nominal origin",
        "world_axes": "OpenCV x right, y down, z forward",
        "physical_up_vector": [0, -1, 0],
        "units": "m",
        "expected_ids": list(TAG_IDS),
        "tags": [
            {
                "id": tag_id,
                "corners_m": corners.tolist(),
                "panel": f"depth_{corners[0, 2]:.2f}m",
            }
            for tag_id, corners in world_tags.items()
        ],
    }
    tag_map_path = synthetic / "two_tag_world_map.json"
    tag_map_path.write_text(json.dumps(tag_map, indent=2) + "\n", encoding="utf-8")
    markers = {
        tag_id: cv2.imread(
            str(tags_dir / f"apriltag36h11_id{tag_id}_raw.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        for tag_id in TAG_IDS
    }
    if any(marker is None for marker in markers.values()):
        raise RuntimeError("missing generated raw tag image")
    project, pixels_to_rays = load_projection()
    video_path = synthetic / "two_tag_fisheye_ground_truth.mp4"
    video_writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
        FPS, (FRAME_SIZE, FRAME_SIZE),
    )
    if not video_writer.isOpened():
        raise RuntimeError("could not create synthetic fisheye video")
    truth_rows = []
    initial_rows = []
    y, x = np.mgrid[:FRAME_SIZE, :FRAME_SIZE]
    radial_background = np.sqrt(
        (x - FRAME_SIZE / 2) ** 2 + (y - FRAME_SIZE / 2) ** 2
    ) / (FRAME_SIZE / 2)
    base_background = np.clip(
        205 - 48 * radial_background + 7 * np.sin(x / 95) + 5 * np.cos(y / 71),
        105, 225,
    ).astype(np.uint8)
    for frame in range(FRAME_COUNT):
        time_s = frame / FPS
        position, rotation = camera_pose(time_s)
        image = cv2.cvtColor(base_background, cv2.COLOR_GRAY2BGR)
        for tag_id, corners_world in world_tags.items():
            render_tag(
                image,
                markers[tag_id],
                corners_world.mean(0),
                position,
                rotation,
                project,
                pixels_to_rays,
            )
        image = cv2.GaussianBlur(image, (3, 3), 0.55)
        cv2.putText(
            image, "SYNTHETIC TWO A3 TAGS / DJI STREAM 1 MODEL",
            (42, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
            (60, 60, 60), 2, cv2.LINE_AA,
        )
        video_writer.write(image)
        quaternion = rotation.as_quat()
        truth_rows.append([frame, time_s, *position, *quaternion])
        offset_rotation = rotation * Rotation.from_euler(
            "xyz", [0.4, -0.3, 0.5], degrees=True,
        )
        initial_rows.append([
            frame, time_s,
            *(position + np.asarray([0.005, -0.004, 0.006])),
            *offset_rotation.as_quat(),
        ])
    video_writer.release()
    fields = [
        "frame", "timestamp", "camera_x_m", "camera_y_m", "camera_z_m",
        "qx", "qy", "qz", "qw",
    ]
    truth_path = synthetic / "ground_truth_pose.csv"
    initial_path = synthetic / "perturbed_initial_pose.csv"
    for path, rows in ((truth_path, truth_rows), (initial_path, initial_rows)):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(fields)
            writer.writerows(rows)
    return video_path, tag_map_path, initial_path


def load_pose(path: Path) -> tuple[np.ndarray, np.ndarray, Rotation]:
    rows = [
        row for row in csv.DictReader(path.open(newline="", encoding="utf-8"))
        if row.get("camera_x_m")
    ]
    times = np.asarray([float(row["timestamp"]) for row in rows])
    positions = np.asarray([
        [float(row[key]) for key in ("camera_x_m", "camera_y_m", "camera_z_m")]
        for row in rows
    ])
    rotations = Rotation.from_quat([
        [float(row[key]) for key in ("qx", "qy", "qz", "qw")]
        for row in rows
    ])
    return times, positions, rotations


def render_demo(
    output: Path,
    source_video: Path,
    times: np.ndarray,
    estimated: np.ndarray,
    truth: np.ndarray,
    position_error: np.ndarray,
    rotation_error: np.ndarray,
    report: dict,
) -> None:
    capture = cv2.VideoCapture(str(source_video))
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1280, 720),
    )
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    detector = cv2.aruco.ArucoDetector(dictionary)
    sample_frames = np.rint(times * FPS).astype(int)
    all_xy = np.vstack((truth[:, :2], estimated[:, :2]))
    low = all_xy.min(0)
    span = np.maximum(all_xy.max(0) - low, 1e-4)
    project, _ = load_projection()
    world_tags = {
        200: tag_corners((-0.62, -0.16, 1.20)),
        201: tag_corners((-0.32, 0.14, 1.30)),
    }

    def points(values: np.ndarray) -> np.ndarray:
        normalized = (values[:, :2] - low) / span
        return np.c_[
            770 + normalized[:, 0] * 430,
            620 - normalized[:, 1] * 315,
        ].astype(np.int32)

    for frame_index in range(FRAME_COUNT):
        ok, frame = capture.read()
        if not ok:
            break
        display = cv2.resize(frame, (720, 720), interpolation=cv2.INTER_AREA)
        truth_position, truth_rotation = camera_pose(frame_index / FPS)
        for tag_id, corners_world in world_tags.items():
            quad = project(
                truth_rotation.inv().apply(corners_world - truth_position)
            ) * (720.0 / FRAME_SIZE)
            quad_int = np.round(quad).astype(np.int32)
            cv2.polylines(display, [quad_int], True, (255, 210, 40), 2, cv2.LINE_AA)
            cv2.putText(
                display, f"map ID {tag_id}", tuple(quad_int[0]),
                cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 210, 40), 1,
            )
        quads, ids, _ = detector.detectMarkers(display)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(display, quads, ids)
        canvas = np.full((720, 1280, 3), (12, 18, 25), np.uint8)
        canvas[:, :720] = display
        sample = int(np.argmin(np.abs(sample_frames - frame_index)))
        cv2.putText(canvas, "CURRENT LOCATOR / TWO A3 TAGS", (742, 45), cv2.FONT_HERSHEY_SIMPLEX, .72, (80, 215, 245), 2)
        cv2.putText(canvas, f"frame {frame_index + 1}/{FRAME_COUNT}  IDs 200 + 201", (742, 82), cv2.FONT_HERSHEY_SIMPLEX, .58, (210, 220, 230), 1)
        cv2.putText(canvas, f"position error  {position_error[sample]:.3f} mm", (742, 140), cv2.FONT_HERSHEY_SIMPLEX, .70, (120, 230, 145), 2)
        cv2.putText(canvas, f"orientation error  {rotation_error[sample]:.4f} deg", (742, 178), cv2.FONT_HERSHEY_SIMPLEX, .62, (120, 230, 145), 2)
        cv2.putText(canvas, f"valid ratio  {report['frames']['valid_ratio'] * 100:.1f}%", (742, 216), cv2.FONT_HERSHEY_SIMPLEX, .62, (230, 190, 80), 2)
        cv2.rectangle(canvas, (742, 260), (1240, 650), (55, 70, 82), 1)
        ground_truth_points = points(truth[: sample + 1])
        estimated_points = points(estimated[: sample + 1])
        if len(ground_truth_points) > 1:
            cv2.polylines(canvas, [ground_truth_points], False, (60, 190, 255), 4, cv2.LINE_AA)
            cv2.polylines(canvas, [estimated_points], False, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.putText(canvas, "orange = ground truth", (760, 290), cv2.FONT_HERSHEY_SIMPLEX, .50, (60, 190, 255), 1)
        cv2.putText(canvas, "white = current locator", (995, 290), cv2.FONT_HERSHEY_SIMPLEX, .50, (235, 235, 235), 1)
        cv2.putText(canvas, "SYNTHETIC VALIDATION - NOT PHYSICAL ACCURACY", (748, 690), cv2.FONT_HERSHEY_SIMPLEX, .48, (100, 115, 130), 1)
        writer.write(canvas)
    capture.release()
    writer.release()


def evaluate(root: Path, video: Path, locator_dir: Path) -> dict:
    estimated_t, estimated_p, estimated_r = load_pose(locator_dir / "pose.csv")
    truth_positions = []
    truth_quaternions = []
    for time_s in estimated_t:
        position, rotation = camera_pose(float(time_s))
        truth_positions.append(position)
        truth_quaternions.append(rotation.as_quat())
    truth_p = np.asarray(truth_positions)
    truth_r = Rotation.from_quat(truth_quaternions)
    position_error = np.linalg.norm(estimated_p - truth_p, axis=1) * 1000
    rotation_error = np.degrees((truth_r.inv() * estimated_r).magnitude())
    summary = json.loads((locator_dir / "summary.json").read_text(encoding="utf-8"))

    def metrics(values: np.ndarray) -> dict:
        return {
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }

    report = {
        "schema_version": "two-tag-synthetic-experiment/1.0",
        "status": "PASS" if (
            summary["valid_ratio"] >= 0.95
            and np.percentile(position_error, 95) <= 10
            and np.percentile(rotation_error, 95) <= 1
        ) else "FAIL",
        "synthetic_ground_truth": True,
        "initial_pose_perturbation": {
            "translation_mm": [5.0, -4.0, 6.0],
            "rotation_xyz_deg": [0.4, -0.3, 0.5],
        },
        "tags": {
            "family": "AprilTag 36h11",
            "ids": list(TAG_IDS),
            "outer_size_m": TAG_SIZE_M,
            "depths_m": [1.20, 1.30],
        },
        "frames": {
            "video": FRAME_COUNT,
            "locator_samples": int(summary["common_frames"]),
            "valid": int(summary["valid_frames"]),
            "valid_ratio": summary["valid_ratio"],
        },
        "position_error_mm": metrics(position_error),
        "orientation_error_deg": metrics(rotation_error),
        "angular_fit_rmse_deg": summary["angular_rmse_deg"],
        "artifacts": {
            "video": str(video),
            "pose_csv": str(locator_dir / "pose.csv"),
        },
        "limitations": [
            "synthetic tags are rendered by calibrated fisheye ray/plane intersection",
            "does not test print flatness, blur, glare, rolling shutter, wall survey, or dual-lens fusion",
        ],
    }
    (root / "experiment_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    render_demo(
        root / "two_tag_locator_demo.mp4", video, estimated_t,
        estimated_p, truth_p, position_error, rotation_error, report,
    )
    return report


def main() -> int:
    root = Path("/home/cenxi/Videos/umi-captures/20260827/two-tag-a3-experiment-v9")
    tags_dir = root / "tags"
    video, tag_map, initial = render_inputs(root, tags_dir)
    locator_dir = root / "locator-output"
    locator_dir.mkdir()
    command = [
        str(ROOT / ".venv/bin/python"), "-m", "osmo360.localization.raw_fisheye_world_pose",
        str(video), "--calibration", str(CALIBRATION), "--tag-map", str(tag_map),
        "--output-dir", str(locator_dir), "--panoforge-root", str(PANO_ROOT),
        "--stream", "1", "--source-width", str(FRAME_SIZE),
        "--source-height", str(FRAME_SIZE), "--common-duration-s",
        str((FRAME_COUNT - 1) / FPS), "--sample-fps", "10",
        "--initial-pose", str(initial), "--min-tags", "2",
        "--max-angular-rmse-deg", "1.5", "--radial-model", "stitch",
        "--edge-rectification",
    ]
    subprocess.run(command, check=True)
    report = evaluate(root, video, locator_dir)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
