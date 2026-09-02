#!/usr/bin/env python3
"""Calibrate and visualize a real one-camera/two-screen AprilTag trajectory test."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from osmo360.localization.raw_fisheye_world_pose import (
    detect_rectified_tags,
    make_ray_converter,
    solve_pose,
)
from tools.run_two_tag_synthetic_experiment import verify_freeze


from tools._root import ROOT
PANO_ROOT = ROOT.parent / "panoforge-test"
TAG_IDS = (200, 201)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag-size-mm", type=float, default=240.0)
    parser.add_argument("--stream", type=int, default=1)
    parser.add_argument("--sample-fps", type=float, default=10.0)
    parser.add_argument("--observation-cache", type=Path)
    return parser.parse_args()


def local_corners(size_m: float) -> np.ndarray:
    half = size_m / 2
    return np.asarray([
        [-half, -half, 0.0], [half, -half, 0.0],
        [half, half, 0.0], [-half, half, 0.0],
    ])



def make_dense_rectified_maps(calibration: dict, stream: int):
    from app.core.maps import _radial_model

    lens = calibration["lenses"][stream]
    radial_model = _radial_model(lens)
    size = 700
    fov = np.radians(65.0)
    coordinate = (
        (np.arange(size) + 0.5 - size / 2.0) / (size / 2.0)
        * np.tan(fov / 2.0)
    )
    xx, yy = np.meshgrid(coordinate, coordinate)
    base = np.stack([xx, yy, np.ones_like(xx)], axis=-1)
    base /= np.linalg.norm(base, axis=-1, keepdims=True)
    maps = []
    for yaw in (-70, -35, 0, 35, 70):
        for pitch in (-70, -35, 0, 35, 70):
            centre_rotation = Rotation.from_euler("yx", [yaw, pitch], degrees=True)
            rays = centre_rotation.apply(base.reshape(-1, 3)).reshape(base.shape)
            theta = np.arccos(np.clip(rays[..., 2], -1.0, 1.0))
            planar_radius = np.linalg.norm(rays[..., :2], axis=-1)
            image_radius = radial_model(theta)
            xmap = (
                lens["cx"] + image_radius * rays[..., 0]
                / np.maximum(planar_radius, 1e-9)
            ).astype(np.float32)
            ymap = (
                lens["cy"] + image_radius * rays[..., 1]
                / np.maximum(planar_radius, 1e-9)
            ).astype(np.float32)
            maps.append((xmap, ymap))
    return maps

def detector_and_geometry(args: argparse.Namespace):
    geometry_args = SimpleNamespace(
        calibration=args.calibration.resolve(), panoforge_root=PANO_ROOT.resolve(),
        source_width=1920, source_height=1920, stream=args.stream,
        radial_model="stitch", edge_rectification=True,
        rectified_view_size=700, rectification_radial_model="stitch",
    )
    convert, calibration = make_ray_converter(geometry_args)
    rectified_maps = make_dense_rectified_maps(calibration, args.stream)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.adaptiveThreshWinSizeMax = 63
    return cv2.aruco.ArucoDetector(dictionary, parameters), convert, rectified_maps


def detect_samples(args: argparse.Namespace, detector, rectified_maps) -> tuple[list[dict], float, int]:
    capture = cv2.VideoCapture(str(args.video))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, round(fps / args.sample_fps))
    samples = []
    frame_index = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        if frame_index % stride:
            frame_index += 1
            continue
        direct_quads, direct_ids, _ = detector.detectMarkers(image)
        detections = {} if direct_ids is None else {
            int(tag_id): quad.reshape(4, 2).astype(np.float32)
            for tag_id, quad in zip(direct_ids.flatten(), direct_quads)
            if int(tag_id) in TAG_IDS
        }
        direct_ids_set = set(detections)
        for tag_id, quad in detect_rectified_tags(image, detector, rectified_maps):
            if tag_id in TAG_IDS and tag_id not in detections:
                detections[tag_id] = quad
        samples.append({
            "frame": frame_index,
            "time_s": frame_index / fps,
            "direct_ids": sorted(direct_ids_set),
            "detections": {str(tag_id): quad.tolist() for tag_id, quad in detections.items()},
        })
        frame_index += 1
    capture.release()
    return samples, fps, frame_count


def samples_from_cache(path: Path) -> tuple[list[dict], float, int, int]:
    cache = np.load(path)
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    stride = int(metadata["frame_stride"])
    selected = {}
    for index, (frame, tag_id) in enumerate(
        zip(cache["frame_index"], cache["tag_id"])
    ):
        tag_id = int(tag_id)
        if tag_id not in TAG_IDS:
            continue
        key = (int(frame), tag_id)
        if key not in selected or cache["area_px2"][index] > cache["area_px2"][selected[key]]:
            selected[key] = index
    samples = []
    for frame in range(0, int(metadata["frame_count"]), stride):
        detections = {}
        direct_ids = []
        for tag_id in TAG_IDS:
            index = selected.get((frame, tag_id))
            if index is None:
                continue
            detections[str(tag_id)] = cache["corners_px"][index].tolist()
            if str(cache["detection_source"][index]) == "direct_raw":
                direct_ids.append(tag_id)
        samples.append({
            "frame": frame,
            "time_s": frame / float(metadata["fps"]),
            "direct_ids": sorted(direct_ids),
            "detections": detections,
        })
    cache.close()
    return samples, float(metadata["fps"]), int(metadata["frame_count"]), stride


def pose_from_tag(corners_m: np.ndarray, quad: np.ndarray, convert) -> tuple[np.ndarray, Rotation, float]:
    rays = convert(np.asarray(quad, dtype=float))
    mean_ray = rays.mean(axis=0)
    mean_ray /= np.linalg.norm(mean_ray)
    tangent_from_camera = Rotation.align_vectors(
        np.asarray([[0.0, 0.0, 1.0]]), mean_ray.reshape(1, 3)
    )[0]
    tangent_rays = tangent_from_camera.apply(rays)
    normalized = tangent_rays[:, :2] / tangent_rays[:, 2:3]
    result = cv2.solvePnPGeneric(
        corners_m.astype(np.float64), normalized.astype(np.float64),
        np.eye(3, dtype=np.float64), None, flags=cv2.SOLVEPNP_IPPE,
    )
    candidates = []
    camera_from_tangent = tangent_from_camera.inv()
    for rvec, tvec in zip(result[1], result[2]):
        tangent_from_tag = Rotation.from_rotvec(rvec.reshape(3))
        tangent_translation = tvec.reshape(3)
        points_tangent = tangent_from_tag.apply(corners_m) + tangent_translation
        if np.all(points_tangent[:, 2] > 0):
            camera_from_tag = camera_from_tangent * tangent_from_tag
            camera_translation = camera_from_tangent.apply(tangent_translation)
            tag_from_camera = camera_from_tag.inv()
            position_tag_camera = -tag_from_camera.apply(camera_translation)
            position, rotation, angular = solve_pose(
                corners_m, rays, (position_tag_camera, tag_from_camera),
                regularize=False,
            )
            candidates.append((
                position, rotation,
                float(np.sqrt(np.mean(angular ** 2))),
            ))
    if not candidates:
        raise ValueError("no positive-depth planar pose branch")
    return min(candidates, key=lambda item: item[2])


def compose(parent_child: tuple[np.ndarray, Rotation], child_grandchild: tuple[np.ndarray, Rotation]) -> tuple[np.ndarray, Rotation]:
    p_ab, r_ab = parent_child;p_bc, r_bc = child_grandchild
    return p_ab + r_ab.apply(p_bc), r_ab * r_bc


def inverse(transform: tuple[np.ndarray, Rotation]) -> tuple[np.ndarray, Rotation]:
    position, rotation = transform
    inv_rotation = rotation.inv()
    return -inv_rotation.apply(position), inv_rotation


def calibrate_map(samples: list[dict], convert, size_m: float) -> tuple[dict, list[dict], dict]:
    corners = local_corners(size_m)
    relative = []
    pose_observations = []
    failures = 0
    for sample in samples:
        poses = {}
        for tag_id_text, quad_list in sample["detections"].items():
            tag_id = int(tag_id_text)
            try:
                position, rotation, rmse = pose_from_tag(corners, np.asarray(quad_list), convert)
            except ValueError:
                failures += 1
                continue
            if rmse <= 1.0:
                poses[tag_id] = (position, rotation, rmse)
        if poses:
            pose_observations.append({
                "time_s": sample["time_s"],
                "frame": sample["frame"],
                "poses": poses,
            })
        if 200 in poses and 201 in poses:
            tag200_camera = poses[200][:2]
            tag201_camera = poses[201][:2]
            tag200_tag201 = compose(tag200_camera, inverse(tag201_camera))
            relative.append(tag200_tag201)
    if len(relative) < 5:
        raise RuntimeError(f"only {len(relative)} joint two-tag pose samples; cannot calibrate map")
    translations = np.asarray([item[0] for item in relative])
    rotations = Rotation.from_quat([item[1].as_quat() for item in relative])
    centre_translation = np.median(translations, axis=0)
    centre_rotation = rotations.mean()
    translation_residual_mm = np.linalg.norm(translations - centre_translation, axis=1) * 1000
    rotation_residual_deg = np.degrees((centre_rotation.inv() * rotations).magnitude())
    keep = (
        translation_residual_mm <= np.percentile(translation_residual_mm, 90)
    ) & (
        rotation_residual_deg <= np.percentile(rotation_residual_deg, 90)
    )
    centre_translation = np.median(translations[keep], axis=0)
    centre_rotation = Rotation.from_quat(rotations.as_quat()[keep]).mean()
    tag200 = corners
    tag201 = centre_rotation.apply(corners) + centre_translation
    pose_samples = []
    tag200_tag201 = (centre_translation, centre_rotation)
    for observation in pose_observations:
        poses = observation["poses"]
        if 200 in poses:
            tag200_camera = poses[200][:2]
        elif 201 in poses:
            tag200_camera = compose(tag200_tag201, poses[201][:2])
        else:
            continue
        pose_samples.append({
            "time_s": observation["time_s"],
            "frame": observation["frame"],
            "tag200_camera": tag200_camera,
        })
    tag_map = {
        "schema_version": "world-apriltag-map/1.0",
        "map_id": "two-screen-self-calibrated-v1",
        "calibration_status": "SELF_CALIBRATED_FROM_SAME_CAPTURE_NOT_ABSOLUTE_GROUND_TRUTH",
        "world_frame": "tag200_map",
        "world_origin": "center of screen AprilTag ID200",
        "world_axes": "ID200 local x right, y down, z out of screen",
        "physical_up_vector": [0, -1, 0],
        "units": "m",
        "expected_ids": [200, 201],
        "tag_size_assumption_m": size_m,
        "tags": [
            {"id": 200, "corners_m": tag200.tolist(), "panel": "screen_id200"},
            {"id": 201, "corners_m": tag201.tolist(), "panel": "screen_id201"},
        ],
    }
    calibration_audit = {
        "joint_samples": len(relative), "kept_samples": int(keep.sum()),
        "single_tag_pose_failures": failures,
        "tag200_to_tag201_translation_m": centre_translation.tolist(),
        "tag200_to_tag201_quaternion_xyzw": centre_rotation.as_quat().tolist(),
        "relative_translation_scatter_mm": {
            "median": float(np.median(translation_residual_mm)),
            "p95": float(np.percentile(translation_residual_mm, 95)),
            "max": float(translation_residual_mm.max()),
        },
        "relative_orientation_scatter_deg": {
            "median": float(np.median(rotation_residual_deg)),
            "p95": float(np.percentile(rotation_residual_deg, 95)),
            "max": float(rotation_residual_deg.max()),
        },
    }
    return tag_map, pose_samples, calibration_audit


def write_initial_pose(path: Path, pose_samples: list[dict]) -> None:
    fields = ["frame", "timestamp", "camera_x_m", "camera_y_m", "camera_z_m", "qx", "qy", "qz", "qw"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields);writer.writeheader()
        for sample in pose_samples:
            position, rotation = sample["tag200_camera"]
            quaternion = rotation.as_quat()
            writer.writerow({
                "frame": sample["frame"], "timestamp": f"{sample['time_s']:.6f}",
                "camera_x_m": position[0], "camera_y_m": position[1], "camera_z_m": position[2],
                "qx": quaternion[0], "qy": quaternion[1], "qz": quaternion[2], "qw": quaternion[3],
            })


def load_locator_pose(path: Path) -> tuple[np.ndarray, np.ndarray, Rotation, list[dict]]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    valid = [row for row in rows if row.get("quality_status") == "valid"]
    times = np.asarray([float(row["timestamp"]) for row in valid])
    positions = np.asarray([[float(row[key]) for key in ("camera_x_m", "camera_y_m", "camera_z_m")] for row in valid])
    rotations = Rotation.from_quat([[float(row[key]) for key in ("qx", "qy", "qz", "qw")] for row in valid])
    return times, positions, rotations, rows


def stats(values: np.ndarray) -> dict:
    return {"median": float(np.median(values)), "p95": float(np.percentile(values, 95)), "max": float(np.max(values))}


def evaluate(locator_dir: Path, samples: list[dict], fps: float, frame_count: int, calibration_audit: dict, tag_size_m: float) -> tuple[dict, tuple]:
    times, positions, rotations, rows = load_locator_pose(locator_dir / "pose.csv")
    position_steps = np.linalg.norm(np.diff(positions, axis=0), axis=1) * 1000
    orientation_steps = np.degrees((rotations[:-1].inv() * rotations[1:]).magnitude())
    path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    window = max(2, min(5, len(positions) // 4))
    start_position = np.median(positions[:window], axis=0)
    end_position = np.median(positions[-window:], axis=0)
    start_rotation = Rotation.from_quat(rotations.as_quat()[:window]).mean()
    end_rotation = Rotation.from_quat(rotations.as_quat()[-window:]).mean()
    start_end_position_mm = float(np.linalg.norm(end_position - start_position) * 1000)
    start_end_orientation_deg = float(np.degrees((start_rotation.inv() * end_rotation).magnitude()))
    summary = json.loads((locator_dir / "summary.json").read_text(encoding="utf-8"))
    direct_both = sum(set(sample["direct_ids"]) == set(TAG_IDS) for sample in samples)
    any_both = sum(set(sample["detections"]) == {"200", "201"} for sample in samples)
    report = {
        "schema_version": "physical-two-screen-camera-test/1.0",
        "status": "DIAGNOSTIC_SELF_CALIBRATED",
        "absolute_accuracy_verified": False,
        "tag_size": {"assumed_black_outer_size_m": tag_size_m, "measured": False},
        "source": {"video": str(args.video.resolve()), "fps": fps, "frames": frame_count, "duration_s": frame_count / fps},
        "coverage": {
            "samples": len(samples), "direct_both": direct_both,
            "direct_both_ratio": direct_both / len(samples),
            "with_rectification_both": any_both,
            "with_rectification_both_ratio": any_both / len(samples),
            "locator_valid_ratio": summary["valid_ratio"],
        },
        "angular_rmse_deg": summary["angular_rmse_deg"],
        "trajectory": {
            "valid_samples": len(times), "path_length_m": path_length,
            "position_step_mm": stats(position_steps),
            "orientation_step_deg": stats(orientation_steps),
            "start_end_position_difference_mm": start_end_position_mm,
            "start_end_orientation_difference_deg": start_end_orientation_deg,
        },
        "map_self_calibration": calibration_audit,
        "limitations": [
            "both screen Tag sizes are assumed to be 240 mm and must be physically measured",
            "Tag map is estimated from this same capture, so this is continuity/repeatability evidence, not absolute accuracy",
            "screen refresh, glare, and display scaling remain part of the physical test",
        ],
    }
    return report, (times, positions, rotations, rows)


def render_demo(output: Path, video: Path, samples: list[dict], pose_data: tuple, report: dict, fps: float, frame_count: int) -> None:
    times, positions, rotations, _ = pose_data
    capture = cv2.VideoCapture(str(video))
    output_fps = 30.0
    output_frames = round(frame_count / fps * output_fps)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (1280, 720))
    slerp = Slerp(times, rotations)
    query_times = np.clip(np.arange(output_frames) / output_fps, times[0], times[-1])
    query_positions = np.column_stack([np.interp(query_times, times, positions[:, axis]) for axis in range(3)])
    query_rotations = slerp(query_times)
    xy = positions[:, [0, 2]];low = xy.min(0);span = np.maximum(xy.max(0) - low, 1e-4)
    colors = {200: (60, 220, 255), 201: (100, 230, 120)}
    current_source_frame = -1
    frame = None
    for output_index, time_s in enumerate(query_times):
        source_frame = min(frame_count - 1, round(time_s * fps))
        ok = True
        while current_source_frame < source_frame:
            ok, frame = capture.read()
            current_source_frame += 1
            if not ok:
                break
        if not ok or frame is None:
            break
        view = cv2.resize(frame, (720, 720), interpolation=cv2.INTER_AREA)
        sample = min(samples, key=lambda item: abs(item["time_s"] - time_s))
        for tag_id_text, quad in sample["detections"].items():
            tag_id = int(tag_id_text);scaled = np.round(np.asarray(quad) * 720 / 1920).astype(np.int32)
            cv2.polylines(view, [scaled], True, colors[tag_id], 3, cv2.LINE_AA)
            cv2.putText(view, f"ID {tag_id}", tuple(scaled[0]), cv2.FONT_HERSHEY_SIMPLEX, .55, colors[tag_id], 2)
        canvas = np.full((720, 1280, 3), (12, 18, 25), np.uint8);canvas[:, :720] = view
        cv2.putText(canvas, f"REAL {args.video.stem.upper()} / TWO SCREEN TAGS", (740, 42), cv2.FONT_HERSHEY_SIMPLEX, .58, (80, 215, 245), 2)
        cv2.putText(canvas, f"time {time_s:5.2f} s   tag size ASSUMED 240 mm", (740, 78), cv2.FONT_HERSHEY_SIMPLEX, .50, (215, 220, 228), 1)
        cv2.putText(canvas, f"coverage rectified {report['coverage']['with_rectification_both_ratio']*100:.1f}%", (740, 125), cv2.FONT_HERSHEY_SIMPLEX, .60, (120, 230, 145), 2)
        cv2.putText(canvas, f"angular RMSE P95 {report['angular_rmse_deg']['p95']:.3f} deg", (740, 160), cv2.FONT_HERSHEY_SIMPLEX, .58, (120, 230, 145), 2)
        cv2.putText(canvas, f"path length {report['trajectory']['path_length_m']:.3f} m", (740, 195), cv2.FONT_HERSHEY_SIMPLEX, .58, (230, 190, 80), 2)
        cv2.rectangle(canvas, (740, 235), (1245, 640), (50, 65, 78), 1)
        track = query_positions[:output_index + 1, [0, 2]]
        normalized = (track - low) / span
        points = np.c_[770 + normalized[:, 0] * 430, 600 - normalized[:, 1] * 320].astype(np.int32)
        if len(points)>1:cv2.polylines(canvas,[points],False,(60,190,255),3,cv2.LINE_AA)
        cv2.circle(canvas,tuple(points[-1]),6,(245,245,245),-1,cv2.LINE_AA)
        cv2.putText(canvas,"camera trajectory in tag200 map (X-Z)",(760,270),cv2.FONT_HERSHEY_SIMPLEX,.52,(200,210,220),1)
        cv2.putText(canvas,"SELF-CALIBRATED MAP - NOT ABSOLUTE ACCURACY",(744,690),cv2.FONT_HERSHEY_SIMPLEX,.46,(105,120,132),1)
        writer.write(canvas)
    capture.release();writer.release()


def main() -> int:
    global args
    args = parse_args()
    verify_freeze()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.observation_cache:
        geometry_args = SimpleNamespace(
            calibration=args.calibration.resolve(),
            panoforge_root=PANO_ROOT.resolve(),
            source_width=1920, source_height=1920,
            stream=args.stream, radial_model="stitch",
        )
        convert, _ = make_ray_converter(geometry_args)
        samples, fps, frame_count, cache_stride = samples_from_cache(
            args.observation_cache.resolve()
        )
    else:
        detector, convert, rectified_maps = detector_and_geometry(args)
        samples, fps, frame_count = detect_samples(
            args, detector, rectified_maps
        )
        cache_stride = max(1, round(fps / args.sample_fps))
    (args.output_dir / "detections.json").write_text(
        json.dumps(samples) + "\n", encoding="utf-8"
    )
    tag_map, pose_samples, calibration_audit = calibrate_map(
        samples, convert, args.tag_size_mm / 1000
    )
    tag_map_path = args.output_dir / "two_screen_tag_map.json"
    tag_map_path.write_text(
        json.dumps(tag_map, indent=2) + "\n", encoding="utf-8"
    )
    initial_path = args.output_dir / "single_tag_initial_pose.csv"
    write_initial_pose(initial_path, pose_samples)
    locator_dir = args.output_dir / "locator-output"
    if args.observation_cache:
        command = [
            str(ROOT / ".venv/bin/python"),
            "-m",
            "tools.raw_fisheye_world_pose_cached",
            "--observation-cache", str(args.observation_cache.resolve()),
            "--tag-map", str(tag_map_path),
            "--initial-pose", str(initial_path),
            "--sample-stride", str(cache_stride),
            "--min-tags", "1",
            "--max-angular-rmse-deg", "1.5",
            "--prior-policy", "initial-first",
            "--regularize-prior",
            "--output-dir", str(locator_dir),
        ]
    else:
        command = [
            str(ROOT / ".venv/bin/python"),
            "-m",
            "osmo360.localization.raw_fisheye_world_pose",
            str(args.video.resolve()),
            "--calibration", str(args.calibration.resolve()),
            "--tag-map", str(tag_map_path),
            "--output-dir", str(locator_dir),
            "--panoforge-root", str(PANO_ROOT),
            "--stream", str(args.stream),
            "--source-width", "1920", "--source-height", "1920",
            "--common-duration-s", str(frame_count / fps - 1 / fps),
            "--sample-fps", str(args.sample_fps),
            "--initial-pose", str(initial_path),
            "--min-tags", "2", "--max-angular-rmse-deg", "1.5",
            "--radial-model", "stitch", "--edge-rectification",
            "--rectified-view-size", "1400",
        ]
    subprocess.run(command, check=True)
    report, pose_data = evaluate(locator_dir, samples, fps, frame_count, calibration_audit, args.tag_size_mm / 1000)
    (args.output_dir / "physical_experiment_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    render_demo(args.output_dir / "physical_two_tag_camera_demo.mp4", args.video, samples, pose_data, report, fps, frame_count)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
