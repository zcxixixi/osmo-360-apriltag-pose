#!/usr/bin/env python3
"""Render an auditable Insta360 AprilTag/OptiTrack comparison video."""

from __future__ import annotations

import argparse
import csv
import json
import struct
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


BG = (18, 20, 25)
WHITE = (235, 238, 242)
MUTED = (145, 154, 166)
GREEN = (84, 214, 132)
AMBER = (59, 193, 255)
RED = (82, 82, 245)
CYAN = (245, 196, 72)


def text(image: np.ndarray, value: str, xy: tuple[int, int], scale: float = 0.65,
         color: tuple[int, int, int] = WHITE, thickness: int = 1) -> None:
    cv2.putText(image, value, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


def direct_row(row: dict[str, str], min_tags: int = 2) -> bool:
    return (
        row.get("quality_status") == "valid"
        and bool(row.get("camera_x_m"))
        and int(row.get("detected_tag_count") or 0) >= min_tags
        and row.get("measurement_source", "direct") in ("", "direct")
    )


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def matched_rows(path: Path) -> dict[int, dict[str, str]]:
    return {int(row["video_frame"]): row for row in load_rows(path)}


def world_bounds(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    values = []
    for row in rows:
        for prefix in ("truth", "estimate"):
            values.append([float(row[f"{prefix}_{axis}_m"]) for axis in "xyz"])
    points = np.asarray(values)
    low, high = np.percentile(points, [1, 99], axis=0)
    center = (low + high) / 2.0
    radius = max(float(np.max(high - low)) / 2.0, 0.15)
    return center - radius, center + radius


def project(point: np.ndarray, low: np.ndarray, high: np.ndarray,
            origin: tuple[int, int], size: tuple[int, int]) -> tuple[int, int]:
    # Isometric projection: world Z is visibly vertical while XY gives depth.
    normalized = (point - low) / np.maximum(high - low, 1e-9) - 0.5
    u = normalized[0] - 0.55 * normalized[1]
    v = -normalized[2] + 0.32 * normalized[0] + 0.32 * normalized[1]
    return (
        int(origin[0] + size[0] * (0.5 + 0.72 * u)),
        int(origin[1] + size[1] * (0.5 + 0.72 * v)),
    )


def draw_axes(image: np.ndarray, position: np.ndarray, quaternion: np.ndarray,
              low: np.ndarray, high: np.ndarray, origin: tuple[int, int],
              size: tuple[int, int], label: str) -> None:
    rotation = Rotation.from_quat(quaternion).as_matrix()
    length = max(float(np.max(high - low)) * 0.08, 0.04)
    start = project(position, low, high, origin, size)
    for axis, color in zip(range(3), ((55, 80, 245), (75, 220, 105), (245, 155, 45))):
        end = project(position + rotation[:, axis] * length, low, high, origin, size)
        cv2.arrowedLine(image, start, end, color, 2, cv2.LINE_AA, tipLength=0.25)
    text(image, label, (start[0] + 6, start[1] - 7), 0.44, WHITE, 1)


def load_gripper_edges(mesh_dir: Path, max_triangles_per_mesh: int = 80) -> np.ndarray:
    """Load a light wireframe from the supplied CAD STL assembly."""
    edge_sets = []
    for name in ("base_link.STL", "Link1.STL", "Link2.STL", "Link3.STL"):
        data = (mesh_dir / name).read_bytes()
        count = struct.unpack_from("<I", data, 80)[0]
        dtype = np.dtype([
            ("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2"),
        ])
        triangles = np.frombuffer(data, dtype=dtype, offset=84, count=count)["vertices"].astype(float)
        indices = np.unique(np.linspace(0, len(triangles) - 1,
                                        min(max_triangles_per_mesh, len(triangles))).astype(int))
        selected = triangles[indices]
        edge_sets.extend((selected[:, [0, 1]], selected[:, [1, 2]], selected[:, [2, 0]]))
    edges = np.concatenate(edge_sets)
    # CAD assembly origin is at the base. A small fixed offset places its body
    # symmetrically around the tracked camera pose without altering orientation.
    edges -= np.asarray([0.025, 0.0, 0.025])
    return edges


def draw_gripper(image: np.ndarray, position: np.ndarray, quaternion: np.ndarray,
                 edges: np.ndarray, low: np.ndarray, high: np.ndarray,
                 origin: tuple[int, int], size: tuple[int, int],
                 color: tuple[int, int, int], label: str, label_y_offset: int = -7) -> None:
    rotation = Rotation.from_quat(quaternion).as_matrix()
    world_edges = edges @ rotation.T + position
    overlay = image.copy()
    for segment in world_edges:
        first = project(segment[0], low, high, origin, size)
        second = project(segment[1], low, high, origin, size)
        cv2.line(overlay, first, second, color, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.68, image, 0.32, 0, image)
    anchor = project(position, low, high, origin, size)
    cv2.circle(image, anchor, 4, color, -1, cv2.LINE_AA)
    text(image, label, (anchor[0] + 7, anchor[1] + label_y_offset), 0.44, color, 1)


def transcode_h264(intermediate: Path, output: Path) -> None:
    command = [
        "gst-launch-1.0", "-q", "filesrc", f"location={intermediate}", "!",
        "qtdemux", "!", "queue", "!", "avdec_mpeg4", "!", "videoconvert", "!",
        "x264enc", "speed-preset=medium", "bitrate=9000", "key-int-max=50", "!",
        "video/x-h264,stream-format=avc,alignment=au", "!",
        "mp4mux", "faststart=true", "!", "filesink", f"location={output}",
    ]
    subprocess.run(command, check=True)


def make_truth_sampler(evaluation_dir: Path, report: dict):
    rows = load_rows(evaluation_dir / "mocap_normalized.csv")
    valid_rows = [row for row in rows if row["valid"] == "1"]
    times = np.asarray([float(row["timestamp"]) for row in valid_rows])
    positions = np.asarray([[float(row[f"{axis}_m"]) for axis in "xyz"] for row in valid_rows])
    rotations = Rotation.from_quat(np.asarray([
        [float(row[f"q{axis}"]) for axis in "xyzw"] for row in valid_rows
    ]))
    slerp = Slerp(times, rotations)
    body_camera = np.asarray(report["hand_eye"]["T_body_camera"], dtype=float)
    offset = float(report["time_alignment"]["optimized_offset_s"])

    def sample(video_time: float):
        timestamp = video_time + offset
        right = int(np.searchsorted(times, timestamp))
        if right <= 0 or right >= len(times) or times[right] - times[right - 1] > 0.05:
            return None
        alpha_position = np.asarray([
            np.interp(timestamp, times, positions[:, axis]) for axis in range(3)
        ])
        world_body = np.eye(4)
        world_body[:3, :3] = slerp([timestamp]).as_matrix()[0]
        world_body[:3, 3] = alpha_position
        truth = world_body @ body_camera
        return truth[:3, 3], Rotation.from_matrix(truth[:3, :3]).as_quat()

    return sample


def kalman_rts_scalar(measurements: np.ndarray, dt: float, acceleration_sigma: float,
                      measurement_sigma: float) -> np.ndarray:
    """Constant-velocity Kalman filter followed by a fixed-interval RTS pass."""
    observed = np.isfinite(measurements)
    output = np.full(len(measurements), np.nan)
    if not observed.any():
        return output
    first, last = int(np.flatnonzero(observed)[0]), int(np.flatnonzero(observed)[-1])
    transition = np.asarray([[1.0, dt], [0.0, 1.0]])
    process = acceleration_sigma**2 * np.asarray([
        [dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2],
    ])
    observation = np.asarray([[1.0, 0.0]])
    measurement_var = measurement_sigma**2
    count = last - first + 1
    filtered_state = np.zeros((count, 2))
    filtered_covariance = np.zeros((count, 2, 2))
    predicted_state = np.zeros((count, 2))
    predicted_covariance = np.zeros((count, 2, 2))
    state = np.asarray([measurements[first], 0.0])
    covariance = np.diag([measurement_var, 1.0])
    for local, index in enumerate(range(first, last + 1)):
        if local:
            state = transition @ state
            covariance = transition @ covariance @ transition.T + process
        predicted_state[local] = state
        predicted_covariance[local] = covariance
        if observed[index]:
            innovation = measurements[index] - (observation @ state).item()
            innovation_covariance = (observation @ covariance @ observation.T).item() + measurement_var
            gain = covariance @ observation.T / innovation_covariance
            state = state + gain[:, 0] * innovation
            covariance = (np.eye(2) - gain @ observation) @ covariance
        filtered_state[local] = state
        filtered_covariance[local] = covariance
    smoothed_state = filtered_state.copy()
    smoothed_covariance = filtered_covariance.copy()
    for local in range(count - 2, -1, -1):
        gain = filtered_covariance[local] @ transition.T @ np.linalg.inv(
            predicted_covariance[local + 1]
        )
        smoothed_state[local] += gain @ (
            smoothed_state[local + 1] - predicted_state[local + 1]
        )
        smoothed_covariance[local] += gain @ (
            smoothed_covariance[local + 1] - predicted_covariance[local + 1]
        ) @ gain.T
    output[first:last + 1] = smoothed_state[:, 0]
    return output


def smoothed_visual_poses(matched: dict[int, dict[str, str]], frame_count: int,
                          fps: float, prediction_limit_s: float = 0.25
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    observed = np.zeros(frame_count, dtype=bool)
    positions = np.full((frame_count, 3), np.nan)
    quaternions, frames = [], []
    for frame, row in sorted(matched.items()):
        if not 0 <= frame < frame_count:
            continue
        observed[frame] = True
        positions[frame] = [float(row[f"estimate_{axis}_m"]) for axis in "xyz"]
        quaternions.append([float(row[f"estimate_q{axis}"]) for axis in "xyzw"])
        frames.append(frame)
    dt = 1.0 / fps
    smoothed_positions = np.column_stack([
        kalman_rts_scalar(positions[:, axis], dt, 2.5, 0.012) for axis in range(3)
    ])
    euler_measurements = np.full((frame_count, 3), np.nan)
    if frames:
        euler = Rotation.from_quat(np.asarray(quaternions)).as_euler("xyz")
        euler = np.unwrap(euler, axis=0)
        euler_measurements[np.asarray(frames)] = euler
    smoothed_euler = np.column_stack([
        kalman_rts_scalar(euler_measurements[:, axis], dt, 8.0, np.radians(1.8))
        for axis in range(3)
    ])
    smoothed_quaternions = np.full((frame_count, 4), np.nan)
    finite_orientation = np.isfinite(smoothed_euler).all(axis=1)
    smoothed_quaternions[finite_orientation] = Rotation.from_euler(
        "xyz", smoothed_euler[finite_orientation]
    ).as_quat()
    indices = np.arange(frame_count)
    previous = np.maximum.accumulate(np.where(observed, indices, -frame_count))
    following = np.minimum.accumulate(np.where(observed, indices, frame_count)[::-1])[::-1]
    distance = np.minimum(indices - previous, following - indices)
    visible = distance <= int(round(prediction_limit_s * fps))
    visible &= np.isfinite(smoothed_positions).all(axis=1) & finite_orientation
    return smoothed_positions, smoothed_quaternions, visible, observed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("pose_csv", type=Path)
    parser.add_argument("evaluation_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gripper-mesh-dir", type=Path)
    parser.add_argument("--output-fps", type=float, default=25.0)
    args = parser.parse_args()

    pose_rows = load_rows(args.pose_csv)
    matched = matched_rows(args.evaluation_dir / "matched_errors.csv")
    report = json.loads((args.evaluation_dir / "mocap_evaluation.json").read_text(encoding="utf-8"))
    matched_list = list(matched.values())
    low, high = world_bounds(matched_list)
    truth_path = np.asarray([[float(row[f"truth_{axis}_m"]) for axis in "xyz"] for row in matched_list])
    estimate_path = np.asarray([[float(row[f"estimate_{axis}_m"]) for axis in "xyz"] for row in matched_list])
    matched_frames = np.asarray([int(row["video_frame"]) for row in matched_list])
    gripper_edges = load_gripper_edges(args.gripper_mesh_dir) if args.gripper_mesh_dir else None
    truth_sampler = make_truth_sampler(args.evaluation_dir, report)

    cap = cv2.VideoCapture(str(args.video), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    smooth_positions, smooth_quaternions, smooth_visible, smooth_measured = smoothed_visual_poses(
        matched, source_frame_count, source_fps
    )
    estimate_path = smooth_positions[matched_frames]
    pose_frames = np.asarray([int(row["frame"]) for row in pose_rows])
    pose_step = int(np.median(np.diff(pose_frames))) if len(pose_frames) > 1 else 1
    requested_step = max(1, round(source_fps / args.output_fps))
    render_step = max(pose_step, requested_step)
    output_fps = source_fps / render_step
    args.output.parent.mkdir(parents=True, exist_ok=True)
    intermediate = args.output.with_suffix(".mp4v.mp4")
    writer = cv2.VideoWriter(
        str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (1920, 960)
    )
    if not writer.isOpened():
        raise SystemExit("cannot create intermediate comparison video")

    pose_by_frame = {int(row["frame"]): row for row in pose_rows}
    previous_direct = False
    rendered = 0
    frame_number = 0
    truth_color, estimate_color = CYAN, GREEN
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_number % render_step:
            frame_number += 1
            continue
        canvas = np.full((960, 1920, 3), BG, dtype=np.uint8)
        canvas[:640, :1280] = cv2.resize(frame, (1280, 640), interpolation=cv2.INTER_AREA)
        canvas[:640, :1280] = cv2.addWeighted(canvas[:640, :1280], 0.82,
                                               np.zeros((640, 1280, 3), np.uint8), 0.18, 0)
        row = pose_by_frame.get(frame_number)
        is_direct = bool(row and direct_row(row) and frame_number in matched)
        predicted = bool(frame_number < len(smooth_visible) and smooth_visible[frame_number] and not is_direct)
        recovered = is_direct and not previous_direct and rendered > 0
        if is_direct:
            status, status_color = (
                ("RECOVERED DIRECT + KALMAN/RTS", AMBER)
                if recovered else ("MEASURED + KALMAN/RTS", GREEN)
            )
        elif predicted:
            status, status_color = "PREDICTED SHORT GAP (KALMAN/RTS)", AMBER
        else:
            status, status_color = "LOST", RED
        previous_direct = is_direct

        text(canvas, "INSTA360 X6  /  OPTITRACK AUDIT", (30, 42), 0.86, WHITE, 2)
        text(canvas, status, (30, 82), 0.75, status_color, 2)
        text(canvas, "Formal metrics use direct multi-tag frames only", (30, 112), 0.50, WHITE)

        panel_x = 1300
        text(canvas, "HELD-OUT 6DoF COMPARISON", (panel_x, 42), 0.76, WHITE, 2)
        if report["publishable_accuracy"]:
            text(canvas, "FORMAL ACCURACY", (panel_x, 75), 0.62, GREEN, 2)
        else:
            text(canvas, "DIAGNOSTIC ONLY - SYNC/ATTITUDE CHECK FAILED", (panel_x, 75), 0.48, RED, 2)
        sync = report["time_alignment"]
        text(canvas, f"offset {sync['optimized_offset_s']:+.3f} s", (panel_x, 108), 0.52, MUTED)
        text(canvas, f"corr linear {sync['linear_correlation']:.3f}", (panel_x, 136), 0.52, GREEN)
        text(canvas, f"corr angular {sync['angular_correlation']:.3f}", (panel_x, 164), 0.52, RED)
        text(canvas, f"corr combined {sync['motion_correlation']:.3f} / 0.800", (panel_x, 192), 0.52, WHITE)

        plot_origin, plot_size = (1300, 220), (590, 490)
        cv2.rectangle(canvas, plot_origin, (plot_origin[0] + plot_size[0], plot_origin[1] + plot_size[1]),
                      (55, 61, 72), 1)
        upto = int(np.searchsorted(matched_frames, frame_number, side="right"))
        for path, color in ((truth_path[:upto], truth_color), (estimate_path[:upto], estimate_color)):
            if len(path) >= 2:
                pts = np.asarray([project(point, low, high, plot_origin, plot_size) for point in path], np.int32)
                cv2.polylines(canvas, [pts], False, color, 2, cv2.LINE_AA)
        current = matched.get(frame_number)
        estimate_position = estimate_quaternion = None
        if frame_number < len(smooth_visible) and smooth_visible[frame_number]:
            estimate_position = smooth_positions[frame_number]
            estimate_quaternion = smooth_quaternions[frame_number]
        sampled_truth = truth_sampler(frame_number / source_fps)
        if sampled_truth is not None:
            truth_position, truth_quaternion = sampled_truth
            if gripper_edges is None:
                draw_axes(canvas, truth_position, truth_quaternion, low, high,
                          plot_origin, plot_size, "OptiTrack")
            else:
                draw_gripper(canvas, truth_position, truth_quaternion, gripper_edges,
                             low, high, plot_origin, plot_size, truth_color,
                             "OptiTrack 6DoF", -12)
        if estimate_position is not None and estimate_quaternion is not None:
            if gripper_edges is None:
                draw_axes(canvas, estimate_position, estimate_quaternion, low, high, plot_origin, plot_size, "Visual")
            else:
                draw_gripper(canvas, estimate_position, estimate_quaternion, gripper_edges,
                             low, high, plot_origin, plot_size, estimate_color,
                             "Optimized visual 6DoF", 20)
            if sampled_truth is not None:
                smooth_position_error = 1000.0 * float(np.linalg.norm(estimate_position - truth_position))
                smooth_orientation_error = np.degrees(
                    (Rotation.from_quat(truth_quaternion).inv()
                     * Rotation.from_quat(estimate_quaternion)).magnitude()
                )
                text(canvas, f"smoothed position error {smooth_position_error:.1f} mm",
                     (panel_x, 750), 0.58, WHITE)
                text(canvas, f"smoothed attitude error {smooth_orientation_error:.1f} deg",
                     (panel_x, 782), 0.58, WHITE)
        else:
            text(canvas, "Visual 6DoF unavailable", (panel_x, 750), 0.58, RED)

        if gripper_edges is not None and sampled_truth is not None:
            inset_origin, inset_size = (1320, 235), (300, 210)
            cv2.rectangle(canvas, inset_origin,
                          (inset_origin[0] + inset_size[0], inset_origin[1] + inset_size[1]),
                          (68, 75, 88), -1)
            cv2.rectangle(canvas, inset_origin,
                          (inset_origin[0] + inset_size[0], inset_origin[1] + inset_size[1]),
                          WHITE, 1)
            center = truth_position if estimate_position is None else (truth_position + estimate_position) / 2.0
            inset_low, inset_high = center - 0.18, center + 0.18
            draw_gripper(canvas, truth_position, truth_quaternion, gripper_edges,
                         inset_low, inset_high, inset_origin, inset_size, truth_color,
                         "OptiTrack", -10)
            if estimate_position is not None and estimate_quaternion is not None:
                draw_gripper(canvas, estimate_position, estimate_quaternion, gripper_edges,
                             inset_low, inset_high, inset_origin, inset_size, estimate_color,
                             "Visual", 18)
            text(canvas, "CURRENT GRIPPER POSE ZOOM", (inset_origin[0] + 10, inset_origin[1] + 20),
                 0.38, WHITE, 1)

        text(canvas, "Tag map  130 -> 131 -> 129 -> 128", (30, 690), 0.63, WHITE, 2)
        active_ids = set((row or {}).get("detected_ids", "").split())
        for index, tag_id in enumerate(("130", "131", "129", "128")):
            left = 30 + index * 180
            color = GREEN if tag_id in active_ids else (67, 70, 78)
            cv2.rectangle(canvas, (left, 720), (left + 150, 800), color, 2)
            text(canvas, f"ID {tag_id}", (left + 35, 770), 0.62, color, 2)
        coverage = report["visual"]["direct_coverage_ratio"] * 100.0
        text(canvas, f"direct multi-tag coverage {coverage:.1f}%", (30, 845), 0.60, WHITE)
        text(canvas, f"t={frame_number/source_fps:06.2f}s  frame={frame_number}", (30, 890), 0.62, MUTED)
        text(canvas, "cyan: OptiTrack    green: visual Kalman + RTS", (panel_x, 835), 0.51, WHITE)
        legend = "CAD gripper pose | short gaps predicted" if gripper_edges is not None else "Axes: X red / Y green / Z blue"
        text(canvas, legend, (panel_x, 868), 0.51, MUTED)
        writer.write(canvas)
        rendered += 1
        frame_number += 1
    writer.release()
    cap.release()
    transcode_h264(intermediate, args.output)
    intermediate.unlink(missing_ok=True)
    print(json.dumps({"output": str(args.output.resolve()), "frames": rendered,
                      "fps": output_fps}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
