#!/usr/bin/env python3
"""Render a synchronized panorama + recent-tail 3D trajectory video."""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
from collections import deque
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("pose_csv", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--smooth", type=float, default=0.55)
    parser.add_argument("--median-window", type=int, default=1)
    parser.add_argument("--filter", choices=("ema", "kalman"), default="ema")
    parser.add_argument("--kalman-measurement-noise", type=float, default=0.04)
    parser.add_argument("--kalman-accel-noise", type=float, default=0.8)
    parser.add_argument("--kalman-angle-noise", type=float, default=2.0)
    parser.add_argument("--kalman-angular-accel-noise", type=float, default=30.0)
    parser.add_argument("--tail-seconds", type=float, default=2.0)
    parser.add_argument("--prediction-max-age", type=float, default=0.32)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float)
    return parser.parse_args()


def load_and_filter(path: Path, smooth: float, median_window: int = 1) -> np.ndarray:
    poses = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["quality_status"] == "valid" and row["camera_x_m"]:
                poses.append(
                    [
                        float(row["timestamp"]),
                        float(row["camera_x_m"]),
                        float(row["camera_y_m"]),
                        float(row["camera_z_m"]),
                        float(row["roll_deg"]),
                        float(row["pitch_deg"]),
                        float(row["yaw_deg"]),
                    ]
                )
    data = np.asarray(poses, dtype=np.float64)
    if not len(data):
        raise ValueError("pose CSV has no valid 6DoF samples")
    # Euler angles must be continuous before median/EMA/Kalman filtering so a
    # +179 -> -179 degree wrap is not mistaken for a 358 degree rotation.
    breaks = np.flatnonzero(np.diff(data[:, 0]) > 0.65) + 1
    for segment in np.split(np.arange(len(data)), breaks):
        data[segment, 4:7] = np.degrees(
            np.unwrap(np.radians(data[segment, 4:7]), axis=0)
        )
    median_window = max(1, int(median_window))
    if median_window % 2 == 0:
        median_window += 1
    if median_window > 1:
        # Offline centered median removes isolated planar-PnP spikes. Never let
        # the window cross a long detection gap.
        source = data[:, 1:].copy()
        breaks = np.flatnonzero(np.diff(data[:, 0]) > 0.65) + 1
        for segment in np.split(np.arange(len(data)), breaks):
            radius = median_window // 2
            for index in segment:
                lo = max(int(segment[0]), index - radius)
                hi = min(int(segment[-1]) + 1, index + radius + 1)
                data[index, 1:] = np.median(source[lo:hi], axis=0)
    alpha = 1.0 - smooth
    for index in range(1, len(data)):
        if data[index, 0] - data[index - 1, 0] <= 0.65:
            data[index, 1:] = data[index - 1, 1:] + alpha * (
                data[index, 1:] - data[index - 1, 1:]
            )
    return data


def kalman_rts_filter(
    data: np.ndarray,
    measurement_noise: float,
    accel_noise: float,
    angle_noise: float = 2.0,
    angular_accel_noise: float = 30.0,
) -> np.ndarray:
    """Offline 6DoF constant-velocity Kalman filter followed by RTS smoothing.

    Long observation gaps are separate segments so the smoother never invents
    a connection across a lost AprilGrid interval.
    """
    result = data.copy()
    breaks = np.flatnonzero(np.diff(data[:, 0]) > 0.65) + 1
    for indices in np.split(np.arange(len(data)), breaks):
        if not len(indices):
            continue
        count = len(indices)
        dimensions = data.shape[1] - 1
        if dimensions != 6:
            raise ValueError("expected XYZ + roll/pitch/yaw samples")
        measurement_std = np.array(
            [measurement_noise] * 3 + [angle_noise] * 3, dtype=np.float64
        )
        acceleration_std = np.array(
            [accel_noise] * 3 + [angular_accel_noise] * 3, dtype=np.float64
        )
        filtered_x: list[np.ndarray] = []
        filtered_p: list[np.ndarray] = []
        predicted_x: list[np.ndarray] = []
        predicted_p: list[np.ndarray] = []
        transitions: list[np.ndarray] = []
        state = np.r_[data[indices[0], 1:], np.zeros(dimensions)]
        velocity_std = np.array([0.5] * 3 + [30.0] * 3)
        covariance = np.diag(np.r_[measurement_std**2, velocity_std**2])
        h = np.c_[np.eye(dimensions), np.zeros((dimensions, dimensions))]
        r = np.diag(measurement_std**2)
        for local, index in enumerate(indices):
            if local == 0:
                f = np.eye(dimensions * 2)
                prior_state, prior_covariance = state.copy(), covariance.copy()
            else:
                dt = float(data[index, 0] - data[indices[local - 1], 0])
                f = np.eye(dimensions * 2)
                f[:dimensions, dimensions:] = np.eye(dimensions) * dt
                q = np.zeros((dimensions * 2, dimensions * 2))
                spectral_density = np.diag(acceleration_std**2)
                q[:dimensions, :dimensions] = spectral_density * dt**3 / 3.0
                q[:dimensions, dimensions:] = spectral_density * dt**2 / 2.0
                q[dimensions:, :dimensions] = spectral_density * dt**2 / 2.0
                q[dimensions:, dimensions:] = spectral_density * dt
                prior_state = f @ state
                prior_covariance = f @ covariance @ f.T + q
            innovation = data[index, 1:] - h @ prior_state
            innovation_covariance = h @ prior_covariance @ h.T + r
            gain = np.linalg.solve(innovation_covariance, h @ prior_covariance).T
            state = prior_state + gain @ innovation
            covariance = (np.eye(dimensions * 2) - gain @ h) @ prior_covariance
            predicted_x.append(prior_state)
            predicted_p.append(prior_covariance)
            filtered_x.append(state.copy())
            filtered_p.append(covariance.copy())
            transitions.append(f)
        smoothed_x = [value.copy() for value in filtered_x]
        smoothed_p = [value.copy() for value in filtered_p]
        for local in range(count - 2, -1, -1):
            f = transitions[local + 1]
            smoother_gain = np.linalg.solve(
                predicted_p[local + 1], f @ filtered_p[local]
            ).T
            smoothed_x[local] = filtered_x[local] + smoother_gain @ (
                smoothed_x[local + 1] - predicted_x[local + 1]
            )
            smoothed_p[local] = filtered_p[local] + smoother_gain @ (
                smoothed_p[local + 1] - predicted_p[local + 1]
            ) @ smoother_gain.T
        for index, state in zip(indices, smoothed_x):
            result[index, 1:] = state[:dimensions]
    return result


def sample_pose(
    data: np.ndarray, now: float, prediction_max_age: float
) -> tuple[np.ndarray | None, str, float]:
    right = int(np.searchsorted(data[:, 0], now, side="right"))
    if right == 0:
        delta = data[0, 0] - now
        return (data[0, 1:].copy(), "ENTERING", max(0.0, 1.0 - delta / 0.25)) if delta <= 0.25 else (None, "SEARCHING", 0.0)
    previous = data[right - 1]
    if right < len(data):
        following = data[right]
        gap = following[0] - previous[0]
        if gap <= 0.65:
            u = np.clip((now - previous[0]) / gap, 0.0, 1.0)
            u = u * u * (3.0 - 2.0 * u)
            return previous[1:] * (1.0 - u) + following[1:] * u, "TRACKED", 1.0
    age = now - previous[0]
    if age <= prediction_max_age:
        velocity = np.zeros(data.shape[1] - 1)
        if right >= 2:
            prior = data[right - 2]
            dt = previous[0] - prior[0]
            if 0 < dt <= 0.65:
                limits = np.array([0.8, 0.8, 0.8, 90.0, 90.0, 90.0])
                velocity = np.clip((previous[1:] - prior[1:]) / dt, -limits, limits)
        opacity = max(0.0, 1.0 - age / prediction_max_age)
        return previous[1:] + velocity * age, "PREDICTED", opacity
    return None, "SEARCHING", 0.0


class Projector:
    def __init__(self, xyz: np.ndarray, rect: tuple[int, int, int, int]):
        self.x, self.y, self.w, self.h = rect
        projected = np.asarray([self.raw(point) for point in xyz])
        low, high = projected.min(axis=0), projected.max(axis=0)
        span = np.maximum(high - low, 1e-6)
        self.scale = min((self.w - 54) / span[0], (self.h - 58) / span[1])
        center = (low + high) / 2
        self.offset = np.array([self.x + self.w / 2, self.y + self.h / 2]) - center * self.scale

    @staticmethod
    def raw(point: np.ndarray) -> np.ndarray:
        x, y, z = point
        return np.array([(x - y) * 0.88, -(z - 0.55) + (x + y) * 0.22])

    def __call__(self, point: np.ndarray) -> tuple[int, int]:
        pixel = self.raw(point) * self.scale + self.offset
        return int(round(pixel[0])), int(round(pixel[1]))


def rpy_to_rotation(rpy_deg: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.radians(rpy_deg)
    rx = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
    )
    ry = np.array(
        [[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]]
    )
    rz = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )
    return rz @ ry @ rx


def draw_pose_axes(
    canvas: np.ndarray,
    projector: Projector,
    position: np.ndarray,
    rpy_deg: np.ndarray,
    axis_length: float = 0.12,
) -> None:
    origin = projector(position)
    rotation = rpy_to_rotation(rpy_deg)
    colors = ((40, 55, 245), (70, 220, 90), (245, 110, 55))  # X red, Y green, Z blue
    labels = ("Xc", "Yc", "Zc")
    for axis, color, label in zip(np.eye(3), colors, labels):
        endpoint = projector(position + rotation @ axis * axis_length)
        cv2.arrowedLine(canvas, origin, endpoint, color, 3, cv2.LINE_AA, tipLength=0.22)
        cv2.putText(
            canvas, label, (endpoint[0] + 3, endpoint[1] - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA,
        )


def draw_board_axes(
    canvas: np.ndarray,
    projector: Projector,
    axis_length: float = 0.18,
) -> None:
    """Draw the fixed AprilGrid/board frame used by position and orientation."""
    origin_3d = np.zeros(3, dtype=np.float64)
    origin = projector(origin_3d)
    colors = ((40, 55, 245), (70, 220, 90), (245, 110, 55))
    labels = ("Xb", "Yb", "Zb")
    cv2.circle(canvas, origin, 4, (225, 230, 236), -1, cv2.LINE_AA)
    for axis, color, label in zip(np.eye(3), colors, labels):
        endpoint = projector(origin_3d + axis * axis_length)
        cv2.arrowedLine(canvas, origin, endpoint, color, 2, cv2.LINE_AA, tipLength=0.20)
        cv2.putText(
            canvas, label, (endpoint[0] + 3, endpoint[1] - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA,
        )


def blend_rect(image: np.ndarray, rect: tuple[int, int, int, int], opacity: float) -> None:
    x, y, w, h = rect
    roi = image[y : y + h, x : x + w]
    black = np.zeros_like(roi)
    cv2.addWeighted(black, opacity, roi, 1.0 - opacity, 0, roi)


def draw_hud(
    canvas: np.ndarray,
    projector: Projector,
    history: deque,
    current: np.ndarray | None,
    current_rpy: np.ndarray | None,
    state: str,
    opacity: float,
    now: float,
    duration: float,
    progress_ratio: float,
    filter_label: str,
) -> None:
    panel = (842, 26, 410, 394)
    blend_rect(canvas, panel, 0.68)
    cv2.rectangle(canvas, (panel[0], panel[1]), (panel[0] + panel[2], panel[1] + panel[3]), (82, 92, 106), 1, cv2.LINE_AA)
    cv2.putText(canvas, "6DoF CAMERA POSE", (862, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (235, 240, 246), 1, cv2.LINE_AA)
    cv2.putText(canvas, "board frame + camera frame", (862, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (157, 171, 188), 1, cv2.LINE_AA)

    grid_points = []
    for x_value in np.linspace(-0.7, 0.45, 6):
        a = projector(np.array([x_value, -0.70, 0.0]))
        b = projector(np.array([x_value, 0.05, 0.0]))
        cv2.line(canvas, a, b, (62, 72, 84), 1, cv2.LINE_AA)
    for y_value in np.linspace(-0.7, 0.05, 5):
        a = projector(np.array([-0.70, y_value, 0.0]))
        b = projector(np.array([0.45, y_value, 0.0]))
        cv2.line(canvas, a, b, (62, 72, 84), 1, cv2.LINE_AA)
    draw_board_axes(canvas, projector)

    samples = list(history)
    for index in range(1, len(samples)):
        t0, p0 = samples[index - 1]
        t1, p1 = samples[index]
        if p0 is None or p1 is None or t1 - t0 > 0.12:
            continue
        age = max(0.0, now - t1)
        strength = max(0.08, 1.0 - age / 2.0)
        color = (int(248 * strength), int(189 * strength), int(56 * strength))
        cv2.line(canvas, projector(p0), projector(p1), color, max(1, int(4 * strength)), cv2.LINE_AA)

    if current is not None:
        point = projector(current)
        halo = int(14 + 5 * math.sin(now * 6.0) ** 2)
        overlay = canvas.copy()
        cv2.circle(overlay, point, halo, (245, 185, 56), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.16 * opacity, canvas, 1.0 - 0.16 * opacity, 0, canvas)
        cv2.circle(canvas, point, 7, (255, 205, 84), -1, cv2.LINE_AA)
        cv2.circle(canvas, point, 9, (245, 246, 248), 1, cv2.LINE_AA)
        if current_rpy is not None:
            draw_pose_axes(canvas, projector, current, current_rpy)

    state_color = (93, 215, 139) if state == "TRACKED" else (72, 174, 240) if state in {"PREDICTED", "ENTERING"} else (128, 139, 154)
    cv2.circle(canvas, (865, 390), 5, state_color, -1, cv2.LINE_AA)
    cv2.putText(canvas, state, (878, 396), cv2.FONT_HERSHEY_SIMPLEX, 0.52, state_color, 1, cv2.LINE_AA)

    blend_rect(canvas, (0, 640, 1280, 80), 0.78)
    cv2.putText(canvas, f"{now:05.2f} s", (28, 678), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (244, 247, 250), 2, cv2.LINE_AA)
    if current is not None:
        coords = f"X {current[0]:+.3f} m    Y {current[1]:+.3f} m    Z {current[2]:+.3f} m"
        cv2.putText(canvas, coords, (205, 669), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 220, 230), 1, cv2.LINE_AA)
    if current_rpy is not None:
        wrapped = (current_rpy + 180.0) % 360.0 - 180.0
        angles = f"R {wrapped[0]:+.1f} deg    P {wrapped[1]:+.1f} deg    Y {wrapped[2]:+.1f} deg"
        cv2.putText(canvas, angles, (205, 699), cv2.FONT_HERSHEY_SIMPLEX, 0.49, (166, 183, 202), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"adaptive IPPE | {filter_label}", (886, 678), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (157, 171, 188), 1, cv2.LINE_AA)
    cv2.line(canvas, (0, 714), (1280, 714), (43, 51, 61), 4)
    cv2.line(canvas, (0, 714), (int(1280 * np.clip(progress_ratio, 0.0, 1.0)), 714), (245, 185, 56), 4)


def main() -> int:
    args = parse_args()
    if args.filter == "kalman":
        data = load_and_filter(args.pose_csv, 0.0, args.median_window)
        data = kalman_rts_filter(
            data,
            args.kalman_measurement_noise,
            args.kalman_accel_noise,
            args.kalman_angle_noise,
            args.kalman_angular_accel_noise,
        )
    else:
        data = load_and_filter(args.pose_csv, args.smooth, args.median_window)
    capture = cv2.VideoCapture(str(args.video), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise SystemExit("cannot open video")
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_duration = frame_count / source_fps
    start_time = float(np.clip(args.start, 0.0, source_duration))
    output_duration = min(
        args.duration if args.duration is not None else source_duration - start_time,
        source_duration - start_time,
    )
    end_time = start_time + output_duration
    board_reference = np.vstack(
        [np.zeros((1, 3)), np.eye(3) * 0.18]
    )
    projector = Projector(
        np.vstack([data[:, 1:4], board_reference]), (862, 94, 368, 276)
    )
    history: deque[tuple[float, np.ndarray | None]] = deque()
    temporary = args.output.with_suffix(".visual.tmp.mp4")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    encode = subprocess.Popen(
        [
            str(args.ffmpeg), "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", "1280x720", "-r", str(args.fps), "-i", "-", "-an", "-c:v", "libx264",
            "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-y", str(temporary),
        ],
        stdin=subprocess.PIPE,
    )
    output_index = input_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        now = input_index / source_fps
        input_index += 1
        if now < start_time:
            continue
        if now > end_time:
            break
        local_time = now - start_time
        if local_time + 1e-6 < output_index / args.fps:
            continue
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        canvas[:640] = cv2.resize(frame, (1280, 640), interpolation=cv2.INTER_AREA)
        pose, state, opacity = sample_pose(data, now, args.prediction_max_age)
        position = None if pose is None else pose[:3]
        orientation = None if pose is None else pose[3:6]
        history.append((now, None if position is None else position.copy()))
        while history and now - history[0][0] > args.tail_seconds:
            history.popleft()
        draw_hud(
            canvas, projector, history, position, orientation, state, opacity, now,
            output_duration, local_time / output_duration,
            "Kalman + RTS" if args.filter == "kalman" else f"{args.smooth * 100:.0f}% smoothing",
        )
        assert encode.stdin is not None
        encode.stdin.write(canvas.tobytes())
        output_index += 1
    capture.release()
    assert encode.stdin is not None
    encode.stdin.close()
    if encode.wait() != 0:
        raise SystemExit("video encoder failed")
    subprocess.run(
        [
            str(args.ffmpeg), "-v", "error", "-i", str(temporary), "-ss", str(start_time),
            "-t", str(output_duration), "-i", str(args.video),
            "-map", "0:v:0", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-movflags", "+faststart", "-y", str(args.output),
        ],
        check=True,
    )
    temporary.unlink(missing_ok=True)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
