#!/usr/bin/env python3
"""Render a synchronized panorama + recent-tail 3D trajectory video."""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
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
    parser.add_argument(
        "--layout", choices=("overlay", "analysis"), default="overlay",
        help="overlay: compact HUD; analysis: split-screen full trajectory and large 6DoF axes",
    )
    parser.add_argument(
        "--reference-frame", choices=("board", "start"), default="board",
        help="board: absolute AprilGrid frame; start: first valid camera pose is XYZ/RPY zero",
    )
    parser.add_argument(
        "--state-label",
        help="override TRACKED/PREDICTED state text (for example FUSED/ESTIMATED)",
    )
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--claw-angle-csv", type=Path)
    parser.add_argument("--claw-mesh-dir", type=Path)
    parser.add_argument(
        "--camera-to-gripper-json", type=Path,
        help="fixed gripper-to-camera transform used to convert camera pose into gripper-base pose",
    )
    parser.add_argument("--audio-source", type=Path)
    parser.add_argument("--video-fit", choices=("stretch", "contain"), default="stretch")
    return parser.parse_args()


def load_claw_angles(path: Path) -> np.ndarray:
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append([
                float(row["time_s"]), float(row["opening_angle_deg"]),
                float(row["joint1_deg"]), float(row["joint2_deg"]),
                float(row.get("measured", "1")), float(row.get("confidence", "1")),
            ])
    if not rows:
        raise ValueError("claw angle CSV has no samples")
    return np.asarray(rows, dtype=np.float64)


def sample_claw_angle(data: np.ndarray, now: float) -> tuple[np.ndarray, bool, float]:
    index = int(np.clip(np.searchsorted(data[:, 0], now), 0, len(data) - 1))
    if index and abs(data[index - 1, 0] - now) < abs(data[index, 0] - now):
        index -= 1
    values = np.array([np.interp(now, data[:, 0], data[:, column]) for column in (1, 2, 3)])
    return values, bool(data[index, 4] >= 0.5), float(data[index, 5])


def load_binary_stl(path: Path, triangle_budget: int = 950) -> np.ndarray:
    raw = path.read_bytes()
    count = struct.unpack_from("<I", raw, 80)[0]
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    triangles = np.frombuffer(raw, dtype=dtype, offset=84, count=count)["vertices"].astype(np.float64)
    if len(triangles) > triangle_budget:
        triangles = triangles[::max(1, len(triangles) // triangle_budget)][:triangle_budget]
    return triangles


def load_claw_meshes(mesh_dir: Path) -> dict[str, np.ndarray]:
    return {
        "base": load_binary_stl(mesh_dir / "base_link.STL", 500),
        "left": load_binary_stl(mesh_dir / "Link1.STL", 850),
        "right": load_binary_stl(mesh_dir / "Link2.STL", 850),
    }


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


def load_pose_quality(path: Path) -> np.ndarray:
    """Load every processed frame so rendered state labels remain auditable."""
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append([float(row["timestamp"]), row["quality_status"] == "valid"])
    if not rows:
        raise ValueError("pose CSV has no quality samples")
    return np.asarray(rows, dtype=np.float64)


def measurement_state(quality: np.ndarray, now: float, fallback: str) -> str:
    """Distinguish a decoded visual pose from interpolation and prediction."""
    if fallback in {"SEARCHING", "PREDICTED", "ENTERING"}:
        return fallback
    index = int(np.clip(np.searchsorted(quality[:, 0], now), 0, len(quality) - 1))
    if index and abs(quality[index - 1, 0] - now) < abs(quality[index, 0] - now):
        index -= 1
    nominal_dt = float(np.median(np.diff(quality[:, 0]))) if len(quality) > 1 else 0.02
    if abs(quality[index, 0] - now) <= max(0.012, nominal_dt * 0.55):
        return "VISUAL MEASURED" if quality[index, 1] >= 0.5 else "INTERPOLATED"
    return "INTERPOLATED"


def apply_gripper_extrinsic(data: np.ndarray, path: Path) -> np.ndarray:
    """Convert board<-camera poses to board<-gripper-base poses."""
    transform = json.loads(path.read_text(encoding="utf-8"))
    rotation_camera_gripper = np.asarray(
        transform["rotation_gripper_to_camera"], dtype=np.float64
    )
    translation_camera_gripper = np.asarray(
        transform["translation_gripper_origin_in_camera_m"], dtype=np.float64
    )
    if rotation_camera_gripper.shape != (3, 3) or translation_camera_gripper.shape != (3,):
        raise ValueError("invalid camera-to-gripper transform JSON")
    result = data.copy()
    for index in range(len(result)):
        rotation_board_camera = rpy_to_rotation(data[index, 4:7])
        result[index, 1:4] = (
            data[index, 1:4] + rotation_board_camera @ translation_camera_gripper
        )
        result[index, 4:7] = rotation_to_rpy(
            rotation_board_camera @ rotation_camera_gripper
        )
    return result


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


def rotation_to_rpy(rotation: np.ndarray) -> np.ndarray:
    """Return ZYX roll/pitch/yaw angles in degrees for a rotation matrix."""
    sy = math.hypot(rotation[0, 0], rotation[1, 0])
    if sy >= 1e-8:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])


def make_start_relative(data: np.ndarray) -> np.ndarray:
    """Express every camera pose in the first valid camera pose frame."""
    result = data.copy()
    start_position = data[0, 1:4].copy()
    start_rotation = rpy_to_rotation(data[0, 4:7])
    board_to_start = start_rotation.T
    for index in range(len(result)):
        current_rotation = rpy_to_rotation(data[index, 4:7])
        result[index, 1:4] = board_to_start @ (data[index, 1:4] - start_position)
        result[index, 4:7] = rotation_to_rpy(board_to_start @ current_rotation)
    return result


def draw_pose_axes(
    canvas: np.ndarray,
    projector: Projector,
    position: np.ndarray,
    rpy_deg: np.ndarray,
    axis_length: float = 0.12,
    frame_suffix: str = "c",
) -> None:
    origin = projector(position)
    rotation = rpy_to_rotation(rpy_deg)
    colors = ((40, 55, 245), (70, 220, 90), (245, 110, 55))  # X red, Y green, Z blue
    labels = (f"X{frame_suffix}", f"Y{frame_suffix}", f"Z{frame_suffix}")
    for axis, color, label in zip(np.eye(3), colors, labels):
        endpoint = projector(position + rotation @ axis * axis_length)
        cv2.arrowedLine(canvas, origin, endpoint, color, 3, cv2.LINE_AA, tipLength=0.22)
        cv2.putText(
            canvas, label, (endpoint[0] + 3, endpoint[1] - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA,
        )


def _rz_degrees(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def draw_claw_model(
    canvas: np.ndarray,
    projector: Projector,
    position: np.ndarray,
    rpy_deg: np.ndarray,
    joint1_deg: float,
    joint2_deg: float,
    meshes: dict[str, np.ndarray],
    model_scale: float = 1.0,
) -> None:
    """Draw the URDF meshes at the current gripper-base pose."""
    base_rotation = rpy_to_rotation(rpy_deg)
    pivots = {
        "left": np.array([0.015, 0.01425, 0.01510]),
        "right": np.array([0.015, -0.01425, 0.01570]),
    }
    colors = {
        "base": (170, 178, 190),
        "left": (82, 231, 126),
        "right": (30, 218, 252),
    }
    angles = {"base": 0.0, "left": joint1_deg, "right": joint2_deg}
    for name in ("base", "left", "right"):
        local = meshes[name].copy()
        if name != "base":
            local = local @ _rz_degrees(angles[name]).T + pivots[name]
        local *= model_scale
        world = local @ base_rotation.T + position
        color = colors[name]
        # A sparse translucent wire surface stays legible over the metric grid.
        overlay = canvas.copy()
        for triangle in world:
            polygon = np.array([projector(point) for point in triangle], np.int32)
            cv2.fillConvexPoly(overlay, polygon, color, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.10, canvas, 0.90, 0, canvas)
        for triangle in world[::3]:
            polygon = np.array([projector(point) for point in triangle], np.int32)
            cv2.polylines(canvas, [polygon], True, color, 1, cv2.LINE_AA)

    origin = projector(position)
    cv2.circle(canvas, origin, 5, (245, 246, 248), -1, cv2.LINE_AA)
    # Physical 24 mm black-code square at the measured CAD base location.
    tag_local = np.array([
        [-0.013, -0.012, 0.004], [0.011, -0.012, 0.004],
        [0.011, 0.012, 0.004], [-0.013, 0.012, 0.004],
    ]) * model_scale
    tag_world = tag_local @ base_rotation.T + position
    tag_polygon = np.array([projector(point) for point in tag_world], np.int32)
    cv2.fillConvexPoly(canvas, tag_polygon, (28, 30, 34), cv2.LINE_AA)
    cv2.polylines(canvas, [tag_polygon], True, (255, 72, 238), 2, cv2.LINE_AA)
    tag_center = tuple(np.round(tag_polygon.mean(axis=0)).astype(int))
    cv2.putText(canvas, "ID2", (tag_center[0] + 4, tag_center[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255, 105, 244), 1, cv2.LINE_AA)

    # Make the fixed CAD relationship explicit instead of leaving the tag and
    # moving links as two unrelated drawings.  ID2 is rigidly attached to the
    # base; the two dashed legs terminate at the physical revolute axes.
    tag_origin_local = np.array([-0.001, 0.0, 0.004]) * model_scale
    tag_origin_world = tag_origin_local @ base_rotation.T + position
    tag_origin_px = projector(tag_origin_world)
    for name, colour in (("left", colors["left"]), ("right", colors["right"])):
        pivot_local = pivots[name] * model_scale
        pivot_world = pivot_local @ base_rotation.T + position
        pivot_px = projector(pivot_world)
        for dash in range(0, 12, 2):
            a = dash / 12.0
            b = min((dash + 1) / 12.0, 1.0)
            p0 = tuple(np.round(np.asarray(tag_origin_px) * (1.0-a) + np.asarray(pivot_px) * a).astype(int))
            p1 = tuple(np.round(np.asarray(tag_origin_px) * (1.0-b) + np.asarray(pivot_px) * b).astype(int))
            cv2.line(canvas, p0, p1, colour, 1, cv2.LINE_AA)
        cv2.circle(canvas, pivot_px, 3, colour, -1, cv2.LINE_AA)


def draw_aprilgrid_anchor(canvas: np.ndarray, projector: Projector) -> None:
    """Draw the measured 8x8 world anchor in the same metric frame as the claw."""
    tag_size = 0.088
    pitch = tag_size * 1.30
    extent = 7 * pitch + tag_size
    half = extent / 2.0
    outline = np.array([
        projector(np.array([-half, half, 0.0])),
        projector(np.array([half, half, 0.0])),
        projector(np.array([half, -half, 0.0])),
        projector(np.array([-half, -half, 0.0])),
    ], np.int32)
    cv2.fillConvexPoly(canvas, outline, (25, 29, 35), cv2.LINE_AA)
    cv2.polylines(canvas, [outline], True, (118, 134, 151), 2, cv2.LINE_AA)
    for col in range(8):
        for row in range(8):
            x0 = col * pitch - half
            y0 = half - row * pitch
            square = np.array([
                projector(np.array([x0, y0, 0.0])),
                projector(np.array([x0 + tag_size, y0, 0.0])),
                projector(np.array([x0 + tag_size, y0 - tag_size, 0.0])),
                projector(np.array([x0, y0 - tag_size, 0.0])),
            ], np.int32)
            cv2.polylines(canvas, [square], True, (72, 82, 95), 1, cv2.LINE_AA)
    label = projector(np.array([-half, half, 0.0]))
    cv2.putText(canvas, "WORLD APRILGRID  ID 0-63", (label[0], label[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (159, 176, 195), 1, cv2.LINE_AA)


def draw_claw_model_inset(
    canvas: np.ndarray,
    rpy_deg: np.ndarray,
    claw_values: np.ndarray,
    meshes: dict[str, np.ndarray],
) -> None:
    """Large live URDF view in the right analysis panel."""
    rect = (1320, 108, 560, 165)
    cv2.rectangle(canvas, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]),
                  (48, 56, 67), 1, cv2.LINE_AA)
    cv2.putText(canvas, "LIVE URDF GRIPPER", (1334, 130), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (155, 174, 194), 1, cv2.LINE_AA)
    cv2.putText(canvas, "TAG ID2 FIXED TO BASE  |  PIVOTS 28.5 mm", (1560, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (238, 105, 229), 1, cv2.LINE_AA)
    extent = 0.17
    fit = np.asarray([[x, y, z] for x in (-extent, extent)
                      for y in (-extent, extent) for z in (-0.06, 0.14)], dtype=np.float64)
    model_projector = Projector(fit, (1340, 136, 520, 122))
    draw_claw_model(canvas, model_projector, np.zeros(3), rpy_deg,
                    float(claw_values[1]), float(claw_values[2]), meshes, model_scale=3.15)
    draw_pose_axes(canvas, model_projector, np.zeros(3), rpy_deg, axis_length=0.085)


def draw_board_axes(
    canvas: np.ndarray,
    projector: Projector,
    axis_length: float = 0.18,
    frame_suffix: str = "b",
) -> None:
    """Draw the fixed AprilGrid/board frame used by position and orientation."""
    origin_3d = np.zeros(3, dtype=np.float64)
    origin = projector(origin_3d)
    colors = ((40, 55, 245), (70, 220, 90), (245, 110, 55))
    labels = (f"X{frame_suffix}", f"Y{frame_suffix}", f"Z{frame_suffix}")
    cv2.circle(canvas, origin, 4, (225, 230, 236), -1, cv2.LINE_AA)
    for axis, color, label in zip(np.eye(3), colors, labels):
        endpoint = projector(origin_3d + axis * axis_length)
        cv2.arrowedLine(canvas, origin, endpoint, color, 2, cv2.LINE_AA, tipLength=0.20)
        cv2.putText(
            canvas, label, (endpoint[0] + 3, endpoint[1] - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA,
        )


def draw_frame_transform(
    canvas: np.ndarray,
    projector: Projector,
    camera_position: np.ndarray,
    reference_name: str = "board",
    target_name: str = "camera",
) -> None:
    """Connect board and camera frames with the translation in board axes."""
    if float(np.linalg.norm(camera_position)) < 1e-4:
        return
    x, y, z = camera_position
    points = (
        np.array([0.0, 0.0, 0.0]),
        np.array([x, 0.0, 0.0]),
        np.array([x, y, 0.0]),
        np.array([x, y, z]),
    )
    colors = ((40, 55, 245), (70, 220, 90), (245, 110, 55))
    labels = (f"dX {x:+.2f}m", f"dY {y:+.2f}m", f"dZ {z:+.2f}m")
    for start, end, color, label in zip(points[:-1], points[1:], colors, labels):
        p0, p1 = projector(start), projector(end)
        cv2.arrowedLine(canvas, p0, p1, color, 2, cv2.LINE_AA, tipLength=0.08)
        midpoint = ((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2)
        cv2.putText(
            canvas, label, (midpoint[0] + 5, midpoint[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA,
        )
    board_px, camera_px = projector(points[0]), projector(points[-1])
    # A neutral direct link makes the complete board->camera transform obvious;
    # colored component legs above explain how its translation is composed.
    for index in range(0, 20, 2):
        a = index / 20.0
        b = min((index + 1) / 20.0, 1.0)
        q0 = (
            int(board_px[0] + (camera_px[0] - board_px[0]) * a),
            int(board_px[1] + (camera_px[1] - board_px[1]) * a),
        )
        q1 = (
            int(board_px[0] + (camera_px[0] - board_px[0]) * b),
            int(board_px[1] + (camera_px[1] - board_px[1]) * b),
        )
        cv2.line(canvas, q0, q1, (202, 211, 222), 1, cv2.LINE_AA)
    transform_midpoint = (
        (board_px[0] + camera_px[0]) // 2,
        (board_px[1] + camera_px[1]) // 2,
    )
    cv2.putText(
        canvas, f"T {reference_name}->{target_name}", (transform_midpoint[0] + 7, transform_midpoint[1] + 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (202, 211, 222), 1, cv2.LINE_AA,
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
    reference_frame: str,
) -> None:
    panel = (842, 26, 410, 394)
    blend_rect(canvas, panel, 0.68)
    cv2.rectangle(canvas, (panel[0], panel[1]), (panel[0] + panel[2], panel[1] + panel[3]), (82, 92, 106), 1, cv2.LINE_AA)
    cv2.putText(canvas, "6DoF CAMERA POSE", (862, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (235, 240, 246), 1, cv2.LINE_AA)
    reference_label = "start frame" if reference_frame == "start" else "board frame"
    cv2.putText(canvas, f"{reference_label} + camera frame", (862, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (157, 171, 188), 1, cv2.LINE_AA)

    grid_points = []
    for x_value in np.linspace(-0.7, 0.45, 6):
        a = projector(np.array([x_value, -0.70, 0.0]))
        b = projector(np.array([x_value, 0.05, 0.0]))
        cv2.line(canvas, a, b, (62, 72, 84), 1, cv2.LINE_AA)
    for y_value in np.linspace(-0.7, 0.05, 5):
        a = projector(np.array([-0.70, y_value, 0.0]))
        b = projector(np.array([0.45, y_value, 0.0]))
        cv2.line(canvas, a, b, (62, 72, 84), 1, cv2.LINE_AA)
    draw_board_axes(canvas, projector, frame_suffix="s" if reference_frame == "start" else "b")

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
        draw_frame_transform(canvas, projector, current, reference_name=reference_frame)
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


def _nice_grid_step(span: float) -> float:
    """Return a readable metric grid step for the current trajectory extent."""
    target = max(span / 5.0, 1e-3)
    exponent = 10.0 ** math.floor(math.log10(target))
    normalized = target / exponent
    multiplier = 1.0 if normalized <= 1.0 else 2.0 if normalized <= 2.0 else 5.0
    return multiplier * exponent


def draw_analysis_hud(
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
    xyz_bounds: tuple[np.ndarray, np.ndarray],
    axis_length: float,
    reference_frame: str,
    claw_values: np.ndarray | None = None,
    claw_measured: bool = False,
    claw_confidence: float = 0.0,
    claw_meshes: dict[str, np.ndarray] | None = None,
) -> None:
    """Large split-screen visualization intended for trajectory inspection."""
    panel_x = 1280
    canvas[:, panel_x:] = (20, 24, 30)
    cv2.line(canvas, (panel_x, 0), (panel_x, 720), (72, 82, 96), 2, cv2.LINE_AA)
    title = "GRIPPER BASE 6DoF TRAJECTORY" if claw_values is not None else "ROBOT CAMERA 6DoF"
    cv2.putText(canvas, title, (1312, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.76, (240, 244, 248), 2, cv2.LINE_AA)
    subtitle = "FULL TRACK RELATIVE TO START POSE" if reference_frame == "start" else "FULL TRACK IN APRILGRID / WORLD FRAME"
    cv2.putText(canvas, subtitle, (1312, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (151, 170, 190), 1, cv2.LINE_AA)
    if claw_values is not None:
        cv2.putText(
            canvas, "TAG ID2 RIGID TO GRIPPER BASE  |  CAD PIVOTS 28.5 mm",
            (1312, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
            (226, 112, 220), 1, cv2.LINE_AA,
        )
    low, high = xyz_bounds
    x_step = _nice_grid_step(float(high[0] - low[0]))
    y_step = _nice_grid_step(float(high[1] - low[1]))
    x0 = math.floor(low[0] / x_step) * x_step
    x1 = math.ceil(high[0] / x_step) * x_step
    y0 = math.floor(low[1] / y_step) * y_step
    y1 = math.ceil(high[1] / y_step) * y_step
    for x_value in np.arange(x0, x1 + x_step * 0.5, x_step):
        a = projector(np.array([x_value, y0, 0.0]))
        b = projector(np.array([x_value, y1, 0.0]))
        cv2.line(canvas, a, b, (48, 56, 67), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{x_value:+.2f}", (a[0] - 14, a[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (111, 126, 143), 1, cv2.LINE_AA)
    for y_value in np.arange(y0, y1 + y_step * 0.5, y_step):
        a = projector(np.array([x0, y_value, 0.0]))
        b = projector(np.array([x1, y_value, 0.0]))
        cv2.line(canvas, a, b, (48, 56, 67), 1, cv2.LINE_AA)

    if reference_frame == "board":
        draw_aprilgrid_anchor(canvas, projector)

    draw_board_axes(
        canvas, projector, axis_length=axis_length,
        frame_suffix="s" if reference_frame == "start" else "b",
    )
    samples = list(history)
    first_point = next((p for _, p in samples if p is not None), None)
    if first_point is not None:
        start_px = projector(first_point)
        cv2.circle(canvas, start_px, 7, (92, 220, 130), -1, cv2.LINE_AA)
        cv2.putText(canvas, "START", (start_px[0] + 9, start_px[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (92, 220, 130), 1, cv2.LINE_AA)
    for index in range(1, len(samples)):
        t0, p0 = samples[index - 1]
        t1, p1 = samples[index]
        if p0 is None or p1 is None or t1 - t0 > 0.12:
            continue
        cv2.line(canvas, projector(p0), projector(p1), (242, 180, 62), 4, cv2.LINE_AA)

    if current is not None:
        draw_frame_transform(
            canvas, projector, current, reference_name=reference_frame,
            target_name="gripper" if claw_values is not None else "camera",
        )
        point = projector(current)
        cv2.circle(canvas, point, 12, (255, 210, 80), 2, cv2.LINE_AA)
        cv2.circle(canvas, point, 5, (255, 238, 176), -1, cv2.LINE_AA)
        if current_rpy is not None:
            draw_pose_axes(
                canvas, projector, current, current_rpy, axis_length=axis_length,
                frame_suffix="g" if claw_values is not None else "c",
            )
            if claw_values is not None and claw_meshes is not None:
                draw_claw_model(canvas, projector, current, current_rpy,
                                float(claw_values[1]), float(claw_values[2]), claw_meshes)

    state_color = ((93, 215, 139) if state in {"TRACKED", "VISUAL MEASURED"}
                   else (72, 174, 240) if state in {"PREDICTED", "ENTERING", "INTERPOLATED"}
                   else (128, 139, 154))
    cv2.circle(canvas, (1318, 578), 6, state_color, -1, cv2.LINE_AA)
    cv2.putText(canvas, state, (1332, 585), cv2.FONT_HERSHEY_SIMPLEX, 0.56, state_color, 1, cv2.LINE_AA)
    if claw_values is not None:
        claw_state = "DOTS MEASURED" if claw_measured else "DOTS RECOVERED"
        claw_color = (88, 224, 133) if claw_measured else (72, 174, 240)
        cv2.putText(canvas, f"GRIPPER {claw_values[0]:5.1f} deg  {claw_state}  Q {claw_confidence:.2f}",
                    (1312, 612), cv2.FONT_HERSHEY_SIMPLEX, 0.48, claw_color, 1, cv2.LINE_AA)
    if current is not None:
        cv2.putText(canvas, f"POSITION  X {current[0]:+.3f}  Y {current[1]:+.3f}  Z {current[2]:+.3f} m", (1312, 642), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 228, 237), 1, cv2.LINE_AA)
    if current_rpy is not None:
        wrapped = (current_rpy + 180.0) % 360.0 - 180.0
        cv2.putText(canvas, f"ATTITUDE  R {wrapped[0]:+.1f}  P {wrapped[1]:+.1f}  Y {wrapped[2]:+.1f} deg", (1312, 672), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (177, 195, 214), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{now:05.2f} / {duration:05.2f} s   |   {filter_label}", (1312, 704), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (135, 151, 169), 1, cv2.LINE_AA)

    # Video-side legend and progress bar.
    cv2.rectangle(canvas, (0, 640), (1280, 720), (18, 22, 28), -1)
    fixed_axes = "start axes: Xs/Ys/Zs" if reference_frame == "start" else "board axes: Xb/Yb/Zb"
    moving_axes = "gripper axes: Xg/Yg/Zg" if claw_values is not None else "camera axes: Xc/Yc/Zc"
    cv2.putText(canvas, f"X red   Y green   Z blue   |   fixed {fixed_axes}   moving {moving_axes}", (28, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.51, (203, 214, 225), 1, cv2.LINE_AA)
    cv2.line(canvas, (0, 714), (1280, 714), (43, 51, 61), 4)
    cv2.line(canvas, (0, 714), (int(1280 * np.clip(progress_ratio, 0.0, 1.0)), 714), (245, 185, 56), 4)


def main() -> int:
    args = parse_args()
    pose_quality = load_pose_quality(args.pose_csv)
    claw_data = load_claw_angles(args.claw_angle_csv) if args.claw_angle_csv else None
    claw_meshes = load_claw_meshes(args.claw_mesh_dir) if args.claw_mesh_dir else None
    if (claw_data is None) != (claw_meshes is None):
        raise ValueError("--claw-angle-csv and --claw-mesh-dir must be used together")
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
    if args.camera_to_gripper_json:
        data = apply_gripper_extrinsic(data, args.camera_to_gripper_json)
    if args.reference_frame == "start":
        data = make_start_relative(data)
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
    xyz = data[:, 1:4]
    xyz_low = np.minimum(xyz.min(axis=0), np.zeros(3))
    xyz_high = np.maximum(xyz.max(axis=0), np.zeros(3))
    xyz_span = np.maximum(xyz_high - xyz_low, 1e-3)
    axis_length = float(np.clip(np.max(xyz_span) * 0.16, 0.10, 0.24))
    board_reference = np.vstack([np.zeros((1, 3)), np.eye(3) * axis_length])
    if args.reference_frame == "board":
        grid_half = (7 * 0.088 * 1.30 + 0.088) / 2.0
        board_reference = np.vstack([
            board_reference,
            [[x, y, 0.0] for x in (-grid_half, grid_half) for y in (-grid_half, grid_half)],
        ])
    fit_low = xyz_low - axis_length * 0.65
    fit_high = xyz_high + axis_length * 0.65
    fit_corners = np.asarray(
        [[x, y, z] for x in (fit_low[0], fit_high[0])
         for y in (fit_low[1], fit_high[1]) for z in (fit_low[2], fit_high[2])],
        dtype=np.float64,
    )
    if args.layout == "analysis":
        # With a gripper model there is only one 3D view: the model lives in
        # the trajectory coordinate system.  Give that unified view the full
        # panel instead of duplicating the gripper in a separate inset.
        projector_rect = (1312, 92, 576, 460) if claw_data is not None else (1312, 292, 576, 260)
        canvas_size = (720, 1920)
    else:
        projector_rect = (862, 94, 368, 276)
        canvas_size = (720, 1280)
    projector = Projector(np.vstack([xyz, board_reference, fit_corners]), projector_rect)
    history: deque[tuple[float, np.ndarray | None]] = deque()
    temporary = args.output.with_suffix(".visual.tmp.mp4")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    encode = subprocess.Popen(
        [
            str(args.ffmpeg), "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{canvas_size[1]}x{canvas_size[0]}", "-r", str(args.fps), "-i", "-", "-an", "-c:v", "libx264",
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
        canvas = np.zeros((*canvas_size, 3), dtype=np.uint8)
        if args.video_fit == "contain":
            source_h, source_w = frame.shape[:2]
            scale = min(1280.0 / source_w, 640.0 / source_h)
            fitted_w = max(1, int(round(source_w * scale)))
            fitted_h = max(1, int(round(source_h * scale)))
            fitted = cv2.resize(frame, (fitted_w, fitted_h), interpolation=cv2.INTER_AREA)
            x0 = (1280 - fitted_w) // 2
            y0 = (640 - fitted_h) // 2
            canvas[y0:y0 + fitted_h, x0:x0 + fitted_w] = fitted
        else:
            canvas[:640, :1280] = cv2.resize(frame, (1280, 640), interpolation=cv2.INTER_AREA)
        claw_values = None
        claw_measured = False
        claw_confidence = 0.0
        if claw_data is not None:
            claw_values, claw_measured, claw_confidence = sample_claw_angle(claw_data, now)
            blend_rect(canvas, (24, 24, 555, 108), 0.72)
            cv2.putText(canvas, f"GRIPPER OPENING  {claw_values[0]:5.1f} deg", (46, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.88, (38, 232, 255), 2, cv2.LINE_AA)
            label = "DOTS MEASURED" if claw_measured else "DOTS RECOVERED"
            color = (88, 224, 133) if claw_measured else (72, 174, 240)
            cv2.putText(canvas, f"J1 {claw_values[1]:+5.1f}  J2 {claw_values[2]:+5.1f} deg   {label}",
                        (46, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        pose, state, opacity = sample_pose(data, now, args.prediction_max_age)
        state = measurement_state(pose_quality, now, state)
        if args.state_label:
            state = args.state_label
        position = None if pose is None else pose[:3]
        orientation = None if pose is None else pose[3:6]
        history.append((now, None if position is None else position.copy()))
        if args.layout == "overlay":
            while history and now - history[0][0] > args.tail_seconds:
                history.popleft()
        filter_label = "Kalman + RTS" if args.filter == "kalman" else f"{args.smooth * 100:.0f}% smoothing"
        if args.layout == "analysis":
            draw_analysis_hud(
                canvas, projector, history, position, orientation, state, opacity, now,
                output_duration, local_time / output_duration, filter_label,
                (xyz_low, xyz_high), axis_length, args.reference_frame,
                claw_values, claw_measured, claw_confidence, claw_meshes,
            )
        else:
            draw_hud(
                canvas, projector, history, position, orientation, state, opacity, now,
                output_duration, local_time / output_duration, filter_label,
                args.reference_frame,
            )
        assert encode.stdin is not None
        encode.stdin.write(canvas.tobytes())
        output_index += 1
    capture.release()
    assert encode.stdin is not None
    encode.stdin.close()
    if encode.wait() != 0:
        raise SystemExit("video encoder failed")
    audio_source = args.audio_source if args.audio_source is not None else args.video
    subprocess.run(
        [
            str(args.ffmpeg), "-v", "error", "-i", str(temporary), "-ss", str(start_time),
            "-t", str(output_duration), "-i", str(audio_source),
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
