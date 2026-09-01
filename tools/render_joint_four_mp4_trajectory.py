#!/usr/bin/env python3
"""Render four synchronized fisheye videos beside shared-map 6DoF telemetry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from osmo360.ffmpeg_runtime import project_ffmpeg_runtime
from osmo360.localization.cached_a3_bootstrap import (
    MAXIMUM_TRUSTED_INTERPOLATION_GAP_S,
)


BG = (13, 17, 23)
PANEL = (24, 31, 41)
PLOT_BG = (17, 24, 34)
CARD = (29, 38, 50)
GRID = (55, 67, 82)
MUTED = (132, 145, 161)
WHITE = (238, 242, 247)
LEFT = (238, 196, 72)
RIGHT = (118, 151, 255)
AXIS_X = (92, 105, 255)
AXIS_Y = (113, 222, 126)
AXIS_Z = (255, 184, 82)
WARNING = (65, 177, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("tracking_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--duration", type=float, default=0.0,
        help="seconds; zero uses the complete common timeline",
    )
    parser.add_argument(
        "--view-preset",
        choices=("legacy-oblique", "tag-map-front-above", "flu-front-above"),
        default="legacy-oblique",
        help=(
            "fixed 3D view; tag-map-front-above keeps the native Tag-map world, "
            "while flu-front-above expects a FLU world"
        ),
    )
    parser.add_argument(
        "--ffmpeg", type=Path,
        help="override the verified project FFmpeg runtime",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class TrackSample:
    position: np.ndarray
    rotation: Rotation
    rpy_deg: np.ndarray
    linear_speed_m_s: float
    angular_speed_deg_s: float
    state: str


class Track:
    def __init__(
        self,
        rows: list[dict[str, str]],
        prefix: str,
        *,
        maximum_gap_s: float = MAXIMUM_TRUSTED_INTERPOLATION_GAP_S,
    ):
        if not 0 < maximum_gap_s <= MAXIMUM_TRUSTED_INTERPOLATION_GAP_S:
            raise ValueError("maximum_gap_s must be positive and no greater than 0.25")
        valid = [
            row for row in rows
            if row.get(f"{prefix}_camera_x_m")
        ]
        self.times = np.asarray([float(row["timestamp_s"]) for row in valid])
        self.positions = np.asarray([
            [float(row[f"{prefix}_camera_{axis}_m"]) for axis in "xyz"]
            for row in valid
        ])
        self.rotations = Rotation.from_quat(np.asarray([
            [float(row[f"{prefix}_q{axis}"]) for axis in "xyzw"]
            for row in valid
        ]))
        self.states = np.asarray([
            row.get(f"{prefix}_pose_state", "UNKNOWN") for row in valid
        ])
        if len(self.times) < 2 or np.any(np.diff(self.times) <= 0):
            raise ValueError(f"{prefix} joint trajectory has no increasing valid track")
        self.maximum_gap_s = maximum_gap_s
        gaps = np.diff(self.times)
        self.segment_ids = np.cumsum(np.r_[False, gaps > maximum_gap_s])
        self.slerp = Slerp(self.times, self.rotations)
        raw_rpy = self.rotations.as_euler("xyz", degrees=True)
        self.rpy_deg = np.degrees(np.unwrap(np.radians(raw_rpy), axis=0))
        delta_t = np.diff(self.times)
        linear_segments = np.linalg.norm(np.diff(self.positions, axis=0), axis=1) / delta_t
        angular_segments = np.degrees(
            (self.rotations[:-1].inv() * self.rotations[1:]).magnitude()
        ) / delta_t
        linear_segments[gaps > maximum_gap_s] = 0.0
        angular_segments[gaps > maximum_gap_s] = 0.0
        self.linear_speed_m_s = np.r_[linear_segments[0], linear_segments]
        self.angular_speed_deg_s = np.r_[angular_segments[0], angular_segments]

    def segment_slices(self, upto: int | None = None) -> list[slice]:
        stop = len(self.times) if upto is None else min(max(upto, 0), len(self.times))
        if stop == 0:
            return []
        split = np.flatnonzero(np.diff(self.segment_ids[:stop]) != 0) + 1
        edges = np.r_[0, split, stop]
        return [slice(int(start), int(end)) for start, end in zip(edges[:-1], edges[1:])]

    def sample(self, now: float) -> TrackSample | None:
        tolerance = 1e-6
        if now < self.times[0] - tolerance or now > self.times[-1] + tolerance:
            return None
        upper = int(np.searchsorted(self.times, now, side="left"))
        if upper < len(self.times) and abs(float(self.times[upper] - now)) <= tolerance:
            clipped = float(self.times[upper])
        else:
            if upper == 0 or upper >= len(self.times):
                return None
            lower = upper - 1
            if self.times[upper] - self.times[lower] > self.maximum_gap_s + tolerance:
                return None
            clipped = float(now)
        position = np.asarray([
            np.interp(clipped, self.times, self.positions[:, axis])
            for axis in range(3)
        ])
        rpy = np.asarray([
            np.interp(clipped, self.times, self.rpy_deg[:, axis])
            for axis in range(3)
        ])
        nearest = int(np.argmin(np.abs(self.times - clipped)))
        return TrackSample(
            position=position,
            rotation=self.slerp([clipped])[0],
            rpy_deg=rpy,
            linear_speed_m_s=float(np.interp(
                clipped, self.times, self.linear_speed_m_s
            )),
            angular_speed_deg_s=float(np.interp(
                clipped, self.times, self.angular_speed_deg_s
            )),
            state=str(self.states[nearest]),
        )


class VideoSampler:
    def __init__(self, path: Path):
        self.capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
        if not self.capture.isOpened():
            raise ValueError(f"cannot open video: {path}")
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.index = 0
        self.frame: np.ndarray | None = None

    def sample(self, now: float, size: tuple[int, int]) -> np.ndarray:
        target = max(0, int(round(now * self.fps)))
        while self.index <= target:
            ok, frame = self.capture.read()
            if not ok:
                break
            self.frame = frame
            self.index += 1
        if self.frame is None:
            return np.full((size[1], size[0], 3), PANEL, dtype=np.uint8)
        return cv2.resize(self.frame, size, interpolation=cv2.INTER_AREA)

    def release(self) -> None:
        self.capture.release()


def text(
    canvas: np.ndarray,
    value: str,
    origin: tuple[int, int],
    scale: float = 0.55,
    color: tuple[int, int, int] = WHITE,
    thickness: int = 1,
) -> None:
    cv2.putText(
        canvas, value, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
        color, thickness, cv2.LINE_AA,
    )


def _dim(color: tuple[int, int, int], amount: float = 0.38) -> tuple[int, int, int]:
    return tuple(int(channel * amount) for channel in color)


def translucent_box(
    canvas: np.ndarray,
    upper_left: tuple[int, int],
    lower_right: tuple[int, int],
    color: tuple[int, int, int],
    alpha: float = 0.86,
) -> None:
    x0, y0 = upper_left
    x1, y1 = lower_right
    overlay = canvas[y0:y1, x0:x1].copy()
    overlay[:] = color
    cv2.addWeighted(overlay, alpha, canvas[y0:y1, x0:x1], 1.0 - alpha, 0,
                    canvas[y0:y1, x0:x1])


def load_map(path: Path) -> tuple[dict, list[dict[str, object]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tags = [
        {
            "id": int(tag["id"]),
            "panel": str(tag.get("panel", "grid_A")),
            "corners": np.asarray(tag["corners_m"], dtype=float),
        }
        for tag in payload["tags"]
    ]
    return payload, tags


class Projector:
    def __init__(
        self,
        points: np.ndarray,
        origin: tuple[int, int],
        size: tuple[int, int],
        *,
        preset: str = "legacy-oblique",
        focus: np.ndarray | None = None,
    ):
        self.preset = preset
        self.origin = np.asarray(origin, dtype=float)
        self.size = np.asarray(size, dtype=float)
        self.eye: np.ndarray | None = None
        self.target: np.ndarray | None = None
        if preset == "legacy-oblique":
            self.view = Rotation.from_euler("xz", [62.0, -38.0], degrees=True)
            transformed = self.view.apply(points)[:, :2]
        elif preset in {"tag-map-front-above", "flu-front-above"}:
            self.view = None
            panel_focus = (
                np.asarray(focus, dtype=float)
                if focus is not None
                else np.asarray(points, dtype=float).mean(axis=0)
            )
            if preset == "tag-map-front-above":
                # Native Tag map: wall is Z=0 and physical up is -Y.
                self.eye = panel_focus + np.asarray([0.0, -0.85, -1.55])
                self.target = panel_focus + np.asarray([0.0, 0.0, -0.28])
                world_up = np.asarray([0.0, -1.0, 0.0])
            else:
                # FLU world: wall is X=0 and physical up is +Z.
                self.eye = panel_focus + np.asarray([1.55, 0.0, 0.85])
                self.target = panel_focus + np.asarray([0.28, 0.0, 0.0])
                world_up = np.asarray([0.0, 0.0, 1.0])
            forward = self.target - self.eye
            self.forward = forward / np.linalg.norm(forward)
            right = np.cross(self.forward, world_up)
            self.right = right / np.linalg.norm(right)
            self.camera_up = np.cross(self.right, self.forward)
            transformed = np.asarray([
                self._perspective_value(point) for point in points
            ])
        else:
            raise ValueError(f"unknown view preset: {preset}")
        low = transformed.min(axis=0)
        high = transformed.max(axis=0)
        span = np.maximum(high - low, 0.1)
        padding = span * 0.10 + 0.015
        low -= padding
        high += padding
        scale = min(self.size[0] / (high[0] - low[0]),
                    self.size[1] / (high[1] - low[1]))
        content = (high - low) * scale
        self.offset = self.origin + (self.size - content) / 2.0
        self.low = low
        self.scale = scale

    def _perspective_value(self, point: np.ndarray) -> np.ndarray:
        relative = np.asarray(point, dtype=float) - self.eye
        depth = float(np.dot(relative, self.forward))
        if depth <= 1e-3:
            raise ValueError("3D point lies behind the fixed front-above camera")
        return np.asarray([
            np.dot(relative, self.right) / depth,
            np.dot(relative, self.camera_up) / depth,
        ])

    def __call__(self, point: np.ndarray) -> tuple[int, int]:
        value = (
            self.view.apply(np.asarray(point, dtype=float))[:2]
            if self.preset == "legacy-oblique"
            else self._perspective_value(point)
        )
        pixel = self.offset + (value - self.low) * self.scale
        return int(round(pixel[0])), int(round(
            self.origin[1] + self.size[1] - (pixel[1] - self.origin[1])
        ))


def draw_polyline(
    canvas: np.ndarray,
    projector: Projector,
    points: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 2,
    closed: bool = False,
) -> None:
    if len(points) < 2:
        return
    pixels = np.asarray([projector(point) for point in points], dtype=np.int32)
    cv2.polylines(canvas, [pixels], closed, color, thickness, cv2.LINE_AA)


def draw_grid_and_axes(
    canvas: np.ndarray,
    projector: Projector,
    point_bounds: tuple[np.ndarray, np.ndarray],
) -> None:
    low, high = point_bounds
    if projector.preset == "flu-front-above":
        y0 = np.floor((low[1] - 0.05) / 0.1) * 0.1
        y1 = np.ceil((high[1] + 0.05) / 0.1) * 0.1
        z0 = np.floor((low[2] - 0.05) / 0.1) * 0.1
        z1 = np.ceil((high[2] + 0.05) / 0.1) * 0.1
        for y in np.arange(y0, y1 + 0.001, 0.1):
            major = abs((y * 10) % 2) < 0.01
            draw_polyline(
                canvas, projector, np.asarray([[0, y, z0], [0, y, z1]]),
                GRID if major else _dim(GRID, 0.65), 1,
            )
        for z in np.arange(z0, z1 + 0.001, 0.1):
            major = abs((z * 10) % 2) < 0.01
            draw_polyline(
                canvas, projector, np.asarray([[0, y0, z], [0, y1, z]]),
                GRID if major else _dim(GRID, 0.65), 1,
            )
    else:
        x0 = np.floor((low[0] - 0.05) / 0.1) * 0.1
        x1 = np.ceil((high[0] + 0.05) / 0.1) * 0.1
        y0 = np.floor((low[1] - 0.05) / 0.1) * 0.1
        y1 = np.ceil((high[1] + 0.05) / 0.1) * 0.1
        for x in np.arange(x0, x1 + 0.001, 0.1):
            major = abs((x * 10) % 2) < 0.01
            draw_polyline(
                canvas, projector, np.asarray([[x, y0, 0], [x, y1, 0]]),
                GRID if major else _dim(GRID, 0.65), 1,
            )
        for y in np.arange(y0, y1 + 0.001, 0.1):
            major = abs((y * 10) % 2) < 0.01
            draw_polyline(
                canvas, projector, np.asarray([[x0, y, 0], [x1, y, 0]]),
                GRID if major else _dim(GRID, 0.65), 1,
            )
    origin = np.zeros(3)
    if projector.preset == "flu-front-above":
        labels = ("X FWD", "Y LEFT", "Z UP")
    elif projector.preset == "tag-map-front-above":
        labels = ("X MAP", "Y MAP", "Z MAP")
    else:
        labels = ("X", "Y", "Z")
    for endpoint, color, label in zip(
        (np.asarray([0.22, 0, 0]), np.asarray([0, 0.22, 0]),
         np.asarray([0, 0, 0.22])),
        (AXIS_X, AXIS_Y, AXIS_Z),
        labels,
    ):
        p0, p1 = projector(origin), projector(endpoint)
        cv2.arrowedLine(canvas, p0, p1, color, 3, cv2.LINE_AA, tipLength=0.14)
        text(canvas, label, (p1[0] + 5, p1[1] - 5), 0.40, color, 2)
    center = projector(origin)
    cv2.circle(canvas, center, 5, WHITE, -1, cv2.LINE_AA)
    text(canvas, "WORLD / GRID A", (center[0] + 9, center[1] + 18), 0.38, MUTED)


def draw_tags(
    canvas: np.ndarray, projector: Projector, tags: list[dict[str, object]]
) -> None:
    overlay = canvas.copy()
    panel_points: dict[str, list[np.ndarray]] = {"grid_A": [], "grid_B": []}
    for tag in tags:
        corners = np.asarray(tag["corners"])
        panel = str(tag["panel"])
        pixels = np.asarray([projector(point) for point in corners], dtype=np.int32)
        fill = (55, 72, 91) if panel == "grid_A" else (67, 57, 87)
        cv2.fillConvexPoly(overlay, pixels, fill, cv2.LINE_AA)
        panel_points.setdefault(panel, []).extend(corners)
    cv2.addWeighted(overlay, 0.52, canvas, 0.48, 0, canvas)
    for tag in tags:
        corners = np.asarray(tag["corners"])
        panel = str(tag["panel"])
        outline = (105, 128, 153) if panel == "grid_A" else (135, 108, 159)
        draw_polyline(canvas, projector, corners, outline, 1, closed=True)
    for panel, points in panel_points.items():
        if not points:
            continue
        center = projector(np.mean(np.asarray(points), axis=0))
        text(
            canvas, "APRILGRID A" if panel == "grid_A" else "APRILGRID B",
            (center[0] - 46, center[1] - 8), 0.36, WHITE, 1,
        )


def draw_track(
    canvas: np.ndarray,
    projector: Projector,
    track: Track,
    now: float,
    color: tuple[int, int, int],
) -> None:
    upto = int(np.searchsorted(track.times, now, side="right"))
    for segment in track.segment_slices(upto):
        draw_polyline(canvas, projector, track.positions[segment], _dim(color), 2)
        indices = np.arange(segment.start, segment.stop)
        recent = indices[track.times[indices] >= now - 2.0]
        if len(recent):
            draw_polyline(canvas, projector, track.positions[recent], color, 4)
        for index in range(segment.start + 1, segment.stop):
            if "INTERPOLATED" in (track.states[index - 1], track.states[index]):
                p0 = projector(track.positions[index - 1])
                p1 = projector(track.positions[index])
                cv2.line(canvas, p0, p1, _dim(color, 0.65), 2, cv2.LINE_AA)


def draw_camera(
    canvas: np.ndarray,
    projector: Projector,
    sample: TrackSample,
    color: tuple[int, int, int],
    label: str,
) -> None:
    position = sample.position
    center = projector(position)
    local_frustum = (
        np.asarray([
            [0.12, -0.045, -0.032], [0.12, 0.045, -0.032],
            [0.12, 0.045, 0.032], [0.12, -0.045, 0.032],
        ])
        if projector.preset in {"tag-map-front-above", "flu-front-above"}
        else np.asarray([
            [-0.045, -0.032, 0.12], [0.045, -0.032, 0.12],
            [0.045, 0.032, 0.12], [-0.045, 0.032, 0.12],
        ])
    )
    frustum = sample.rotation.apply(local_frustum) + position
    pixels = [projector(point) for point in frustum]
    cv2.polylines(canvas, [np.asarray(pixels, dtype=np.int32)], True,
                  _dim(color, 0.8), 2, cv2.LINE_AA)
    for point in pixels:
        cv2.line(canvas, center, point, _dim(color, 0.8), 1, cv2.LINE_AA)
    for axis_index, (axis, axis_color) in enumerate(
        zip(np.eye(3), (AXIS_X, AXIS_Y, AXIS_Z))
    ):
        endpoint = projector(position + sample.rotation.apply(axis * 0.09))
        cv2.arrowedLine(canvas, center, endpoint, axis_color, 2, cv2.LINE_AA,
                        tipLength=0.16)
        if projector.preset in {"tag-map-front-above", "flu-front-above"}:
            text(canvas, f"{('Xc', 'Yc', 'Zc')[axis_index]}",
                 (endpoint[0] + 3, endpoint[1] - 3), 0.28, axis_color, 1)
    cv2.circle(canvas, center, 9, color, -1, cv2.LINE_AA)
    cv2.circle(canvas, center, 4, WHITE, -1, cv2.LINE_AA)
    text(canvas, label, (center[0] + 12, center[1] - 9), 0.52, color, 2)


def draw_depth_to_reference_plane(
    canvas: np.ndarray,
    projector: Projector,
    position: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    if projector.preset not in {"tag-map-front-above", "flu-front-above"}:
        return
    if projector.preset == "tag-map-front-above":
        foot = np.asarray([position[0], position[1], 0.0])
        depth = abs(float(position[2]))
        depth_label = f"wall depth {depth:.2f} m"
    else:
        foot = np.asarray([0.0, position[1], position[2]])
        depth_label = f"X depth {position[0]:.2f} m"
    samples = np.linspace(position, foot, 17)
    for index in range(0, len(samples) - 1, 2):
        cv2.line(
            canvas, projector(samples[index]), projector(samples[index + 1]),
            _dim(color, 0.72), 2, cv2.LINE_AA,
        )
    foot_pixel = projector(foot)
    cv2.circle(canvas, foot_pixel, 4, _dim(color, 0.8), 1, cv2.LINE_AA)
    middle = projector((position + foot) * 0.5)
    text(canvas, depth_label, (middle[0] + 5, middle[1] - 5), 0.30, color, 1)


def draw_sparkline(
    canvas: np.ndarray,
    rect: tuple[int, int, int, int],
    track: Track,
    values: np.ndarray,
    now: float,
    labels: str,
) -> None:
    x, y, width, height = rect
    cv2.rectangle(canvas, (x, y), (x + width, y + height), PLOT_BG, -1)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), GRID, 1)
    low = float(np.min(values))
    high = float(np.max(values))
    margin = max((high - low) * 0.08, 1e-4)
    low -= margin
    high += margin
    x_values = x + (track.times - track.times[0]) / (
        track.times[-1] - track.times[0]
    ) * width
    palette = (AXIS_X, AXIS_Y, AXIS_Z)
    upto = max(1, int(np.searchsorted(track.times, now, side="right")))
    for axis, color in enumerate(palette):
        y_values = y + height - (values[:, axis] - low) / (high - low) * height
        points = np.c_[x_values, y_values].round().astype(np.int32)
        for segment in track.segment_slices():
            cv2.polylines(
                canvas, [points[segment]], False, _dim(color, 0.32), 1, cv2.LINE_AA
            )
        for segment in track.segment_slices(upto):
            cv2.polylines(
                canvas, [points[segment]], False, color, 1, cv2.LINE_AA
            )
    if low <= 0 <= high:
        zero_y = int(round(y + height - (0 - low) / (high - low) * height))
        cv2.line(canvas, (x, zero_y), (x + width, zero_y), GRID, 1, cv2.LINE_AA)
    cursor_x = int(round(x + np.clip(
        (now - track.times[0]) / (track.times[-1] - track.times[0]), 0, 1
    ) * width))
    cv2.line(canvas, (cursor_x, y), (cursor_x, y + height), WHITE, 1, cv2.LINE_AA)
    for index, label in enumerate(labels):
        text(canvas, label, (x + 7 + index * 26, y + 13), 0.30,
             palette[index], 1)
    text(canvas, f"{low:+.1f}..{high:+.1f}", (x + width - 76, y + 13),
         0.27, MUTED)


def draw_telemetry_card(
    canvas: np.ndarray,
    rect: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int],
    track: Track,
    sample: TrackSample | None,
    now: float,
    coordinate_frame: str,
    camera_frame: str,
) -> None:
    x, y, width, height = rect
    translucent_box(canvas, (x, y), (x + width, y + height), CARD, 0.94)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), _dim(color, 0.65), 1)
    cv2.circle(canvas, (x + 17, y + 20), 6, color, -1, cv2.LINE_AA)
    text(canvas, f"{label} CAMERA", (x + 30, y + 25), 0.50, WHITE, 2)
    state = "UNTRUSTED" if sample is None else sample.state
    state_color = color if state == "MEASURED" else WARNING
    state_width = min(210, max(92, 18 + len(state) * 8))
    cv2.rectangle(canvas, (x + width - state_width - 10, y + 8),
                  (x + width - 10, y + 31), _dim(state_color, 0.40), -1)
    text(canvas, state, (x + width - state_width + 1, y + 24),
         0.33, state_color, 1)

    text(canvas, f"POSITION XYZ  [m] / {coordinate_frame}",
         (x + 12, y + 51), 0.35, MUTED)
    position_text = (
        "X N/A  Y N/A  Z N/A"
        if sample is None else "  ".join(
            f"{axis} {value:+.3f}" for axis, value in zip("XYZ", sample.position)
        )
    )
    text(canvas, position_text, (x + 12, y + 72), 0.43, WHITE, 1)
    camera_label = "CAMERA FLU RPY" if camera_frame == "CAMERA FLU" else "CAM RPY"
    text(canvas, f"{camera_label}  [deg, xyz] / {coordinate_frame}",
         (x + 12, y + 96), 0.35, MUTED)
    rpy_text = (
        "R N/A  P N/A  Y N/A"
        if sample is None else "  ".join(
            f"{axis} {value:+.1f}" for axis, value in zip("RPY", sample.rpy_deg)
        )
    )
    text(canvas, rpy_text, (x + 12, y + 117), 0.43, WHITE, 1)
    speed_text = (
        "SPEED  N/A                  OMEGA  N/A"
        if sample is None else (
            f"SPEED  {sample.linear_speed_m_s:5.2f} m/s     "
            f"OMEGA  {sample.angular_speed_deg_s:5.1f} deg/s"
        )
    )
    text(canvas, speed_text, (x + 12, y + 142), 0.37, color, 1)
    text(canvas, "XYZ OVER FULL CLIP", (x + 12, y + 162), 0.30, MUTED)
    draw_sparkline(
        canvas, (x + 12, y + 168, width - 24, 57),
        track, track.positions, now, "XYZ",
    )
    text(canvas, "RPY OVER FULL CLIP", (x + 12, y + 244), 0.30, MUTED)
    draw_sparkline(
        canvas, (x + 12, y + 250, width - 24, 57),
        track, track.rpy_deg, now, "RPY",
    )


def render_gradient_panel(canvas: np.ndarray) -> None:
    x0, y0, x1, y1 = 980, 55, 1900, 1005
    top = np.asarray([39, 48, 61], dtype=float)
    bottom = np.asarray([15, 21, 29], dtype=float)
    for y in range(y0, y1):
        mix = (y - y0) / max(y1 - y0 - 1, 1)
        canvas[y, x0:x1] = np.round(top * (1 - mix) + bottom * mix).astype(np.uint8)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (70, 82, 98), 1)


def provenance_lines(report: dict, world_map: dict) -> tuple[str, str]:
    claims = report.get("claims", {})
    if claims.get("trajectory_source") == "user_provided_csv":
        world_reexpressed = claims.get(
            "world_coordinate_reexpressed", claims.get("coordinate_reexpressed", False)
        )
        camera_reexpressed = claims.get("camera_coordinate_reexpressed", False)
        if camera_reexpressed and not world_reexpressed:
            first = "USER-PROVIDED CSV / TAG-MAP POSITION UNCHANGED / CAMERA AXES -> FLU"
        elif world_reexpressed:
            first = "USER-PROVIDED CSV / RE-EXPRESSED IN WORLD FLU / PHYSICAL POSE UNCHANGED"
        else:
            first = "USER-PROVIDED CSV / POSES UNCHANGED"
        second = (
            "DISPLAY MAP BINDING UNVERIFIED - CSV CONTAINS NO MAP HASH"
            if not claims.get("display_map_binding_verified", False)
            else f"Map: {world_map['map_id']}"
        )
        return first, second
    return (
        "Capture-local self-calibration; not external ground truth",
        f"Map: {world_map['map_id']}",
    )


def _extract_frame(video: Path, timestamp_s: float, output: Path) -> None:
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_MSEC, max(timestamp_s, 0.0) * 1000.0)
    ok, image = capture.read()
    capture.release()
    if not ok or not cv2.imwrite(str(output), image):
        raise ValueError(f"cannot extract video frame at {timestamp_s:.3f}s")


def main() -> int:
    args = parse_args()
    dataset = args.dataset_root.resolve(strict=True)
    tracking = args.tracking_dir.resolve(strict=True)
    with (tracking / "joint_trajectory.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    report = json.loads((tracking / "report.json").read_text(encoding="utf-8"))
    maximum_gap_s = float(
        report.get("trajectories", {}).get("joint", {}).get(
            "maximum_allowed_interpolation_gap_s",
            MAXIMUM_TRUSTED_INTERPOLATION_GAP_S,
        )
    )
    left = Track(rows, "left", maximum_gap_s=maximum_gap_s)
    right = Track(rows, "right", maximum_gap_s=maximum_gap_s)
    world_map, tags = load_map(tracking / "session_world_map.json")
    start = max(left.times[0], right.times[0])
    end = min(left.times[-1], right.times[-1])
    if args.duration > 0:
        end = min(end, start + args.duration)
    if end <= start:
        raise ValueError("joint trajectory has no common render interval")

    video_paths = {
        "LEFT BACK / H5 KB": dataset / "video/Left_back.mp4",
        "LEFT FORWARD": dataset / "video/Left_forward.mp4",
        "RIGHT BACK / H5 KB": dataset / "video/Right_back.mp4",
        "RIGHT FORWARD": dataset / "video/Right_forward.mp4",
    }
    samplers = {name: VideoSampler(path) for name, path in video_paths.items()}
    tag_points = np.vstack([np.asarray(tag["corners"]) for tag in tags])
    all_world_points = np.vstack((left.positions, right.positions, tag_points))
    low, high = all_world_points.min(axis=0), all_world_points.max(axis=0)
    if args.view_preset == "flu-front-above":
        grid_corners = np.asarray([
            [0, low[1] - 0.08, low[2] - 0.08],
            [0, high[1] + 0.08, low[2] - 0.08],
            [0, low[1] - 0.08, high[2] + 0.08],
            [0, high[1] + 0.08, high[2] + 0.08],
            [0.22, 0, 0], [0, 0.22, 0], [0, 0, 0.22],
        ])
    else:
        grid_corners = np.asarray([
            [low[0] - 0.08, low[1] - 0.08, 0.0],
            [high[0] + 0.08, low[1] - 0.08, 0.0],
            [low[0] - 0.08, high[1] + 0.08, 0.0],
            [high[0] + 0.08, high[1] + 0.08, 0.0],
            [0.22, 0, 0], [0, 0.22, 0], [0, 0, 0.22],
        ])
    projector = Projector(
        np.vstack((all_world_points, grid_corners)), (1000, 72), (880, 565),
        preset=args.view_preset,
        focus=tag_points.mean(axis=0),
    )
    coordinate_frame = {
        "flu-front-above": "WORLD FLU",
        "tag-map-front-above": "TAG MAP",
    }.get(args.view_preset, world_map.get("world_frame", "WORLD"))
    camera_frame = (
        "CAMERA FLU"
        if args.view_preset in {"tag-map-front-above", "flu-front-above"}
        else "SOURCE CAMERA"
    )
    provenance_first, provenance_second = provenance_lines(report, world_map)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    intermediate = args.output.with_name(args.output.stem + ".mp4v.mp4")
    writer = cv2.VideoWriter(
        str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (1920, 1080)
    )
    if not writer.isOpened():
        raise ValueError("cannot create comparison video")
    duration = end - start
    total = max(1, int(round(duration * args.fps)))
    names = list(video_paths)
    try:
        for index in range(total):
            now = start + index / args.fps
            canvas = np.full((1080, 1920, 3), BG, dtype=np.uint8)
            text(canvas, "4x FISHEYE VIDEO", (24, 37), 0.72, WHITE, 2)
            if args.view_preset == "flu-front-above":
                title = "JOINT 6DoF / FIXED FRONT-ABOVE / WORLD FLU"
            elif args.view_preset == "tag-map-front-above":
                title = "JOINT 6DoF / FIXED FRONT-ABOVE / TAG MAP + CAM FLU"
            else:
                title = "JOINT 6DoF / ONE SHARED APRILGRID MAP"
            text(canvas, title, (985, 37), 0.68, WHITE, 2)
            for tile, name in enumerate(names):
                x = 20 + (tile % 2) * 475
                y = 55 + (tile // 2) * 475
                frame = samplers[name].sample(now, (455, 420))
                canvas[y:y + 420, x:x + 455] = frame
                color = LEFT if name.startswith("LEFT") else RIGHT
                cv2.rectangle(canvas, (x, y), (x + 455, y + 420), color, 2)
                translucent_box(canvas, (x + 6, y + 6), (x + 222, y + 35), BG, 0.70)
                text(canvas, name, (x + 12, y + 27), 0.45, color, 2)

            render_gradient_panel(canvas)
            cv2.rectangle(canvas, (995, 67), (1885, 645), PLOT_BG, -1)
            cv2.rectangle(canvas, (995, 67), (1885, 645), GRID, 1)
            draw_grid_and_axes(canvas, projector, (low, high))
            draw_tags(canvas, projector, tags)
            left_sample = left.sample(now)
            right_sample = right.sample(now)
            draw_track(canvas, projector, left, now, LEFT)
            draw_track(canvas, projector, right, now, RIGHT)
            if left_sample is not None:
                draw_depth_to_reference_plane(
                    canvas, projector, left_sample.position, LEFT
                )
                draw_camera(canvas, projector, left_sample, LEFT, "LEFT")
            if right_sample is not None:
                draw_depth_to_reference_plane(
                    canvas, projector, right_sample.position, RIGHT
                )
                draw_camera(canvas, projector, right_sample, RIGHT, "RIGHT")
            if args.view_preset == "flu-front-above":
                view_note = "FIXED VIEW: IN FRONT OF + ABOVE BOTH GRIDS / DASH = X DEPTH"
            elif args.view_preset == "tag-map-front-above":
                view_note = "FIXED VIEW: TAG-WALL FRONT + PHYSICAL ABOVE / UP = -Y MAP"
            else:
                view_note = "SOLID = recent 2 s   DIM = history/interpolation"
            text(canvas, view_note, (1008, 630), 0.35, MUTED)
            draw_telemetry_card(
                canvas, (995, 655, 435, 330), "LEFT", LEFT,
                left, left_sample, now, coordinate_frame, camera_frame,
            )
            draw_telemetry_card(
                canvas, (1450, 655, 435, 330), "RIGHT", RIGHT,
                right, right_sample, now, coordinate_frame, camera_frame,
            )
            text(canvas, f"t = {now:6.3f} s", (1003, 1038), 0.62, WHITE, 2)
            text(canvas, "MEASURED = PnP observation", (1240, 1036), 0.38, MUTED)
            text(canvas, "UNTRUSTED = pose shown, not a trusted measurement", (1482, 1036),
                 0.36, WARNING)
            text(canvas, provenance_first, (22, 1015), 0.43, MUTED)
            text(canvas, provenance_second, (22, 1045), 0.40, WARNING)
            writer.write(canvas)
    finally:
        writer.release()
        for sampler in samplers.values():
            sampler.release()

    ffmpeg = (
        args.ffmpeg.resolve(strict=True)
        if args.ffmpeg is not None
        else project_ffmpeg_runtime().ffmpeg
    )
    subprocess.run([
        str(ffmpeg), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(intermediate), "-i", str(video_paths["LEFT BACK / H5 KB"]),
        "-map", "0:v:0", "-map", "1:a?", "-t", f"{duration:.6f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart", str(args.output),
    ], check=True)
    intermediate.unlink(missing_ok=True)
    cover = args.output.with_name(args.output.stem + "_cover.jpg")
    jump_check = args.output.with_name(args.output.stem + "_t7p1.jpg")
    _extract_frame(args.output, duration * 0.5, cover)
    _extract_frame(args.output, min(max(7.1 - start, 0.0), duration), jump_check)
    audit = {
        "schema_version": "joint-four-mp4-trajectory-comparison/2.2",
        "output": str(args.output.resolve()),
        "cover": str(cover.resolve()),
        "jump_check_frame": str(jump_check.resolve()),
        "sha256": _sha256(args.output),
        "width": 1920,
        "height": 1080,
        "fps": args.fps,
        "duration_s": duration,
        "frame_count": total,
        "ffmpeg_sha256": _sha256(ffmpeg),
        "map_id": world_map["map_id"],
        "tracking_status": report["status"],
        "joint_valid_ratio": report["trajectories"]["joint"]["joint_valid_ratio"],
        "joint_pose_ratio": report["trajectories"]["joint"]["joint_pose_ratio"],
        "untrusted_long_gap_frames": report["trajectories"]["joint"].get(
            "untrusted_long_gap_frames", 0
        ),
        "maximum_trusted_interpolation_gap_s": maximum_gap_s,
        "view_preset": args.view_preset,
        "coordinate_frame": coordinate_frame,
        "camera_frame": camera_frame,
        "view_eye_world": (
            None if projector.eye is None else projector.eye.tolist()
        ),
        "view_target_world": (
            None if projector.target is None else projector.target.tolist()
        ),
        "temporal_outlier_rejected_frames": {
            side: report["trajectories"][side].get(
                "temporal_outlier_rejected_frames", 0
            ) for side in ("left", "right")
        },
        "claims": report["claims"],
    }
    audit_path = args.output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
