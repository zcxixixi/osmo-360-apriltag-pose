#!/usr/bin/env python3
"""Render four synchronized fisheye videos beside one shared-map 3D trajectory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from tools._root import ROOT


FFMPEG = ROOT / "work/tools/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg"
if not FFMPEG.is_file():
    FFMPEG = Path("/usr/bin/ffmpeg")

BG = (20, 23, 29)
PANEL = (31, 36, 44)
MUTED = (120, 132, 148)
WHITE = (235, 239, 244)
CYAN = (238, 196, 72)
GREEN = (108, 220, 126)
RED = (85, 90, 245)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("tracking_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=0.0,
                        help="seconds; zero uses the complete common timeline")
    parser.add_argument("--ffmpeg", type=Path, default=FFMPEG)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Track:
    def __init__(self, rows: list[dict[str, str]], prefix: str):
        valid = [
            row for row in rows
            if row.get(f"{prefix}_quality_status") in {"valid", "interpolated"}
            and row.get(f"{prefix}_camera_x_m")
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
        if len(self.times) < 2 or np.any(np.diff(self.times) <= 0):
            raise ValueError(f"{prefix} joint trajectory has no increasing valid track")
        self.slerp = Slerp(self.times, self.rotations)

    def sample(self, now: float) -> tuple[np.ndarray, Rotation, float]:
        clipped = float(np.clip(now, self.times[0], self.times[-1]))
        position = np.asarray([
            np.interp(clipped, self.times, self.positions[:, axis])
            for axis in range(3)
        ])
        rotation = self.slerp([clipped])[0]
        age = float(np.min(np.abs(self.times - now)))
        return position, rotation, age


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


def load_map(path: Path) -> tuple[dict, list[np.ndarray]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tags = [np.asarray(tag["corners_m"], dtype=float) for tag in payload["tags"]]
    return payload, tags


class Projector:
    def __init__(self, points: np.ndarray, origin: tuple[int, int], size: tuple[int, int]):
        # Fixed oblique world view; bounds stay fixed throughout the video.
        self.view = Rotation.from_euler("xz", [58.0, -32.0], degrees=True)
        transformed = self.view.apply(points)
        low, high = np.percentile(transformed[:, :2], [1, 99], axis=0)
        padding = np.maximum((high - low) * 0.10, 0.05)
        self.low = low - padding
        self.high = high + padding
        self.origin = np.asarray(origin, dtype=float)
        self.size = np.asarray(size, dtype=float)

    def __call__(self, point: np.ndarray) -> tuple[int, int]:
        value = self.view.apply(np.asarray(point, dtype=float))[:2]
        normalized = (value - self.low) / np.maximum(self.high - self.low, 1e-9)
        pixel = self.origin + normalized * self.size
        return int(round(pixel[0])), int(round(self.origin[1] + self.size[1] - normalized[1] * self.size[1]))


def draw_polyline(
    canvas: np.ndarray, projector: Projector, points: np.ndarray,
    color: tuple[int, int, int], thickness: int = 2,
) -> None:
    if len(points) < 2:
        return
    pixels = np.asarray([projector(point) for point in points], dtype=np.int32)
    cv2.polylines(canvas, [pixels], False, color, thickness, cv2.LINE_AA)


def draw_camera(
    canvas: np.ndarray,
    projector: Projector,
    position: np.ndarray,
    rotation: Rotation,
    color: tuple[int, int, int],
    label: str,
) -> None:
    center = projector(position)
    cv2.circle(canvas, center, 8, color, -1, cv2.LINE_AA)
    length = 0.08
    for axis, axis_color in zip(np.eye(3), (RED, GREEN, CYAN)):
        endpoint = projector(position + rotation.apply(axis * length))
        cv2.line(canvas, center, endpoint, axis_color, 3, cv2.LINE_AA)
    text(canvas, label, (center[0] + 12, center[1] - 8), 0.55, color, 2)


def main() -> int:
    args = parse_args()
    dataset = args.dataset_root.resolve(strict=True)
    tracking = args.tracking_dir.resolve(strict=True)
    joint_path = tracking / "joint_trajectory.csv"
    world_map_path = tracking / "session_world_map.json"
    report_path = tracking / "report.json"
    with joint_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    left = Track(rows, "left")
    right = Track(rows, "right")
    world_map, tags = load_map(world_map_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
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
    all_points = np.vstack((left.positions, right.positions, *tags))
    plot_origin, plot_size = (1010, 100), (860, 850)
    projector = Projector(all_points, plot_origin, plot_size)
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
            text(canvas, "SYNCHRONIZED JOINT 6DoF / ONE SHARED APRILGRID MAP", (1005, 37), 0.72, WHITE, 2)
            for tile, name in enumerate(names):
                x = 20 + (tile % 2) * 475
                y = 55 + (tile // 2) * 475
                frame = samplers[name].sample(now, (455, 420))
                canvas[y:y + 420, x:x + 455] = frame
                color = CYAN if name.startswith("LEFT") else GREEN
                cv2.rectangle(canvas, (x, y), (x + 455, y + 420), color, 2)
                text(canvas, name, (x + 10, y + 27), 0.48, color, 2)

            cv2.rectangle(canvas, (990, 55), (1900, 1000), PANEL, -1)
            cv2.rectangle(canvas, (990, 55), (1900, 1000), (70, 78, 90), 1)
            for tag in tags:
                closed = np.vstack((tag, tag[0]))
                draw_polyline(canvas, projector, closed, (70, 82, 96), 1)
            left_now, left_rotation, left_age = left.sample(now)
            right_now, right_rotation, right_age = right.sample(now)
            left_upto = int(np.searchsorted(left.times, now, side="right"))
            right_upto = int(np.searchsorted(right.times, now, side="right"))
            draw_polyline(canvas, projector, left.positions[:left_upto], CYAN, 3)
            draw_polyline(canvas, projector, right.positions[:right_upto], GREEN, 3)
            draw_camera(canvas, projector, left_now, left_rotation, CYAN, "LEFT")
            draw_camera(canvas, projector, right_now, right_rotation, GREEN, "RIGHT")
            origin = projector(np.zeros(3))
            cv2.drawMarker(canvas, origin, WHITE, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
            text(canvas, "GRID A ORIGIN", (origin[0] + 10, origin[1] + 22), 0.42, MUTED)
            text(canvas, f"t = {now:6.3f} s", (1010, 1030), 0.62, WHITE, 2)
            text(canvas, f"LEFT sample age {left_age * 1000:4.0f} ms", (1250, 1030), 0.48, CYAN)
            text(canvas, f"RIGHT sample age {right_age * 1000:4.0f} ms", (1540, 1030), 0.48, GREEN)
            text(canvas, "Capture-local self-calibration; not external ground truth", (22, 1015), 0.50, MUTED)
            text(canvas, f"Map: {world_map['map_id']}", (22, 1045), 0.42, MUTED)
            writer.write(canvas)
    finally:
        writer.release()
        for sampler in samplers.values():
            sampler.release()

    ffmpeg = args.ffmpeg.resolve(strict=True)
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
    capture = cv2.VideoCapture(str(args.output))
    capture.set(cv2.CAP_PROP_POS_MSEC, duration * 500.0)
    ok, image = capture.read()
    capture.release()
    if not ok or not cv2.imwrite(str(cover), image):
        raise ValueError("cannot extract Feishu video cover")
    audit = {
        "schema_version": "joint-four-mp4-trajectory-comparison/1.0",
        "output": str(args.output.resolve()),
        "cover": str(cover.resolve()),
        "sha256": _sha256(args.output),
        "width": 1920,
        "height": 1080,
        "fps": args.fps,
        "duration_s": duration,
        "frame_count": total,
        "map_id": world_map["map_id"],
        "tracking_status": report["status"],
        "joint_valid_ratio": report["trajectories"]["joint"]["joint_valid_ratio"],
        "claims": report["claims"],
    }
    audit_path = args.output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
