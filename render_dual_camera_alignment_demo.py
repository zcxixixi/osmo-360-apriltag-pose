#!/usr/bin/env python3
"""Render two synchronized camera trajectories in the left-camera frame."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from render_mocap_comparison import (
    AMBER, BG, CYAN, GREEN, MUTED, RED, WHITE,
    draw_gripper, load_gripper_edges, project, text,
)
from render_trajectory_overlay_video import kalman_rts_filter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("left_video", type=Path)
    parser.add_argument("right_video", type=Path)
    parser.add_argument("aligned_csv", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gripper-mesh-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-duration", type=float, default=8.0)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    return parser.parse_args()


def load_track(path: Path, prefix: str) -> tuple[np.ndarray, np.ndarray, Rotation]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    times = np.asarray([float(row["left_timestamp_s"]) for row in rows])
    positions = np.asarray([
        [float(row[f"{prefix}_{axis}_m"]) for axis in "xyz"] for row in rows
    ])
    rotations = Rotation.from_quat(np.asarray([
        [float(row[f"{prefix}_q{axis}"]) for axis in "xyzw"] for row in rows
    ]))
    euler = np.unwrap(rotations.as_euler("xyz"), axis=0)
    smooth = kalman_rts_filter(
        np.column_stack((times, positions, np.degrees(euler))),
        measurement_noise=0.025, accel_noise=0.8,
        angle_noise=2.0, angular_accel_noise=35.0,
    )
    return times, smooth[:, 1:4], Rotation.from_euler("xyz", smooth[:, 4:7], degrees=True)


def sample(times: np.ndarray, positions: np.ndarray, rotations: Rotation, now: float):
    position = np.asarray([np.interp(now, times, positions[:, axis]) for axis in range(3)])
    quaternion = Slerp(times, rotations)([np.clip(now, times[0], times[-1])]).as_quat()[0]
    nearest = float(np.min(np.abs(times - now)))
    return position, quaternion, nearest


class VideoSampler:
    """Sequential video sampler; avoids an expensive decoder seek per frame."""

    def __init__(self, path: Path, start_s: float):
        self.capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
        if not self.capture.isOpened():
            raise SystemExit(f"cannot open panorama video: {path}")
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.current = max(0, int(round(start_s * self.fps)))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.current)
        self.last: np.ndarray | None = None

    def frame_at(self, timestamp: float, size: tuple[int, int]) -> np.ndarray:
        target = max(0, int(round(timestamp * self.fps)))
        if target < self.current:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, target)
            self.current = target
        while self.current <= target:
            ok, frame = self.capture.read()
            if not ok:
                break
            self.last = frame
            self.current += 1
        if self.last is None:
            return np.full((size[1], size[0], 3), (30, 32, 38), np.uint8)
        return cv2.resize(self.last, size, interpolation=cv2.INTER_AREA)

    def release(self) -> None:
        self.capture.release()


def main() -> int:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    left_t, left_p, left_r = load_track(args.aligned_csv, "left")
    right_t, right_p, right_r = load_track(args.aligned_csv, "right_aligned")
    start = float(max(left_t[0], right_t[0]))
    end = float(min(left_t[-1], right_t[-1], start + args.max_duration))
    if end <= start:
        raise SystemExit("no shared render interval")
    all_points = np.vstack((left_p, right_p))
    low, high = np.percentile(all_points, [1, 99], axis=0)
    center = (low + high) / 2.0
    radius = max(float(np.max(high - low)) / 2.0 + 0.12, 0.30)
    low, high = center - radius, center + radius
    # The CAD is an orientation glyph here, not a metrically scaled obstacle.
    # Keep it compact enough that both poses remain readable in one plot.
    edges = load_gripper_edges(args.gripper_mesh_dir) * 0.45
    offset = float(report["time_alignment"]["offset_s"])

    left_video = VideoSampler(args.left_video, start)
    right_video = VideoSampler(args.right_video, start + offset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    intermediate = args.output.with_suffix(".mp4v.mp4")
    writer = cv2.VideoWriter(
        str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (1920, 1080)
    )
    if not writer.isOpened():
        raise SystemExit("cannot create demo video")

    total = int(round((end - start) * args.fps))
    plot_origin, plot_size = (40, 555), (1240, 485)
    for frame_index in range(total):
        now = start + frame_index / args.fps
        left_pos, left_quat, left_age = sample(left_t, left_p, left_r, now)
        right_pos, right_quat, right_age = sample(right_t, right_p, right_r, now)
        canvas = np.full((1080, 1920, 3), BG, dtype=np.uint8)
        canvas[55:505, 20:950] = left_video.frame_at(now, (930, 450))
        canvas[55:505, 970:1900] = right_video.frame_at(now + offset, (930, 450))
        cv2.rectangle(canvas, (20, 55), (950, 505), CYAN, 2)
        cv2.rectangle(canvas, (970, 55), (1900, 505), GREEN, 2)
        text(canvas, "LEFT REFERENCE", (30, 42), 0.72, CYAN, 2)
        text(canvas, "RIGHT  (audio synchronized)", (980, 42), 0.72, GREEN, 2)

        cv2.rectangle(canvas, plot_origin,
                      (plot_origin[0] + plot_size[0], plot_origin[1] + plot_size[1]),
                      (60, 66, 78), 1)
        upto = int(np.searchsorted(left_t, now, side="right"))
        for points, color in ((left_p[:upto], CYAN), (right_p[:upto], GREEN)):
            if len(points) > 1:
                pixels = np.asarray([project(p, low, high, plot_origin, plot_size) for p in points], np.int32)
                cv2.polylines(canvas, [pixels], False, color, 2, cv2.LINE_AA)
        draw_gripper(canvas, left_pos, left_quat, edges, low, high,
                     plot_origin, plot_size, CYAN, "LEFT gripper", -14)
        draw_gripper(canvas, right_pos, right_quat, edges, low, high,
                     plot_origin, plot_size, GREEN, "RIGHT aligned gripper", 22)
        origin_px = project(np.zeros(3), low, high, plot_origin, plot_size)
        cv2.circle(canvas, origin_px, 5, WHITE, -1, cv2.LINE_AA)
        text(canvas, "LEFT-FRAME 6DoF + CAD GRIPPER", (55, 585), 0.64, WHITE, 2)
        text(canvas, "gripper glyph scale 45% (orientation display)", (55, 1018), 0.42, MUTED)

        panel_x = 1320
        text(canvas, "DUAL CAMERA ALIGNMENT", (panel_x, 585), 0.70, WHITE, 2)
        text(canvas, "DEMO / REVIEW", (panel_x, 622), 0.72, AMBER, 2)
        text(canvas, f"pair  {report['capture_pair_id']}", (panel_x, 658), 0.43, MUTED)
        text(canvas, f"audio offset  {offset:+.4f} s", (panel_x, 700), 0.56, WHITE)
        text(canvas, f"audio corr    {report['time_alignment']['correlation']:.3f}",
             (panel_x, 732), 0.56, WHITE)
        pos = report["position_residual_m"]
        ori = report["orientation_residual_deg"]
        text(canvas, f"position P95  {1000.0 * pos['p95']:.1f} mm", (panel_x, 780), 0.56, AMBER)
        text(canvas, f"attitude P95  {ori['p95']:.1f} deg", (panel_x, 812), 0.56, AMBER)
        state = "MEASURED" if max(left_age, right_age) <= 0.25 else "PREDICTED / SPARSE"
        state_color = GREEN if state == "MEASURED" else AMBER
        text(canvas, state, (panel_x, 860), 0.62, state_color, 2)
        text(canvas, "Kalman + RTS visualization", (panel_x, 900), 0.52, MUTED)
        text(canvas, "Formal calibration requires denser right poses", (panel_x, 935), 0.48, MUTED)
        text(canvas, f"t={now-start:05.2f}s / {end-start:05.2f}s", (panel_x, 990), 0.56, WHITE)
        writer.write(canvas)

    writer.release()
    left_video.release(); right_video.release()
    command = [
        str(args.ffmpeg), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(intermediate), "-c:v", "libx264", "-crf", "18",
        "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(args.output),
    ]
    import subprocess
    subprocess.run(command, check=True)
    intermediate.unlink(missing_ok=True)
    print(json.dumps({"output": str(args.output.resolve()), "frames": total,
                      "fps": args.fps, "duration_s": end - start}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
