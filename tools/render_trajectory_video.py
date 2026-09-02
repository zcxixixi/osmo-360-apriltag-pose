#!/usr/bin/env python3
"""Render an animated 3D trajectory from an offline pose CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("pose_csv", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--smooth", type=float, default=0.55)
    parser.add_argument("--gap", type=float, default=0.45)
    return parser.parse_args()


def load_valid(path: Path) -> np.ndarray:
    values = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["quality_status"] != "valid" or not row["camera_x_m"]:
                continue
            values.append(
                [
                    float(row["timestamp"]),
                    float(row["camera_x_m"]),
                    float(row["camera_y_m"]),
                    float(row["camera_z_m"]),
                ]
            )
    return np.asarray(values, dtype=float)


def filter_segments(data: np.ndarray, strength: float, gap: float) -> tuple[np.ndarray, list[slice]]:
    alpha = 1.0 - strength
    filtered = data.copy()
    starts = [0]
    for index in range(1, len(data)):
        if data[index, 0] - data[index - 1, 0] > gap:
            starts.append(index)
        else:
            filtered[index, 1:] = (
                filtered[index - 1, 1:] + alpha * (data[index, 1:] - filtered[index - 1, 1:])
            )
    ends = starts[1:] + [len(data)]
    return filtered, [slice(start, end) for start, end in zip(starts, ends)]


def main() -> int:
    args = arguments()
    data = load_valid(args.pose_csv)
    if not len(data):
        raise SystemExit("no valid poses")
    filtered, segments = filter_segments(data, args.smooth, args.gap)
    duration = float(np.ceil(data[-1, 0] + 1.0))
    frames = int(duration * args.fps)
    xyz = data[:, 1:]
    padding = np.maximum(np.ptp(xyz, axis=0) * 0.12, 0.06)
    low, high = xyz.min(axis=0) - padding, xyz.max(axis=0) + padding

    plt.style.use("dark_background")
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="#090c10")
    ax = fig.add_subplot(111, projection="3d", facecolor="#090c10")
    fig.subplots_adjust(left=0.02, right=0.88, bottom=0.08, top=0.90)
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]))
    ax.set_xlabel("X / m", labelpad=10)
    ax.set_ylabel("Y / m", labelpad=10)
    ax.set_zlabel("Z / m", labelpad=8)
    ax.view_init(elev=25, azim=-58)
    ax.grid(alpha=0.20)
    ax.set_box_aspect(np.maximum(high - low, 0.1))
    fig.suptitle("Insta360 X5 · AprilGrid 三维动作轨迹", fontsize=20, color="#f5f7fa")

    raw_lines = []
    smooth_lines = []
    for _segment in segments:
        raw_line, = ax.plot([], [], [], color="#f59e0b", lw=1.2, ls=(0, (2, 4)), alpha=0.42)
        smooth_line, = ax.plot([], [], [], color="#38bdf8", lw=3.2, solid_capstyle="round")
        raw_lines.append(raw_line)
        smooth_lines.append(smooth_line)
    raw_point, = ax.plot([], [], [], marker="o", ms=5, color="#f59e0b", linestyle="none", alpha=0.78)
    smooth_point, = ax.plot([], [], [], marker="o", ms=11, color="#38bdf8", markeredgecolor="#e0f2fe", markeredgewidth=1.5, linestyle="none")
    ax.scatter([0], [0], [0], marker="s", s=45, color="#f8fafc", alpha=0.85)
    time_text = fig.text(0.89, 0.83, "", fontsize=18, color="#f5f7fa", ha="left")
    state_text = fig.text(0.89, 0.77, "", fontsize=12, color="#94a3b8", ha="left")
    fig.text(0.89, 0.17, "橙色  原始测量\n蓝色  55% 低通滤波\n断点  不跨段插值", fontsize=11, color="#cbd5e1", ha="left", linespacing=1.7)

    def update(frame: int):
        now = frame / args.fps
        active = None
        for line_index, segment in enumerate(segments):
            segment_data = data[segment]
            segment_filtered = filtered[segment]
            visible = segment_data[:, 0] <= now
            raw = segment_data[visible, 1:]
            smooth = segment_filtered[visible, 1:]
            if len(raw):
                raw_lines[line_index].set_data_3d(raw[:, 0], raw[:, 1], raw[:, 2])
                smooth_lines[line_index].set_data_3d(smooth[:, 0], smooth[:, 1], smooth[:, 2])
            else:
                raw_lines[line_index].set_data_3d([], [], [])
                smooth_lines[line_index].set_data_3d([], [], [])
            if segment_data[0, 0] <= now <= segment_data[-1, 0] + 0.22:
                active = (segment_data, segment_filtered)
        if active is None:
            raw_point.set_data_3d([], [], [])
            smooth_point.set_data_3d([], [], [])
            state = "等待有效姿态"
        else:
            segment_data, segment_filtered = active
            index = min(np.searchsorted(segment_data[:, 0], now, side="right") - 1, len(segment_data) - 1)
            index = max(index, 0)
            raw = segment_data[index, 1:]
            smooth = segment_filtered[index, 1:]
            raw_point.set_data_3d([raw[0]], [raw[1]], [raw[2]])
            smooth_point.set_data_3d([smooth[0]], [smooth[1]], [smooth[2]])
            state = "有效姿态进入滤波器"
        time_text.set_text(f"{now:05.2f} s")
        state_text.set_text(state)
        return [*raw_lines, *smooth_lines, raw_point, smooth_point, time_text, state_text]

    animation = FuncAnimation(fig, update, frames=frames, interval=1000 / args.fps, blit=False)
    matplotlib.rcParams["animation.ffmpeg_path"] = str(args.ffmpeg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(
        fps=args.fps,
        codec="libx264",
        bitrate=6000,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    animation.save(args.output, writer=writer, dpi=100)
    plt.close(fig)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
