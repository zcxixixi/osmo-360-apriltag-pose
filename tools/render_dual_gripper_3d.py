#!/usr/bin/env python3
"""Package timeline export, deterministic WebGL rendering and panorama PIP composition."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from osmo360.visualization.node_runtime import resolve_node_binary

from tools._root import ROOT
FFMPEG = ROOT / "work/tools/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("pair_dir", type=Path)
    parser.add_argument("--duration", type=float, default=3.0, help="seconds; 0 renders the full timeline")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--layout", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trim-start", type=float, default=0.0)
    parser.add_argument("--trim-end", type=float)
    parser.add_argument("--ffmpeg", type=Path, default=FFMPEG)
    parser.add_argument("--swap-sides", action="store_true",
                        help="display view2 as physical left and view1 as physical right")
    parser.add_argument(
        "--attitude-mode", choices=("visual", "imu-full", "imu-level"), default="imu-level",
        help="default keeps physically level grippers flat while using DJI IMU heading",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("none of the expected inputs exist: " + ", ".join(map(str, paths)))


def main() -> int:
    args = parse_args()
    pair = args.pair_dir.resolve()
    layout = (args.layout or pair / "manual-start-layout.json").resolve()
    timeline_path = pair / "dual_gripper_timeline.json"
    audit_path = pair / "dual_gripper_render_audit.json"
    full = args.duration <= 0
    output = (args.output or pair / ("dual_gripper_3d_final.mp4" if full else "dual_gripper_3d_preview.mp4")).resolve()
    base = output.with_name(output.stem + "_base.mp4")
    ffmpeg = args.ffmpeg.resolve()
    left_angles = first_existing(
        pair / "left-claw-angle/claw_angle_robust.csv",
        pair / "view1-claw-angle/claw_angle_robust.csv",
    )
    right_angles = first_existing(
        pair / "right-claw-angle-fitted/claw_angle_two_points.csv",
        pair / "view2-claw-angle/claw_angle_two_points.csv",
    )
    left_pose = first_existing(
        pair / "view1-60fps-full/pose.csv",
        pair / "left-trajectory-5fps/pose.csv", pair / "view1-5fps/pose.csv",
    )
    right_pose = first_existing(
        pair / "view2-60fps-full/pose.csv",
        pair / "right-trajectory-5fps/pose.csv", pair / "view2-5fps/pose.csv",
    )
    alignment_dir = first_existing(pair / "alignment-60fps", pair / "alignment-5fps")
    left_video = first_existing(
        pair / "left_360_3840x1920.mp4", pair / "view1_360_3840x1920.mp4",
    )
    right_video = first_existing(
        pair / "right_360_3840x1920.mp4", pair / "view2_360_3840x1920.mp4",
    )

    export_command = [
        sys.executable, "-m", "tools.export_dual_gripper_timeline",
        str(alignment_dir / "aligned_trajectories.csv"),
        str(alignment_dir / "alignment_report.json"), str(layout),
        str(pair / "claw-calibration/left_camera_to_gripper.json"),
        str(pair / "claw-calibration/right_camera_to_gripper.json"),
        str(left_angles), str(right_angles), str(left_pose), str(right_pose),
        str(timeline_path),
        "--fps", str(args.fps),
        "--view1-imu", str(pair / "view1-metadata/imu_perframe.csv"),
        "--view2-imu", str(pair / "view2-metadata/imu_perframe.csv"),
        "--attitude-mode", args.attitude_mode,
    ]
    if args.trim_start:
        export_command.extend(("--start-s", str(args.trim_start)))
    if args.trim_end is not None:
        export_command.extend(("--end-s", str(args.trim_end)))
    if args.swap_sides:
        export_command.append("--swap-sides")
    run(export_command)
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    duration = timeline["duration_s"] if full else min(args.duration, timeline["duration_s"])
    run([
        str(resolve_node_binary()), str(ROOT / "dual_gripper_3d/render_frames.mjs"),
        "--timeline", str(timeline_path),
        "--mesh-dir", str(ROOT / "assets/osmo_rig/osmo定位.SLDASM/meshes"),
        "--output", str(base), "--ffmpeg", str(ffmpeg),
        "--duration", str(duration), "--fps", str(args.fps),
    ])

    offset = float(timeline["sync"]["offset_s"])
    source_start = float(timeline["source_interval_s"]["start"])
    if args.swap_sides:
        display_left_video, display_right_video = right_video, left_video
        display_left_start, display_right_start = source_start + offset, source_start
    else:
        display_left_video, display_right_video = left_video, right_video
        display_left_start, display_right_start = source_start, source_start + offset
    run([
        str(ffmpeg), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(base),
        "-ss", f"{display_left_start:.6f}", "-i", str(display_left_video),
        "-ss", f"{display_right_start:.6f}", "-i", str(display_right_video),
        "-filter_complex",
        "[1:v]setpts=PTS-STARTPTS,scale=320:160:force_original_aspect_ratio=decrease,pad=320:160:(ow-iw)/2:(oh-ih)/2:black[lv];"
        "[2:v]setpts=PTS-STARTPTS,scale=320:160:force_original_aspect_ratio=decrease,pad=320:160:(ow-iw)/2:(oh-ih)/2:black[rv];"
        "[0:v][lv]overlay=1210:56:shortest=1[tmp];[tmp][rv]overlay=1554:56:shortest=1[v]",
        "-map", "[v]", "-map", "1:a?", "-t", f"{duration:.6f}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output),
    ])
    base.unlink(missing_ok=True)

    visible = [frame for frame in timeline["frames"] if frame["t"] <= duration + 1e-6]
    max_opening_frame = max(range(len(visible)), key=lambda i: max(
        visible[i]["left"]["opening"], visible[i]["right"]["opening"],
    ))
    screenshot_times = {
        "start": 0.0,
        "middle": duration / 2.0,
        "max_opening": visible[max_opening_frame]["t"],
    }
    screenshots = {}
    for label, timestamp in screenshot_times.items():
        path = pair / f"dual_gripper_3d_{label}.jpg"
        run([
            str(ffmpeg), "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{timestamp:.6f}",
            "-i", str(output), "-frames:v", "1", "-q:v", "2", str(path),
        ])
        screenshots[label] = str(path.resolve())

    first = visible[0]
    layout_data = json.loads(layout.read_text(encoding="utf-8"))
    initial_errors = {}
    for side in ("left", "right"):
        target = layout_data["grippers_in_center_frame"][side]["translation_m"]
        error = sum((first[side]["p"][axis] - target[axis]) ** 2 for axis in range(3)) ** 0.5
        initial_errors[side] = error * 1000
    status_counts = {
        side: dict(Counter(frame[side]["pose_state"] for frame in visible)) for side in ("left", "right")
    }
    audit = {
        "schema_version": "dual-gripper-render-audit/v1",
        "capture_pair_id": timeline["capture_pair_id"],
        "layout_calibration_id": timeline["layout_calibration_id"],
        "output": str(output), "timeline": str(timeline_path),
        "width": 1920, "height": 1080, "fps": args.fps, "duration_s": duration,
        "frame_count": len(visible), "audio_offset_s": offset,
        "source_interval_s": timeline["source_interval_s"],
        "side_mapping": timeline["side_mapping"],
        "attitude": timeline["attitude"],
        "initial_position_error_mm": initial_errors,
        "pose_state_frames": status_counts,
        "opening_range_deg": {
            side: [min(frame[side]["opening"] for frame in visible),
                   max(frame[side]["opening"] for frame in visible)] for side in ("left", "right")
        },
        "screenshots": screenshots,
        "claims": {"manual_start_layout_is_prior": True, "metrology_ground_truth": False},
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
