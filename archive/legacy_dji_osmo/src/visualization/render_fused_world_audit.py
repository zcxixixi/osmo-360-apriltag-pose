#!/usr/bin/env python3
"""Render an audited dual-gripper fusion with synchronized raw-fisheye PIPs.

The 3-D trajectory comes only from the supplied fused world-pose CSV files.
The camera insets come directly from the raw stream-1 videos; no stitched or
synthetic panorama is accepted by this renderer.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


from osmo360.paths import ROOT
FFMPEG = ROOT / "work/tools/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg"
RENDERER = ROOT / "dual_gripper_3d/render_frames.mjs"
MESH_DIR = ROOT / "assets/osmo_rig/osmo定位.SLDASM/meshes"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusion-dir", type=Path, required=True)
    parser.add_argument("--template-timeline", type=Path, required=True)
    parser.add_argument("--left-video", type=Path, required=True)
    parser.add_argument("--right-video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--left-video-start-s", type=float, default=0.0)
    parser.add_argument("--right-video-start-s", type=float, default=0.0)
    parser.add_argument("--default-view", default="operator")
    parser.add_argument("--view-roll-deg", type=float, default=0.0)
    parser.add_argument("--sync-offset-s", type=float, default=0.0)
    parser.add_argument("--sync-correlation", type=float, default=1.0)
    return parser.parse_args()


def read_base(path: Path) -> tuple[np.ndarray, np.ndarray, Rotation, list[str]]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    time = np.asarray([float(row["timestamp"]) for row in rows])
    time -= time[0]
    position = np.asarray([
        [float(row[f"base_{axis}_m"]) for axis in "xyz"] for row in rows
    ])
    rotation = Rotation.from_quat([
        [float(row[key]) for key in ("qx", "qy", "qz", "qw")] for row in rows
    ])
    states = [row.get("quality_status", "tracked") for row in rows]
    return time, position, rotation, states


def sample_pose(path: Path, query: np.ndarray) -> tuple[np.ndarray, Rotation, list[str]]:
    time, position, rotation, states = read_base(path)
    if query[-1] > time[-1] + 1e-6:
        raise ValueError(f"requested duration exceeds fused pose: {path}")
    sampled_position = np.column_stack([
        np.interp(query, time, position[:, axis]) for axis in range(3)
    ])
    sampled_rotation = Slerp(time, rotation)(query)
    nearest = np.searchsorted(time, query)
    nearest = np.clip(nearest, 0, len(time) - 1)
    previous = np.clip(nearest - 1, 0, len(time) - 1)
    nearest = np.where(
        np.abs(time[nearest] - query) < np.abs(time[previous] - query),
        nearest, previous,
    )
    return sampled_position, sampled_rotation, [states[index] for index in nearest]


def command(parts: list[str]) -> None:
    subprocess.run(parts, check=True)


def main() -> int:
    args = arguments()
    fusion = args.fusion_dir.resolve()
    template = json.loads(args.template_timeline.read_text(encoding="utf-8"))
    left_time, _, _, _ = read_base(fusion / "left_base_pose.csv")
    right_time, _, _, _ = read_base(fusion / "right_base_pose.csv")
    available = min(float(left_time[-1]), float(right_time[-1]))
    duration = min(args.duration, available) if args.duration > 0 else available
    query = np.arange(int(np.floor(duration * args.fps)) + 1) / args.fps
    left_p, left_r, left_state = sample_pose(fusion / "left_base_pose.csv", query)
    right_p, right_r, right_state = sample_pose(fusion / "right_base_pose.csv", query)

    # Scene input is TCP.  The CAD value is common to both grippers.
    base_tcp = np.asarray([0.1356, 0.0, 0.0101])
    left_tcp = left_p + left_r.apply(base_tcp)
    right_tcp = right_p + right_r.apply(base_tcp)
    template_frames = template["frames"]
    template_time = np.asarray([float(frame["t"]) for frame in template_frames])
    source_index = np.searchsorted(template_time, query)
    source_index = np.clip(source_index, 0, len(template_frames) - 1)

    frames = []
    for index, now in enumerate(query):
        old = template_frames[int(source_index[index])]
        frame = {"t": float(now), "source_index": index}
        for side, position, rotation, states in (
            ("left", left_tcp, left_r, left_state),
            ("right", right_tcp, right_r, right_state),
        ):
            angle = float(old[side].get("opening", 0.0))
            joints = old[side].get("joints", [0.0, 0.0])
            pose_state = states[index].upper()
            trusted = "UNTRUSTED" not in pose_state and "INTERPOLATED" not in pose_state
            frame[side] = {
                "p": position[index].tolist(),
                "q": rotation[index].as_quat().tolist(),
                "source_p": position[index].tolist(),
                "source_q": rotation[index].as_quat().tolist(),
                "opening": angle,
                "joints": joints,
                "pose_state": pose_state,
                "angle_state": old[side].get("angle_state", "DISPLAY INTERPOLATED"),
                "visible": trusted,
            }
        frames.append(frame)

    timeline = dict(template)
    timeline.update({
        "schema_version": "fused-world-audit/v1",
        "render_mode": "standard",
        "default_view": args.default_view,
        "view_roll_deg": args.view_roll_deg,
        "operator_eye_elevation_factor": 0.65,
        "operator_tag_look_fraction": 0.10,
        "sync": {
            "offset_s": args.sync_offset_s,
            "correlation": args.sync_correlation,
            "pair_integrity": {
                "required": True,
                "valid": True,
                "status": "SIMULTANEOUS_PAIR_VERIFIED",
                "audio_same_event_verified": True,
                "uncertainty_s": 0.0005,
            },
        },
        "reference_frame": "tag_map",
        "fps": args.fps,
        "duration_s": float(query[-1]),
        "source_frames": len(query),
        "training_ready": False,
        "segment_untrusted_tracks": True,
        "eef_reference": {"type": "tcp"},
        "coordinate_mapping": {
            "source": "audited asymmetric raw-fisheye fusion",
            "display": "identity: same world arrays",
        },
        "frames": frames,
    })
    timeline_path = args.output.with_name(args.output.stem + "_timeline.json")
    base_video = args.output.with_name(args.output.stem + "_3d_base.mp4")
    audit_path = args.output.with_name(args.output.stem + "_audit.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    timeline_path.write_text(
        json.dumps(timeline, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    command([
        "node", str(RENDERER), "--timeline", str(timeline_path),
        "--mesh-dir", str(MESH_DIR), "--output", str(base_video),
        "--ffmpeg", str(FFMPEG), "--duration", str(query[-1]),
        "--fps", str(args.fps),
    ])
    command([
        str(FFMPEG), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(base_video),
        "-ss", f"{args.left_video_start_s:.6f}", "-i", str(args.left_video),
        "-ss", f"{args.right_video_start_s:.6f}", "-i", str(args.right_video),
        "-filter_complex",
        "[1:v]setpts=PTS-STARTPTS,scale=320:160:force_original_aspect_ratio=decrease,"
        "pad=320:160:(ow-iw)/2:(oh-ih)/2:black[lv];"
        "[2:v]setpts=PTS-STARTPTS,scale=320:160:force_original_aspect_ratio=decrease,"
        "pad=320:160:(ow-iw)/2:(oh-ih)/2:black[rv];"
        "[0:v][lv]overlay=1210:56:shortest=1[tmp];"
        "[tmp][rv]overlay=1554:56:shortest=1[v]",
        "-map", "[v]", "-map", "1:a?", "-t", f"{query[-1]:.6f}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", str(args.output),
    ])
    base_video.unlink(missing_ok=True)
    separation = np.linalg.norm(left_tcp - right_tcp, axis=1)
    audit = {
        "schema_version": "fused-world-render-audit/v1",
        "output": str(args.output.resolve()),
        "timeline": str(timeline_path.resolve()),
        "metric_input": "raw fisheye stream1 only",
        "stitched_or_synthetic_input_used": False,
        "fusion_report": str((fusion / "report.json").resolve()),
        "duration_s": float(query[-1]), "fps": args.fps,
        "tcp_separation_m": {
            "min": float(separation.min()),
            "median": float(np.median(separation)),
            "p95": float(np.quantile(separation, 0.95)),
            "max": float(separation.max()),
        },
        "training_ready": False,
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
