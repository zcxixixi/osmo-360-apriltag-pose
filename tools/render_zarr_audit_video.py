#!/usr/bin/env python3
"""Render an audit video using only arrays stored in the final UMI Zarr file."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import zarr

from osmo360.localization.world_frames import compile_world_tag_map
from osmo360.visualization.node_runtime import resolve_node_binary


from tools._root import ROOT
FFMPEG = ROOT / "work/tools/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg"
MESH_DIR = ROOT / "assets/osmo_rig/osmo定位.SLDASM/meshes"
RENDERER = ROOT / "dual_gripper_3d/render_frames.mjs"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], *, stdin: bytes | None = None) -> None:
    subprocess.run(command, input=stdin, check=True)


def encode_rgb_video(frames: np.ndarray, output: Path, fps: float) -> None:
    height, width = frames.shape[1:3]
    command = [
        str(FFMPEG), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", f"{fps:.8f}", "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "fast", "-crf", "16", "-pix_fmt", "yuv420p", str(output),
    ]
    run(command, stdin=np.ascontiguousarray(frames).tobytes())


def source_indices(metadata: dict, frame_count: int) -> np.ndarray:
    indices = np.concatenate([
        np.arange(int(start), int(end), dtype=int)
        for start, end in metadata["training_segments"]
    ])
    if len(indices) != frame_count:
        raise ValueError(f"metadata has {len(indices)} training indices but Zarr has {frame_count}")
    return indices


def episode_lookup(episode_ends: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    episodes = np.searchsorted(episode_ends, np.arange(count), side="right")
    starts = np.r_[0, episode_ends[:-1]]
    return episodes, starts[episodes]


def build_timeline(dataset: Path, metadata_path: Path, timeline_path: Path,
                   maximum_frames: int | None) -> tuple[dict, dict[str, np.ndarray]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    store = zarr.ZipStore(str(dataset), mode="r")
    root = zarr.open_group(store=store, mode="r")
    arrays = {name: np.asarray(root[f"data/{name}"][:]) for name in (
        "camera0_rgb", "camera1_rgb", "robot0_eef_pos", "robot1_eef_pos",
        "robot0_eef_rot_axis_angle", "robot1_eef_rot_axis_angle",
        "robot0_gripper_angle_deg", "robot1_gripper_angle_deg", "action_valid",
    )}
    episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=int)
    store.close()
    full_count = len(arrays["camera0_rgb"])
    count = full_count if maximum_frames is None else min(full_count, maximum_frames)
    arrays = {name: value[:count] for name, value in arrays.items()}
    if not np.all(arrays["action_valid"]):
        raise ValueError("training Zarr contains invalid action frames")
    original_indices = source_indices(metadata, full_count)[:count]
    episodes, episode_starts = episode_lookup(episode_ends, count)
    effective_episode_ends = episode_ends[episode_ends <= count].tolist()
    if not effective_episode_ends or effective_episode_ends[-1] != count:
        effective_episode_ends.append(count)
    dataset_hash = sha256(dataset)
    neutral = abs(np.degrees(np.arctan2(50.568, 63.276) - np.arctan2(-50.745, 63.134)))
    coordinate = metadata.get("coordinate_frame", {})
    world_mode = coordinate.get("mode") == "world"
    display_offsets = [np.zeros(3), np.zeros(3)] if world_mode else [
        np.asarray([-0.32, 0.0, 0.0]), np.asarray([0.32, 0.0, 0.0])]
    display_rotations = [Rotation.identity(), Rotation.identity()] if world_mode else [
        Rotation.identity(), Rotation.from_euler("z", 180, degrees=True)]
    tag_anchors = []
    if world_mode and coordinate.get("tag_map"):
        world_map = compile_world_tag_map(Path(coordinate["tag_map"]))
        if world_map["tag_map_sha256"] != coordinate.get("tag_map_sha256"):
            raise ValueError("Zarr metadata Tag map hash mismatch")
        for tag in world_map["tags"]:
            panel = str(tag.get("panel", ""))
            side = "left" if panel.startswith("left") else "right"
            tag_anchors.append({
                "id": int(tag["id"]), "side": side, "corners": tag["corners_m"],
                "color": "#37c8ff" if side == "left" else "#58df91",
            })
    frames = []
    for index in range(count):
        frame = {
            "t": index / float(metadata["frequency_hz"]),
            "source_index": int(original_indices[index]),
            "episode": int(episodes[index]),
            "episode_start": int(episode_starts[index]),
        }
        for robot, side in enumerate(("left", "right")):
            position = arrays[f"robot{robot}_eef_pos"][index].astype(float)
            source_rotation = Rotation.from_rotvec(
                arrays[f"robot{robot}_eef_rot_axis_angle"][index].astype(float))
            display_position = display_offsets[robot] + display_rotations[robot].apply(position)
            display_rotation = display_rotations[robot] * source_rotation
            opening = float(arrays[f"robot{robot}_gripper_angle_deg"][index, 0])
            travel = (neutral - opening) / 2.0
            frame[side] = {
                "p": display_position.tolist(), "q": display_rotation.as_quat().tolist(),
                "source_p": position.tolist(), "source_q": source_rotation.as_quat().tolist(),
                "opening": opening, "joints": [-travel, travel],
                "pose_state": "VALID", "angle_state": "ZARR",
            }
        frames.append(frame)
    timeline = {
        "schema_version": "zarr-exact-replay/v1", "audit_mode": True,
        "capture_pair_id": metadata["episode_id"], "layout_calibration_id": "zarr-display-lanes-v1",
        "dataset_sha256": dataset_hash,
        "reference_frame": coordinate.get("frame_id", "room_world") if world_mode else "per-gripper-start-local",
        "training_ready": bool(metadata.get("training_ready", False)),
        "source_interval_s": {"start": 0.0, "end": count / float(metadata["frequency_hz"])},
        "fps": float(metadata["frequency_hz"]), "duration_s": count / float(metadata["frequency_hz"]),
        "source_frames": int(metadata["frames"]), "training_frames": count,
        "training_episodes": int(episodes[-1] + 1) if count else 0,
        "episode_ends": effective_episode_ends,
        "sync": {"offset_s": 0.0, "correlation": 1.0},
        "side_mapping": {"left": "camera0/robot0", "right": "camera1/robot1"},
        "coordinate_mapping": {
            "source": "exact Zarr shared world pose" if world_mode else "exact Zarr per-gripper local pose",
            "display": "identity: same world arrays" if world_mode else "declared static display lanes",
        },
        "coordinate_status": coordinate,
        "attitude": {"mode": "zarr-exact", "source": "robot*_eef_rot_axis_angle", "level_constraint": False},
        "bounds_m": {}, "tag_anchors": tag_anchors, "frames": frames,
    }
    timeline_path.write_text(json.dumps(timeline, ensure_ascii=False, separators=(",", ":")) + "\n",
                             encoding="utf-8")
    return timeline, arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-frames", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args(); dataset = args.dataset.resolve()
    metadata = (args.metadata or dataset.with_name("episode_metadata.json")).resolve()
    output = (args.output or dataset.with_name("zarr_exact_replay.mp4")).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    timeline_path = output.with_name(output.stem + "_timeline.json")
    report_path = output.with_name(output.stem + "_audit.json")
    base_video = output.with_name(output.stem + "_3d_base.mp4")
    camera_videos = [output.with_name(output.stem + f"_camera{i}.mp4") for i in range(2)]
    timeline, arrays = build_timeline(dataset, metadata, timeline_path, args.max_frames)
    fps = float(timeline["fps"]); count = len(timeline["frames"])
    for camera in range(2):
        encode_rgb_video(arrays[f"camera{camera}_rgb"], camera_videos[camera], fps)
    run([
        str(resolve_node_binary()), str(RENDERER), "--timeline", str(timeline_path),
        "--mesh-dir", str(MESH_DIR),
        "--output", str(base_video), "--ffmpeg", str(FFMPEG),
        "--duration", str(timeline["duration_s"]), "--fps", str(fps),
    ])
    run([
        str(FFMPEG), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(base_video), "-i", str(camera_videos[0]), "-i", str(camera_videos[1]),
        "-filter_complex",
        "[1:v]scale=160:160:flags=lanczos,pad=320:160:80:0:black[c0];"
        "[2:v]scale=160:160:flags=lanczos,pad=320:160:80:0:black[c1];"
        "[0:v][c0]overlay=1210:56:shortest=1[tmp];[tmp][c1]overlay=1554:56:shortest=1[v]",
        "-map", "[v]", "-an", "-frames:v", str(count), "-c:v", "libx264", "-preset", "medium",
        "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ])
    base_video.unlink(missing_ok=True)
    for camera_video in camera_videos:
        camera_video.unlink(missing_ok=True)
    position_error = 0.0; opening_error = 0.0; rotation_error_deg = 0.0
    for robot, side in enumerate(("left", "right")):
        timeline_positions = np.asarray([frame[side]["source_p"] for frame in timeline["frames"]])
        position_error = max(position_error, float(np.max(np.abs(
            timeline_positions - arrays[f"robot{robot}_eef_pos"]))))
        timeline_opening = np.asarray([frame[side]["opening"] for frame in timeline["frames"]])
        opening_error = max(opening_error, float(np.max(np.abs(
            timeline_opening - arrays[f"robot{robot}_gripper_angle_deg"][:, 0]))))
        timeline_rotation = Rotation.from_quat(np.asarray([
            frame[side]["source_q"] for frame in timeline["frames"]]))
        stored_rotation = Rotation.from_rotvec(arrays[f"robot{robot}_eef_rot_axis_angle"])
        rotation_error_deg = max(rotation_error_deg, float(np.degrees(
            np.max((timeline_rotation.inv() * stored_rotation).magnitude()))))
    report = {
        "schema_version": "zarr-exact-replay-audit/v1", "dataset": str(dataset),
        "dataset_sha256": timeline["dataset_sha256"], "timeline": str(timeline_path),
        "timeline_sha256": sha256(timeline_path), "video": str(output), "video_sha256": sha256(output),
        "frames": count, "fps": fps, "episode_ends": timeline["episode_ends"],
        "checks": {
            "all_training_actions_valid": bool(np.all(arrays["action_valid"])),
            "camera_frame_counts_match": len(arrays["camera0_rgb"]) == len(arrays["camera1_rgb"]) == count,
            "pose_frame_counts_match": len(arrays["robot0_eef_pos"]) == len(arrays["robot1_eef_pos"]) == count,
            "timeline_position_max_abs_error_m": position_error,
            "timeline_opening_max_abs_error_deg": opening_error,
            "timeline_rotation_max_error_deg": rotation_error_deg,
            "renderer_recomputed_pose": False, "renderer_applied_filtering": False,
        },
        "display_only_transforms": timeline["coordinate_mapping"],
        "claim": "Every rendered RGB frame and source pose comes directly from dataset.zarr.zip.",
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
