#!/usr/bin/env python3
"""Render a 60 fps, display-only interpolation of a full audited episode."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from tools.render_zarr_audit_video import FFMPEG, MESH_DIR, RENDERER, encode_rgb_video
from vla_dataset_export import (
    apply_camera_to_tcp, load_pose_csv, resample_pose, smooth_positions, smooth_rotations,
)
from world_frames import compile_world_tag_map


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("arrays", type=Path)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--episode-spec", type=Path)
    parser.add_argument("--left-tag-map", type=Path)
    parser.add_argument("--right-tag-map", type=Path)
    parser.add_argument("--manual-layout", type=Path,
                        help="place each start-local trajectory in one shared centre frame")
    parser.add_argument("--tag-anchors", type=Path,
                        help="precomputed room-frame tag anchor JSON")
    parser.add_argument("--world-frame", action="store_true",
                        help="input poses are already expressed in one shared world frame")
    parser.add_argument("--world-tag-map", type=Path,
                        help="render the same compiled global Tag map used by the dataset")
    parser.add_argument("--timeline-only", action="store_true",
                        help="write the interactive timeline without encoding an MP4")
    parser.add_argument("--segment-untrusted-tracks", action="store_true",
                        help="break trajectory lines instead of drawing across recovered gaps")
    parser.add_argument(
        "--allow-invalid-pair", action="store_true",
        help="render an explicitly invalid multi-camera pair for forensic inspection only",
    )
    args = parser.parse_args()

    arrays_path = args.arrays.resolve(); output = args.output.resolve()
    metadata = json.loads(args.metadata.resolve().read_text(encoding="utf-8"))
    pair_integrity = metadata.get("sync", {}).get("pair_integrity", {})
    if pair_integrity.get("required") and not pair_integrity.get("valid") and not args.allow_invalid_pair:
        raise ValueError(
            "refusing to render an invalid multi-camera pair: "
            f"{pair_integrity.get('status', 'UNKNOWN')}; "
            "record simultaneously or provide verified hardware/audio synchronization"
        )
    episode_spec = {}
    hardware = {}
    if args.episode_spec:
        episode_spec = json.loads(args.episode_spec.resolve().read_text(encoding="utf-8"))
        hardware_ref = episode_spec.get("hardware_config")
        if hardware_ref:
            hardware_path = (args.episode_spec.resolve().parent / hardware_ref).resolve()
            hardware = json.loads(hardware_path.read_text(encoding="utf-8"))
    source = np.load(arrays_path)
    source_t = np.asarray(source["timestamp_s"], dtype=float)
    source_t = source_t - source_t[0]
    source_hz = float(metadata["frequency_hz"])
    duration = len(source_t) / source_hz
    query_t = np.arange(round(duration * args.fps), dtype=float) / args.fps
    query_t = np.minimum(query_t, source_t[-1])
    nearest = np.clip(np.rint(query_t * source_hz).astype(int), 0, len(source_t) - 1)

    neutral = abs(np.degrees(np.arctan2(50.568, 63.276) - np.arctan2(-50.745, 63.134)))
    display_offsets = [np.asarray([-0.32, 0.0, 0.0]), np.asarray([0.32, 0.0, 0.0])]
    display_rotations = [Rotation.identity(), Rotation.from_euler("z", 180, degrees=True)]
    layout_id = "display-lanes-v1"
    if args.world_frame:
        display_offsets = [np.zeros(3), np.zeros(3)]
        display_rotations = [Rotation.identity(), Rotation.identity()]
        layout_id = "measured-room-corner-frame"
    if args.manual_layout:
        layout = json.loads(args.manual_layout.resolve().read_text(encoding="utf-8"))
        display_offsets = [
            np.asarray(layout["grippers_in_center_frame"][side]["translation_m"], dtype=float)
            for side in ("left", "right")
        ]
        display_rotations = [
            Rotation.from_euler(
                "xyz", layout["grippers_in_center_frame"][side]["rotation_rpy_deg"], degrees=True)
            for side in ("left", "right")
        ]
        layout_id = str(layout.get("calibration_id", args.manual_layout.stem))
    robot_specs_by_name = {
        item.get("name"): item for item in episode_spec.get("robots", [])
    }
    display_enabled = [
        bool(robot_specs_by_name.get(side, {}).get("display_enabled", True))
        for side in ("left", "right")
    ]
    robot_data = []
    for robot in range(2):
        pos0 = np.asarray(source[f"robot{robot}_eef_pos"], dtype=float)
        pos = np.column_stack([np.interp(query_t, source_t, pos0[:, axis]) for axis in range(3)])
        rot0 = Rotation.from_rotvec(np.asarray(source[f"robot{robot}_eef_rot_axis_angle"], dtype=float))
        rot = Slerp(source_t, rot0)(query_t)
        opening0 = np.asarray(source[f"robot{robot}_gripper_angle_deg"], dtype=float)[:, 0]
        opening = np.interp(query_t, source_t, opening0)
        angle_available = np.isfinite(opening)
        opening = np.where(angle_available, opening, neutral)
        measured = np.asarray(source[f"robot{robot}_pose_measured"], dtype=bool)[nearest]
        tracked = np.asarray(source[f"robot{robot}_pose_tracked"], dtype=bool)[nearest]
        angle_measured = (
            np.asarray(source[f"robot{robot}_gripper_measured"], dtype=bool)[nearest]
            & angle_available
        )
        robot_data.append((pos, rot, opening, measured, tracked, angle_measured, angle_available))

    tag_anchors = []
    if args.world_tag_map:
        if not args.world_frame:
            raise ValueError("--world-tag-map requires --world-frame")
        world_map = compile_world_tag_map(args.world_tag_map.resolve())
        metadata_hash = metadata.get("coordinate_frame", {}).get("tag_map_sha256")
        if metadata_hash and metadata_hash != world_map["tag_map_sha256"]:
            raise ValueError("visualizer Tag map hash does not match dataset metadata")
        for tag in world_map["tags"]:
            panel = str(tag.get("panel", ""))
            side = "left" if panel.startswith("left") else "right"
            tag_anchors.append({
                "id": int(tag["id"]), "side": side,
                "corners": tag["corners_m"],
                "color": "#37c8ff" if side == "left" else "#58df91",
            })
    if args.tag_anchors:
        tag_anchors = json.loads(args.tag_anchors.resolve().read_text(encoding="utf-8"))
    if args.episode_spec and args.left_tag_map and args.right_tag_map:
        spec_path = args.episode_spec.resolve()
        spec = episode_spec
        hardware_path = (spec_path.parent / spec["hardware_config"]).resolve()
        hardware = json.loads(hardware_path.read_text(encoding="utf-8"))
        absolute_t = float(spec["start_s"]) + np.arange(len(source_t), dtype=float) / source_hz
        for robot, (side, map_path) in enumerate((("left", args.left_tag_map), ("right", args.right_tag_map))):
            robot_spec = next(item for item in spec["robots"] if item["name"] == side)
            local_t = absolute_t + float(robot_spec.get("source_time_offset_s", 0.0))
            pose_path = (spec_path.parent / robot_spec["trajectory_csv"]).resolve()
            pose = load_pose_csv(pose_path)
            world_p, world_r, _, _ = resample_pose(pose, local_t)
            world_p, world_r = apply_camera_to_tcp(
                world_p, world_r, hardware["robots"][side]["camera_to_tcp"])
            world_p, _, _ = smooth_positions(world_p, absolute_t)
            world_r = smooth_rotations(world_r)
            origin_p, origin_r = world_p[0], world_r[0]
            tag_map = json.loads(map_path.resolve().read_text(encoding="utf-8"))
            for tag in tag_map["tags"]:
                corners_world = np.asarray(tag["corners_m"], dtype=float)
                corners_local = origin_r.inv().apply(corners_world - origin_p)
                corners_display = display_offsets[robot] + display_rotations[robot].apply(corners_local)
                tag_anchors.append({
                    "id": int(tag["id"]), "side": side,
                    "corners": corners_display.tolist(),
                    "color": "#37c8ff" if side == "left" else "#58df91",
                })

    frames = []
    for index, now in enumerate(query_t):
        frame = {"t": float(now), "source_index": int(nearest[index])}
        for robot, side in enumerate(("left", "right")):
            pos, rot, opening, measured, tracked, angle_measured, angle_available = robot_data[robot]
            display_p = display_offsets[robot] + display_rotations[robot].apply(pos[index])
            display_q = (display_rotations[robot] * rot[index]).as_quat()
            travel = (neutral - opening[index]) / 2.0
            if measured[index]:
                pose_state = "MEASURED"
            elif tracked[index]:
                pose_state = "FLOW / FILTERED"
            else:
                pose_state = "DISPLAY INTERPOLATED"
            frame[side] = {
                "p": display_p.tolist(), "q": display_q.tolist(),
                "source_p": pos[index].tolist(), "source_q": rot[index].as_quat().tolist(),
                "opening": float(opening[index]), "joints": [-float(travel), float(travel)],
                "pose_state": pose_state,
                "visible": display_enabled[robot],
                "angle_state": (
                    "MEASURED" if angle_measured[index]
                    else "DISPLAY INTERPOLATED" if angle_available[index]
                    else "UNAVAILABLE"
                ),
            }
        frames.append(frame)

    validation_events = []
    if episode_spec:
        robot_specs = {item["name"]: item for item in episode_spec.get("robots", [])}
        start_s = float(episode_spec.get("start_s", 0.0))
        for event in episode_spec.get("validation_events", []):
            observations = {}
            positions = []
            for side in ("left", "right"):
                observation = event.get("observations", {}).get(side)
                if not observation or side not in robot_specs:
                    continue
                source_time_s = float(observation["source_time_s"])
                source_offset_s = float(robot_specs[side].get("source_time_offset_s", 0.0))
                timeline_time_s = source_time_s - start_s - source_offset_s
                frame_index = int(np.clip(round(timeline_time_s * args.fps), 0, len(frames) - 1))
                position = frames[frame_index][side]["p"]
                observations[side] = {
                    "action": str(observation.get("action", "OBSERVE")),
                    "source_time_s": source_time_s,
                    "timeline_time_s": timeline_time_s,
                    "frame_index": frame_index,
                    "position_m": position,
                }
                positions.append(np.asarray(position, dtype=float))
            distance_m = None
            if len(positions) == 2:
                distance_m = float(np.linalg.norm(positions[0] - positions[1]))
            validation_events.append({
                "id": str(event["id"]), "label": str(event.get("label", event["id"])),
                "observations": observations, "distance_m": distance_m,
            })

    display_tag_to_base = {}
    for side in ("left", "right"):
        base_tag = hardware.get("robots", {}).get(side, {}).get("base_to_eef_reference")
        if not isinstance(base_tag, dict):
            continue
        rotation_base_tag = Rotation.from_quat(base_tag["quaternion_xyzw"])
        rotation_tag_base = rotation_base_tag.inv()
        translation_tag_base = -rotation_tag_base.apply(
            np.asarray(base_tag["translation_m"], dtype=float)
        )
        display_tag_to_base[side] = {
            "translation_m": translation_tag_base.tolist(),
            "quaternion_xyzw": rotation_tag_base.as_quat().tolist(),
            "tag_outer_size_m": float(
                hardware.get("robots", {}).get(side, {})
                .get("camera_to_eef_reference", {}).get("tag_outer_size_m", 0.020)
            ),
            "source": str(base_tag.get("source", "hardware base_to_eef_reference inverse")),
        }

    timeline = {
        "schema_version": "smooth-episode-preview/v1", "audit_mode": False,
        "render_mode": str(episode_spec.get("render_mode", "standard")),
        "default_view": str(episode_spec.get("default_view", "operator")),
        "view_roll_deg": float(episode_spec.get("view_roll_deg", 0.0)),
        "capture_pair_id": metadata["episode_id"], "layout_calibration_id": layout_id,
        "reference_frame": (
            metadata.get("coordinate_frame", {}).get("frame_id", "UNDECLARED_COMMON_FRAME")
            if args.world_frame else "per-gripper-start-local"
        ), "fps": args.fps,
        "duration_s": duration, "source_frames": len(source_t), "training_frames": int(metadata["training_frames"]),
        "training_episodes": int(metadata["training_episodes"]),
        "training_ready": bool(metadata.get("training_ready", False)),
        "sync": {
            "offset_s": 0.0,
            "correlation": float(metadata.get("sync", {}).get("correlation", 1.0)),
            "pair_integrity": pair_integrity,
        },
        "side_mapping": {"left": "camera0/robot0", "right": "camera1/robot1"},
        "coordinate_mapping": {
            "source": "shared room_world pose" if args.world_frame else "per-gripper start-local pose",
            "display": "identity: same world arrays" if args.world_frame else (
                "shared manual centre frame" if args.manual_layout else "independent display lanes"
            ),
        },
        "coordinate_status": metadata.get("coordinate_frame", {}),
        "eef_reference": metadata.get("eef_reference", {"type": "tcp"}),
        "display_tag_to_base": display_tag_to_base,
        "attitude": {"mode": "SLERP display interpolation", "level_constraint": False},
        "bounds_m": {}, "tag_anchors": tag_anchors, "validation_events": validation_events,
        "segment_untrusted_tracks": bool(args.segment_untrusted_tracks), "frames": frames,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    timeline_path = output.with_name(output.stem + "_timeline.json")
    base_video = output.with_name(output.stem + "_3d_base.mp4")
    camera_videos = [output.with_name(output.stem + f"_camera{i}.mp4") for i in range(2)]
    timeline_path.write_text(json.dumps(timeline, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    if args.timeline_only:
        print(timeline_path)
        return 0
    has_camera_insets = all(f"camera{camera}_rgb" in source.files for camera in range(2))
    if has_camera_insets:
        for camera in range(2):
            encode_rgb_video(source[f"camera{camera}_rgb"], camera_videos[camera], source_hz)
    run(["node", str(RENDERER), "--timeline", str(timeline_path), "--mesh-dir", str(MESH_DIR),
         "--output", str(base_video), "--ffmpeg", str(FFMPEG), "--duration", str(duration), "--fps", str(args.fps)])
    if has_camera_insets:
        run([str(FFMPEG), "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(base_video), "-i", str(camera_videos[0]), "-i", str(camera_videos[1]),
             "-filter_complex",
             "[1:v]scale=160:160:flags=lanczos,pad=320:160:80:0:black[c0];"
             "[2:v]scale=160:160:flags=lanczos,pad=320:160:80:0:black[c1];"
             "[0:v][c0]overlay=1210:56:shortest=1[tmp];[tmp][c1]overlay=1554:56:shortest=1[v]",
             "-map", "[v]", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)])
        base_video.unlink(missing_ok=True)
    else:
        base_video.replace(output)
    for path in camera_videos:
        path.unlink(missing_ok=True)
    report = {
        "schema_version": "smooth-episode-preview-audit/v1", "source_arrays": str(arrays_path),
        "output": str(output), "source_fps": source_hz, "display_fps": args.fps,
        "source_frames": len(source_t), "display_frames": len(query_t),
        "rendered_tag_ids": [item["id"] for item in tag_anchors],
        "layout_calibration_id": layout_id,
        "position": "linear interpolation of already Kalman+RTS filtered episode arrays",
        "orientation": "quaternion SLERP", "training_data_modified": False,
        "camera_insets": has_camera_insets,
        "warning": "DISPLAY INTERPOLATED frames are visualization only and are not training measurements",
    }
    output.with_name(output.stem + "_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
