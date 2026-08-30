#!/usr/bin/env python3
"""Render a single 100 FPS gripper capture in the established Three.js scene."""

from __future__ import annotations

import csv
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from fuse_asymmetric_gripper_world_pose import camera_to_base
from render_single_gripper_motion_demo import (
    compose_base_track,
    fill_for_display,
    load_camera_track,
    load_gripper_signals,
    make_cad_opening_model,
    path_length,
    sample_pose,
    stats,
)
from rig_revision import load_rig_revision, sha256
from vla_dataset_export import smooth_positions, smooth_rotations
from world_frames import compile_world_tag_map


ROOT = Path(__file__).resolve().parent
RENDERER = ROOT / "dual_gripper_3d/render_frames.mjs"
SINGLE_SCENE = ROOT / "dual_gripper_3d/single_gripper_scene.html"
FFMPEG = ROOT / "work/tools/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("front_video", type=Path)
    parser.add_argument("--source-osv", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--camera-pose-csv", type=Path)
    parser.add_argument("--camera-pose-summary", type=Path)
    parser.add_argument("--force-angle-csv", type=Path, required=True)
    parser.add_argument("--force-angle-audit", type=Path, required=True)
    parser.add_argument("--rig-revision", type=Path, required=True)
    parser.add_argument("--marker-layout", type=Path, required=True)
    parser.add_argument("--world-tag-map", type=Path)
    parser.add_argument("--new-cad-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--timeline-only", action="store_true")
    parser.add_argument("--allow-diagnostic-rig", action="store_true")
    parser.add_argument(
        "--camera-local-basetag",
        action="store_true",
        help="use the rig's BaseTag-fitted camera-to-base transform without a world-pose claim",
    )
    parser.add_argument(
        "--camera-hardware-model",
        choices=("dji-osmo-360", "insta360-x5"),
        default="dji-osmo-360",
    )
    parser.add_argument("--right-base-pose-csv", type=Path)
    parser.add_argument(
        "--display-filter",
        choices=("raw", "kalman-rts"),
        default="raw",
    )
    parser.add_argument(
        "--right-tracking-source",
        default="reciprocal BaseTag3",
    )
    parser.add_argument("--right-force-angle-audit", type=Path)
    return parser.parse_args()


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def single_capture_id(source_osv: Path, fps: float) -> str:
    fps_label = f"{fps:.6f}".rstrip("0").rstrip(".")
    return f"{source_osv.stem}-single-{fps_label}fps"


def tag_anchors(compiled: dict) -> list[dict]:
    anchors = []
    for tag in compiled["tags"]:
        panel = str(tag.get("panel", ""))
        side = "left" if panel.startswith(("left", "grid_A")) else "right"
        anchors.append(
            {
                "id": int(tag["id"]),
                "side": side,
                "corners": tag["corners_m"],
                "color": "#37c8ff" if side == "left" else "#58df91",
            }
        )
    return anchors


def main() -> int:
    args = parse_args()
    required_names = (
        "front_video",
        "source_osv",
        "calibration",
        "force_angle_csv",
        "force_angle_audit",
        "rig_revision",
        "marker_layout",
        "new_cad_source",
    )
    paths = {
        name: getattr(args, name).resolve(strict=True)
        for name in required_names
    }
    if args.camera_local_basetag:
        if args.camera_pose_csv or args.camera_pose_summary or args.world_tag_map:
            raise ValueError(
                "camera-local BaseTag mode must not accept world camera-pose or tag-map inputs"
            )
        if args.display_filter != "raw":
            raise ValueError("camera-local BaseTag mode requires the raw display filter")
    else:
        for name in ("camera_pose_csv", "camera_pose_summary", "world_tag_map"):
            value = getattr(args, name)
            if value is None:
                raise ValueError(f"--{name.replace('_', '-')} is required in world mode")
            paths[name] = value.resolve(strict=True)
    right_pose_path = (
        args.right_base_pose_csv.resolve(strict=True)
        if args.right_base_pose_csv else None
    )
    right_rows = {}
    if right_pose_path is not None:
        with right_pose_path.open(newline="", encoding="utf-8") as handle:
            right_rows = {
                int(row["frame"]): row for row in csv.DictReader(handle)
                if row.get("base_x_m")
            }
    left_force_audit = json.loads(
        paths["force_angle_audit"].read_text(encoding="utf-8")
    )
    left_force_valid = left_force_audit.get("force", {}).get(
        "validated_for_display",
        left_force_audit.get("source", {}).get("camera_profile")
        != "insta360-x5-front",
    )
    with paths["force_angle_csv"].open(newline="", encoding="utf-8") as handle:
        left_diagnostic_rows = {
            int(row["frame"]): row for row in csv.DictReader(handle)
        }
    left_has_black_dot_gap = any(
        row.get("black_dot_gap_px") not in (None, "", "nan", "NaN")
        for row in left_diagnostic_rows.values()
    )
    right_force_audit_path = (
        args.right_force_angle_audit.resolve(strict=True)
        if args.right_force_angle_audit else None
    )
    right_force_audit = (
        json.loads(right_force_audit_path.read_text(encoding="utf-8"))
        if right_force_audit_path else None
    )
    right_force_valid = bool(
        right_force_audit
        and right_force_audit.get("force", {}).get(
            "validated_for_display",
            right_force_audit.get("source", {}).get("camera_profile")
            != "insta360-x5-front",
        )
    )
    if not FFMPEG.is_file():
        raise ValueError(f"archived project FFmpeg is unavailable: {FFMPEG}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    rig = load_rig_revision(
        paths["rig_revision"],
        allow_diagnostic_world=args.allow_diagnostic_rig,
    )
    calibration = json.loads(paths["calibration"].read_text(encoding="utf-8"))
    serial = calibration.get("serial")
    role = next(
        (
            name
            for name, robot in rig["hardware"]["robots"].items()
            if robot["camera_serial"] == serial
        ),
        None,
    )
    if role is None:
        raise ValueError(f"camera serial {serial!r} is not bound by the rig revision")
    marker_layout = json.loads(paths["marker_layout"].read_text(encoding="utf-8"))
    if marker_layout.get("status") != "ACTIVE_FOR_IMAGE_RELATIVE_YELLOW_LAYOUT_ONLY":
        raise ValueError("single-gripper WebGL requires the corrected marker-layout r2")
    marker_source = Path(marker_layout["source"]["path"])
    if sha256(marker_source) != marker_layout["source"]["sha256"]:
        raise ValueError("marker-layout DXF hash mismatch")

    capture = cv2.VideoCapture(str(paths["front_video"]))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if args.camera_hardware_model == "dji-osmo-360" and abs(fps - 100.0) > 1e-6:
        raise ValueError(f"expected the complete 100 FPS stream, got {fps}")
    if args.max_frames > 0:
        frame_count = min(frame_count, args.max_frames)
    frame_times = np.arange(frame_count, dtype=float) / fps
    camera_base = camera_to_base(rig["hardware"], role)
    base_camera = camera_base.inverse()
    if args.camera_local_basetag:
        positions = np.tile(camera_base.p, (frame_count, 1))
        rotations = Rotation.from_quat(
            np.tile(camera_base.r.as_quat(), (frame_count, 1))
        )
        source_positions = positions.copy()
        source_rotations = rotations
        direct_frame_times = frame_times
        display_filter_audit = {
            "mode": "raw",
            "source_pose_preserved": True,
            "camera_local_basetag": True,
        }
    else:
        camera = load_camera_track(paths["camera_pose_csv"])
        direct_position, direct_rotation = compose_base_track(camera, camera_base)
        positions, rotations = sample_pose(
            camera.time_s, direct_position, direct_rotation, frame_times
        )
        source_positions = positions.copy()
        source_rotations = rotations
        direct_frame_times = camera.time_s
        display_filter_audit = {
            "mode": args.display_filter,
            "source_pose_preserved": True,
        }
        if args.display_filter == "kalman-rts":
            positions, _, left_filter_audit = smooth_positions(positions, frame_times)
            rotations = smooth_rotations(rotations)
            display_filter_audit["left"] = left_filter_audit
    right_display = {}
    if right_rows:
        right_keys = sorted(right_rows)
        right_times = np.asarray(right_keys, dtype=float) / fps
        right_positions = np.asarray([
            [float(right_rows[frame][f"base_{axis}_m"]) for axis in "xyz"]
            for frame in right_keys
        ])
        right_rotations = Rotation.from_quat([
            [float(right_rows[frame][key]) for key in ("qx", "qy", "qz", "qw")]
            for frame in right_keys
        ])
        if args.display_filter == "kalman-rts":
            right_positions, _, right_filter_audit = smooth_positions(
                right_positions, right_times
            )
            right_rotations = smooth_rotations(right_rotations)
            display_filter_audit["right"] = right_filter_audit
        for offset, frame in enumerate(right_keys):
            right_display[frame] = (
                right_positions[offset], right_rotations[offset]
            )
    all_signals = load_gripper_signals(
        paths["force_angle_csv"],
        int(cv2.VideoCapture(str(paths["front_video"])).get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    included = all_signals.included_angle_deg[:frame_count]
    opening = all_signals.opening_angle_deg[:frame_count]
    intensity = all_signals.contact_intensity_percent[:frame_count]
    source_intensity = all_signals.contact_intensity_percent[:frame_count]
    signal_states = all_signals.measurement_state[:frame_count]

    cad = rig["cad_revision"]
    if cad is None:
        raise ValueError("rig revision has no renderable CAD revision")
    mesh_dir = (ROOT / cad["mesh_directory"]).resolve()
    cad_model = make_cad_opening_model(mesh_dir, rig["geometry"])
    widths_mm = np.asarray([
        1000.0 * cad_model.width_m(value) if np.isfinite(value) else np.nan
        for value in opening
    ])

    compiled_map = (
        None
        if args.camera_local_basetag
        else compile_world_tag_map(paths["world_tag_map"])
    )
    frames = []
    display_rotation_correction = Rotation.from_euler(
        "x", 0.0 if args.camera_local_basetag else 180.0, degrees=True
    )
    display_rotations = rotations * display_rotation_correction
    for index, now in enumerate(frame_times):
        quaternion = display_rotations[index].as_quat()
        source_quaternion = source_rotations[index].as_quat()
        nearest_age = float(np.min(np.abs(direct_frame_times - now)))
        pose_state = (
            f"FIXED BASETAG{rig['hardware']['robots'][role]['base_tag_id']}-FITTED CAMERA→BASE_LINK"
            if args.camera_local_basetag
            else f"DISPLAY FILTER {args.display_filter.upper()}"
            if args.display_filter != "raw"
            else f"MEASURED {fps:.2f} HZ" if nearest_age <= 0.5 / fps
            else "DIRECT-BRACKETED DISPLAY"
        )
        angle_state = signal_states[index]
        one_sided_angle = "ONE_SIDED" in angle_state
        angle_available = bool(
            np.isfinite(opening[index])
            and (np.isfinite(included[index]) or one_sided_angle)
            and angle_state != "UNAVAILABLE"
        )
        force_available = bool(
            left_force_valid
            and np.isfinite(intensity[index])
            and (angle_state == "MEASURED" or angle_state.startswith("RECOVERED"))
        )
        left_contact_measurement_state = (
            angle_state if force_available else
            "UNAVAILABLE" if left_force_valid else "REJECTED_X5_FORCE_MODEL"
        )
        contact_state = (
            "HIGH" if force_available and intensity[index] >= 70.0 else
            "CONTACT" if force_available and intensity[index] >= 20.0 else
            "LOW / FREE" if force_available else "UNAVAILABLE"
        )
        joint1, joint2 = (
            cad_model.joint_angles(opening[index])
            if angle_available else (None, None)
        )
        left_diagnostic = left_diagnostic_rows.get(index, {})
        left_gap_text = left_diagnostic.get("black_dot_gap_px", "")
        left_gap = (
            float(left_gap_text)
            if left_gap_text not in ("", "nan", "NaN")
            else None
        )
        left_gap_residual_text = left_diagnostic.get(
            "opening_conditioned_gap_residual_px", ""
        )
        left_gap_residual = (
            float(left_gap_residual_text)
            if left_gap_residual_text not in ("", "nan", "NaN")
            else None
        )
        left_contact_ground_truth = left_diagnostic.get(
            "contact_ground_truth", "UNLABELED"
        )
        left = {
            "p": positions[index].tolist(),
            "q": quaternion.tolist(),
            "source_p": source_positions[index].tolist(),
            "source_q": source_quaternion.tolist(),
            "opening": float(opening[index]) if angle_available else None,
            "included_angle_deg": (
                float(included[index]) if np.isfinite(included[index]) else None
            ),
            "angle_available": angle_available,
            "contact_measurement_state": left_contact_measurement_state,
            "cad_opening_width_mm": (
                float(widths_mm[index]) if angle_available else None
            ),
            "source_contact_intensity_percent": (
                float(source_intensity[index])
                if np.isfinite(source_intensity[index]) else None
            ),
            "contact_intensity_percent": (
                float(intensity[index]) if force_available else 0.0
            ),
            "contact_state": (
                contact_state if left_force_valid else "REJECTED"
            ),
            "joints": [joint1, joint2] if angle_available else [0.0, 0.0],
            "source_joints": [joint1, joint2] if angle_available else None,
            "black_dot_gap_px": left_gap,
            "opening_conditioned_gap_residual_px": left_gap_residual,
            "contact_ground_truth": left_contact_ground_truth,
            "pose_state": pose_state,
            "angle_state": angle_state,
            "visible": True,
        }
        right_row = right_rows.get(index)
        if right_row is not None:
            right_source_position = [
                float(right_row[f"base_{axis}_m"]) for axis in "xyz"
            ]
            right_source_rotation = Rotation.from_quat(
                [float(right_row[key]) for key in ("qx", "qy", "qz", "qw")]
            )
            right_display_position, right_display_rotation = right_display.get(
                index, (np.asarray(right_source_position), right_source_rotation)
            )
            right_quaternion = (
                right_display_rotation * display_rotation_correction
            ).as_quat().tolist()
            right_opening = float(right_row.get("opening_angle_deg", 0.0))
            right_included = float(right_row.get("included_jaw_angle_deg", 0.0))
            right_raw_contact = float(right_row.get("contact_intensity_percent", np.nan))
            right_signal_state = right_row.get(
                "gripper_measurement_state", "UNAVAILABLE"
            )
            right_opening = right_opening if np.isfinite(right_opening) else 0.0
            right_included = right_included if np.isfinite(right_included) else 0.0
            right_source_contact = (
                right_raw_contact
                if np.isfinite(right_raw_contact)
                and right_signal_state != "UNAVAILABLE"
                else None
            )
            right_contact = right_raw_contact if np.isfinite(right_raw_contact) else 0.0
            right_joint1, right_joint2 = cad_model.joint_angles(right_opening)
            right = {
                "p": right_display_position.tolist(), "q": right_quaternion,
                "source_p": right_source_position,
                "source_q": right_source_rotation.as_quat().tolist(),
                "opening": right_opening, "included_angle_deg": right_included,
                "cad_opening_width_mm": float(cad_model.width_m(right_opening) * 1000),
                "source_contact_intensity_percent": right_source_contact,
                "contact_intensity_percent": (
                    right_contact if right_force_valid else 0.0
                ),
                "contact_state": (
                    (
                        "HIGH" if right_contact >= 70.0 else
                        "CONTACT" if right_contact >= 20.0 else "LOW / FREE"
                    )
                    if right_force_valid else "REJECTED"
                ),
                "joints": [right_joint1, right_joint2],
                "pose_state": right_row["measurement_source"],
                "angle_state": right_row.get(
                    "gripper_measurement_state", "UNAVAILABLE"
                ),
                "contact_measurement_state": (
                    right_row.get("gripper_measurement_state", "UNAVAILABLE")
                    if right_force_valid else "REJECTED_X5_FORCE_MODEL"
                ),
                "visible": True,
            }
        else:
            right = {
                "p": positions[index].tolist(), "q": quaternion.tolist(),
                "source_p": source_positions[index].tolist(),
                "source_q": source_quaternion.tolist(),
                "opening": 0.0, "included_angle_deg": 0.0,
                "cad_opening_width_mm": 0.0,
                "contact_intensity_percent": 0.0,
                "contact_state": "HIDDEN", "joints": [0.0, 0.0],
                "pose_state": "HIDDEN", "angle_state": "HIDDEN",
                "visible": False,
                "contact_measurement_state": "UNAVAILABLE",
            }
        frames.append({"t": float(now), "source_index": index, "left": left, "right": right})

    base_tag_id = int(rig["hardware"]["robots"][role]["base_tag_id"])
    timeline = {
        "schema_version": "single-gripper-webgl/v1",
        "audit_mode": False,
        "render_mode": "single_gripper_world_diagnostic",
        "default_view": "camera_mount" if args.camera_local_basetag else "human_corner",
        "view_roll_deg": 0.0,
        "operator_eye_elevation_factor": 0.10,
        "camera_serial": serial,
        "operator_tag_look_fraction": 0.40,
        "capture_pair_id": single_capture_id(paths["source_osv"], fps),
        "camera_hardware_model": args.camera_hardware_model,
        "mounted_camera": {
            "model": args.camera_hardware_model,
            "serial": serial,
            "body_size_m": (
                [0.0382, 0.046, 0.1245]
                if args.camera_hardware_model == "insta360-x5"
                else [0.075, 0.052, 0.040]
            ),
            "T_base_camera": {
                "translation_m": base_camera.p.tolist(),
                "quaternion_xyzw": base_camera.r.as_quat().tolist(),
            },
            "geometry_status": (
                (
                    f"official X5 body dimensions; BaseTag{base_tag_id}-fitted "
                    "camera-to-base mount"
                )
                if args.camera_hardware_model == "insta360-x5"
                else "legacy display-only Osmo model"
            ),
        },
        "layout_calibration_id": (
            f"camera-local-BaseTag{base_tag_id}"
            if args.camera_local_basetag else compiled_map["map_id"]
        ),
        "reference_frame": (
            "panorama_camera"
            if args.camera_local_basetag
            else compiled_map.get("world_frame", "tag_map")
        ),
        "fps": fps,
        "duration_s": frame_count / fps,
        "source_frames": frame_count,
        "training_frames": 0,
        "training_episodes": 0,
        "training_ready": False,
        "sync": {"offset_s": 0.0, "correlation": 1.0},
        "primary_hardware_role": role,
        "side_mapping": {
            "left": f"physical-{role}/X5-mounted",
            "right": args.right_tracking_source if right_rows else "hidden",
        },
        "right_gripper_tracking": {
            "enabled": bool(right_rows),
            "source": args.right_tracking_source,
            "frame_range": (
                [min(right_rows), max(right_rows)] if right_rows else None
            ),
            "jaw_angle_available": bool(
                right_rows and "opening_angle_deg" in next(iter(right_rows.values()))
            ),
            "contact_intensity_available": bool(
                right_rows
                and "contact_intensity_percent" in next(iter(right_rows.values()))
            ),
        },
        "display_filter": display_filter_audit,
        "contact_ground_truth": left_force_audit.get("contact_ground_truth"),
        "angle_models": {
            "left": left_force_audit.get("angle"),
            "right": (
                right_force_audit.get("angle")
                if right_force_audit is not None else None
            ),
        },
        "contact_events": left_force_audit.get("contact_events"),
        "localization": (
            {
                "method": "rigid camera-to-BaseTag-to-base_link calibration",
                "base_tag_id": base_tag_id,
                "frame_id": "panorama_camera",
                "pose_state": "fixed rigid mount; no world-pose claim",
                "camera_to_base": {
                    "translation_m": camera_base.p.tolist(),
                    "quaternion_xyzw": camera_base.r.as_quat().tolist(),
                },
            }
            if args.camera_local_basetag else None
        ),
        "force_models": {
            "left": {
                "validated_for_display": bool(left_force_valid),
                "source_audit": str(paths["force_angle_audit"]),
                "quantity": left_force_audit.get("force", {}).get("quantity"),
                "application_point": left_force_audit.get("force", {}).get(
                    "application_point"
                ),
                "direction": left_force_audit.get("force", {}).get("direction"),
                "fixed_scale_across_captures": left_force_audit.get(
                    "force", {}
                ).get("fixed_scale_across_captures", False),
                "display_metric": (
                    "black_dot_gap_px"
                    if not left_force_valid and left_has_black_dot_gap
                    else "raw_deformation_score"
                ),
            },
            "right": {
                "validated_for_display": bool(right_force_valid),
                "source_audit": (
                    str(right_force_audit_path)
                    if right_force_audit_path else None
                ),
            },
            "raw_diagnostics_preserved": True,
        },
        "coordinate_mapping": (
            {
                "source": (
                    f"existing BaseTag{base_tag_id}-fitted rigid "
                    "camera_to_base calibration"
                ),
                "display": "panorama-camera coordinates; no world transform",
                "display_orientation_correction": {
                    "axis": None,
                    "angle_deg": 0.0,
                    "scope": "none",
                    "source_pose_modified": False,
                },
            }
            if args.camera_local_basetag
            else {
                "source": f"official stitched {args.camera_hardware_model} Grid camera pose composed with serial-bound camera_to_base",
                "display": "tag_map position identity; display-only base-local X roll +180 deg",
                "display_orientation_correction": {
                    "axis": "base_link +X",
                    "angle_deg": 180.0,
                    "scope": "rendering only",
                    "source_pose_modified": False,
                },
            }
        ),
        "coordinate_status": (
            {
                "mode": "camera_local_basetag",
                "frame_id": "panorama_camera",
                "calibration_status": rig["hardware"]["calibration_status"],
                "frame_convention": {
                    "up_vector": None,
                    "units": "m",
                    "transform_direction": "T_panorama_camera_base_link",
                },
            }
            if args.camera_local_basetag
            else {
                "mode": "world",
                "frame_id": compiled_map.get("world_frame", "tag_map"),
                "tag_map_sha256": compiled_map["tag_map_sha256"],
                "calibration_status": compiled_map.get(
                    "calibration_status", "DIAGNOSTIC"
                ),
                "frame_convention": {
                    "up_vector": compiled_map.get(
                        "physical_up_vector", [0.0, -1.0, 0.0]
                    ),
                    "units": "m",
                },
            }
        ),
        "eef_reference": {"type": "base_link"},
        "jaw_joint_origins_m": rig["geometry"]["jaw_joint_origins_m"],
        "attitude": {
            "mode": (
                f"fixed BaseTag{base_tag_id}-fitted camera-to-base rotation"
                if args.camera_local_basetag
                else f"visual Grid pose at {fps:.3f} Hz with display-only SLERP brackets"
            ),
            "level_constraint": False,
        },
        "bounds_m": {},
        "tag_anchors": [] if args.camera_local_basetag else tag_anchors(compiled_map),
        "validation_events": [],
        "segment_untrusted_tracks": False,
        "frames": frames,
    }
    timeline_path = output_dir / "single_gripper_webgl_timeline.json"
    timeline_path.write_text(
        json.dumps(timeline, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    if args.timeline_only:
        print(timeline_path)
        return 0

    base_video = output_dir / "single_gripper_webgl_base.mp4"
    output_video = output_dir / "single_gripper_webgl_100fps.mp4"
    run(
        [
            "node",
            str(RENDERER),
            "--timeline",
            str(timeline_path),
            "--mesh-dir",
            str(mesh_dir),
            "--output",
            str(base_video),
            "--ffmpeg",
            str(FFMPEG),
            "--duration",
            str(frame_count / fps),
            "--fps",
            str(fps),
            "--view-preset",
            "human_corner",
            "--scene",
            str(SINGLE_SCENE),
        ]
    )
    run(
        [
            str(FFMPEG),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(base_video),
            "-i",
            str(paths["front_video"]),
            "-i",
            str(paths["source_osv"]),
            "-filter_complex",
            "[1:v]scale=160:160:flags=lanczos,pad=320:160:80:0:black[c0];"
            "[0:v][c0]overlay=1210:56:shortest=1[v]",
            "-map",
            "[v]",
            "-map",
            "2:a?",
            "-t",
            str(frame_count / fps),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(output_video),
        ]
    )
    base_video.unlink()

    pose_summary = json.loads(paths["camera_pose_summary"].read_text(encoding="utf-8"))
    force_audit = json.loads(paths["force_angle_audit"].read_text(encoding="utf-8"))
    position_steps = np.linalg.norm(np.diff(direct_position, axis=0), axis=1)
    orientation_steps = np.degrees(
        (direct_rotation[:-1].inv() * direct_rotation[1:]).magnitude()
    )
    audit = {
        "schema_version": "single-gripper-webgl-audit/1.0",
        "status": "DIAGNOSTIC",
        "source": {
            "osv": str(paths["source_osv"]),
            "osv_sha256": sha256(paths["source_osv"]),
            "front_lens": str(paths["front_video"]),
            "front_lens_sha256": sha256(paths["front_video"]),
            "camera_serial": serial,
            "hardware_role": role,
            "base_tag_id": rig["hardware"]["robots"][role]["base_tag_id"],
            "fps": fps,
            "frames": frame_count,
        },
        "renderer": {
            "scene": str(SINGLE_SCENE),
            "scene_sha256": sha256(SINGLE_SCENE),
            "renderer": str(RENDERER),
            "renderer_sha256": sha256(RENDERER),
            "ffmpeg": str(FFMPEG.resolve()),
            "ffmpeg_sha256": sha256(FFMPEG),
            "view": "human_corner Tag-wall overview",
            "display_orientation_correction": "base-local X roll +180 deg; source quaternion preserved",
        },
        "rig": {
            "revision_id": rig["revision"]["revision_id"],
            "revision_sha256": rig["revision_sha256"],
            "rendered_cad_revision_id": cad["revision_id"],
            "newest_editable_source": str(paths["new_cad_source"]),
            "newest_editable_source_sha256": sha256(paths["new_cad_source"]),
            "warning": "UMI-III is source-only; v52 meshes are the current renderable diagnostic CAD.",
        },
        "marker_layout": {
            "revision_id": marker_layout["revision_id"],
            "path": str(paths["marker_layout"]),
            "sha256": sha256(paths["marker_layout"]),
        },
        "trajectory": {
            "direct_frames_10hz": len(camera.time_s),
            "valid_ratio": pose_summary["valid_ratio"],
            "rejected_frames": pose_summary["common_frames"] - pose_summary["valid_frames"],
            "angular_rmse_deg": stats(camera.angular_rmse_deg),
            "position_step_m": stats(position_steps),
            "orientation_step_deg": stats(orientation_steps),
            "path_length_m": path_length(direct_position),
            "holdout_status": "NONE_DIAGNOSTIC_CAPTURE_ONLY",
        },
        "jaw": {
            "included_angle_range_deg": [float(included.min()), float(included.max())],
            "opening_angle_range_deg": [float(opening.min()), float(opening.max())],
            "cad_opening_range_mm": [float(widths_mm.min()), float(widths_mm.max())],
        },
        "contact_intensity": {
            "range_percent": [float(intensity.min()), float(intensity.max())],
            "zero_ratio": float(np.mean(intensity == 0.0)),
            "source_audit": str(paths["force_angle_audit"]),
            "onset_policy": force_audit["force"]["onset_policy"],
            "warning": "Capture-local continuous deformation proxy; not Newtons.",
        },
        "outputs": {
            "video": str(output_video),
            "video_sha256": sha256(output_video),
            "timeline": str(timeline_path),
            "timeline_sha256": sha256(timeline_path),
        },
        "training_ready": False,
    }
    audit_path = output_dir / "audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
