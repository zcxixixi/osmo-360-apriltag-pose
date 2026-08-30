#!/usr/bin/env python3
"""Render one gripper's world trajectory, jaw geometry, and contact intensity."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation, Slerp
import trimesh

from calibrate_basetag_reciprocal import Transform
from fuse_asymmetric_gripper_world_pose import camera_to_base
from render_dual_camera_alignment_demo import UrdfWireframe, load_urdf_wireframe
from render_gripper_force_angle_demo import (
    AMBER,
    BG,
    CYAN,
    GREEN,
    MUTED,
    PANEL,
    RED,
    WHITE,
    draw_text,
    observe_frame,
    transcode_h264,
)
from rig_revision import load_rig_revision, sha256


@dataclass
class CameraTrack:
    time_s: np.ndarray
    position_m: np.ndarray
    rotation: Rotation
    fit_error: np.ndarray
    fit_error_name: str


@dataclass
class GripperSignals:
    included_angle_deg: np.ndarray
    opening_angle_deg: np.ndarray
    contact_intensity_percent: np.ndarray
    measurement_state: list[str]


@dataclass
class CadOpeningModel:
    left_vertices: np.ndarray
    right_vertices: np.ndarray
    joint1_origin_m: np.ndarray
    joint2_origin_m: np.ndarray
    closed_joint_rotation_deg: float

    def joint_angles(self, opening_angle_deg: float) -> tuple[float, float]:
        half = max(float(opening_angle_deg), 0.0) / 2.0
        return -self.closed_joint_rotation_deg + half, self.closed_joint_rotation_deg - half

    def width_m(self, opening_angle_deg: float) -> float:
        joint1, joint2 = self.joint_angles(opening_angle_deg)
        left = Rotation.from_euler("z", joint1, degrees=True).apply(self.left_vertices)
        right = Rotation.from_euler("z", joint2, degrees=True).apply(self.right_vertices)
        left += self.joint1_origin_m
        right += self.joint2_origin_m
        return max(0.0, float(left[:, 1].min() - right[:, 1].max()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("front_video", type=Path)
    parser.add_argument("--source-osv", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--camera-pose-csv", type=Path, required=True)
    parser.add_argument("--camera-pose-summary", type=Path, required=True)
    parser.add_argument("--force-angle-csv", type=Path, required=True)
    parser.add_argument("--force-angle-audit", type=Path, required=True)
    parser.add_argument("--rig-revision", type=Path, required=True)
    parser.add_argument("--marker-layout", type=Path, required=True)
    parser.add_argument("--new-cad-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--camera-profile",
        choices=("osmo-front", "insta360-x5-front"),
        default="osmo-front",
    )
    parser.add_argument("--allow-diagnostic-rig", action="store_true")
    return parser.parse_args()


def load_camera_track(path: Path) -> CameraTrack:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["quality_status"] == "valid"]
    if len(rows) < 2:
        raise ValueError(f"camera trajectory has only {len(rows)} valid rows")
    error_key = (
        "angular_rmse_deg"
        if "angular_rmse_deg" in rows[0]
        else "reprojection_rmse_px"
    )
    return CameraTrack(
        time_s=np.asarray([float(row["timestamp"]) for row in rows]),
        position_m=np.asarray(
            [[float(row[f"camera_{axis}_m"]) for axis in "xyz"] for row in rows]
        ),
        rotation=Rotation.from_quat(
            [[float(row[key]) for key in ("qx", "qy", "qz", "qw")] for row in rows]
        ),
        fit_error=np.asarray([float(row[error_key]) for row in rows]),
        fit_error_name=error_key,
    )


def compose_base_track(
    camera: CameraTrack, camera_base: Transform
) -> tuple[np.ndarray, Rotation]:
    positions = []
    quaternions = []
    for position, rotation in zip(camera.position_m, camera.rotation):
        world_base = Transform(position, rotation).compose(camera_base)
        positions.append(world_base.p)
        quaternions.append(world_base.r.as_quat())
    return np.asarray(positions), Rotation.from_quat(quaternions)


def sample_pose(
    time_s: np.ndarray,
    positions: np.ndarray,
    rotations: Rotation,
    query_s: np.ndarray,
) -> tuple[np.ndarray, Rotation]:
    sampled_position = np.column_stack(
        [np.interp(query_s, time_s, positions[:, axis]) for axis in range(3)]
    )
    sampled_rotation = Slerp(time_s, rotations)(
        np.clip(query_s, time_s[0], time_s[-1])
    )
    return sampled_position, sampled_rotation


def read_float(value: str) -> float:
    return float(value) if value not in (None, "") else np.nan


def load_gripper_signals(path: Path, frame_count: int) -> GripperSignals:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != frame_count:
        raise ValueError(
            f"gripper signal/video frame mismatch: {len(rows)} rows vs {frame_count} frames"
        )
    return GripperSignals(
        included_angle_deg=np.asarray(
            [read_float(row["included_jaw_angle_deg"]) for row in rows]
        ),
        opening_angle_deg=np.asarray([read_float(row["opening_angle_deg"]) for row in rows]),
        contact_intensity_percent=np.asarray(
            [read_float(row["relative_force_percent"]) for row in rows]
        ),
        measurement_state=[row["measurement_state"] for row in rows],
    )


def fill_for_display(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        raise ValueError("signal has no finite samples")
    return np.interp(np.arange(len(values)), valid, values[valid])


def make_cad_opening_model(mesh_dir: Path, geometry: dict) -> CadOpeningModel:
    left = trimesh.load_mesh(mesh_dir / "Link1.STL", process=False).vertices
    right = trimesh.load_mesh(mesh_dir / "Link2.STL", process=False).vertices
    # Only the distal pad region defines jaw opening. Gear and mount surfaces
    # are excluded because their nearest separation is not the gripped width.
    left = left[left[:, 0] > 0.075]
    right = right[right[:, 0] > 0.075]
    joint1 = np.asarray(geometry["jaw_joint_origins_m"]["joint1"], dtype=float)
    joint2 = np.asarray(geometry["jaw_joint_origins_m"]["joint2"], dtype=float)

    def signed_width(per_jaw_rotation_deg: float) -> float:
        transformed_left = Rotation.from_euler(
            "z", -per_jaw_rotation_deg, degrees=True
        ).apply(left) + joint1
        transformed_right = Rotation.from_euler(
            "z", per_jaw_rotation_deg, degrees=True
        ).apply(right) + joint2
        return float(transformed_left[:, 1].min() - transformed_right[:, 1].max())

    closed = float(brentq(signed_width, 0.0, 15.0))
    return CadOpeningModel(left, right, joint1, joint2, closed)


def path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())

def stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95.0)),
        "max": float(np.max(values)),
    }


def view_coordinates(points: np.ndarray, center: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_euler("xyz", [58.0, 0.0, 38.0], degrees=True)
    return rotation.apply(points - center)


def make_projector(
    trajectory: np.ndarray, plot_origin: tuple[int, int], plot_size: tuple[int, int]
):
    center = (trajectory.min(axis=0) + trajectory.max(axis=0)) / 2.0
    view = view_coordinates(trajectory, center)
    low = view[:, :2].min(axis=0) - 0.12
    high = view[:, :2].max(axis=0) + 0.12
    span = np.maximum(high - low, 1e-6)
    scale = 0.92 * min(plot_size[0] / span[0], plot_size[1] / span[1])
    middle = (low + high) / 2.0

    def project(points: np.ndarray) -> np.ndarray:
        value = view_coordinates(np.asarray(points), center)
        pixels = (value[:, :2] - middle) * scale
        pixels[:, 1] *= -1.0
        pixels += np.array(
            [plot_origin[0] + plot_size[0] / 2.0, plot_origin[1] + plot_size[1] / 2.0]
        )
        return pixels

    return project


def draw_world_scene(
    canvas: np.ndarray,
    projector,
    trajectory: np.ndarray,
    current_index: int,
    base_position: np.ndarray,
    base_rotation: Rotation,
    urdf: UrdfWireframe,
    joint_angles: tuple[float, float],
    plot_origin: tuple[int, int],
    plot_size: tuple[int, int],
) -> None:
    cv2.rectangle(
        canvas,
        plot_origin,
        (plot_origin[0] + plot_size[0], plot_origin[1] + plot_size[1]),
        (52, 64, 78),
        1,
    )
    trail = projector(trajectory[: current_index + 1])
    if len(trail) > 1:
        cv2.polylines(
            canvas, [np.round(trail).astype(np.int32)], False, CYAN, 3, cv2.LINE_AA
        )
    full = projector(trajectory)
    cv2.polylines(
        canvas, [np.round(full).astype(np.int32)], False, (48, 59, 72), 1, cv2.LINE_AA
    )
    edges = urdf.articulate(*joint_angles)
    world = base_rotation.apply(edges.reshape(-1, 3)) + base_position
    pixels = np.round(projector(world).reshape(edges.shape[:-1] + (2,))).astype(int)
    for first, second in pixels:
        cv2.line(canvas, tuple(first), tuple(second), AMBER, 1, cv2.LINE_AA)
    point = tuple(np.round(projector(base_position[None])[0]).astype(int))
    cv2.circle(canvas, point, 7, WHITE, -1, cv2.LINE_AA)


def draw_video_markers(
    frame: np.ndarray, camera_profile: str = "osmo-front"
) -> tuple[np.ndarray, int, int]:
    observation = observe_frame(frame, camera_profile)
    for points, color in (
        (observation.yellow_left, CYAN),
        (observation.yellow_right, GREEN),
    ):
        if points is None:
            continue
        pixels = np.round(points).astype(int)
        cv2.polylines(frame, [pixels], False, color, 7, cv2.LINE_AA)
        for point in pixels:
            cv2.circle(frame, tuple(point), 18, color, 5, cv2.LINE_AA)
    black_count = 0
    if observation.dot_left is not None:
        cv2.circle(
            frame,
            tuple(np.round(observation.dot_left.point).astype(int)),
            20,
            RED,
            5,
            cv2.LINE_AA,
        )
        black_count += 1
    if observation.dot_right is not None:
        cv2.circle(
            frame,
            tuple(np.round(observation.dot_right.point).astype(int)),
            20,
            RED,
            5,
            cv2.LINE_AA,
        )
        black_count += 1
    yellow_count = 3 * int(observation.yellow_left is not None) + 3 * int(
        observation.yellow_right is not None
    )
    return frame, yellow_count, black_count


def render(
    video: Path,
    output: Path,
    fps: float,
    positions: np.ndarray,
    rotations: Rotation,
    signals: GripperSignals,
    widths_mm: np.ndarray,
    cad_model: CadOpeningModel,
    urdf: UrdfWireframe,
    rig_id: str,
    camera_profile: str = "osmo-front",
) -> tuple[np.ndarray, np.ndarray]:
    capture = cv2.VideoCapture(str(video))
    intermediate = output.with_name(output.stem + "_mp4v.mp4")
    writer = cv2.VideoWriter(
        str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 1080)
    )
    if not writer.isOpened():
        raise ValueError(f"cannot create video: {output}")
    opening_display = fill_for_display(signals.opening_angle_deg)
    included_display = fill_for_display(signals.included_angle_deg)
    contact_display = fill_for_display(signals.contact_intensity_percent)
    projector = make_projector(positions, (1010, 95), (870, 640))
    marker_counts = []
    for index in range(len(positions)):
        ok, frame = capture.read()
        if not ok:
            break
        frame, yellow_count, black_count = draw_video_markers(frame, camera_profile)
        marker_counts.append([yellow_count, black_count])
        scale = frame.shape[1] / 1920.0
        crop = frame[
            round(850 * scale):round(1620 * scale),
            round(450 * scale):round(1470 * scale),
        ]
        crop = cv2.resize(crop, (940, 710), interpolation=cv2.INTER_AREA)
        canvas = np.full((1080, 1920, 3), BG, dtype=np.uint8)
        canvas[95:805, 30:970] = crop
        cv2.rectangle(canvas, (30, 95), (970, 805), (58, 70, 85), 2)
        draw_text(canvas, "FRONT LENS / SINGLE GRIPPER", (35, 66), 0.72, WHITE, 2)
        draw_text(canvas, "yellow triads = jaw angle   red = pad marker", (35, 840), 0.48, MUTED)

        opening = opening_display[index]
        included = included_display[index]
        intensity = contact_display[index]
        joint_angles = cad_model.joint_angles(opening)
        draw_world_scene(
            canvas,
            projector,
            positions,
            index,
            positions[index],
            rotations[index],
            urdf,
            joint_angles,
            (1010, 95),
            (870, 640),
        )
        draw_text(canvas, "TAG_MAP BASE_LINK TRAJECTORY + CURRENT EXPORTED CAD", (1020, 66), 0.60, WHITE, 2)
        draw_text(canvas, rig_id, (1020, 765), 0.40, MUTED)

        draw_text(canvas, "INCLUDED JAW ANGLE", (45, 910), 0.54, MUTED)
        draw_text(canvas, f"{included:5.1f} deg", (45, 975), 1.18, CYAN, 3)
        draw_text(canvas, "OPENING ANGLE", (315, 910), 0.54, MUTED)
        draw_text(canvas, f"{opening:5.1f} deg", (315, 975), 1.18, GREEN, 3)
        draw_text(canvas, "CAD OPENING", (585, 910), 0.54, MUTED)
        draw_text(canvas, f"{widths_mm[index]:5.1f} mm", (585, 975), 1.18, AMBER, 3)
        draw_text(canvas, "RELATIVE CONTACT INTENSITY", (1015, 835), 0.54, MUTED)
        intensity_color = GREEN if intensity < 35 else AMBER if intensity < 70 else RED
        draw_text(canvas, f"{intensity:5.1f} %", (1015, 900), 1.25, intensity_color, 3)
        cv2.rectangle(canvas, (1015, 925), (1515, 955), (47, 56, 67), -1)
        cv2.rectangle(
            canvas,
            (1015, 925),
            (1015 + round(5.0 * np.clip(intensity, 0, 100)), 955),
            intensity_color,
            -1,
        )
        draw_text(canvas, "capture-local diagnostic, not Newtons", (1015, 988), 0.46, AMBER, 2)
        position = positions[index]
        draw_text(
            canvas,
            f"base_link XYZ  {position[0]:+.3f}  {position[1]:+.3f}  {position[2]:+.3f} m",
            (1015, 1030),
            0.50,
            WHITE,
        )
        draw_text(
            canvas,
            f"t={index / fps:05.2f}s  markers {yellow_count}/6 yellow {black_count}/2 black",
            (1530, 988),
            0.38,
            MUTED,
        )
        writer.write(canvas)
    capture.release()
    writer.release()
    transcode_h264(intermediate, output)
    intermediate.unlink()
    return np.asarray(marker_counts), np.column_stack(
        [included_display, opening_display, widths_mm, contact_display]
    )


def write_timeline(
    path: Path,
    fps: float,
    positions: np.ndarray,
    rotations: Rotation,
    signals: GripperSignals,
    display_signals: np.ndarray,
    parent_frame: str,
) -> None:
    quaternions = rotations.as_quat()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame",
                "time_s",
                "base_x_m",
                "base_y_m",
                "base_z_m",
                "qx",
                "qy",
                "qz",
                "qw",
                "included_jaw_angle_deg",
                "opening_angle_deg",
                "cad_opening_width_mm",
                "contact_intensity_percent",
                "gripper_measurement_state",
                "pose_state",
                "parent_frame",
                "child_frame",
            ]
        )
        for index, (position, quaternion) in enumerate(zip(positions, quaternions)):
            writer.writerow(
                [
                    index,
                    f"{index / fps:.9f}",
                    *position,
                    *quaternion,
                    *display_signals[index],
                    signals.measurement_state[index],
                    "DIRECT_OR_SHORT_BRACKETED_VISUAL",
                    parent_frame,
                    "base_link",
                ]
            )


def main() -> int:
    args = parse_args()
    paths = {
        name: getattr(args, name).resolve(strict=True)
        for name in (
            "front_video",
            "source_osv",
            "calibration",
            "camera_pose_csv",
            "camera_pose_summary",
            "force_angle_csv",
            "force_angle_audit",
            "rig_revision",
            "marker_layout",
            "new_cad_source",
        )
    }
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
            role
            for role, robot in rig["hardware"]["robots"].items()
            if robot["camera_serial"] == serial
        ),
        None,
    )
    if role is None:
        raise ValueError(f"camera serial {serial!r} is not bound by the rig revision")
    marker_layout = json.loads(paths["marker_layout"].read_text(encoding="utf-8"))
    if sha256(Path(marker_layout["source"]["path"])) != marker_layout["source"]["sha256"]:
        raise ValueError("marker-layout source DXF hash mismatch")

    capture = cv2.VideoCapture(str(paths["front_video"]))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    frame_times = np.arange(frame_count, dtype=float) / fps
    camera = load_camera_track(paths["camera_pose_csv"])
    base_positions_direct, base_rotations_direct = compose_base_track(
        camera, camera_to_base(rig["hardware"], role)
    )
    base_positions, base_rotations = sample_pose(
        camera.time_s,
        base_positions_direct,
        base_rotations_direct,
        frame_times,
    )
    signals = load_gripper_signals(paths["force_angle_csv"], frame_count)
    opening_display = fill_for_display(signals.opening_angle_deg)

    cad = rig["cad_revision"]
    if cad is None:
        raise ValueError("rig revision does not pin a renderable CAD revision")
    root = Path(__file__).resolve().parent
    mesh_dir = (root / cad["mesh_directory"]).resolve()
    urdf_path = (root / cad["urdf"]["path"]).resolve()
    cad_model = make_cad_opening_model(mesh_dir, rig["geometry"])
    widths_mm = np.asarray([1000.0 * cad_model.width_m(value) for value in opening_display])
    urdf = load_urdf_wireframe(urdf_path, max_triangles_per_mesh=180)

    video_path = output_dir / "single_gripper_motion_force_demo.mp4"
    timeline_path = output_dir / "single_gripper_timeline.csv"
    marker_counts, display_signals = render(
        paths["front_video"],
        video_path,
        fps,
        base_positions,
        base_rotations,
        signals,
        widths_mm,
        cad_model,
        urdf,
        rig["revision"]["revision_id"],
        args.camera_profile,
    )
    write_timeline(
        timeline_path,
        fps,
        base_positions,
        base_rotations,
        signals,
        display_signals,
        rig["world_map"].get("world_frame", "tag_map"),
    )

    pose_summary = json.loads(paths["camera_pose_summary"].read_text(encoding="utf-8"))
    force_audit = json.loads(paths["force_angle_audit"].read_text(encoding="utf-8"))
    direct_pose_hz = float(1.0 / np.median(np.diff(camera.time_s)))
    audit = {
        "schema_version": "single-gripper-motion-force-demo/1.0",
        "status": "DIAGNOSTIC",
        "source": {
            "osv": str(paths["source_osv"]),
            "osv_sha256": sha256(paths["source_osv"]),
            "front_lens": str(paths["front_video"]),
            "front_lens_sha256": sha256(paths["front_video"]),
            "camera_serial": serial,
            "hardware_role": role,
            "base_tag_id": rig["hardware"]["robots"][role]["base_tag_id"],
            "frame_count": frame_count,
            "fps": fps,
        },
        "rig": {
            "revision_id": rig["revision"]["revision_id"],
            "revision_sha256": rig["revision_sha256"],
            "rendered_cad_revision_id": cad["revision_id"],
            "newest_editable_source": str(paths["new_cad_source"]),
            "newest_editable_source_sha256": sha256(paths["new_cad_source"]),
            "warning": "UMI-III is source-only; rendered v52 meshes are not yet proven geometry-identical to it.",
        },
        "marker_layout": {
            "revision_id": marker_layout["revision_id"],
            "path": str(paths["marker_layout"]),
            "sha256": sha256(paths["marker_layout"]),
            "source_dxf_sha256": marker_layout["source"]["sha256"],
        },
        "trajectory": {
            "frame": pose_summary.get("parent_frame", "tag_map"),
            "reference": "base_link",
            "direct_pose_frames": len(camera.time_s),
            "direct_pose_valid_ratio": pose_summary.get(
                "valid_ratio", pose_summary.get("valid_pose_ratio")
            ),
            "direct_pose_rejected_frames": (
                pose_summary.get("common_frames", pose_summary.get("processed_frames"))
                - pose_summary.get("valid_frames", pose_summary.get("valid_pose_frames"))
            ),
            camera.fit_error_name: stats(camera.fit_error),
            "position_step_m": stats(
                np.linalg.norm(np.diff(base_positions_direct, axis=0), axis=1)
            ),
            "orientation_step_deg": stats(
                np.degrees(
                    (base_rotations_direct[:-1].inv() * base_rotations_direct[1:]).magnitude()
                )
            ),
            "path_length_m": path_length(base_positions_direct),
            "start_to_end_displacement_m": float(
                np.linalg.norm(base_positions_direct[-1] - base_positions_direct[0])
            ),
            "render_sampling": f"SLERP/linear between {direct_pose_hz:.3f} Hz visual poses",
            "holdout_status": "NONE_DIAGNOSTIC_CAPTURE_ONLY",
        },
        "jaw": {
            "included_angle_range_deg": [
                float(np.nanmin(display_signals[:, 0])),
                float(np.nanmax(display_signals[:, 0])),
            ],
            "opening_angle_range_deg": [
                float(np.nanmin(display_signals[:, 1])),
                float(np.nanmax(display_signals[:, 1])),
            ],
            "cad_width_range_mm": [
                float(np.nanmin(display_signals[:, 2])),
                float(np.nanmax(display_signals[:, 2])),
            ],
            "closed_per_jaw_rotation_deg": cad_model.closed_joint_rotation_deg,
            "warning": "CAD width uses the rendered v52 distal-pad geometry and is diagnostic pending UMI-III export.",
        },
        "contact_intensity": {
            "range_percent": [
                float(np.nanmin(display_signals[:, 3])),
                float(np.nanmax(display_signals[:, 3])),
            ],
            "yellow_direct_ratio": float(np.mean(marker_counts[:, 0] == 6)),
            "tracked_black_pair_ratio": float(np.mean(marker_counts[:, 1] == 2)),
            "source_force_audit": str(paths["force_angle_audit"]),
            "capture_local_normalization": force_audit["force"]["full_scale_99th_percentile"],
            "warning": "Capture-local pad deformation proxy; not Newtons and not cross-episode calibrated.",
        },
        "outputs": {
            "video": str(video_path),
            "video_sha256": sha256(video_path),
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
