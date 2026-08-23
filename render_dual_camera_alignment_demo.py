#!/usr/bin/env python3
"""Render synchronized 6DoF estimates with URDF grippers in one frame."""

from __future__ import annotations

import argparse
import csv
import json
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
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
    parser.add_argument("--gripper-mesh-dir", type=Path)
    parser.add_argument("--gripper-urdf", type=Path)
    parser.add_argument(
        "--camera-to-gripper-json", type=Path,
        help="fixed T_camera_gripper applied before drawing the URDF",
    )
    parser.add_argument("--left-camera-to-gripper-json", type=Path)
    parser.add_argument("--right-camera-to-gripper-json", type=Path)
    parser.add_argument("--left-claw-angle-csv", type=Path)
    parser.add_argument("--right-claw-angle-csv", type=Path)
    parser.add_argument(
        "--gripper-layout-json", type=Path,
        help="manual center-frame starting poses; each trajectory is rebased to this frame",
    )
    parser.add_argument(
        "--right-gripper-yaw-deg", type=float, default=0.0,
        help="right gripper mounting yaw relative to the calibrated left mounting",
    )
    parser.add_argument(
        "--coordinate-frame", choices=("left-aligned", "board"),
        default="left-aligned",
        help="board keeps both raw poses in their shared AprilGrid frame",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-duration", type=float, default=8.0)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    return parser.parse_args()


@dataclass
class UrdfWireframe:
    links: dict[str, np.ndarray]
    joints: dict[str, tuple[str, np.ndarray, np.ndarray]]

    def articulate(self, joint1_deg: float = 0.0, joint2_deg: float = 0.0) -> np.ndarray:
        angles = {"joint1": joint1_deg, "joint2": joint2_deg}
        transforms = {"base_link": np.eye(4)}
        pending = dict(self.joints)
        while pending:
            changed = False
            for child, (parent, local, axis) in list(pending.items()):
                if parent not in transforms:
                    continue
                angle = angles.get(child.replace("Link", "joint"), 0.0)
                motion = np.eye(4)
                motion[:3, :3] = Rotation.from_rotvec(
                    np.radians(angle) * axis
                ).as_matrix()
                transforms[child] = transforms[parent] @ local @ motion
                del pending[child]
                changed = True
            if not changed:
                break
        result = []
        for link, edges in self.links.items():
            transform = transforms.get(link)
            if transform is None:
                continue
            result.append(edges @ transform[:3, :3].T + transform[:3, 3])
        if not result:
            raise ValueError("URDF has no transformable link edges")
        return np.concatenate(result)


def load_urdf_wireframe(path: Path, max_triangles_per_mesh: int = 100) -> UrdfWireframe:
    """Load native-link wireframes and the URDF kinematic tree."""
    root = ET.parse(path).getroot()
    joints: dict[str, tuple[str, np.ndarray, np.ndarray]] = {}
    for joint in root.findall("joint"):
        child = joint.find("child").attrib["link"]
        parent = joint.find("parent").attrib["link"]
        origin = joint.find("origin")
        xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
        rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
        axis_node = joint.find("axis")
        axis = np.fromstring(
            axis_node.attrib.get("xyz", "0 0 0") if axis_node is not None else "0 0 0",
            sep=" ",
        )
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
        transform[:3, 3] = xyz
        joints[child] = (parent, transform, axis)

    links = {}
    mesh_root = path.parent.parent / "meshes"
    for link in root.findall("link"):
        visual = link.find("visual")
        if visual is None:
            continue
        mesh = visual.find("./geometry/mesh")
        if mesh is None:
            continue
        data = (mesh_root / Path(mesh.attrib["filename"]).name).read_bytes()
        count = struct.unpack_from("<I", data, 80)[0]
        dtype = np.dtype([
            ("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)),
            ("attribute", "<u2"),
        ])
        triangles = np.frombuffer(data, dtype=dtype, offset=84, count=count)["vertices"].astype(float)
        indices = np.unique(np.linspace(
            0, len(triangles) - 1, min(max_triangles_per_mesh, len(triangles))
        ).astype(int))
        vertices = triangles[indices]
        links[link.attrib["name"]] = np.concatenate(
            (vertices[:, [0, 1]], vertices[:, [1, 2]], vertices[:, [2, 0]])
        )
    if not links:
        raise ValueError(f"URDF contains no renderable STL meshes: {path}")
    return UrdfWireframe(links, joints)


def load_claw_angles(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    values = []
    neutral_opening_deg = abs(
        np.degrees(np.arctan2(50.568, 63.276) - np.arctan2(-50.745, 63.134))
    )
    for row in rows:
        opening = float(row["opening_angle_deg"])
        if row.get("joint1_deg") not in (None, ""):
            joint1, joint2 = float(row["joint1_deg"]), float(row["joint2_deg"])
            measured = float(row.get("measured", "1"))
        else:
            # The fitted-pivot solver returns opening only. Drive the symmetric
            # CAD mechanism around its zero-angle opening configuration.
            travel = (neutral_opening_deg - opening) / 2.0
            joint1, joint2 = -travel, travel
            measured = float(np.isfinite(float(row.get("raw_angle_deg", "nan"))))
        values.append([float(row["time_s"]), opening, joint1, joint2,
                       measured, float(row.get("confidence", "1"))])
    return np.asarray(values, dtype=float)


def sample_claw_angles(data: np.ndarray, now: float) -> tuple[np.ndarray, bool, float]:
    values = np.asarray([np.interp(now, data[:, 0], data[:, column]) for column in (1, 2, 3)])
    index = int(np.argmin(np.abs(data[:, 0] - now)))
    return values, bool(data[index, 4] >= 0.5), float(data[index, 5])


def load_track(
    path: Path, prefix: str, camera_to_gripper_json: Path | None = None,
    mounting_yaw_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, Rotation]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    times = np.asarray([float(row["left_timestamp_s"]) for row in rows])
    positions = np.asarray([
        [float(row[f"{prefix}_{axis}_m"]) for axis in "xyz"] for row in rows
    ])
    rotations = Rotation.from_quat(np.asarray([
        [float(row[f"{prefix}_q{axis}"]) for axis in "xyzw"] for row in rows
    ]))
    if camera_to_gripper_json:
        extrinsic = json.loads(camera_to_gripper_json.read_text(encoding="utf-8"))
        rotation_camera_gripper = Rotation.from_matrix(np.asarray(
            extrinsic["rotation_gripper_to_camera"], dtype=float,
        ))
        translation_camera_gripper = np.asarray(
            extrinsic["translation_gripper_origin_in_camera_m"], dtype=float,
        )
        if translation_camera_gripper.shape != (3,):
            raise ValueError("invalid camera-to-gripper translation")
        positions = positions + rotations.apply(translation_camera_gripper)
        rotations = rotations * rotation_camera_gripper
    if mounting_yaw_deg:
        rotations = rotations * Rotation.from_euler("z", mounting_yaw_deg, degrees=True)
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


def rebase_track_to_start_layout(
    positions: np.ndarray, rotations: Rotation, target: dict,
) -> tuple[np.ndarray, Rotation]:
    """Preserve measured motion while replacing the track's starting pose."""
    target_position = np.asarray(target["translation_m"], dtype=float)
    target_rotation = Rotation.from_euler(
        "xyz", np.asarray(target["rotation_rpy_deg"], dtype=float), degrees=True,
    )
    if target_position.shape != (3,):
        raise ValueError("layout translation must contain XYZ")
    start_position = positions[0]
    start_rotation = rotations[0]
    relative_positions = start_rotation.inv().apply(positions - start_position)
    rebased_positions = target_position + target_rotation.apply(relative_positions)
    rebased_rotations = target_rotation * start_rotation.inv() * rotations
    return rebased_positions, rebased_rotations


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
    if bool(args.gripper_mesh_dir) == bool(args.gripper_urdf):
        raise SystemExit("supply exactly one of --gripper-mesh-dir or --gripper-urdf")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    left_extrinsic = args.left_camera_to_gripper_json or args.camera_to_gripper_json
    right_extrinsic = args.right_camera_to_gripper_json or args.camera_to_gripper_json
    left_t, left_p, left_r = load_track(args.aligned_csv, "left", left_extrinsic)
    right_prefix = "right_raw" if args.coordinate_frame == "board" else "right_aligned"
    right_t, right_p, right_r = load_track(
        args.aligned_csv, right_prefix, right_extrinsic,
        args.right_gripper_yaw_deg,
    )
    if args.gripper_layout_json:
        layout = json.loads(args.gripper_layout_json.read_text(encoding="utf-8"))
        center_poses = layout["grippers_in_center_frame"]
        left_p, left_r = rebase_track_to_start_layout(left_p, left_r, center_poses["left"])
        right_p, right_r = rebase_track_to_start_layout(right_p, right_r, center_poses["right"])
    start = float(max(left_t[0], right_t[0]))
    end = float(min(left_t[-1], right_t[-1], start + args.max_duration))
    if end <= start:
        raise SystemExit("no shared render interval")
    # The AprilGrid origin is part of the visualization, even though both
    # cameras normally remain half a metre or more in front of the board.
    all_points = np.vstack((left_p, right_p, np.zeros((1, 3))))
    low, high = np.percentile(all_points, [1, 99], axis=0)
    low = np.minimum(low, 0.0)
    high = np.maximum(high, 0.0)
    center = (low + high) / 2.0
    radius = max(float(np.max(high - low)) / 2.0 + 0.12, 0.30)
    low, high = center - radius, center + radius
    # The CAD is an orientation glyph here, not a metrically scaled obstacle.
    # Keep it compact enough that both poses remain readable in one plot.
    urdf_model = load_urdf_wireframe(args.gripper_urdf) if args.gripper_urdf else None
    edges = urdf_model.articulate() if urdf_model else load_gripper_edges(args.gripper_mesh_dir) * 0.45
    angle_args = (args.left_claw_angle_csv, args.right_claw_angle_csv)
    if bool(angle_args[0]) != bool(angle_args[1]):
        raise SystemExit("supply both --left-claw-angle-csv and --right-claw-angle-csv")
    if angle_args[0] and urdf_model is None:
        raise SystemExit("animated claw angles require --gripper-urdf")
    left_claw = load_claw_angles(angle_args[0]) if angle_args[0] else None
    right_claw = load_claw_angles(angle_args[1]) if angle_args[1] else None
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
        left_angle = sample_claw_angles(left_claw, now) if left_claw is not None else None
        right_angle = sample_claw_angles(right_claw, now + offset) if right_claw is not None else None
        left_edges = (
            urdf_model.articulate(left_angle[0][1], left_angle[0][2])
            if urdf_model and left_angle else edges
        )
        right_edges = (
            urdf_model.articulate(right_angle[0][1], right_angle[0][2])
            if urdf_model and right_angle else edges
        )
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
        # AprilGrid plane (Zb=0). It is the common metric reference, not a
        # decorative camera-local grid.
        grid_color = (47, 52, 61)
        for value in np.linspace(low[0], high[0], 9):
            first = project(np.asarray([value, low[1], 0.0]), low, high, plot_origin, plot_size)
            second = project(np.asarray([value, high[1], 0.0]), low, high, plot_origin, plot_size)
            cv2.line(canvas, first, second, grid_color, 1, cv2.LINE_AA)
        for value in np.linspace(low[1], high[1], 9):
            first = project(np.asarray([low[0], value, 0.0]), low, high, plot_origin, plot_size)
            second = project(np.asarray([high[0], value, 0.0]), low, high, plot_origin, plot_size)
            cv2.line(canvas, first, second, grid_color, 1, cv2.LINE_AA)
        upto = int(np.searchsorted(left_t, now, side="right"))
        for points, color in ((left_p[:upto], CYAN), (right_p[:upto], GREEN)):
            if len(points) > 1:
                pixels = np.asarray([project(p, low, high, plot_origin, plot_size) for p in points], np.int32)
                cv2.polylines(canvas, [pixels], False, color, 2, cv2.LINE_AA)
        draw_gripper(canvas, left_pos, left_quat, left_edges, low, high,
                     plot_origin, plot_size, CYAN, "LEFT gripper", -14)
        right_label = "RIGHT gripper (board)" if args.coordinate_frame == "board" else "RIGHT aligned gripper"
        draw_gripper(canvas, right_pos, right_quat, right_edges, low, high,
                     plot_origin, plot_size, GREEN, right_label, 22)

        # Make the relationship explicit: both glyphs live in the same metric
        # frame and this segment is their instantaneous relative displacement.
        left_px = project(left_pos, low, high, plot_origin, plot_size)
        right_px = project(right_pos, low, high, plot_origin, plot_size)
        cv2.line(canvas, left_px, right_px, AMBER, 2, cv2.LINE_AA)
        midpoint = ((left_px[0] + right_px[0]) // 2, (left_px[1] + right_px[1]) // 2)
        separation_mm = 1000.0 * float(np.linalg.norm(right_pos - left_pos))
        text(canvas, f"delta {separation_mm:.1f} mm", (midpoint[0] + 5, midpoint[1] - 6),
             0.42, AMBER, 1)

        origin_px = project(np.zeros(3), low, high, plot_origin, plot_size)
        cv2.circle(canvas, origin_px, 5, WHITE, -1, cv2.LINE_AA)
        axis_length = min(0.12, radius * 0.30)
        for vector, color, label in (
            (np.asarray([axis_length, 0.0, 0.0]), RED, "Xb"),
            (np.asarray([0.0, axis_length, 0.0]), GREEN, "Yb"),
            (np.asarray([0.0, 0.0, axis_length]), CYAN, "Zb"),
        ):
            endpoint = project(vector, low, high, plot_origin, plot_size)
            cv2.arrowedLine(canvas, origin_px, endpoint, color, 2, cv2.LINE_AA, tipLength=0.20)
            text(canvas, label, (endpoint[0] + 4, endpoint[1] - 4), 0.40, color, 1)
        frame_title = "APRILGRID BOARD FRAME + URDF GRIPPERS" if args.coordinate_frame == "board" else "LEFT-FRAME 6DoF + CAD GRIPPER"
        text(canvas, frame_title, (55, 585), 0.64, WHITE, 2)
        has_gripper_extrinsic = bool(left_extrinsic or right_extrinsic)
        glyph_note = (
            "URDF + calibrated camera-to-gripper extrinsic"
            if args.gripper_urdf and has_gripper_extrinsic else
            "URDF zero-joint assembly, metric STL scale"
            if args.gripper_urdf else "gripper glyph scale 45% (orientation display)"
        )
        text(canvas, glyph_note, (55, 1018), 0.42, MUTED)

        panel_x = 1320
        panel_title = "SHARED BOARD 6DoF" if args.coordinate_frame == "board" else "DUAL CAMERA ALIGNMENT"
        text(canvas, panel_title, (panel_x, 585), 0.70, WHITE, 2)
        text(canvas, "DIRECT POSES / REVIEW", (panel_x, 622), 0.62, AMBER, 2)
        text(canvas, f"pair  {report['capture_pair_id']}", (panel_x, 658), 0.43, MUTED)
        text(canvas, f"audio offset  {offset:+.4f} s", (panel_x, 700), 0.56, WHITE)
        mount_label = "manual start layout" if args.gripper_layout_json else f"R mount {args.right_gripper_yaw_deg:+.0f} deg"
        text(canvas, f"audio corr {report['time_alignment']['correlation']:.3f}  |  {mount_label}",
             (panel_x, 732), 0.47, WHITE)
        pos = report["position_residual_m"]
        ori = report["orientation_residual_deg"]
        text(canvas, f"position P95  {1000.0 * pos['p95']:.1f} mm", (panel_x, 780), 0.56, AMBER)
        text(canvas, f"attitude P95  {ori['p95']:.1f} deg", (panel_x, 812), 0.56, AMBER)
        text(canvas, f"live delta    {separation_mm:.1f} mm", (panel_x, 844), 0.56, WHITE)
        if left_angle and right_angle:
            left_state = "M" if left_angle[1] else "R"
            right_state = "M" if right_angle[1] else "R"
            text(canvas, f"opening L {left_angle[0][0]:5.1f} deg [{left_state}]",
                 (panel_x, 874), 0.52, CYAN)
            text(canvas, f"opening R {right_angle[0][0]:5.1f} deg [{right_state}]",
                 (panel_x, 904), 0.52, GREEN)
        state = "MEASURED" if max(left_age, right_age) <= 0.25 else "PREDICTED / SPARSE"
        state_color = GREEN if state == "MEASURED" else AMBER
        state_y = 940 if left_angle and right_angle else 882
        text(canvas, state, (panel_x, state_y), 0.62, state_color, 2)
        text(canvas, "Kalman + RTS visualization", (panel_x, state_y + 38), 0.52, MUTED)
        text(canvas, "Raw poses share the AprilGrid board frame", (panel_x, state_y + 70), 0.45, MUTED)
        time_y = 1050 if left_angle and right_angle else 990
        text(canvas, f"t={now-start:05.2f}s / {end-start:05.2f}s", (panel_x, time_y), 0.56, WHITE)
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
