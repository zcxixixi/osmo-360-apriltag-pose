#!/usr/bin/env python3
"""Visualize an AprilGrid pose from an Osmo 360 RTSP/RTMP video stream."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


DEFAULT_SOURCE = "rtsp://127.0.0.1:8554/osmo/live"


@dataclass(frozen=True)
class Grid:
    rows: int
    cols: int
    tag_size: float
    spacing_ratio: float
    first_id: int = 0
    id_order: str = "column-major"

    @property
    def pitch(self) -> float:
        return self.tag_size * (1.0 + self.spacing_ratio)

    @property
    def width(self) -> float:
        return (self.cols - 1) * self.pitch + self.tag_size

    @property
    def height(self) -> float:
        return (self.rows - 1) * self.pitch + self.tag_size

    def corners(self, tag_id: int) -> np.ndarray | None:
        index = tag_id - self.first_id
        if index < 0 or index >= self.rows * self.cols:
            return None
        if self.id_order == "column-major":
            # Kalibr AprilGrid IDs advance down each column, then move right.
            col, row = divmod(index, self.rows)
        elif self.id_order == "row-major":
            # Wall-panel PDFs advance left-to-right, then top-to-bottom.
            row, col = divmod(index, self.cols)
        else:
            raise ValueError(f"unsupported AprilGrid ID order: {self.id_order}")
        x0 = col * self.pitch - self.width / 2.0
        y0 = self.height / 2.0 - row * self.pitch
        s = self.tag_size
        # OpenCV marker order: top-left, top-right, bottom-right, bottom-left.
        # Board frame: X right, Y up, Z out of the printed board.
        return np.array(
            [
                [x0, y0, 0.0],
                [x0 + s, y0, 0.0],
                [x0 + s, y0 - s, 0.0],
                [x0, y0 - s, 0.0],
            ],
            dtype=np.float32,
        )

    def center(self, tag_id: int) -> np.ndarray | None:
        marker_corners = self.corners(tag_id)
        if marker_corners is None:
            return None
        return marker_corners.mean(axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect a 6x6 AprilGrid and visualize camera-relative coordinates."
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="RTSP/RTMP URL or video file")
    parser.add_argument("--tag-size", type=float, required=True, help="black tag edge length in meters")
    parser.add_argument("--spacing", type=float, default=0.30, help="gap/tag-size ratio (default: 0.30)")
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--first-id", type=int, default=0)
    parser.add_argument("--min-tags", type=int, default=6)
    parser.add_argument("--camera-yaml", type=Path, help="OpenCV YAML containing camera_matrix and dist_coeff")
    parser.add_argument("--hfov", type=float, default=90.0, help="fallback horizontal FOV in degrees")
    parser.add_argument("--csv", type=Path, default=Path("pose.csv"))
    parser.add_argument("--jsonl", type=Path, help="write one detection record for every frame")
    parser.add_argument("--summary", type=Path, help="write session coverage summary on exit")
    parser.add_argument("--live-status", type=Path, help="periodically update a small live status JSON")
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


def load_calibration(path: Path) -> tuple[np.ndarray, np.ndarray]:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"cannot open calibration file: {path}")
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeff = fs.getNode("dist_coeff").mat()
    fs.release()
    if camera_matrix is None or camera_matrix.shape != (3, 3):
        raise RuntimeError("camera_yaml must contain a 3x3 camera_matrix")
    if dist_coeff is None:
        dist_coeff = np.zeros((5, 1), dtype=np.float64)
    return camera_matrix.astype(np.float64), dist_coeff.astype(np.float64)


def approximate_calibration(width: int, height: int, hfov_deg: float) -> tuple[np.ndarray, np.ndarray]:
    focal = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    camera_matrix = np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return camera_matrix, np.zeros((5, 1), dtype=np.float64)


def rotation_to_rpy(rotation: np.ndarray) -> tuple[float, float, float]:
    sy = math.hypot(rotation[0, 0], rotation[1, 0])
    singular = sy < 1e-8
    if not singular:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = 0.0
    return tuple(math.degrees(v) for v in (roll, pitch, yaw))


def draw_text(frame: np.ndarray, lines: list[str], good: bool) -> None:
    overlay = frame.copy()
    box_height = 32 + 27 * len(lines)
    cv2.rectangle(overlay, (12, 12), (610, box_height), (8, 8, 8), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)
    color = (80, 230, 100) if good else (60, 170, 255)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (26, 44 + i * 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


def open_capture(source: str) -> cv2.VideoCapture:
    if source.startswith("rtsp://"):
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def main() -> int:
    args = parse_args()
    grid = Grid(args.rows, args.cols, args.tag_size, args.spacing, args.first_id)
    capture = open_capture(args.source)
    if not capture.isOpened():
        print(f"Cannot open stream: {args.source}", file=sys.stderr)
        print("Start Osmo 360 RTMP livestream first, then retry.", file=sys.stderr)
        return 2

    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    # The RTMP feed is compressed and the grid can occupy most of a 720p frame.
    # Widen the default threshold/perimeter search range for this use case.
    detector_params.adaptiveThreshWinSizeMin = 3
    detector_params.adaptiveThreshWinSizeMax = 83
    detector_params.adaptiveThreshWinSizeStep = 4
    detector_params.minMarkerPerimeterRate = 0.01
    detector_params.maxMarkerPerimeterRate = 8.0
    # Reject weak decodes caused by the compressed 720p livestream.
    detector_params.errorCorrectionRate = 0.4
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    ok, frame = capture.read()
    if not ok:
        print("Stream opened but no frame arrived.", file=sys.stderr)
        return 3
    height, width = frame.shape[:2]
    calibrated = args.camera_yaml is not None
    if calibrated:
        camera_matrix, dist_coeff = load_calibration(args.camera_yaml)
    else:
        camera_matrix, dist_coeff = approximate_calibration(width, height, args.hfov)
        print("WARNING: using approximate intrinsics; XYZ is demo-grade only.", file=sys.stderr)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    csv_file = args.csv.open("w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["frame", "unix_time", "camera_x_m", "camera_y_m", "camera_z_m", "roll_deg", "pitch_deg", "yaw_deg", "tags", "inliers", "reprojection_rmse_px", "ids"])
    csv_file.flush()

    jsonl_file = None
    if args.jsonl:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        jsonl_file = args.jsonl.open("w", encoding="utf-8", buffering=1)

    previous_rvec: np.ndarray | None = None
    previous_tvec: np.ndarray | None = None
    frame_index = 0
    pose_frames = 0
    seen_counts: dict[int, int] = {}
    try:
        while ok:
            frame_time = time.time()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            object_points: list[np.ndarray] = []
            image_points: list[np.ndarray] = []
            used_ids: list[int] = []
            frame_detections: list[dict[str, object]] = []

            if ids is not None:
                if not args.no_display:
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                for marker_corners, raw_id in zip(corners, ids.flatten()):
                    board_center = grid.center(int(raw_id))
                    if board_center is None:
                        continue
                    marker_id = int(raw_id)
                    pixel_corners = marker_corners.reshape(4, 2).astype(np.float32)
                    # Pose is solved from marker centers. This avoids depending on
                    # per-marker corner orientation while RANSAC rejects wrong IDs.
                    object_points.append(board_center)
                    image_points.append(pixel_corners.mean(axis=0))
                    used_ids.append(marker_id)
                    frame_detections.append({"id": marker_id, "corners_px": pixel_corners.round(3).tolist()})
                    seen_counts[marker_id] = seen_counts.get(marker_id, 0) + 1

            pose_ok = False
            pose_record: dict[str, object] | None = None
            if len(used_ids) >= args.min_tags:
                obj = np.asarray(object_points, dtype=np.float32)
                img = np.asarray(image_points, dtype=np.float32)
                pose_ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                    obj,
                    img,
                    camera_matrix,
                    dist_coeff,
                    rvec=previous_rvec,
                    tvec=previous_tvec,
                    useExtrinsicGuess=previous_rvec is not None,
                    iterationsCount=120,
                    reprojectionError=4.0,
                    confidence=0.995,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if pose_ok and inliers is not None and len(inliers) >= args.min_tags:
                    rvec, tvec = cv2.solvePnPRefineLM(obj[inliers[:, 0]], img[inliers[:, 0]], camera_matrix, dist_coeff, rvec, tvec)
                    previous_rvec, previous_tvec = rvec.copy(), tvec.copy()
                    if not args.no_display:
                        cv2.drawFrameAxes(frame, camera_matrix, dist_coeff, rvec, tvec, grid.tag_size * 2.0, 3)

                    board_to_camera, _ = cv2.Rodrigues(rvec)
                    camera_to_board = board_to_camera.T
                    camera_position = (-camera_to_board @ tvec).reshape(3)
                    roll, pitch, yaw = rotation_to_rpy(camera_to_board)
                    projected, _ = cv2.projectPoints(obj[inliers[:, 0]], rvec, tvec, camera_matrix, dist_coeff)
                    residual = projected.reshape(-1, 2) - img[inliers[:, 0]]
                    reprojection_rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
                    pose_record = {
                        "camera_xyz_m": camera_position.round(7).tolist(),
                        "camera_rpy_deg": [round(roll, 4), round(pitch, 4), round(yaw, 4)],
                        "inliers": int(len(inliers)),
                        "reprojection_rmse_px": round(reprojection_rmse, 4),
                    }
                    writer.writerow([frame_index, f"{frame_time:.6f}", *[f"{v:.6f}" for v in camera_position], f"{roll:.3f}", f"{pitch:.3f}", f"{yaw:.3f}", len(used_ids), len(inliers), f"{reprojection_rmse:.4f}", " ".join(map(str, sorted(used_ids)))])
                    csv_file.flush()
                    pose_frames += 1
                    accuracy = "CALIBRATED" if calibrated else "APPROX INTRINSICS"
                    draw_text(
                        frame,
                        [
                            f"Camera in AprilGrid frame ({accuracy})",
                            f"X {camera_position[0]:+.3f} m   Y {camera_position[1]:+.3f} m   Z {camera_position[2]:+.3f} m",
                            f"Roll {roll:+.1f}   Pitch {pitch:+.1f}   Yaw {yaw:+.1f} deg",
                            f"Tags: {len(used_ids)}  IDs: {used_ids[:12]}",
                        ],
                        True,
                    )
                else:
                    pose_ok = False

            if not pose_ok:
                if not args.no_display:
                    draw_text(frame, [f"Waiting for AprilGrid: {len(used_ids)}/{args.min_tags} tags", "Family: tag36h11"], False)

            if jsonl_file:
                jsonl_file.write(json.dumps({
                    "frame": frame_index,
                    "unix_time": round(frame_time, 6),
                    "detections": frame_detections,
                    "pose": pose_record,
                }, separators=(",", ":")) + "\n")

            if args.no_display and frame_index % 150 == 0:
                print(f"frame={frame_index} tags={len(used_ids)} seen={sorted(seen_counts)} pose_frames={pose_frames}", flush=True)

            if args.live_status and frame_index % 30 == 0:
                args.live_status.parent.mkdir(parents=True, exist_ok=True)
                args.live_status.write_text(json.dumps({
                    "running": True,
                    "frame": frame_index,
                    "current_ids": sorted(used_ids),
                    "seen_ids": sorted(seen_counts),
                    "pose_frames": pose_frames,
                }), encoding="utf-8")

            if not args.no_display:
                cv2.imshow("Osmo 360 - AprilGrid Pose", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
            frame_index += 1
            ok, frame = capture.read()
    except KeyboardInterrupt:
        pass
    finally:
        summary = {
            "frames": frame_index,
            "pose_frames": pose_frames,
            "seen_ids": sorted(seen_counts),
            "missing_ids": [tag_id for tag_id in range(grid.first_id, grid.first_id + grid.rows * grid.cols) if tag_id not in seen_counts],
            "detections_per_id": {str(key): value for key, value in sorted(seen_counts.items())},
            "assumed_tag_size_m": grid.tag_size,
            "spacing_ratio": grid.spacing_ratio,
            "calibrated_intrinsics": calibrated,
        }
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if args.live_status:
            args.live_status.write_text(json.dumps({"running": False, **summary}), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        if jsonl_file:
            jsonl_file.close()
        csv_file.close()
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
