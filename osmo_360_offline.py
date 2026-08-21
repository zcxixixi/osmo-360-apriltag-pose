#!/usr/bin/env python3
"""Offline AprilGrid pose estimation for stitched 2:1 equirectangular video.

This deliberately does not implement DJI's private OSV stitching.  Feed it a
DJI Studio export, or an independently validated OSV conversion.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import shutil
import signal
import subprocess
import threading
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from osmo_apriltag_demo import Grid, rotation_to_rpy
from projection_backends import ProjectionRequest, make_projection_backend

LOG = logging.getLogger("osmo360.offline")
STOP = False


CAMERA_MODELS = ("auto", "dji-osmo-360", "insta360-x6", "generic")


def infer_camera_model(path: Path, width: int, height: int, requested: str) -> str:
    """Choose a conservative processing profile from the export name and size."""
    if requested != "auto":
        return requested
    name = path.name.lower()
    if "insta360" in name or "_no_flowstate" in name or (
        name.startswith("vid_") and width >= 6000
    ):
        return "insta360-x6"
    if "osmo" in name or name.startswith("cam_") or path.suffix.lower() == ".osv":
        return "dji-osmo-360"
    if width >= 7000 and width == 2 * height:
        return "insta360-x6"
    if width <= 4096 and width == 2 * height:
        return "dji-osmo-360"
    return "generic"


def resolve_decoder(requested: str, camera_model: str) -> str:
    """Resolve decoding independently from projection acceleration.

    Downloading NVDEC frames back to BGR is slower than OpenCV/FFmpeg CPU
    decoding on the validated DJI 3K and Insta360 X6 8K exports. Keep NVDEC as
    an explicit experiment until a zero-copy detector path is available.
    """
    if requested != "auto":
        return requested
    if camera_model in ("dji-osmo-360", "insta360-x6", "generic"):
        return "cpu"
    raise ValueError(f"unknown camera model: {camera_model}")


def resolve_projection(requested: str, camera_model: str, width: int) -> str:
    """Select the measured fastest projection path for the input profile."""
    if requested != "auto":
        return requested
    if camera_model == "dji-osmo-360" and width <= 4096:
        return "cpu"
    return "cuda"


class VideoReader:
    """Sequential OpenCV-compatible reader with optional NVIDIA NVDEC.

    NVDEC performs HEVC entropy decoding on the GPU. Frames are downloaded as
    BGR because AprilTag still runs on the CPU. It is opt-in: measured auto
    profiles use CPU decoding until the rest of the pipeline can stay on-GPU.
    """

    def __init__(
        self, path: Path, decoder: str, ffmpeg_bin: Path | None,
        requested_camera_model: str = "auto",
    ):
        self.path = path
        self.decoder = "cpu"
        self._process: subprocess.Popen | None = None
        self._capture: cv2.VideoCapture | None = None
        probe = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
        if not probe.isOpened():
            raise RuntimeError(f"cannot open input: {path}")
        self.width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(probe.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
        probe.release()
        self.camera_model = infer_camera_model(
            path, self.width, self.height, requested_camera_model
        )
        selected_decoder = resolve_decoder(decoder, self.camera_model)
        if selected_decoder == "nvdec":
            if ffmpeg_bin is None:
                raise RuntimeError("NVDEC requested but no FFmpeg binary is available")
            if self._start_nvdec(ffmpeg_bin):
                self.decoder = "nvdec"
                return
            raise RuntimeError("NVDEC requested but CUDA decode validation failed")
        self._capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
        if not self._capture.isOpened():
            raise RuntimeError(f"cannot open input: {path}")

    def _start_nvdec(self, ffmpeg_bin: Path) -> bool:
        ffprobe = ffmpeg_bin.with_name("ffprobe")
        if not ffmpeg_bin.exists() or not ffprobe.exists():
            return False
        try:
            pixel_format = subprocess.check_output(
                [
                    str(ffprobe), "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=pix_fmt", "-of", "default=nw=1:nk=1",
                    str(self.path),
                ],
                text=True,
                timeout=15,
            ).strip()
            download_format = "p010le" if "10" in pixel_format else "nv12"
            validation = subprocess.run(
                [
                    str(ffmpeg_bin), "-v", "error", "-hwaccel", "cuda",
                    "-hwaccel_output_format", "cuda", "-i", str(self.path),
                    "-map", "0:v:0", "-frames:v", "1", "-an", "-sn",
                    "-vf", f"hwdownload,format={download_format}", "-f", "null", "-",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
            )
            if validation.returncode:
                return False
            command = [
                str(ffmpeg_bin), "-v", "error", "-hwaccel", "cuda",
                "-hwaccel_output_format", "cuda", "-i", str(self.path),
                "-map", "0:v:0", "-an", "-sn",
                "-vf", f"hwdownload,format={download_format},format=bgr24",
                "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1",
            ]
            self._process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            return self._process.stdout is not None
        except (OSError, subprocess.SubprocessError):
            return False

    def is_opened(self) -> bool:
        return self._process is not None or bool(
            self._capture is not None and self._capture.isOpened()
        )

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._capture is not None:
            return self._capture.read()
        if self._process is None or self._process.stdout is None:
            return False, None
        frame = np.empty((self.height, self.width, 3), dtype=np.uint8)
        target = memoryview(frame).cast("B")
        offset = 0
        while offset < len(target):
            count = self._process.stdout.readinto(target[offset:])
            if not count:
                return False, None
            offset += count
        return True, frame

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
        if self._process is not None:
            if self._process.stdout is not None:
                self._process.stdout.close()
            self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()


@dataclass(frozen=True)
class View:
    name: str
    yaw: float
    pitch: float
    fov: float = 100.0
    roll: float = 0.0


# PanoForge's factory-calibrated DJI convention, converted to this module's
# OpenCV panorama axes (x right, y down, z forward). DJI quaternions rotate
# body vectors into the world frame.
_BODY_TO_PANORAMA = np.array(
    [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
)


def quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    """Return a body-to-world rotation for a DJI [w, x, y, z] quaternion."""
    q = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("invalid quaternion")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def propagate_view_with_imu(
    view: View, previous_quaternion: np.ndarray, current_quaternion: np.ndarray
) -> View:
    """Keep a static world-facing perspective basis stable during rotation."""
    previous_basis = view_to_panorama_rotation(view.yaw, view.pitch, view.roll)
    body_to_world_previous = quaternion_to_rotation(previous_quaternion)
    body_to_world_current = quaternion_to_rotation(current_quaternion)
    current_basis = (
        _BODY_TO_PANORAMA
        @ body_to_world_current.T
        @ body_to_world_previous
        @ _BODY_TO_PANORAMA.T
        @ previous_basis
    )
    yaw, pitch, roll = Rotation.from_matrix(current_basis).as_euler(
        "YXZ", degrees=True
    )
    return View(
        view.name,
        ((yaw + 180.0) % 360.0) - 180.0,
        float(np.clip(pitch, -85, 85)),
        view.fov,
        ((roll + 180.0) % 360.0) - 180.0,
    )


def load_imu_quaternions(path: Path | None) -> dict[int, np.ndarray]:
    """Load PanoForge's per-source-frame DJI quaternion export."""
    if path is None:
        return {}
    result: dict[int, np.ndarray] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result[int(row["frame"])] = np.array(
                [float(row[key]) for key in ("qw", "qx", "qy", "qz")],
                dtype=np.float64,
            )
    return result


# Overlap is intentional: a tag near a cardinal-view edge gets another chance.
def make_views(horizontal_step_deg: int = 30, horizontal_fov_deg: float = 110.0) -> tuple[View, ...]:
    """Build overlapping perspective views for AprilTag detection."""
    yaws = range(-180, 180, horizontal_step_deg)
    return tuple(
        [View(f"h{yaw:+04d}", float(yaw), 0.0, horizontal_fov_deg) for yaw in yaws]
        + [
            View("up", 0.0, 90.0, min(horizontal_fov_deg, 120.0)),
            View("down", 0.0, -90.0, min(horizontal_fov_deg, 120.0)),
        ]
    )


DEFAULT_VIEWS = make_views()


@dataclass(frozen=True)
class IndependentTagMap:
    """Measured non-contiguous AprilTag corners in one metric world frame."""

    tags: dict[int, np.ndarray]
    metadata: dict

    @property
    def expected_ids(self) -> list[int]:
        return sorted(self.tags)

    def corners(self, tag_id: int) -> np.ndarray | None:
        corners = self.tags.get(tag_id)
        return None if corners is None else corners.copy()

    def center(self, tag_id: int) -> np.ndarray | None:
        corners = self.corners(tag_id)
        return None if corners is None else corners.mean(axis=0)


def load_tag_map(path: Path) -> IndependentTagMap:
    """Load an explicit per-ID 4x3 corner map in OpenCV marker order."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("tags")
    if not isinstance(entries, list) or not entries:
        raise ValueError("tag map must contain a non-empty tags list")
    tags: dict[int, np.ndarray] = {}
    for entry in entries:
        tag_id = int(entry["id"])
        corners = np.asarray(entry["corners_m"], dtype=np.float32)
        if corners.shape != (4, 3) or not np.isfinite(corners).all():
            raise ValueError(f"tag {tag_id} corners_m must be finite 4x3 values")
        if tag_id in tags:
            raise ValueError(f"duplicate tag id {tag_id}")
        edge_lengths = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
        if edge_lengths.min() <= 0 or edge_lengths.max() / edge_lengths.min() > 1.02:
            raise ValueError(f"tag {tag_id} corners are not a square within 2%")
        tags[tag_id] = corners
    return IndependentTagMap(tags, {key: value for key, value in payload.items() if key != "tags"})


@dataclass
class Pose:
    xyz: np.ndarray
    rotation_camera_to_board: np.ndarray
    rpy: tuple[float, float, float]
    inliers: int
    rmse: float
    view: str
    ids: list[int]


def view_to_panorama_rotation(
    yaw_deg: float, pitch_deg: float, roll_deg: float = 0.0
) -> np.ndarray:
    """Rotation from OpenCV perspective axes into panorama camera axes.

    Axes are x right, y down, z forward. Positive yaw looks right and positive
    pitch looks up, matching py360convert's e2p convention.
    """
    yaw, pitch, roll = np.radians([yaw_deg, pitch_deg, roll_deg])
    ry = np.array(
        [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]]
    )
    rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)],
        ]
    )
    rz = np.array(
        [[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]]
    )
    return ry @ rx @ rz


def pose_view_to_panorama(
    rvec: np.ndarray, tvec: np.ndarray, view: View
) -> tuple[np.ndarray, np.ndarray]:
    """Convert board->view PnP output to camera pose in the board frame."""
    board_to_view, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    view_to_pano = view_to_panorama_rotation(view.yaw, view.pitch, view.roll)
    board_to_pano = view_to_pano @ board_to_view
    board_origin_pano = view_to_pano @ np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    pano_to_board = board_to_pano.T
    camera_xyz_board = (-pano_to_board @ board_origin_pano).reshape(3)
    return camera_xyz_board, pano_to_board


def perspective_intrinsics(size: int, fov_deg: float) -> np.ndarray:
    focal = size / (2 * math.tan(math.radians(fov_deg) / 2))
    return np.array(
        [[focal, 0, size / 2], [0, focal, size / 2], [0, 0, 1]], dtype=np.float64
    )


def make_detector() -> cv2.aruco.ArucoDetector:
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 63
    params.adaptiveThreshWinSizeStep = 4
    params.minMarkerPerimeterRate = 0.008
    params.maxMarkerPerimeterRate = 4.0
    # Conservative decode: compression damage should become a miss, not a false ID.
    params.errorCorrectionRate = 0.25
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    return cv2.aruco.ArucoDetector(dictionary, params)


def detect_view(
    image: np.ndarray, detector: cv2.aruco.ArucoDetector,
    grid: Grid | IndependentTagMap,
) -> list[dict]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _rejected = detector.detectMarkers(gray)
    found: list[dict] = []
    if ids is None:
        return found
    for marker_corners, raw_id in zip(corners, ids.flatten()):
        tag_id = int(raw_id)
        center = grid.center(tag_id)
        if center is None:
            continue
        px = marker_corners.reshape(4, 2).astype(np.float32)
        found.append(
            {
                "id": tag_id,
                "corners_px": px,
                "center_px": px.mean(axis=0),
                "object_center": center,
                "object_corners": grid.corners(tag_id),
                "area_px2": abs(float(cv2.contourArea(px))),
            }
        )
    return found


def track_view_detections(
    previous_image: np.ndarray,
    current_image: np.ndarray,
    previous_detections: list[dict],
    max_forward_backward_error: float = 1.5,
) -> list[dict]:
    """Track known AprilTag corners into the next perspective frame.

    Each tag is retained only when all four corners survive a forward/backward
    Lucas-Kanade consistency check. IDs and board coordinates come from the
    last decoded frame; only image coordinates are predicted.
    """
    if not previous_detections:
        return []
    previous_gray = cv2.cvtColor(previous_image, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
    points = np.concatenate(
        [detection["corners_px"] for detection in previous_detections]
    ).astype(np.float32).reshape(-1, 1, 2)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    predicted, forward_status, _error = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        points,
        None,
        winSize=(31, 31),
        maxLevel=3,
        criteria=criteria,
    )
    if predicted is None or forward_status is None:
        return []
    returned, backward_status, _back_error = cv2.calcOpticalFlowPyrLK(
        current_gray,
        previous_gray,
        predicted,
        None,
        winSize=(31, 31),
        maxLevel=3,
        criteria=criteria,
    )
    if returned is None or backward_status is None:
        return []
    forward_backward = np.linalg.norm(
        returned.reshape(-1, 2) - points.reshape(-1, 2), axis=1
    )
    good = (
        forward_status.ravel().astype(bool)
        & backward_status.ravel().astype(bool)
        & np.isfinite(predicted.reshape(-1, 2)).all(axis=1)
        & (forward_backward <= max_forward_backward_error)
    )
    tracked: list[dict] = []
    predicted = predicted.reshape(-1, 2)
    for index, detection in enumerate(previous_detections):
        corner_slice = slice(index * 4, index * 4 + 4)
        if not good[corner_slice].all():
            continue
        corners = predicted[corner_slice].astype(np.float32)
        height, width = current_gray.shape
        if (
            np.any(corners[:, 0] < 0)
            or np.any(corners[:, 0] >= width)
            or np.any(corners[:, 1] < 0)
            or np.any(corners[:, 1] >= height)
        ):
            continue
        updated = detection.copy()
        updated["corners_px"] = corners
        updated["center_px"] = corners.mean(axis=0)
        updated["area_px2"] = abs(float(cv2.contourArea(corners)))
        tracked.append(updated)
    return tracked


def solve_view(
    detections: list[dict], view: View, size: int, min_tags: int,
    max_rmse_px: float = 3.0, pnp_points: str = "centers",
    pnp_solver: str = "ippe",
) -> Pose | None:
    # A repeated decoded ID in one view is suspicious; keep only the largest.
    best: dict[int, dict] = {}
    for det in detections:
        if det["id"] not in best or det["area_px2"] > best[det["id"]]["area_px2"]:
            best[det["id"]] = det
    detections = list(best.values())
    if len(detections) < min_tags:
        return None
    if pnp_points == "corners":
        # Four sub-pixel-refined corners per tag give stronger constraints than
        # the legacy one-center-per-tag approximation.
        obj = np.concatenate([d["object_corners"] for d in detections]).astype(np.float32)
        img = np.concatenate([d["corners_px"] for d in detections]).astype(np.float32)
    else:
        obj = np.asarray([d["object_center"] for d in detections], np.float32)
        img = np.asarray([d["center_px"] for d in detections], np.float32)
    k = perspective_intrinsics(size, view.fov)
    if pnp_solver == "iterative":
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj, img, k, None, iterationsCount=200, reprojectionError=3.0,
            confidence=0.999, flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok or inliers is None:
            return None
        ii = inliers[:, 0]
        solutions = [(rvec, tvec)]
    else:
        # The whole AprilGrid is planar. Filter decoded outliers with a planar
        # homography first, then let IPPE return both ambiguous pose solutions.
        homography_obj = obj[:, :2].astype(np.float64)
        _h, mask = cv2.findHomography(
            homography_obj, img.astype(np.float64), cv2.RANSAC,
            max(3.0, max_rmse_px),
            maxIters=2000, confidence=0.999,
        )
        if mask is None:
            return None
        ii = np.flatnonzero(mask.ravel())
        min_points = min_tags * (4 if pnp_points == "corners" else 1)
        if len(ii) < min_points:
            return None
        ok, rvecs, tvecs, _errors = cv2.solvePnPGeneric(
            obj[ii], img[ii], k, None, flags=cv2.SOLVEPNP_IPPE,
        )
        if not ok:
            return None
        solutions = list(zip(rvecs, tvecs))
    min_points = min_tags * (4 if pnp_points == "corners" else 1)
    if len(ii) < min_points:
        return None
    best_solution = None
    for candidate_rvec, candidate_tvec in solutions:
        candidate_rvec, candidate_tvec = cv2.solvePnPRefineLM(
            obj[ii], img[ii], k, None, candidate_rvec, candidate_tvec,
        )
        if not (np.isfinite(candidate_rvec).all() and np.isfinite(candidate_tvec).all()):
            continue
        camera_points, _ = cv2.projectPoints(
            obj[ii], candidate_rvec, candidate_tvec, k, None,
        )
        residual = camera_points.reshape(-1, 2) - img[ii]
        candidate_rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
        rotation, _ = cv2.Rodrigues(candidate_rvec)
        depths = (rotation @ obj[ii].T + candidate_tvec.reshape(3, 1))[2]
        if np.all(depths > 0) and math.isfinite(candidate_rmse):
            if best_solution is None or candidate_rmse < best_solution[0]:
                best_solution = (candidate_rmse, candidate_rvec, candidate_tvec)
    if best_solution is None:
        return None
    rmse, rvec, tvec = best_solution
    if rmse > max_rmse_px:
        return None
    xyz, camera_to_board = pose_view_to_panorama(rvec, tvec, view)
    if not (np.isfinite(xyz).all() and np.isfinite(camera_to_board).all()):
        return None
    return Pose(
        xyz,
        camera_to_board,
        rotation_to_rpy(camera_to_board),
        len(ii),
        rmse,
        view.name,
        sorted(
            {
                int(detections[i // 4]["id"])
                if pnp_points == "corners"
                else int(detections[i]["id"])
                for i in ii
            }
        ),
    )


def choose_pose(candidates: list[Pose]) -> Pose | None:
    if not candidates:
        return None
    # Perspective views overlap, so the same board often has multiple valid PnP
    # candidates.  Reprojection fit is the primary reliability signal; choosing
    # merely the view with most tags can select a distorted edge-of-view solution.
    # Inlier support only breaks an equal-fit tie.
    return min(candidates, key=lambda p: (p.rmse, -p.inliers))


def _finite_stats(values: Iterable[float]) -> dict[str, float | None]:
    data = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
    if not len(data):
        return {"mean": None, "median": None, "p95": None, "max": None}
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "p95": float(np.percentile(data, 95)),
        "max": float(np.max(data)),
    }


def generate_plot(session: Path, summary: dict) -> None:
    rows = list(
        csv.DictReader((session / "pose.csv").open(encoding="utf-8", newline=""))
    )
    points, errors, point_times = [], [], []
    for row in rows:
        if row["quality_status"] == "valid" and row["camera_x_m"]:
            points.append([float(row[f"camera_{a}_m"]) for a in "xyz"])
            errors.append(float(row["reprojection_rmse_px"]))
            point_times.append(float(row["timestamp"]))
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 10), facecolor="#0b0d10")
    ratio = summary["valid_pose_ratio"] * 100
    fig.suptitle(
        f"Osmo 360 AprilGrid trajectory · valid {ratio:.1f}% · coverage {summary['tag_coverage_ratio'] * 100:.1f}%",
        fontsize=17,
    )
    if points:
        p = np.asarray(points)
        c = np.linspace(0, 1, len(p))
        all_times = np.asarray([float(row["timestamp"]) for row in rows])
        nominal_dt = float(np.median(np.diff(all_times))) if len(all_times) > 1 else math.inf
        breaks = np.flatnonzero(np.diff(point_times) > nominal_dt * 1.5) + 1
        segments = np.split(p, breaks)
        ax = fig.add_subplot(221, projection="3d")
        for segment in segments:
            ax.plot(*segment.T, color="#57b9ff")
        ax.scatter(*p.T, c=c, cmap="viridis", s=9)
        ax.scatter(*p[0], c="#4ade80", s=90, label="Start")
        ax.scatter(*p[-1], c="#fb5b5b", s=90, label="End")
        ax.scatter(0, 0, 0, c="white", marker="s", s=80, label="AprilGrid origin")
        ax.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)", title="3D trajectory")
        ax.legend()
        for slot, ai, bi, title in [
            (222, 0, 1, "XY"),
            (223, 0, 2, "XZ"),
            (224, 1, 2, "YZ"),
        ]:
            a = fig.add_subplot(slot)
            for segment in segments:
                a.plot(segment[:, ai], segment[:, bi], color="#57b9ff")
            a.scatter(p[:, ai], p[:, bi], c=c, cmap="viridis", s=8)
            a.scatter(0, 0, c="white", marker="s")
            a.scatter(*p[0, [ai, bi]], c="#4ade80", s=60)
            a.scatter(*p[-1, [ai, bi]], c="#fb5b5b", s=60)
            a.set_title(title)
            a.grid(alpha=0.2)
            a.set_aspect("equal", adjustable="datalim")
    else:
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, "No valid AprilGrid pose", ha="center", fontsize=24)
    rmse = summary["reprojection_rmse_px"]["median"]
    fig.text(
        0.02,
        0.02,
        f"IDs {summary['recognized_ids']} · median RMSE {rmse if rmse is not None else 'n/a'} px",
    )
    fig.text(
        0.98,
        0.02,
        "APPROXIMATE / DEMO-GRADE — UNCALIBRATED PANORAMA MODEL",
        ha="right",
        color="#ffc857",
        weight="bold",
    )
    fig.savefig(
        session / "relative_coordinates.png",
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline AprilGrid pose from a stitched 2:1 panorama"
    )
    p.add_argument("input", type=Path)
    p.add_argument("--tag-size", type=float, default=0.088)
    p.add_argument("--spacing", type=float, default=0.30)
    p.add_argument("--rows", type=int, default=6)
    p.add_argument("--cols", type=int, default=6)
    p.add_argument("--first-id", type=int, default=0)
    p.add_argument(
        "--tag-map", type=Path,
        help="JSON explicit non-contiguous tag map; overrides rows/cols/first-id geometry",
    )
    p.add_argument("--sample-fps", type=float, default=5.0)
    p.add_argument("--output-dir", type=Path, default=Path("sessions"))
    p.add_argument("--min-tags", type=int, default=6)
    p.add_argument(
        "--max-rmse-px",
        type=float,
        default=3.0,
        help="reject PnP candidates above this reprojection RMSE",
    )
    p.add_argument("--view-size", type=int, default=960)
    p.add_argument(
        "--pnp-points", choices=("corners", "centers"), default="centers",
        help="use tag centers (default) or experimental tag corners",
    )
    p.add_argument(
        "--pnp-solver", choices=("ippe", "iterative"), default="ippe",
        help="planar IPPE (default) or legacy iterative RANSAC",
    )
    p.add_argument(
        "--max-processed-frames", type=int,
        help="stop after this many sampled frames (for small validation runs)",
    )
    p.add_argument(
        "--full-scan", action=argparse.BooleanOptionalAction, default=True,
        help="global recovery scan (enabled by default; use --no-full-scan for experiments)",
    )
    p.add_argument(
        "--temporal-flow", action=argparse.BooleanOptionalAction, default=True,
        help="bidirectional LK between periodic decodes (enabled by default)",
    )
    p.add_argument(
        "--imu-csv",
        type=Path,
        help="PanoForge imu_perframe.csv; guides the tracked view during rotation",
    )
    p.add_argument(
        "--redetect-interval",
        type=int,
        default=3,
        help="decode tags again after this many optical-flow frames",
    )
    p.add_argument(
        "--global-refresh-interval",
        type=int,
        default=150,
        help="run a non-destructive global refresh after this many tracked frames",
    )
    p.add_argument(
        "--recovery-scan-interval",
        type=int,
        default=15,
        help="while lost, run the global sweep only every N frames",
    )
    p.add_argument(
        "--global-search-size",
        type=int,
        default=720,
        help="low-resolution global scout size; a hit is refined at --view-size",
    )
    p.add_argument(
        "--horizontal-step-deg", type=int, default=30,
        help="yaw spacing between perspective views (default: 30)",
    )
    p.add_argument(
        "--horizontal-fov-deg", type=float, default=110.0,
        help="horizontal FOV of perspective views (default: 110)",
    )
    p.add_argument(
        "--focused-yaws",
        help="comma-separated yaw views for a known scene; skips unused views",
    )
    p.add_argument(
        "--projection-backend",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="projection backend (auto: DJI 3K CPU; X6/high-resolution CUDA)",
    )
    p.add_argument(
        "--decoder", choices=("auto", "cpu", "nvdec"), default="auto",
        help="video decoder (auto: measured-safe CPU; NVDEC remains opt-in)",
    )
    p.add_argument(
        "--camera-model", choices=CAMERA_MODELS, default="auto",
        help="processing profile; auto infers DJI Osmo 360, Insta360 X6, or generic",
    )
    p.add_argument(
        "--ffmpeg-bin", type=Path,
        help="CUDA-enabled ffmpeg; defaults to bundled ffmpeg or PATH",
    )
    p.add_argument(
        "--scan-workers", type=int, default=min(4, max(1, (os.cpu_count() or 4) // 4)),
        help="parallel AprilTag workers for global views (default: up to 4)",
    )
    p.add_argument(
        "--max-speed",
        type=float,
        default=5.0,
        help="reject filtered jumps faster than m/s",
    )
    p.add_argument(
        "--official-stitched",
        action="store_true",
        help="input was exported by DJI Studio",
    )
    p.add_argument("--session-name")
    p.add_argument("--status-file", type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if (
        args.tag_size <= 0
        or args.spacing < 0
        or args.sample_fps <= 0
        or args.min_tags < (2 if args.tag_map else 4)
        or args.max_rmse_px <= 0
        or args.view_size < 160
        or (args.max_processed_frames is not None and args.max_processed_frames <= 0)
        or args.redetect_interval <= 0
        or args.global_refresh_interval <= 0
        or args.recovery_scan_interval <= 0
        or args.scan_workers <= 0
        or not 320 <= args.global_search_size <= args.view_size
        or args.horizontal_step_deg <= 0
        or 360 % args.horizontal_step_deg != 0
        or not 30.0 <= args.horizontal_fov_deg < 180.0
    ):
        raise SystemExit("invalid grid/sampling parameters")
    session = args.output_dir / (
        args.session_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    # The web controller pre-creates the directory for its launcher log.
    session.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(session / "processor.log"),
            logging.StreamHandler(),
        ],
    )
    signal.signal(signal.SIGINT, lambda *_: globals().__setitem__("STOP", True))
    bundled_ffmpeg = Path("work/tools/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg")
    ffmpeg_bin = args.ffmpeg_bin
    if ffmpeg_bin is None:
        ffmpeg_bin = bundled_ffmpeg if bundled_ffmpeg.exists() else (
            Path(shutil.which("ffmpeg")) if shutil.which("ffmpeg") else None
        )
    try:
        cap = VideoReader(
            args.input, args.decoder, ffmpeg_bin,
            requested_camera_model=args.camera_model,
        )
    except RuntimeError as exc:
        LOG.error("cannot initialize video decoder: %s", exc)
        return 2
    width, height = cap.width, cap.height
    source_fps = cap.fps
    total = cap.frame_count
    camera_model = cap.camera_model
    LOG.info("camera processing profile: %s", camera_model)
    LOG.info("video decoder: %s", cap.decoder)
    if width != 2 * height:
        LOG.error("expected 2:1 equirectangular input, got %dx%d", width, height)
        return 3
    projection_name = resolve_projection(
        args.projection_backend, camera_model, width
    )
    try:
        projection_backend = make_projection_backend(projection_name)
    except RuntimeError as exc:
        if args.projection_backend != "auto":
            LOG.error("cannot initialize %s projection: %s", projection_name, exc)
            return 4
        LOG.warning("CUDA projection unavailable (%s); falling back to CPU", exc)
        projection_backend = make_projection_backend("cpu")
    LOG.info("projection backend: %s", projection_backend.name)
    step = max(1, round(source_fps / args.sample_fps)) if source_fps > 0 else 1
    if args.tag_map:
        grid = load_tag_map(args.tag_map)
        expected_ids = grid.expected_ids
        LOG.info("loaded explicit tag map %s with IDs %s", args.tag_map, expected_ids)
    else:
        grid = Grid(args.rows, args.cols, args.tag_size, args.spacing, args.first_id)
        expected_ids = list(range(args.first_id, args.first_id + args.rows * args.cols))
    detector = make_detector()
    scan_executor = ThreadPoolExecutor(
        max_workers=args.scan_workers, thread_name_prefix="apriltag-scan"
    )
    scan_thread = threading.local()
    LOG.info("global scan workers: %d", args.scan_workers)
    imu_quaternions = load_imu_quaternions(args.imu_csv)
    if args.imu_csv:
        LOG.info("loaded %d per-frame IMU orientations", len(imu_quaternions))
    views = make_views(args.horizontal_step_deg, args.horizontal_fov_deg)
    if args.focused_yaws:
        try:
            focused = [float(value) for value in args.focused_yaws.split(",")]
        except ValueError as exc:
            raise SystemExit("invalid --focused-yaws") from exc
        if not focused or any(not -180.0 <= yaw < 180.0 for yaw in focused):
            raise SystemExit("invalid --focused-yaws")
        views = tuple(
            View(f"h{yaw:+04.0f}", yaw, 0.0, args.horizontal_fov_deg)
            for yaw in focused
        )
    seen = Counter()
    rmses: list[float] = []
    jumps: list[float] = []
    processed = valid = frame_no = 0
    previous: tuple[float, np.ndarray] | None = None
    tracked_view: View | None = None
    temporal_image: np.ndarray | None = None
    temporal_detections: list[dict] = []
    temporal_age = 0
    tracked_quaternion: np.ndarray | None = None
    lost_frames = 0
    search_size = min(args.global_search_size, args.view_size)
    candidate_sources: list[tuple[Pose, View, np.ndarray, list[dict]]] = []
    candidate_measurements: dict[int, str] = {}

    def evaluate_view(
        pano: np.ndarray,
        view: View,
        size: int,
        perspective: np.ndarray | None = None,
    ) -> tuple[list[dict], Pose | None, dict]:
        if perspective is None:
            perspective = projection_backend.project_many(
                pano, [ProjectionRequest(view.yaw, view.pitch, view.fov, size, view.roll)]
            )[0]
        detections = detect_view(perspective, detector, grid)
        pose = solve_view(
            detections, view, size, args.min_tags, args.max_rmse_px,
            args.pnp_points, args.pnp_solver,
        )
        if pose is not None:
            candidate_sources.append((pose, view, perspective, detections))
            candidate_measurements[id(pose)] = "direct"
        record = {
            "view": asdict(view),
            "size": size,
            "detections": [
                {"id": d["id"], "corners_px": d["corners_px"].round(2).tolist()}
                for d in detections
            ],
            "pose": None if pose is None else {
                "xyz": pose.xyz.tolist(), "rpy": pose.rpy,
                "inliers": pose.inliers, "rmse": pose.rmse,
            },
        }
        return detections, pose, record

    def evaluate_projected(
        item: tuple[View, np.ndarray, int],
    ) -> tuple[View, np.ndarray, list[dict], Pose | None, dict]:
        view, perspective, size = item
        worker_detector = getattr(scan_thread, "detector", None)
        if worker_detector is None:
            worker_detector = make_detector()
            scan_thread.detector = worker_detector
        detections = detect_view(perspective, worker_detector, grid)
        pose = solve_view(
            detections, view, size, args.min_tags, args.max_rmse_px,
            args.pnp_points, args.pnp_solver,
        )
        record = {
            "view": asdict(view),
            "size": size,
            "detections": [
                {"id": detection["id"], "corners_px": detection["corners_px"].round(2).tolist()}
                for detection in detections
            ],
            "pose": None if pose is None else {
                "xyz": pose.xyz.tolist(), "rpy": pose.rpy,
                "inliers": pose.inliers, "rmse": pose.rmse,
            },
        }
        return view, perspective, detections, pose, record

    def evaluate_views(
        pano: np.ndarray, selected_views: tuple[View, ...], size: int
    ) -> list[tuple[View, list[dict], Pose | None, dict]]:
        perspectives = projection_backend.project_many(
            pano,
            [
                ProjectionRequest(view.yaw, view.pitch, view.fov, size, view.roll)
                for view in selected_views
            ],
        )
        payload = [
            (view, perspective, size)
            for view, perspective in zip(selected_views, perspectives)
        ]
        results = list(scan_executor.map(evaluate_projected, payload))
        output = []
        # Aggregate shared candidate state deterministically in view order.
        for view, perspective, detections, pose, record in results:
            if pose is not None:
                candidate_sources.append((pose, view, perspective, detections))
                candidate_measurements[id(pose)] = "direct"
            output.append((view, detections, pose, record))
        return output

    def recentered_view(base: View, detections: list[dict], size: int) -> View:
        centers = np.asarray([d["center_px"] for d in detections], dtype=float)
        focal = perspective_intrinsics(size, base.fov)[0, 0]
        offset = centers.mean(axis=0) - size / 2.0
        yaw = base.yaw + math.degrees(math.atan2(offset[0], focal))
        pitch = base.pitch - math.degrees(math.atan2(offset[1], focal))
        span = np.ptp(centers, axis=0)
        angular_span = math.degrees(2 * math.atan2(max(span) / 2.0, focal))
        return View(
            "tracked", ((yaw + 180.0) % 360.0) - 180.0,
            float(np.clip(pitch, -85.0, 85.0)),
            float(np.clip(max(angular_span * 1.8, 70.0), 70.0, 100.0)),
            base.roll,
        )

    csv_fields = [
        "frame",
        "timestamp",
        "camera_x_m",
        "camera_y_m",
        "camera_z_m",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
        "raw_camera_x_m",
        "raw_camera_y_m",
        "raw_camera_z_m",
        "detected_tag_count",
        "inlier_count",
        "reprojection_rmse_px",
        "detected_ids",
        "selected_view",
        "measurement_source",
        "quality_status",
    ]
    with (
        (session / "pose.csv").open("w", newline="", encoding="utf-8") as cf,
        (session / "detections.jsonl").open("w", encoding="utf-8") as jf,
    ):
        writer = csv.DictWriter(cf, fieldnames=csv_fields)
        writer.writeheader()
        while not STOP:
            if args.max_processed_frames is not None and processed >= args.max_processed_frames:
                break
            ok, pano = cap.read()
            if not ok:
                break
            if frame_no % step:
                frame_no += 1
                continue
            timestamp = frame_no / source_fps if source_fps > 0 else float(processed)
            processed += 1
            all_ids: set[int] = set()
            view_records: list[dict] = []
            candidates: list[Pose] = []
            candidate_sources = []
            candidate_measurements = {}

            current_quaternion = imu_quaternions.get(frame_no)
            if (
                tracked_view is not None
                and tracked_quaternion is not None
                and current_quaternion is not None
            ):
                tracked_view = propagate_view_with_imu(
                    tracked_view, tracked_quaternion, current_quaternion
                )
                tracked_quaternion = current_quaternion

            temporal_success = False
            force_global_refresh = args.temporal_flow and (
                processed == 1
                or (processed - 1) % args.global_refresh_interval == 0
            )
            if (
                args.temporal_flow
                and tracked_view is not None
                and temporal_image is not None
                and temporal_detections
            ):
                perspective = projection_backend.project_many(
                    pano,
                    [
                        ProjectionRequest(
                            tracked_view.yaw,
                            tracked_view.pitch,
                            tracked_view.fov,
                            args.view_size,
                            tracked_view.roll,
                        )
                    ],
                )[0]
                if temporal_age >= args.redetect_interval:
                    dets = detect_view(perspective, detector, grid)
                    tracking_mode = "redetected"
                    next_temporal_age = 0
                    candidate = solve_view(
                        dets,
                        tracked_view,
                        args.view_size,
                        args.min_tags,
                        args.max_rmse_px,
                        args.pnp_points,
                        args.pnp_solver,
                    )
                    # A blurred frame may be impossible to decode even though
                    # its already-identified corners remain trackable. Do not
                    # throw away a healthy track merely because the periodic
                    # decoder misses; LK still has forward/backward checks and
                    # the resulting pose must pass the normal PnP/RMSE gates.
                    if candidate is None:
                        flowed = track_view_detections(
                            temporal_image, perspective, temporal_detections
                        )
                        flow_candidate = solve_view(
                            flowed,
                            tracked_view,
                            args.view_size,
                            args.min_tags,
                            args.max_rmse_px,
                            args.pnp_points,
                            args.pnp_solver,
                        )
                        if flow_candidate is not None:
                            dets = flowed
                            candidate = flow_candidate
                            tracking_mode = "redetect_fallback_flow"
                            next_temporal_age = temporal_age + 1
                else:
                    dets = track_view_detections(
                        temporal_image, perspective, temporal_detections
                    )
                    tracking_mode = "optical_flow"
                    next_temporal_age = temporal_age + 1
                    candidate = solve_view(
                        dets,
                        tracked_view,
                        args.view_size,
                        args.min_tags,
                        args.max_rmse_px,
                        args.pnp_points,
                        args.pnp_solver,
                    )
                all_ids.update(int(d["id"]) for d in dets)
                view_records.append(
                    {
                        "view": asdict(tracked_view),
                        "size": args.view_size,
                        "tracking_mode": tracking_mode,
                        "detections": [
                            {
                                "id": d["id"],
                                "corners_px": d["corners_px"].round(2).tolist(),
                            }
                            for d in dets
                        ],
                        "pose": None
                        if candidate is None
                        else {
                            "xyz": candidate.xyz.tolist(),
                            "rpy": candidate.rpy,
                            "inliers": candidate.inliers,
                            "rmse": candidate.rmse,
                        },
                    }
                )
                if candidate is not None:
                    candidates.append(candidate)
                    candidate_sources.append(
                        (candidate, tracked_view, perspective, dets)
                    )
                    candidate_measurements[id(candidate)] = (
                        "direct" if tracking_mode == "redetected" else "optical_flow"
                    )
                    temporal_image = perspective
                    temporal_detections = dets
                    temporal_age = next_temporal_age
                    temporal_success = True
                else:
                    temporal_image = None
                    temporal_detections = []
                    temporal_age = 0

            # Global measurement path and immediate fallback when temporal
            # tracking has not locked or has just failed.
            # A high-resolution sweep on every lost frame was the dominant
            # runtime cost. Scout globally at low resolution, periodically
            # while lost, then refine only the best direction at full size.
            # A scheduled refresh is non-destructive: failed refreshes never
            # replace a healthy temporal track.
            run_full_scan = args.full_scan and (
                processed == 1
                or force_global_refresh
                or (
                    not temporal_success
                    and lost_frames % args.recovery_scan_interval == 0
                )
            )
            if run_full_scan:
                coarse: list[tuple[View, list[dict]]] = []
                scan_views = views
                if tracked_view is not None and current_quaternion is not None:
                    scan_views = (*views, tracked_view)
                for view, dets, coarse_candidate, record in evaluate_views(
                    pano, scan_views, search_size
                ):
                    view_records.append(record)
                    coarse.append((view, dets))
                    all_ids.update(int(d["id"]) for d in dets)
                    if search_size == args.view_size and coarse_candidate:
                        candidates.append(coarse_candidate)
                    else:
                        # Low-resolution PnP only locates a direction; never
                        # report it as the final measurement.
                        record["pose"] = None
                base, scout_detections = max(coarse, key=lambda item: len(item[1]))
                if len(scout_detections) >= 2:
                    # If scouting was low-resolution, first obtain a canonical
                    # full-resolution measurement. Then always try one centered
                    # refinement: centering is important for stable LK tracking,
                    # while both candidates remain available for RMSE selection.
                    dets = scout_detections
                    if search_size != args.view_size:
                        dets, candidate, record = evaluate_view(
                            pano, base, args.view_size
                        )
                        view_records.append(record)
                        all_ids.update(int(d["id"]) for d in dets)
                        if candidate:
                            candidates.append(candidate)
                    if len(dets) >= 2:
                        refined_view = recentered_view(base, dets, args.view_size)
                        dets, candidate, record = evaluate_view(
                            pano, refined_view, args.view_size
                        )
                        view_records.append(record)
                        all_ids.update(int(d["id"]) for d in dets)
                        if candidate:
                            candidates.append(candidate)
                            tracked_view = refined_view
                            tracked_quaternion = current_quaternion
                # A distant grid may yield only one decoded tag in a 110-degree
                # scout even though a narrower crop contains enough pixels for
                # PnP. Use that tag as a bearing, then search a small 3x3 local
                # neighborhood at 60 degrees. This is only paid while lost.
                if not candidates and scout_detections:
                    anchor = recentered_view(base, scout_detections, search_size)
                    local_views = tuple(
                        View(
                            f"recovery_{dyaw:+.0f}_{dpitch:+.0f}",
                            ((anchor.yaw + dyaw + 180.0) % 360.0) - 180.0,
                            float(np.clip(anchor.pitch + dpitch, -85.0, 85.0)),
                            60.0,
                            anchor.roll,
                        )
                        for dpitch in (-15.0, 0.0, 15.0)
                        for dyaw in (-15.0, 0.0, 15.0)
                    )
                    local_results = evaluate_views(pano, local_views, args.view_size)
                    for local_view, dets, candidate, record in local_results:
                        view_records.append(record)
                        all_ids.update(int(d["id"]) for d in dets)
                        if candidate:
                            candidates.append(candidate)
                    if candidates:
                        best_local = min(
                            (
                                (candidate, local_view, dets)
                                for local_view, dets, candidate, _record in local_results
                                if candidate is not None
                            ),
                            key=lambda item: item[0].rmse,
                        )
                        _candidate, local_view, dets = best_local
                        tracked_view = (
                            recentered_view(local_view, dets, args.view_size)
                            if len(dets) >= 2
                            else local_view
                        )
                        tracked_quaternion = current_quaternion
                if candidates and tracked_view is None:
                    selected = choose_pose(candidates)
                    selected_coarse = next(
                        (
                            (view, detections)
                            for view, detections in coarse
                            if selected is not None and view.name == selected.view
                        ),
                        None,
                    )
                    if selected_coarse is not None:
                        selected_view, selected_detections = selected_coarse
                        tracked_view = (
                            recentered_view(
                                selected_view, selected_detections, args.view_size
                            )
                            if len(selected_detections) >= 2
                            else selected_view
                        )
                        tracked_quaternion = current_quaternion

            # Fast path: reuse the previous board direction.  The final PnP is
            # still measured from a full-resolution perspective image.
            if not temporal_success and not run_full_scan and tracked_view is not None:
                dets, candidate, record = evaluate_view(pano, tracked_view, args.view_size)
                view_records.append(record)
                all_ids.update(int(d["id"]) for d in dets)
                if candidate:
                    candidates.append(candidate)
                if len(dets) >= 2:
                    updated = recentered_view(tracked_view, dets, args.view_size)
                    heading_change = abs(updated.yaw - tracked_view.yaw) + abs(updated.pitch - tracked_view.pitch)
                    tracked_view = updated
                    tracked_quaternion = current_quaternion
                    if candidate is None or heading_change > 2.0:
                        dets, candidate, record = evaluate_view(pano, tracked_view, args.view_size)
                        view_records.append(record)
                        all_ids.update(int(d["id"]) for d in dets)
                        if candidate:
                            candidates.append(candidate)

            # Recovery path: cheap low-resolution global sweep, followed by one
            # high-resolution recentered measurement.  Never use low-res PnP as
            # the reported pose.
            for tag_id in all_ids:
                seen[tag_id] += 1
            pose = choose_pose(candidates)
            measurement_source = candidate_measurements.get(id(pose), "") if pose else ""
            if args.temporal_flow and pose is not None and not temporal_success:
                source = next(
                    (item for item in candidate_sources if item[0] is pose), None
                )
                if source is not None:
                    _pose, tracked_view, temporal_image, temporal_detections = source
                    temporal_age = 0
                    tracked_quaternion = current_quaternion
            elif args.temporal_flow and pose is None:
                # Keep the last board direction so IMU can continue propagating
                # a cheap local search through blur. Only LK image state is lost.
                temporal_image = None
                temporal_detections = []
                temporal_age = 0
            quality = "insufficient_tags"
            filtered = None
            jump = None
            if pose and np.isfinite(pose.xyz).all() and math.isfinite(pose.rmse):
                quality = "valid"
                filtered = pose.xyz.copy()
                rmses.append(pose.rmse)
                if previous:
                    dt = timestamp - previous[0]
                    jump = float(np.linalg.norm(pose.xyz - previous[1]))
                    jumps.append(jump)
                    if dt > 0 and jump / dt > args.max_speed:
                        quality = "jump_rejected"
                        filtered = None
                if quality == "valid":
                    previous = (timestamp, filtered.copy())
                    valid += 1
            lost_frames = 0 if quality == "valid" else lost_frames + 1
            row = dict.fromkeys(csv_fields, "")
            row.update(
                frame=frame_no,
                timestamp=f"{timestamp:.6f}",
                detected_tag_count=len(all_ids),
                detected_ids=" ".join(map(str, sorted(all_ids))),
                measurement_source=measurement_source,
                quality_status=quality,
            )
            if pose:
                row.update(
                    raw_camera_x_m=f"{pose.xyz[0]:.7f}",
                    raw_camera_y_m=f"{pose.xyz[1]:.7f}",
                    raw_camera_z_m=f"{pose.xyz[2]:.7f}",
                    inlier_count=pose.inliers,
                    reprojection_rmse_px=f"{pose.rmse:.4f}",
                    selected_view=pose.view,
                    roll_deg=f"{pose.rpy[0]:.4f}",
                    pitch_deg=f"{pose.rpy[1]:.4f}",
                    yaw_deg=f"{pose.rpy[2]:.4f}",
                )
                if filtered is not None:
                    row.update(
                        camera_x_m=f"{filtered[0]:.7f}",
                        camera_y_m=f"{filtered[1]:.7f}",
                        camera_z_m=f"{filtered[2]:.7f}",
                    )
            writer.writerow(row)
            cf.flush()
            jf.write(
                json.dumps(
                    {
                        "frame": frame_no,
                        "timestamp": timestamp,
                        "detected_ids": sorted(all_ids),
                        "views": view_records,
                        "selected_view": pose.view if pose else None,
                        "measurement_source": measurement_source,
                        "quality_status": quality,
                        "jump_m": jump,
                    }
                )
                + "\n"
            )
            jf.flush()
            status = {
                "running": True,
                "frame": frame_no,
                "processed_frames": processed,
                "pose_frames": valid,
                "seen_ids": sorted(seen),
            }
            if args.status_file:
                args.status_file.write_text(json.dumps(status), encoding="utf-8")
            if processed % 5 == 0:
                LOG.info(
                    "frame=%d processed=%d ids=%s valid=%d",
                    frame_no,
                    processed,
                    sorted(all_ids),
                    valid,
                )
            frame_no += 1
    cap.release()
    scan_executor.shutdown(wait=True)
    expected = expected_ids
    summary = {
        "input": str(args.input.resolve()),
        "total_frames": total,
        "processed_frames": processed,
        "valid_pose_frames": valid,
        "valid_pose_ratio": valid / processed if processed else 0.0,
        "recognized_ids": sorted(seen),
        "missing_ids": [i for i in expected if i not in seen],
        "detections_per_id": {str(k): v for k, v in sorted(seen.items())},
        "tag_coverage_ratio": len(set(expected) & set(seen)) / len(expected),
        "reprojection_rmse_px": _finite_stats(rmses),
        "adjacent_coordinate_jump_m": _finite_stats(jumps),
        "tag_size_m": (
            grid.metadata.get("tag_size_m")
            if isinstance(grid, IndependentTagMap) else args.tag_size
        ),
        "spacing_ratio": None if isinstance(grid, IndependentTagMap) else args.spacing,
        "rows": None if isinstance(grid, IndependentTagMap) else args.rows,
        "cols": None if isinstance(grid, IndependentTagMap) else args.cols,
        "first_id": None if isinstance(grid, IndependentTagMap) else args.first_id,
        "tag_map": str(args.tag_map.resolve()) if args.tag_map else None,
        "pnp_points": args.pnp_points,
        "pnp_solver": args.pnp_solver,
        "projection_backend": projection_backend.name,
        "camera_model": camera_model,
        "requested_decoder": args.decoder,
        "requested_projection_backend": args.projection_backend,
        "video_decoder": cap.decoder,
        "scan_workers": args.scan_workers,
        "temporal_flow": args.temporal_flow,
        "imu_guided_view": bool(imu_quaternions),
        "redetect_interval": args.redetect_interval,
        "global_refresh_interval": args.global_refresh_interval,
        "official_stitched_panorama": args.official_stitched,
        "measurement_grade_camera_model": False,
        "accuracy_label": "APPROXIMATE / DEMO-GRADE",
        "stopped": STOP,
    }
    (session / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    generate_plot(session, summary)
    if args.status_file:
        args.status_file.write_text(
            json.dumps({"running": False, **summary}), encoding="utf-8"
        )
    LOG.info("complete: %s", json.dumps(summary, ensure_ascii=False))
    print(session.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
