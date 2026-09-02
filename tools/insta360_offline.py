#!/usr/bin/env python3
"""Offline AprilGrid pose estimation for Insta360 stitched panorama video."""

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
import warnings
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from itertools import product
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from osmo360.localization.camera_frames import BODY_TO_PANORAMA_OPENCV
from tools.apriltag_geometry import Grid, rotation_to_rpy
from tools.projection_backends import ProjectionRequest, make_projection_backend
from osmo360.localization.world_frames import RigidTransform, canonical_sha256, compile_world_tag_map

LOG = logging.getLogger("insta360.offline")
STOP = False


CAMERA_MODELS = ("auto", "insta360-x5", "generic")


def infer_camera_model(path: Path, width: int, height: int, requested: str) -> str:
    """Choose an Insta360 processing profile from the export name and size."""
    if requested != "auto":
        return requested
    name = path.name.lower()
    if (
        "insta360" in name
        or name.startswith("vid_")
        or (width >= 6000 and width == 2 * height)
    ):
        return "insta360-x5"
    return "generic"


def resolve_decoder(requested: str, camera_model: str) -> str:
    """Use CPU decode until the AprilTag path supports zero-copy NVDEC."""
    if requested != "auto":
        return requested
    if camera_model in ("insta360-x5", "generic"):
        return "cpu"
    raise ValueError(f"unknown camera model: {camera_model}")


def resolve_projection(requested: str, camera_model: str, width: int) -> str:
    """Select CPU for ordinary inputs and CUDA for high-resolution X5 panoramas."""
    if requested != "auto":
        return requested
    return "cuda" if camera_model == "insta360-x5" and width > 4096 else "cpu"


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

    def seek(self, frame_no: int) -> bool:
        """Seek to an absolute source frame for CPU decoding.

        NVDEC is intentionally sequential in the current implementation, so
        segment processing uses the measured-safe CPU decoder.
        """
        if frame_no < 0 or self._capture is None:
            return False
        return bool(self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_no))

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


# Fixed body-to-panorama OpenCV basis shared with X5 mount calibration.
_BODY_TO_PANORAMA = BODY_TO_PANORAMA_OPENCV


def project_to_so3(matrix: np.ndarray) -> np.ndarray:
    """Return the closest proper rotation to an approximately orthogonal matrix."""
    candidate = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    u, _singular, vt = np.linalg.svd(candidate)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def default_panorama_to_body_rotation() -> np.ndarray:
    """Return the canonical panorama-camera to body rotation."""
    return _BODY_TO_PANORAMA.T


@dataclass(frozen=True)
class ImuPanoramaBridgeEstimate:
    """Fixed per-camera bridge in ``R_parent_camera = A R_world_body X``.

    ``X`` maps panorama-camera vectors into the camera IMU body frame. ``A``
    maps the arbitrary attitude world into the visual tag-map parent frame.
    """

    panorama_to_body: np.ndarray
    imu_world_to_parent: np.ndarray
    observation_count: int
    inlier_count: int
    pair_count: int
    excitation_deg: float
    excitation_axis_ratio: float
    residual_median_deg: float
    residual_p95_deg: float
    residual_max_deg: float
    inlier_mask: np.ndarray

    def predict_camera_to_parent(self, quaternion: np.ndarray) -> np.ndarray:
        body_to_imu_world = quaternion_to_rotation(quaternion)
        return project_to_so3(
            self.imu_world_to_parent
            @ body_to_imu_world
            @ self.panorama_to_body
        )


def _chordal_rotation_mean(rotations: list[np.ndarray]) -> np.ndarray:
    if not rotations:
        raise ValueError("at least one rotation is required")
    return project_to_so3(np.sum(np.asarray(rotations, dtype=np.float64), axis=0))


def estimate_imu_panorama_bridge(
    visual_camera_to_parent: Iterable[np.ndarray],
    imu_body_to_world: Iterable[np.ndarray],
    *,
    min_observations: int = 4,
    min_relative_rotation_deg: float = 2.0,
    max_pair_angle_error_deg: float = 12.0,
    min_excitation_axis_ratio: float = 0.03,
    max_fit_residual_deg: float = 15.0,
) -> ImuPanoramaBridgeEstimate | None:
    """Robustly calibrate the fixed IMU-body to panorama-camera rotation.

    For each trustworthy directly decoded multi-Tag visual attitude ``V_i``
    and synchronous body attitude ``I_i`` the model is::

        V_i = A I_i X

    where ``X`` is the fixed panorama-camera-to-body bridge and ``A`` absorbs
    arbitrary IMU world heading into the tag-map parent frame. Relative
    rotations eliminate ``A``::

        V_i.T V_j = X.T (I_i.T I_j) X

    so their rotation vectors can be aligned on SO(3).  Pair-angle agreement,
    robust IRLS and a final absolute-attitude inlier gate keep planar PnP branch
    flips from contaminating the bridge.
    """
    visual = [project_to_so3(rotation) for rotation in visual_camera_to_parent]
    imu = [project_to_so3(rotation) for rotation in imu_body_to_world]
    if len(visual) != len(imu) or len(visual) < min_observations:
        return None

    imu_vectors: list[np.ndarray] = []
    visual_vectors: list[np.ndarray] = []
    base_weights: list[float] = []
    pair_indices: list[tuple[int, int]] = []
    minimum_angle = math.radians(min_relative_rotation_deg)
    maximum_angle_error = math.radians(max_pair_angle_error_deg)
    for left in range(len(visual)):
        for right in range(left + 1, len(visual)):
            imu_vector = Rotation.from_matrix(
                imu[left].T @ imu[right]
            ).as_rotvec()
            visual_vector = Rotation.from_matrix(
                visual[left].T @ visual[right]
            ).as_rotvec()
            imu_angle = float(np.linalg.norm(imu_vector))
            visual_angle = float(np.linalg.norm(visual_vector))
            if (
                imu_angle < minimum_angle
                or imu_angle > math.radians(150.0)
                or abs(imu_angle - visual_angle) > maximum_angle_error
            ):
                continue
            imu_vectors.append(imu_vector)
            visual_vectors.append(visual_vector)
            # Long baselines are more informative, but cap their leverage.
            base_weights.append(min(imu_angle, math.radians(30.0)))
            pair_indices.append((left, right))
    if len(imu_vectors) < 3:
        return None

    imu_array = np.asarray(imu_vectors, dtype=np.float64)
    visual_array = np.asarray(visual_vectors, dtype=np.float64)
    base_weight_array = np.asarray(base_weights, dtype=np.float64)
    singular = np.linalg.svd(imu_array, compute_uv=False)
    excitation = float(singular[0])
    axis_ratio = float(singular[1] / singular[0]) if singular[0] > 0 else 0.0
    if (
        math.degrees(excitation) < min_relative_rotation_deg * 2.0
        or axis_ratio < min_excitation_axis_ratio
    ):
        return None

    weights = base_weight_array.copy()
    bridge: np.ndarray | None = None
    for _iteration in range(6):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                aligned, _rssd = Rotation.align_vectors(
                    imu_array, visual_array, weights=weights,
                )
        except (ValueError, np.linalg.LinAlgError):
            return None
        bridge = aligned.as_matrix()
        vector_residuals = np.linalg.norm(
            (bridge @ visual_array.T).T - imu_array, axis=1,
        )
        median = float(np.median(vector_residuals))
        mad = float(np.median(np.abs(vector_residuals - median)))
        robust_scale = max(math.radians(0.5), 1.4826 * mad)
        normalized = vector_residuals / (2.5 * robust_scale)
        robust_weights = np.ones_like(normalized)
        outlier = normalized > 1.0
        robust_weights[outlier] = 1.0 / normalized[outlier]
        next_weights = base_weight_array * robust_weights
        if np.allclose(next_weights, weights, rtol=1e-3, atol=1e-8):
            break
        weights = next_weights
    if bridge is None:
        return None
    bridge = project_to_so3(bridge)

    alignment_samples = [
        visual_rotation @ bridge.T @ imu_rotation.T
        for visual_rotation, imu_rotation in zip(visual, imu)
    ]
    world_alignment = _chordal_rotation_mean(alignment_samples)
    residuals = np.asarray(
        [
            rotation_residual_deg(
                visual_rotation,
                world_alignment @ imu_rotation @ bridge,
            )
            for visual_rotation, imu_rotation in zip(visual, imu)
        ],
        dtype=np.float64,
    )
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    data_gate = median + max(3.0, 3.0 * 1.4826 * mad)
    inlier_gate = min(max_fit_residual_deg, data_gate)
    inlier_mask = residuals <= inlier_gate
    if int(np.count_nonzero(inlier_mask)) < min_observations:
        return None

    # One robust refit on observation inliers removes pairwise contamination
    # from a single planar-flip observation.
    inlier_indices = np.flatnonzero(inlier_mask)
    refit_imu_vectors: list[np.ndarray] = []
    refit_visual_vectors: list[np.ndarray] = []
    refit_weights: list[float] = []
    for offset, left in enumerate(inlier_indices):
        for right in inlier_indices[offset + 1 :]:
            imu_vector = Rotation.from_matrix(imu[left].T @ imu[right]).as_rotvec()
            visual_vector = Rotation.from_matrix(
                visual[left].T @ visual[right]
            ).as_rotvec()
            imu_angle = float(np.linalg.norm(imu_vector))
            if (
                imu_angle >= minimum_angle
                and abs(imu_angle - float(np.linalg.norm(visual_vector)))
                <= maximum_angle_error
            ):
                refit_imu_vectors.append(imu_vector)
                refit_visual_vectors.append(visual_vector)
                refit_weights.append(min(imu_angle, math.radians(30.0)))
    if len(refit_imu_vectors) >= 3:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            bridge = Rotation.align_vectors(
                np.asarray(refit_imu_vectors),
                np.asarray(refit_visual_vectors),
                weights=np.asarray(refit_weights),
            )[0].as_matrix()
        bridge = project_to_so3(bridge)
    alignment_samples = [
        visual[index] @ bridge.T @ imu[index].T for index in inlier_indices
    ]
    world_alignment = _chordal_rotation_mean(alignment_samples)
    residuals = np.asarray(
        [
            rotation_residual_deg(
                visual_rotation,
                world_alignment @ imu_rotation @ bridge,
            )
            for visual_rotation, imu_rotation in zip(visual, imu)
        ],
        dtype=np.float64,
    )
    inlier_mask = residuals <= max_fit_residual_deg
    if int(np.count_nonzero(inlier_mask)) < min_observations:
        return None
    inlier_residuals = residuals[inlier_mask]
    return ImuPanoramaBridgeEstimate(
        bridge,
        world_alignment,
        len(visual),
        int(np.count_nonzero(inlier_mask)),
        len(pair_indices),
        math.degrees(excitation),
        axis_ratio,
        float(np.median(inlier_residuals)),
        float(np.percentile(inlier_residuals, 95)),
        float(np.max(inlier_residuals)),
        inlier_mask,
    )


class ImuPanoramaBridgeCalibrator:
    """Incrementally estimate one camera's fixed IMU/panorama SO(3) bridge."""

    def __init__(
        self,
        *,
        min_observations: int = 4,
        max_observations: int = 48,
        max_fit_residual_deg: float = 15.0,
    ) -> None:
        self.min_observations = int(min_observations)
        self.max_observations = int(max_observations)
        self.max_fit_residual_deg = float(max_fit_residual_deg)
        self.frames: list[int] = []
        self._visual: list[np.ndarray] = []
        self._imu: list[np.ndarray] = []
        self.estimate: ImuPanoramaBridgeEstimate | None = None
        self.first_calibrated_frame: int | None = None
        self.last_calibrated_frame: int | None = None
        self.revision_count = 0

    def add_observation(
        self,
        frame: int,
        visual_camera_to_parent: np.ndarray,
        imu_quaternion: np.ndarray,
    ) -> bool:
        """Add one trusted direct multi-Tag attitude; return true on update."""
        if frame in self.frames or len(self.frames) >= self.max_observations:
            return False
        try:
            imu_rotation = quaternion_to_rotation(imu_quaternion)
        except ValueError:
            return False
        self.frames.append(int(frame))
        self._visual.append(project_to_so3(visual_camera_to_parent))
        self._imu.append(imu_rotation)
        candidate = estimate_imu_panorama_bridge(
            self._visual,
            self._imu,
            min_observations=self.min_observations,
            max_fit_residual_deg=self.max_fit_residual_deg,
        )
        if candidate is None:
            return False
        self.estimate = candidate
        self.revision_count += 1
        self.last_calibrated_frame = int(frame)
        if self.first_calibrated_frame is None:
            self.first_calibrated_frame = int(frame)
        return True

    def predict(self, imu_quaternion: np.ndarray) -> np.ndarray | None:
        if self.estimate is None:
            return None
        try:
            return self.estimate.predict_camera_to_parent(imu_quaternion)
        except ValueError:
            return None

    @property
    def panorama_to_body(self) -> np.ndarray | None:
        return None if self.estimate is None else self.estimate.panorama_to_body

    @property
    def status(self) -> str:
        if self.estimate is not None:
            return "calibrated"
        if len(self.frames) < self.min_observations:
            return "collecting"
        return "insufficient_excitation"

    def audit(self) -> dict[str, object]:
        result: dict[str, object] = {
            "model": "R_parent_camera = R_parent_imu_world * "
            "R_imu_world_body * R_body_from_panorama",
            "status": self.status,
            "observation_count": len(self.frames),
            "observation_frames": self.frames.copy(),
            "minimum_observations": self.min_observations,
            "maximum_observations": self.max_observations,
            "maximum_fit_residual_deg": self.max_fit_residual_deg,
            "first_calibrated_frame": self.first_calibrated_frame,
            "last_calibrated_frame": self.last_calibrated_frame,
            "revision_count": self.revision_count,
        }
        estimate = self.estimate
        if estimate is None:
            return result
        bridge_quaternion = Rotation.from_matrix(
            estimate.panorama_to_body
        ).as_quat()
        alignment_quaternion = Rotation.from_matrix(
            estimate.imu_world_to_parent
        ).as_quat()
        result.update(
            inlier_count=estimate.inlier_count,
            inlier_frames=[
                frame
                for frame, is_inlier in zip(self.frames, estimate.inlier_mask)
                if bool(is_inlier)
            ],
            pair_count=estimate.pair_count,
            excitation_deg=estimate.excitation_deg,
            excitation_axis_ratio=estimate.excitation_axis_ratio,
            residual_median_deg=estimate.residual_median_deg,
            residual_p95_deg=estimate.residual_p95_deg,
            residual_max_deg=estimate.residual_max_deg,
            panorama_to_body_matrix=estimate.panorama_to_body.tolist(),
            panorama_to_body_quaternion_xyzw=bridge_quaternion.tolist(),
            imu_world_to_parent_matrix=estimate.imu_world_to_parent.tolist(),
            imu_world_to_parent_quaternion_xyzw=alignment_quaternion.tolist(),
        )
        return result


def is_imu_attitude_source(source: str) -> bool:
    return source in {"imu_relative", "imu_calibrated_bridge"}


def quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    """Return a body-to-world rotation for a ``[w, x, y, z]`` quaternion."""
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
    view: View,
    previous_quaternion: np.ndarray,
    current_quaternion: np.ndarray,
    panorama_to_body: np.ndarray | None = None,
) -> View:
    """Keep a static world-facing perspective basis stable during rotation."""
    previous_basis = view_to_panorama_rotation(view.yaw, view.pitch, view.roll)
    body_to_world_previous = quaternion_to_rotation(previous_quaternion)
    body_to_world_current = quaternion_to_rotation(current_quaternion)
    camera_to_body = (
        default_panorama_to_body_rotation()
        if panorama_to_body is None
        else project_to_so3(panorama_to_body)
    )
    current_basis = (
        camera_to_body.T
        @ body_to_world_current.T
        @ body_to_world_previous
        @ camera_to_body
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


def predict_camera_to_parent_rotation(
    previous_camera_to_parent: np.ndarray,
    previous_quaternion: np.ndarray,
    current_quaternion: np.ndarray,
    panorama_to_body: np.ndarray | None = None,
) -> np.ndarray:
    """Propagate a camera-to-parent attitude with relative IMU rotation.

    Only relative body rotation is used, so arbitrary IMU world heading
    cancels. ``panorama_to_body`` is the calibrated per-camera bridge.
    """
    previous = np.asarray(previous_camera_to_parent, dtype=np.float64).reshape(3, 3)
    body_to_world_previous = quaternion_to_rotation(previous_quaternion)
    body_to_world_current = quaternion_to_rotation(current_quaternion)
    camera_to_body = (
        default_panorama_to_body_rotation()
        if panorama_to_body is None
        else project_to_so3(panorama_to_body)
    )
    predicted = (
        previous
        @ camera_to_body.T
        @ body_to_world_previous.T
        @ body_to_world_current
        @ camera_to_body
    )
    # Project away harmless floating-point drift before scipy/OpenCV consume it.
    return project_to_so3(predicted)


def predict_camera_to_parent_rotation_hypotheses(
    previous_camera_to_parent: np.ndarray,
    previous_quaternion: np.ndarray,
    current_quaternion: np.ndarray,
    panorama_to_body: np.ndarray | None = None,
) -> tuple[tuple[str, np.ndarray], tuple[str, np.ndarray]]:
    """Return nominal and inverse-relative attitude hypotheses.

    The inverse-relative hypothesis is recovery-only: decoded Tag corners must
    independently validate it before use.
    """
    nominal = predict_camera_to_parent_rotation(
        previous_camera_to_parent,
        previous_quaternion,
        current_quaternion,
        panorama_to_body,
    )
    previous = np.asarray(previous_camera_to_parent, dtype=np.float64).reshape(3, 3)
    body_to_world_previous = quaternion_to_rotation(previous_quaternion)
    body_to_world_current = quaternion_to_rotation(current_quaternion)
    camera_to_body = (
        default_panorama_to_body_rotation()
        if panorama_to_body is None
        else project_to_so3(panorama_to_body)
    )
    inverse_relative = (
        previous
        @ camera_to_body.T
        @ body_to_world_current.T
        @ body_to_world_previous
        @ camera_to_body
    )
    inverse_relative = project_to_so3(inverse_relative)
    return (
        ("nominal", nominal),
        ("inverse_relative_recovery", inverse_relative),
    )


def rotation_residual_deg(actual: np.ndarray, expected: np.ndarray) -> float:
    """Return the shortest SO(3) angle between two rotation matrices."""
    relative = np.asarray(expected, dtype=np.float64).reshape(3, 3).T @ np.asarray(
        actual, dtype=np.float64,
    ).reshape(3, 3)
    return float(np.degrees(Rotation.from_matrix(relative).magnitude()))


def load_imu_quaternions(path: Path | None) -> dict[int, np.ndarray]:
    """Load per-source-frame camera IMU quaternion export."""
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


@dataclass(frozen=True)
class CaptureTagInstance:
    """One physical marker instance in a capture-only duplicate-ID map."""

    raw_id: int
    virtual_id: str
    panel: str
    corners_m: np.ndarray

    @property
    def center_m(self) -> np.ndarray:
        return self.corners_m.mean(axis=0)


@dataclass(frozen=True)
class CaptureDuplicateTagMap:
    """Explicit, opt-in map for a legacy capture with reused decoded IDs.

    This type is intentionally separate from :class:`IndependentTagMap`.
    Production world maps keep the global-ID uniqueness invariant; a capture
    can opt into duplicate resolution only by naming every physical instance
    and every panel in a dedicated capture map.
    """

    panels: dict[str, dict[int, CaptureTagInstance]]
    metadata: dict

    @property
    def expected_ids(self) -> list[int]:
        return sorted({raw_id for panel in self.panels.values() for raw_id in panel})

    @property
    def expected_virtual_ids(self) -> list[str]:
        return sorted(
            instance.virtual_id
            for panel in self.panels.values()
            for instance in panel.values()
        )

    @property
    def duplicate_raw_ids(self) -> list[int]:
        counts = Counter(
            raw_id for panel in self.panels.values() for raw_id in panel
        )
        return sorted(raw_id for raw_id, count in counts.items() if count > 1)

    def instance_options(self, raw_id: int) -> list[CaptureTagInstance]:
        return [
            panel[raw_id]
            for panel in self.panels.values()
            if raw_id in panel
        ]


def load_tag_map(path: Path) -> IndependentTagMap:
    """Load an explicit per-ID 4x3 corner map in OpenCV marker order."""
    payload = compile_world_tag_map(path)
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


def load_capture_duplicate_tag_map(path: Path) -> CaptureDuplicateTagMap:
    """Load a capture-scoped panel map that explicitly permits reused IDs.

    The file is deliberately not accepted by ``--tag-map`` and must declare
    ``allow_duplicate_decoded_ids: true``.  Each source panel remains a normal
    unique-ID map; duplication is represented only by distinct virtual IDs
    after its rigid transform into the common world frame.  No scale parameter
    exists in this format, so a Sim(3) cannot be introduced accidentally.
    """
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "capture-duplicate-apriltag-map/1.0":
        raise ValueError("unsupported capture duplicate tag-map schema")
    if payload.get("allow_duplicate_decoded_ids") is not True:
        raise ValueError(
            "capture duplicate map must explicitly set "
            "allow_duplicate_decoded_ids=true"
        )
    world_frame = str(payload.get("world_frame", "")).strip()
    panel_entries = payload.get("panels")
    if not world_frame or not isinstance(panel_entries, list) or not panel_entries:
        raise ValueError("capture duplicate map needs world_frame and panels")

    panels: dict[str, dict[int, CaptureTagInstance]] = {}
    virtual_ids: set[str] = set()
    for panel_entry in panel_entries:
        panel_name = str(panel_entry.get("name", "")).strip()
        if not panel_name or panel_name in panels:
            raise ValueError(f"invalid or repeated panel name {panel_name!r}")
        transform = RigidTransform.from_dict(panel_entry["T_world_map"])
        if transform.parent_frame != world_frame:
            raise ValueError("panel transform parent must equal world_frame")
        source_path = (path.parent / panel_entry["tag_map"]).resolve()
        source_payload = compile_world_tag_map(source_path)
        allowed = set(map(int, panel_entry.get("expected_ids", [])))
        panel_instances: dict[int, CaptureTagInstance] = {}
        for source_tag in source_payload["tags"]:
            raw_id = int(source_tag["id"])
            if allowed and raw_id not in allowed:
                continue
            if raw_id in panel_instances:
                raise ValueError(
                    f"panel {panel_name} repeats decoded id {raw_id}; "
                    "duplicates must be on distinct physical panels"
                )
            virtual_id = f"{panel_name}:{raw_id}"
            if virtual_id in virtual_ids:
                raise ValueError(f"duplicate virtual tag id {virtual_id}")
            corners = transform.apply_points(
                np.asarray(source_tag["corners_m"], dtype=np.float64)
            ).astype(np.float32)
            panel_instances[raw_id] = CaptureTagInstance(
                raw_id, virtual_id, panel_name, corners,
            )
            virtual_ids.add(virtual_id)
        if allowed and set(panel_instances) != allowed:
            raise ValueError(
                f"panel {panel_name} IDs {sorted(panel_instances)} do not match "
                f"expected {sorted(allowed)}"
            )
        if not panel_instances:
            raise ValueError(f"panel {panel_name} contains no tags")
        panels[panel_name] = panel_instances

    instance_payload = [
        {
            "raw_id": instance.raw_id,
            "virtual_id": instance.virtual_id,
            "panel": instance.panel,
            "corners_m": instance.corners_m.astype(float).tolist(),
        }
        for panel in panels.values()
        for instance in panel.values()
    ]
    metadata = {
        key: value for key, value in payload.items()
        if key not in {"panels", "tag_map_sha256"}
    }
    hash_payload = dict(metadata)
    hash_payload["instances"] = instance_payload
    metadata.update(
        source_path=str(path),
        panels=panel_entries,
        tag_map_sha256=canonical_sha256(hash_payload),
        expected_virtual_ids=sorted(virtual_ids),
    )
    result = CaptureDuplicateTagMap(panels, metadata)
    declared_duplicates = sorted(map(int, payload.get("duplicate_raw_ids", [])))
    if declared_duplicates and declared_duplicates != result.duplicate_raw_ids:
        raise ValueError(
            f"declared duplicate IDs {declared_duplicates} do not match "
            f"physical instances {result.duplicate_raw_ids}"
        )
    if not result.duplicate_raw_ids:
        raise ValueError(
            "capture duplicate map contains no repeated decoded IDs; use a "
            "normal production --tag-map instead"
        )
    return result


@dataclass
class Pose:
    xyz: np.ndarray
    rotation_camera_to_board: np.ndarray
    rpy: tuple[float, float, float]
    inliers: int
    rmse: float
    view: str
    ids: list[int]
    attitude_residual_deg: float | None = None
    attitude_source: str = "visual_reprojection"
    visual_attitude_residual_deg: float | None = None
    attitude_hypothesis: str = "nominal"
    capture_panel: str | None = None
    virtual_instance_ids: tuple[str, ...] = ()


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


def find_duplicate_tag_ray_conflicts(
    view_records: Iterable[dict],
    minimum_separation_deg: float = 25.0,
) -> list[int]:
    """Find IDs decoded on two physically distinct panorama bearings.

    Overlapping perspective views intentionally observe the same physical Tag
    more than once.  Their reconstructed panorama rays agree.  The same ID on
    two room panels instead produces widely separated rays; such a frame is
    unsafe for world-map pose/IMU-bridge calibration unless the capture has an
    explicit panel-disambiguation rule.
    """
    rays: dict[int, list[np.ndarray]] = {}
    for record in view_records:
        view_payload = record.get("view")
        size = record.get("size")
        detections = record.get("detections")
        if (
            not isinstance(view_payload, dict)
            or not isinstance(size, (int, float))
            or not isinstance(detections, list)
            or float(size) <= 0
        ):
            continue
        try:
            view = View(
                str(view_payload.get("name", "audit")),
                float(view_payload["yaw"]),
                float(view_payload["pitch"]),
                float(view_payload.get("fov", 100.0)),
                float(view_payload.get("roll", 0.0)),
            )
            intrinsics = perspective_intrinsics(int(size), view.fov)
            view_to_panorama = view_to_panorama_rotation(
                view.yaw, view.pitch, view.roll,
            )
        except (KeyError, TypeError, ValueError):
            continue
        for detection in detections:
            try:
                corners = np.asarray(
                    detection["corners_px"], dtype=np.float64,
                ).reshape(4, 2)
                center = corners.mean(axis=0)
                ray_view = np.array(
                    [
                        (center[0] - intrinsics[0, 2]) / intrinsics[0, 0],
                        (center[1] - intrinsics[1, 2]) / intrinsics[1, 1],
                        1.0,
                    ],
                    dtype=np.float64,
                )
                ray = view_to_panorama @ ray_view
                ray /= np.linalg.norm(ray)
                tag_id = int(detection["id"])
            except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
                continue
            rays.setdefault(tag_id, []).append(ray)
    cosine_gate = math.cos(math.radians(minimum_separation_deg))
    conflicts = []
    for tag_id, tag_rays in rays.items():
        if any(
            float(np.dot(tag_rays[left], tag_rays[right])) < cosine_gate
            for left in range(len(tag_rays))
            for right in range(left + 1, len(tag_rays))
        ):
            conflicts.append(tag_id)
    return sorted(conflicts)


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


def solve_fixed_attitude_translation(
    object_points: np.ndarray,
    image_points: np.ndarray,
    view: View,
    size: int,
    camera_to_board: np.ndarray,
    max_rmse_px: float,
) -> tuple[np.ndarray, float] | None:
    """Solve only camera translation while keeping IMU attitude immutable.

    ``camera_to_board`` is the predicted panorama-camera attitude in the map
    frame.  Directly decoded tag corners constrain the remaining three
    translation parameters.  A linear ray-equation solution initializes a
    robust pixel-domain least-squares refinement; neither stage can alter the
    supplied rotation.
    """
    obj = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    img = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    expected = np.asarray(camera_to_board, dtype=np.float64).reshape(3, 3)
    if (
        len(obj) < 4
        or len(obj) != len(img)
        or not np.isfinite(obj).all()
        or not np.isfinite(img).all()
        or not np.isfinite(expected).all()
    ):
        return None
    view_to_panorama = view_to_panorama_rotation(
        view.yaw, view.pitch, view.roll,
    )
    board_to_view = view_to_panorama.T @ expected.T
    rotated = (board_to_view @ obj.T).T
    intrinsics = perspective_intrinsics(size, view.fov)
    normalized_x = (img[:, 0] - intrinsics[0, 2]) / intrinsics[0, 0]
    normalized_y = (img[:, 1] - intrinsics[1, 2]) / intrinsics[1, 1]
    system = np.zeros((2 * len(obj), 3), dtype=np.float64)
    rhs = np.zeros(2 * len(obj), dtype=np.float64)
    system[0::2, 0] = 1.0
    system[0::2, 2] = -normalized_x
    system[1::2, 1] = 1.0
    system[1::2, 2] = -normalized_y
    rhs[0::2] = normalized_x * rotated[:, 2] - rotated[:, 0]
    rhs[1::2] = normalized_y * rotated[:, 2] - rotated[:, 1]
    initial, _residuals, rank, _singular = np.linalg.lstsq(
        system, rhs, rcond=None,
    )
    if rank < 3 or not np.isfinite(initial).all():
        return None

    # This lower bound guarantees every mapped corner remains in front of the
    # perspective camera throughout optimization.
    minimum_tz = float(-np.min(rotated[:, 2]) + 1e-5)
    if initial[2] <= minimum_tz:
        initial[2] = minimum_tz + 0.01

    def pixel_residual(translation: np.ndarray) -> np.ndarray:
        camera_points = rotated + translation.reshape(1, 3)
        projected = np.column_stack(
            (
                intrinsics[0, 0] * camera_points[:, 0] / camera_points[:, 2]
                + intrinsics[0, 2],
                intrinsics[1, 1] * camera_points[:, 1] / camera_points[:, 2]
                + intrinsics[1, 2],
            )
        )
        return (projected - img).reshape(-1)

    try:
        optimized = least_squares(
            pixel_residual,
            initial,
            bounds=(
                np.array([-np.inf, -np.inf, minimum_tz]),
                np.array([np.inf, np.inf, np.inf]),
            ),
            loss="soft_l1",
            f_scale=max(1.0, min(3.0, max_rmse_px)),
            max_nfev=100,
        )
    except ValueError:
        return None
    translation = optimized.x
    camera_points = rotated + translation.reshape(1, 3)
    if (
        not np.isfinite(translation).all()
        or not np.isfinite(camera_points).all()
        or np.any(camera_points[:, 2] <= 0)
    ):
        return None
    residual = pixel_residual(translation).reshape(-1, 2)
    rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    if not math.isfinite(rmse) or rmse > max_rmse_px:
        return None
    fixed_rvec, _ = cv2.Rodrigues(board_to_view)
    xyz, recovered_attitude = pose_view_to_panorama(
        fixed_rvec, translation.reshape(3, 1), view,
    )
    if (
        not np.isfinite(xyz).all()
        or rotation_residual_deg(recovered_attitude, expected) > 1e-6
    ):
        return None
    return xyz, rmse


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
    grid: Grid | IndependentTagMap | CaptureDuplicateTagMap,
) -> list[dict]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _rejected = detector.detectMarkers(gray)
    found: list[dict] = []
    if ids is None:
        return found
    for marker_corners, raw_id in zip(corners, ids.flatten()):
        tag_id = int(raw_id)
        if isinstance(grid, CaptureDuplicateTagMap):
            options = grid.instance_options(tag_id)
            if not options:
                continue
            px = marker_corners.reshape(4, 2).astype(np.float32)
            found.append(
                {
                    "id": tag_id,
                    "corners_px": px,
                    "center_px": px.mean(axis=0),
                    "area_px2": abs(float(cv2.contourArea(px))),
                    "instance_options": tuple(options),
                }
            )
            continue
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


def detection_audit_record(detection: dict) -> dict:
    """Serialize one decoded marker without losing duplicate-map hypotheses."""
    record = {
        "id": int(detection["id"]),
        "corners_px": detection["corners_px"].round(2).tolist(),
    }
    options = detection.get("instance_options")
    if options:
        record["virtual_instance_options"] = [
            option.virtual_id for option in options
        ]
    if detection.get("virtual_instance_id"):
        record["virtual_instance_id"] = detection["virtual_instance_id"]
        record["panel"] = detection.get("panel")
    return record


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
    expected_rotation_camera_to_board: np.ndarray | None = None,
    attitude_source: str = "visual_reprojection",
    max_attitude_residual_deg: float | None = None,
    allow_imu_translation_fallback: bool = False,
    imu_translation_fallback_rotations: tuple[
        tuple[str, np.ndarray], ...
    ] = (),
    max_imu_translation_rmse_px: float | None = None,
    diagnostics: dict | None = None,
    allow_single_tag_imu_translation_fallback: bool = False,
    max_single_tag_imu_translation_rmse_px: float = 5.0,
) -> Pose | None:
    # Duplicate physical instances are never silently collapsed.  A normal
    # production map treats two decodes of the same ID in one perspective as a
    # hard conflict.  Only an explicit capture duplicate map supplies named
    # instance hypotheses, which are solved panel-by-panel below.
    has_instance_hypotheses = any(
        detection.get("instance_options") for detection in detections
    )
    if has_instance_hypotheses:
        detection_indices = {
            id(detection): index for index, detection in enumerate(detections)
        }
        panel_groups: dict[str, dict[int, CaptureTagInstance]] = {}
        for detection in detections:
            for instance in detection.get("instance_options", ()):
                panel_groups.setdefault(instance.panel, {})[instance.raw_id] = instance
        hypothesis_audit: list[dict[str, object]] = []
        panel_candidates: list[Pose] = []
        # Evaluate which image occurrence belongs to which named panel.  A raw
        # ID can appear twice in a perspective; enumerate that small choice but
        # never place both occurrences at the same world point.
        for panel_name, panel_instances in panel_groups.items():
            by_raw_id: dict[int, list[dict]] = {}
            for detection in detections:
                raw_id = int(detection["id"])
                if raw_id in panel_instances:
                    by_raw_id.setdefault(raw_id, []).append(detection)
            ordered_ids = sorted(by_raw_id)
            if not ordered_ids:
                continue
            combinations = product(*(by_raw_id[raw_id] for raw_id in ordered_ids))
            for combination_index, selected_detections in enumerate(combinations):
                if combination_index >= 128:
                    hypothesis_audit.append(
                        {
                            "panel": panel_name,
                            "status": "combination_limit_reached",
                            "limit": 128,
                        }
                    )
                    break
                mapped: list[dict] = []
                virtual_ids: list[str] = []
                for raw_id, detection in zip(ordered_ids, selected_detections):
                    instance = panel_instances[raw_id]
                    mapped_detection = {
                        key: value for key, value in detection.items()
                        if key != "instance_options"
                    }
                    mapped_detection.update(
                        object_center=instance.center_m.astype(np.float32),
                        object_corners=instance.corners_m.copy(),
                        virtual_instance_id=instance.virtual_id,
                        panel=panel_name,
                    )
                    mapped.append(mapped_detection)
                    virtual_ids.append(instance.virtual_id)
                local_diagnostics: dict = {}
                candidate = solve_view(
                    mapped,
                    view,
                    size,
                    min_tags,
                    max_rmse_px,
                    pnp_points,
                    pnp_solver,
                    expected_rotation_camera_to_board,
                    attitude_source,
                    max_attitude_residual_deg,
                    allow_imu_translation_fallback,
                    imu_translation_fallback_rotations,
                    max_imu_translation_rmse_px,
                    local_diagnostics,
                    allow_single_tag_imu_translation_fallback,
                    max_single_tag_imu_translation_rmse_px,
                )
                audit_entry: dict[str, object] = {
                    "panel": panel_name,
                    "virtual_instance_ids": virtual_ids,
                    "image_occurrence_indices": [
                        detection_indices[id(detection)]
                        for detection in selected_detections
                    ],
                    "status": "rejected" if candidate is None else "accepted",
                }
                if local_diagnostics:
                    audit_entry["solve_diagnostics"] = local_diagnostics
                if candidate is not None:
                    candidate = replace(
                        candidate,
                        capture_panel=panel_name,
                        virtual_instance_ids=tuple(sorted(virtual_ids)),
                    )
                    panel_candidates.append(candidate)
                    audit_entry.update(
                        rmse_px=float(candidate.rmse),
                        attitude_residual_deg=candidate.attitude_residual_deg,
                    )
                hypothesis_audit.append(audit_entry)
        if diagnostics is not None:
            diagnostics.update(
                duplicate_panel_resolution=True,
                duplicate_decoded_ids=sorted(
                    raw_id
                    for raw_id, count in Counter(
                        int(detection["id"]) for detection in detections
                    ).items()
                    if count > 1
                ),
                panel_hypotheses=hypothesis_audit,
            )
        if not panel_candidates:
            return None
        selected = choose_pose(panel_candidates, max_attitude_residual_deg)
        if diagnostics is not None and selected is not None:
            diagnostics.update(
                selected_capture_panel=selected.capture_panel,
                selected_virtual_instance_ids=list(selected.virtual_instance_ids),
            )
        return selected

    duplicate_decoded_ids = sorted(
        raw_id
        for raw_id, count in Counter(
            int(detection["id"]) for detection in detections
        ).items()
        if count > 1
    )
    if duplicate_decoded_ids:
        if diagnostics is not None:
            diagnostics.update(
                visual_pose_failure_reason="duplicate_decoded_id_conflict",
                duplicate_decoded_ids=duplicate_decoded_ids,
                duplicate_resolution="rejected_production_unique_map",
            )
        return None

    def fixed_translation_hypotheses(
        supported: list[dict], gate_px: float,
    ) -> tuple[
        np.ndarray,
        list[tuple[float, str, np.ndarray, np.ndarray]],
        list[dict[str, object]],
    ]:
        fixed_object_points = np.concatenate(
            [detection["object_corners"] for detection in supported]
        ).astype(np.float64)
        fixed_image_points = np.concatenate(
            [detection["corners_px"] for detection in supported]
        ).astype(np.float64)
        primary_hypothesis = (
            "calibrated_bridge"
            if attitude_source == "imu_calibrated_bridge" else "nominal"
        )
        rotation_hypotheses = (
            ((primary_hypothesis, expected_rotation_camera_to_board),)
            + tuple(imu_translation_fallback_rotations)
        )
        fixed_candidates: list[
            tuple[float, str, np.ndarray, np.ndarray]
        ] = []
        attempts: list[dict[str, object]] = []
        for hypothesis_name, hypothesis_rotation in rotation_hypotheses:
            if hypothesis_rotation is None:
                continue
            # Request the finite/depth-valid score first. Applying the gate in
            # this caller keeps rejected recoveries visible in diagnostics.
            fixed = solve_fixed_attitude_translation(
                fixed_object_points,
                fixed_image_points,
                view,
                size,
                hypothesis_rotation,
                math.inf,
            )
            attempt: dict[str, object] = {
                "hypothesis": hypothesis_name,
                "status": "geometry_failed" if fixed is None else "rmse_rejected",
                "rmse_gate_px": float(gate_px),
            }
            if fixed is not None:
                fixed_xyz, fixed_rmse = fixed
                attempt.update(
                    rmse_px=float(fixed_rmse),
                    xyz=np.asarray(fixed_xyz, dtype=float).tolist(),
                )
                if fixed_rmse <= gate_px:
                    attempt["status"] = "accepted"
                    fixed_candidates.append(
                        (
                            fixed_rmse,
                            hypothesis_name,
                            np.asarray(hypothesis_rotation, dtype=np.float64)
                            .reshape(3, 3)
                            .copy(),
                            fixed_xyz,
                        )
                    )
            attempts.append(attempt)
        return fixed_object_points, fixed_candidates, attempts

    def fixed_multitag_pose_without_visual_candidate(
        failure_reason: str,
    ) -> Pose | None:
        """Recover translation when visual PnP cannot form any pose branch."""
        if not (
            allow_imu_translation_fallback
            and is_imu_attitude_source(attitude_source)
            and expected_rotation_camera_to_board is not None
            and len(detections) >= min_tags
        ):
            return None
        fallback_gate = (
            max_rmse_px
            if max_imu_translation_rmse_px is None
            else max_imu_translation_rmse_px
        )
        fixed_object_points, fixed_candidates, attempts = (
            fixed_translation_hypotheses(detections, fallback_gate)
        )
        if diagnostics is not None:
            diagnostics.update(
                visual_pose_failure_reason=failure_reason,
                imu_translation_fallback_without_visual_candidate=True,
                imu_translation_fallback_attempts=attempts,
            )
        if not fixed_candidates:
            return None
        fixed_rmse, hypothesis_name, fixed_rotation, fixed_xyz = min(
            fixed_candidates, key=lambda item: item[0]
        )
        tag_ids = sorted(int(detection["id"]) for detection in detections)
        return Pose(
            fixed_xyz,
            fixed_rotation,
            rotation_to_rpy(fixed_rotation),
            len(fixed_object_points),
            fixed_rmse,
            view.name,
            tag_ids,
            0.0,
            "imu_constrained_visual_translation",
            None,
            hypothesis_name,
        )

    if len(detections) < min_tags:
        if (
            len(detections) == 1
            and allow_single_tag_imu_translation_fallback
            and is_imu_attitude_source(attitude_source)
            and expected_rotation_camera_to_board is not None
        ):
            fixed_object_points, fixed_candidates, attempts = (
                fixed_translation_hypotheses(
                    detections, max_single_tag_imu_translation_rmse_px,
                )
            )
            if diagnostics is not None:
                diagnostics.update(
                    single_tag_imu_translation_attempted=True,
                    single_tag_imu_translation_attempts=attempts,
                )
            if fixed_candidates:
                fixed_rmse, hypothesis_name, fixed_rotation, fixed_xyz = min(
                    fixed_candidates, key=lambda item: item[0]
                )
                tag_ids = [int(detections[0]["id"])]
                return Pose(
                    fixed_xyz,
                    fixed_rotation,
                    rotation_to_rpy(fixed_rotation),
                    len(fixed_object_points),
                    fixed_rmse,
                    view.name,
                    tag_ids,
                    0.0,
                    "imu_constrained_single_tag_translation",
                    None,
                    hypothesis_name,
                )
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
    # IPPE requires Z=0 object coordinates, but the observed plane can have any
    # orientation in the world map.  Find a local right-handed plane frame with
    # SVD, solve there, then transform each candidate back to world coordinates.
    obj64 = obj.astype(np.float64)
    plane_origin = obj64.mean(axis=0)
    _u, singular_values, plane_axes = np.linalg.svd(
        obj64 - plane_origin, full_matrices=False,
    )
    scale = singular_values[0] if len(singular_values) else 0.0
    plane_tolerance = max(1e-7 * math.sqrt(len(obj64)), scale * 1e-6)
    rank_two_planar = bool(
        len(singular_values) >= 3
        and singular_values[1] > plane_tolerance
        and singular_values[2] <= plane_tolerance
    )
    use_iterative = pnp_solver == "iterative" or not rank_two_planar
    if use_iterative:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj, img, k, None, iterationsCount=200, reprojectionError=3.0,
            confidence=0.999, flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok or inliers is None:
            return fixed_multitag_pose_without_visual_candidate(
                "iterative_ransac_failed"
            )
        ii = inliers[:, 0]
        solutions = [(rvec, tvec)]
    else:
        plane_x = plane_axes[0]
        plane_y = plane_axes[1]
        plane_z = np.cross(plane_x, plane_y)
        plane_basis = np.column_stack((plane_x, plane_y, plane_z))
        local_obj = (obj64 - plane_origin) @ plane_basis
        # Remove the tiny SVD residual explicitly: IPPE validates that its
        # object points lie on Z=0.
        local_obj[:, 2] = 0.0
        # Filter decoded outliers with a planar homography first, then let IPPE
        # return both ambiguous pose solutions in the local plane frame.
        homography_obj = local_obj[:, :2]
        _h, mask = cv2.findHomography(
            homography_obj, img.astype(np.float64), cv2.RANSAC,
            max(3.0, max_rmse_px),
            maxIters=2000, confidence=0.999,
        )
        if mask is None:
            return fixed_multitag_pose_without_visual_candidate(
                "planar_homography_failed"
            )
        ii = np.flatnonzero(mask.ravel())
        min_points = min_tags * (4 if pnp_points == "corners" else 1)
        if len(ii) < min_points:
            return fixed_multitag_pose_without_visual_candidate(
                "planar_homography_insufficient_inliers"
            )
        ok, local_rvecs, local_tvecs, _errors = cv2.solvePnPGeneric(
            local_obj[ii], img[ii], k, None, flags=cv2.SOLVEPNP_IPPE,
        )
        if not ok:
            return fixed_multitag_pose_without_visual_candidate(
                "ippe_failed"
            )
        solutions = []
        for local_rvec, local_tvec in zip(local_rvecs, local_tvecs):
            local_to_view, _ = cv2.Rodrigues(local_rvec)
            world_to_view = local_to_view @ plane_basis.T
            world_tvec = (
                np.asarray(local_tvec, dtype=np.float64).reshape(3)
                - world_to_view @ plane_origin
            )
            world_rvec, _ = cv2.Rodrigues(world_to_view)
            solutions.append((world_rvec, world_tvec.reshape(3, 1)))
    min_points = min_tags * (4 if pnp_points == "corners" else 1)
    if len(ii) < min_points:
        return fixed_multitag_pose_without_visual_candidate(
            "pnp_insufficient_inliers"
        )
    inlier_ids = sorted(
        {
            int(detections[index // 4]["id"])
            if pnp_points == "corners"
            else int(detections[index]["id"])
            for index in ii
        }
    )

    def evaluate_solution(
        candidate_rvec: np.ndarray, candidate_tvec: np.ndarray,
    ) -> tuple[float, float | None, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        if not (np.isfinite(candidate_rvec).all() and np.isfinite(candidate_tvec).all()):
            return None
        camera_points, _ = cv2.projectPoints(
            obj[ii], candidate_rvec, candidate_tvec, k, None,
        )
        residual = camera_points.reshape(-1, 2) - img[ii]
        candidate_rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
        rotation, _ = cv2.Rodrigues(candidate_rvec)
        depths = (rotation @ obj[ii].T + candidate_tvec.reshape(3, 1))[2]
        if not (np.all(depths > 0) and math.isfinite(candidate_rmse)):
            return None
        xyz, camera_to_board = pose_view_to_panorama(
            candidate_rvec, candidate_tvec, view,
        )
        if not (np.isfinite(xyz).all() and np.isfinite(camera_to_board).all()):
            return None
        attitude_residual = (
            rotation_residual_deg(
                camera_to_board, expected_rotation_camera_to_board,
            )
            if expected_rotation_camera_to_board is not None
            else None
        )
        return (
            candidate_rmse,
            attitude_residual,
            candidate_rvec,
            candidate_tvec,
            xyz,
            camera_to_board,
        )

    # Select the IPPE branch before nonlinear refinement.  Refining both
    # branches first can make LM cross the shallow planar ambiguity and collapse
    # both seeds onto the same (possibly flipped) reprojection minimum.
    valid_solutions = []
    for candidate_rvec, candidate_tvec in solutions:
        evaluated = evaluate_solution(
            np.asarray(candidate_rvec, dtype=np.float64).copy(),
            np.asarray(candidate_tvec, dtype=np.float64).copy(),
        )
        if evaluated is not None:
            valid_solutions.append(evaluated)
    if not valid_solutions:
        return fixed_multitag_pose_without_visual_candidate(
            "no_depth_valid_visual_candidate"
        )
    if expected_rotation_camera_to_board is None:
        selected = min(valid_solutions, key=lambda item: item[0])
    else:
        # IPPE deliberately returns both planar branches.  Once a trusted
        # temporal attitude exists, choose the physically continuous branch;
        # reprojection RMSE only breaks a residual-angle tie.  The same score is
        # later used across overlapping perspective views.
        selected = min(
            valid_solutions,
            key=lambda item: (float(item[1]), item[0]),
        )
    minimum_visual_attitude_residual = (
        min(float(item[1]) for item in valid_solutions)
        if expected_rotation_camera_to_board is not None
        else None
    )
    if (
        allow_imu_translation_fallback
        and is_imu_attitude_source(attitude_source)
        and expected_rotation_camera_to_board is not None
        and max_attitude_residual_deg is not None
        and minimum_visual_attitude_residual is not None
        and minimum_visual_attitude_residual > max_attitude_residual_deg
        and len(inlier_ids) >= min_tags
    ):
        # Use only corners belonging to geometrically supported, directly
        # decoded tags.  The fallback never runs on propagated LK corners.
        inlier_id_set = set(inlier_ids)
        supported = [
            detection for detection in detections
            if int(detection["id"]) in inlier_id_set
        ]
        fallback_gate = (
            max_rmse_px
            if max_imu_translation_rmse_px is None
            else max_imu_translation_rmse_px
        )
        fixed_object_points, fixed_candidates, fallback_attempts = (
            fixed_translation_hypotheses(supported, fallback_gate)
        )
        if diagnostics is not None:
            diagnostics.update(
                attitude_gate_triggered=True,
                minimum_nominal_visual_attitude_residual_deg=float(
                    minimum_visual_attitude_residual
                ),
                imu_translation_fallback_attempts=fallback_attempts,
            )
        if fixed_candidates:
            fixed_rmse, hypothesis_name, fixed_rotation, fixed_xyz = min(
                fixed_candidates, key=lambda item: item[0]
            )
            selected_visual_residual = min(
                rotation_residual_deg(item[5], fixed_rotation)
                for item in valid_solutions
            )
            return Pose(
                fixed_xyz,
                fixed_rotation,
                rotation_to_rpy(fixed_rotation),
                len(fixed_object_points),
                fixed_rmse,
                view.name,
                inlier_ids,
                0.0,
                "imu_constrained_visual_translation",
                selected_visual_residual,
                hypothesis_name,
            )
    selected_rvec, selected_tvec = selected[2], selected[3]
    refined_rvec, refined_tvec = cv2.solvePnPRefineLM(
        obj[ii], img[ii], k, None, selected_rvec.copy(), selected_tvec.copy(),
    )
    refined = evaluate_solution(refined_rvec, refined_tvec)
    if refined is not None:
        if expected_rotation_camera_to_board is None:
            if refined[0] <= selected[0]:
                selected = refined
        elif (
            float(refined[1]) <= float(selected[1]) + 2.0
            and refined[0] <= selected[0]
        ):
            # Accept numerical refinement only while it remains on the selected
            # physical branch (two degrees is far below a planar flip).
            selected = refined
    rmse, attitude_residual, _rvec, _tvec, xyz, camera_to_board = selected
    if rmse > max_rmse_px:
        return None
    return Pose(
        xyz,
        camera_to_board,
        rotation_to_rpy(camera_to_board),
        len(ii),
        rmse,
        view.name,
        inlier_ids,
        attitude_residual,
        attitude_source,
        attitude_residual,
        (
            "calibrated_bridge"
            if attitude_source == "imu_calibrated_bridge" else "nominal"
        ),
    )


def pose_selection_key(pose: Pose) -> tuple[float, float, int]:
    """Use temporal attitude first, then visual fit and inlier support."""
    residual = (
        float(pose.attitude_residual_deg)
        if pose.attitude_residual_deg is not None
        else math.inf
    )
    return residual, pose.rmse, -pose.inliers


def pose_updates_authoritative_anchor(pose: Pose) -> bool:
    """Whether a published pose may advance temporal position/attitude state.

    A directly decoded single Tag plus fixed IMU attitude is useful as an
    auditable observation, but its translation is too weak to become the
    reference for subsequent IMU propagation.  Keeping the last multi-Tag
    anchor prevents one weak recovery from rotating every later pose.
    """
    return pose.attitude_source != "imu_constrained_single_tag_translation"


@dataclass(frozen=True)
class GuardedVisualReanchor:
    """Auditable decision to replace a failed IMU bridge with direct vision.

    A planar target has two IPPE branches with similar reprojection error.  A
    single perspective crop therefore cannot normally overrule an IMU prior.
    A re-anchor is allowed when two independently projected, directly decoded
    multi-Tag measurements agree in the common map frame, or during a very
    short gap when one crop directly decodes at least three Tags and its full
    pose is continuous with the last authoritative pose.
    """

    pose: Pose
    supporting_views: tuple[str, ...]
    position_spread_m: float
    attitude_spread_deg: float
    speed_m_s: float
    angular_speed_deg_s: float


def choose_guarded_visual_reanchor(
    candidates: list[Pose],
    measurement_sources: dict[int, str],
    candidate_views: dict[int, View],
    previous_position: np.ndarray | None,
    previous_rotation: np.ndarray | None,
    elapsed_s: float | None,
    *,
    min_tags: int,
    imu_gate_deg: float,
    max_rmse_px: float,
    max_speed_m_s: float,
    max_gap_s: float,
    max_position_spread_m: float,
    max_attitude_spread_deg: float,
    max_angular_speed_deg_s: float,
    min_view_separation_deg: float = 5.0,
    min_single_view_tags: int = 3,
    max_single_view_gap_s: float = 0.5,
    max_single_view_attitude_step_deg: float = 45.0,
) -> GuardedVisualReanchor | None:
    """Choose a safe direct-vision re-anchor after IMU self-check failure.

    Every accepted :class:`Pose` here already passed IPPE's positive-depth
    check in :func:`solve_view`.  This second gate prefers two geometrically
    distinct perspective projections.  Planar mirror solutions move with the
    virtual view and consequently fail common-world agreement.  A short-gap
    three-Tag temporal continuity fallback covers frames where only one crop
    decodes cleanly without weakening the two-Tag ambiguity rule.
    """

    if (
        previous_position is None
        or previous_rotation is None
        or elapsed_s is None
        or not math.isfinite(elapsed_s)
        or elapsed_s <= 0.0
        or elapsed_s > max_gap_s
    ):
        return None
    previous_xyz = np.asarray(previous_position, dtype=np.float64).reshape(3)
    previous_attitude = np.asarray(previous_rotation, dtype=np.float64).reshape(3, 3)
    if not (np.isfinite(previous_xyz).all() and np.isfinite(previous_attitude).all()):
        return None

    eligible: list[tuple[Pose, View, float, float]] = []
    for pose in candidates:
        view = candidate_views.get(id(pose))
        residual = pose.attitude_residual_deg
        if (
            view is None
            or measurement_sources.get(id(pose)) != "direct"
            or pose.attitude_source != "imu_relative"
            or residual is None
            or residual <= imu_gate_deg
            or len(set(pose.ids)) < min_tags
            or not math.isfinite(pose.rmse)
            or pose.rmse > max_rmse_px
            or not np.isfinite(pose.xyz).all()
            or not np.isfinite(pose.rotation_camera_to_board).all()
        ):
            continue
        speed = float(np.linalg.norm(pose.xyz - previous_xyz) / elapsed_s)
        angular_speed = float(
            rotation_residual_deg(
                pose.rotation_camera_to_board, previous_attitude,
            )
            / elapsed_s
        )
        if speed > max_speed_m_s or angular_speed > max_angular_speed_deg_s:
            continue
        eligible.append((pose, view, speed, angular_speed))

    best: tuple[
        tuple[float, ...],
        Pose,
        View,
        View,
        float,
        float,
        float,
        float,
    ] | None = None
    for left_index, (left, left_view, left_speed, left_angular_speed) in enumerate(
        eligible
    ):
        for right, right_view, right_speed, right_angular_speed in eligible[
            left_index + 1:
        ]:
            yaw_delta = abs((left_view.yaw - right_view.yaw + 180.0) % 360.0 - 180.0)
            pitch_delta = abs(left_view.pitch - right_view.pitch)
            view_separation = math.hypot(yaw_delta, pitch_delta)
            if view_separation < min_view_separation_deg:
                continue
            if len(set(left.ids) & set(right.ids)) < min_tags:
                continue
            position_spread = float(np.linalg.norm(left.xyz - right.xyz))
            attitude_spread = rotation_residual_deg(
                left.rotation_camera_to_board,
                right.rotation_camera_to_board,
            )
            if (
                position_spread > max_position_spread_m
                or attitude_spread > max_attitude_spread_deg
            ):
                continue
            selected, speed, angular_speed = min(
                (
                    (left, left_speed, left_angular_speed),
                    (right, right_speed, right_angular_speed),
                ),
                key=lambda item: (item[0].rmse, item[1], item[2]),
            )
            # More shared Tags and wider independent bearings are stronger;
            # spread and reprojection fit break ties deterministically.
            score = (
                -float(len(set(left.ids) & set(right.ids))),
                -view_separation,
                position_spread,
                attitude_spread,
                left.rmse + right.rmse,
            )
            candidate = (
                score,
                selected,
                left_view,
                right_view,
                position_spread,
                attitude_spread,
                speed,
                angular_speed,
            )
            if best is None or score < best[0]:
                best = candidate
    if best is None:
        # A short-gap temporal branch check is a conservative fallback when
        # only one perspective crop decodes well.  Two Tags in a single planar
        # view remain ambiguous and are never enough; three or more Tags plus
        # a small pose step can safely distinguish the continuous IPPE branch.
        if elapsed_s > max_single_view_gap_s or len(eligible) != 1:
            return None
        temporal_candidates = [
            item for item in eligible
            if (
                len(set(item[0].ids)) >= max(min_tags, min_single_view_tags)
                and item[3] * elapsed_s <= max_single_view_attitude_step_deg
            )
        ]
        if not temporal_candidates:
            return None
        selected, selected_view, speed, angular_speed = min(
            temporal_candidates,
            key=lambda item: (item[0].rmse, item[2], item[3]),
        )
        reanchored = replace(
            selected,
            attitude_source="visual_multitag_reanchor",
            attitude_hypothesis="guarded_temporal",
            visual_attitude_residual_deg=selected.attitude_residual_deg,
        )
        return GuardedVisualReanchor(
            reanchored,
            (selected_view.name,),
            0.0,
            0.0,
            speed,
            angular_speed,
        )
    (
        _score,
        selected,
        left_view,
        right_view,
        position_spread,
        attitude_spread,
        speed,
        angular_speed,
    ) = best
    reanchored = replace(
        selected,
        attitude_source="visual_multitag_reanchor",
        attitude_hypothesis="guarded_multiview",
        visual_attitude_residual_deg=selected.attitude_residual_deg,
    )
    return GuardedVisualReanchor(
        reanchored,
        (left_view.name, right_view.name),
        position_spread,
        attitude_spread,
        speed,
        angular_speed,
    )


def choose_pose(
    candidates: list[Pose], max_attitude_residual_deg: float | None = None,
) -> Pose | None:
    if not candidates:
        return None
    if max_attitude_residual_deg is not None:
        # A per-view constrained-translation candidate is computed eagerly so
        # its decoded corners remain available.  Use it only if *every* visual
        # attitude across overlapping views fails the IMU gate.
        consistent_visual = [
            pose for pose in candidates
            if (
                not pose.attitude_source.startswith("imu_constrained_")
                and pose.attitude_residual_deg is not None
                and pose.attitude_residual_deg <= max_attitude_residual_deg
            )
        ]
        constrained = [
            pose for pose in candidates
            if pose.attitude_source.startswith("imu_constrained_")
        ]
        if consistent_visual:
            candidates = consistent_visual
        elif constrained:
            candidates = constrained
    # Perspective views overlap.  When a temporal attitude prior exists it is
    # the strongest discriminator against planar mirror/flip solutions.  On the
    # first frame (no residuals) this naturally falls back to visual fit.
    if any(p.attitude_residual_deg is not None for p in candidates):
        return min(candidates, key=pose_selection_key)
    return min(candidates, key=lambda p: (p.rmse, -p.inliers))


def choose_scout_base(
    coarse: list[tuple[View, list[dict], Pose | None]],
    max_attitude_residual_deg: float | None = None,
) -> tuple[View, list[dict], Pose | None]:
    """Select the recovery bearing by geometric validity before tag count.

    A wide edge view can decode many tags while being too distorted for PnP.
    Prefer any scout direction that already has a valid low-resolution pose;
    only fall back to raw tag count when no direction is geometrically valid.
    The scout pose remains bearing-only and is never published as a measurement.
    """
    solved = [item for item in coarse if item[2] is not None]
    if solved:
        poses = [item[2] for item in solved]
        selected = choose_pose(
            poses, max_attitude_residual_deg,  # type: ignore[arg-type]
        )
        return next(item for item in solved if item[2] is selected)
    return max(coarse, key=lambda item: len(item[1]))


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
        f"Insta360 AprilGrid trajectory · valid {ratio:.1f}% · coverage {summary['tag_coverage_ratio'] * 100:.1f}%",
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
        "--grid-id-order", choices=("column-major", "row-major"),
        default="column-major",
        help="ID layout: Kalibr column-major or printed wall-panel row-major",
    )
    p.add_argument(
        "--tag-map", type=Path,
        help="JSON explicit non-contiguous tag map; overrides rows/cols/first-id geometry",
    )
    p.add_argument(
        "--capture-duplicate-tag-map",
        type=Path,
        help=(
            "explicit capture-only map with named physical instances for a "
            "legacy recording that reused decoded IDs; never use for new "
            "production layouts"
        ),
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
        "--start-frame", type=int, default=0,
        help="begin at this absolute source frame while preserving source timestamps",
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
        help=(
            "optional camera IMU quaternion CSV; guides the tracked view and "
            "disambiguates visual attitude during rotation"
        ),
    )
    p.add_argument(
        "--max-attitude-residual-deg",
        type=float,
        default=30.0,
        help=(
            "reject a visual attitude that differs from IMU-propagated attitude "
            "by more than this angle (default: 30 degrees)"
        ),
    )
    p.add_argument(
        "--imu-panorama-bridge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "estimate a fixed per-camera IMU-to-factory-panorama SO(3) bridge "
            "from direct multi-Tag attitudes (enabled by default)"
        ),
    )
    p.add_argument(
        "--imu-bridge-min-observations",
        type=int,
        default=4,
        help="minimum trustworthy direct multi-Tag frames for bridge calibration",
    )
    p.add_argument(
        "--imu-bridge-max-observations",
        type=int,
        default=48,
        help="maximum direct multi-Tag frames retained by the bridge calibrator",
    )
    p.add_argument(
        "--imu-bridge-max-fit-residual-deg",
        type=float,
        default=15.0,
        help="robust inlier gate for IMU/panorama bridge calibration",
    )
    p.add_argument(
        "--max-imu-translation-rmse-px",
        type=float,
        default=7.0,
        help=(
            "dedicated reprojection gate for fixed-IMU-attitude/direct-corner "
            "translation recovery (default: 7 px at the full-resolution view)"
        ),
    )
    p.add_argument(
        "--max-single-tag-imu-gap-frames",
        type=int,
        default=10,
        help=(
            "allow direct single-tag/fixed-IMU translation for at most this "
            "many consecutive sampled lost frames; 0 disables it"
        ),
    )
    p.add_argument(
        "--max-single-tag-imu-gap-s",
        type=float,
        default=0.5,
        help="wall-clock cap for single-tag IMU recovery (default: 0.5 s)",
    )
    p.add_argument(
        "--max-single-tag-imu-translation-rmse-px",
        type=float,
        default=5.0,
        help="strict reprojection gate for single-tag IMU translation recovery",
    )
    p.add_argument(
        "--guarded-visual-reanchor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "allow two agreeing direct multi-Tag views to re-anchor a failed "
            "IMU bridge (enabled by default)"
        ),
    )
    p.add_argument(
        "--max-visual-reanchor-gap-s",
        type=float,
        default=1.5,
        help="maximum time since the last authoritative pose for re-anchor",
    )
    p.add_argument(
        "--max-visual-reanchor-rmse-px",
        type=float,
        default=2.0,
        help="strict per-view reprojection gate for guarded re-anchor",
    )
    p.add_argument(
        "--max-visual-reanchor-position-spread-m",
        type=float,
        default=0.08,
        help="maximum position disagreement between two direct views",
    )
    p.add_argument(
        "--max-visual-reanchor-attitude-spread-deg",
        type=float,
        default=8.0,
        help="maximum attitude disagreement between two direct views",
    )
    p.add_argument(
        "--max-visual-reanchor-angular-speed-deg-s",
        type=float,
        default=240.0,
        help="maximum rotation rate from the last authoritative visual pose",
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
        help="projection backend (auto: high-resolution X5 uses CUDA)",
    )
    p.add_argument(
        "--decoder", choices=("auto", "cpu", "nvdec"), default="auto",
        help="video decoder (auto: measured-safe CPU; NVDEC remains opt-in)",
    )
    p.add_argument(
        "--camera-model", choices=CAMERA_MODELS, default="auto",
        help="processing profile; auto infers Insta360 X5 or generic panorama",
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
        help="input is an official stitched panorama",
    )
    p.add_argument("--session-name")
    p.add_argument("--status-file", type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.tag_map and args.capture_duplicate_tag_map:
        raise SystemExit(
            "--tag-map and --capture-duplicate-tag-map are mutually exclusive"
        )
    uses_explicit_map = bool(args.tag_map or args.capture_duplicate_tag_map)
    if (
        args.tag_size <= 0
        or args.spacing < 0
        or args.sample_fps <= 0
        or args.min_tags < (2 if uses_explicit_map else 4)
        or args.max_rmse_px <= 0
        or args.max_imu_translation_rmse_px <= 0
        or args.max_single_tag_imu_gap_frames < 0
        or args.max_single_tag_imu_gap_s < 0
        or args.max_single_tag_imu_translation_rmse_px <= 0
        or args.imu_bridge_min_observations < 4
        or args.imu_bridge_max_observations < args.imu_bridge_min_observations
        or not 0 < args.imu_bridge_max_fit_residual_deg <= 45
        or args.max_visual_reanchor_gap_s <= 0
        or args.max_visual_reanchor_rmse_px <= 0
        or args.max_visual_reanchor_position_spread_m <= 0
        or not 0 < args.max_visual_reanchor_attitude_spread_deg <= 180
        or args.max_visual_reanchor_angular_speed_deg_s <= 0
        or not 0 < args.max_attitude_residual_deg <= 180
        or args.view_size < 160
        or (args.max_processed_frames is not None and args.max_processed_frames <= 0)
        or args.start_frame < 0
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
    if args.start_frame >= total or (
        args.start_frame and not cap.seek(args.start_frame)
    ):
        LOG.error("cannot seek to start frame %d of %d", args.start_frame, total)
        return 2
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
    if args.capture_duplicate_tag_map:
        grid = load_capture_duplicate_tag_map(args.capture_duplicate_tag_map)
        expected_ids = grid.expected_ids
        LOG.warning(
            "CAPTURE-ONLY duplicate-ID resolver enabled for %s; raw duplicate "
            "IDs=%s virtual instances=%s",
            args.capture_duplicate_tag_map,
            grid.duplicate_raw_ids,
            grid.expected_virtual_ids,
        )
    elif args.tag_map:
        grid = load_tag_map(args.tag_map)
        expected_ids = grid.expected_ids
        LOG.info("loaded explicit tag map %s with IDs %s", args.tag_map, expected_ids)
    else:
        grid = Grid(
            args.rows, args.cols, args.tag_size, args.spacing,
            args.first_id, args.grid_id_order,
        )
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
    imu_bridge_calibrator = ImuPanoramaBridgeCalibrator(
        min_observations=args.imu_bridge_min_observations,
        max_observations=args.imu_bridge_max_observations,
        max_fit_residual_deg=args.imu_bridge_max_fit_residual_deg,
    )
    imu_bridge_enabled = bool(args.imu_panorama_bridge and imu_quaternions)
    if imu_bridge_enabled:
        LOG.info(
            "per-camera IMU/panorama bridge calibration enabled "
            "(min=%d max=%d)",
            args.imu_bridge_min_observations,
            args.imu_bridge_max_observations,
        )
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
    attitude_residuals: list[float] = []
    visual_attitude_residuals: list[float] = []
    processed = valid = attitude_rejected = imu_constrained_translations = 0
    imu_constrained_single_tag_translations = 0
    guarded_visual_reanchors = 0
    guarded_reanchor_position_spreads: list[float] = []
    guarded_reanchor_attitude_spreads: list[float] = []
    imu_bridge_updates = 0
    imu_bridge_duplicate_conflict_frames = 0
    imu_bridge_duplicate_conflict_ids: Counter[int] = Counter()
    attitude_hypotheses_used: Counter[str] = Counter()
    frame_no = args.start_frame
    previous: tuple[float, np.ndarray] | None = None
    previous_attitude: tuple[np.ndarray, np.ndarray | None] | None = None
    tracked_view: View | None = None
    temporal_image: np.ndarray | None = None
    temporal_detections: list[dict] = []
    temporal_age = 0
    tracked_quaternion: np.ndarray | None = None
    lost_frames = 0
    search_size = min(args.global_search_size, args.view_size)
    candidate_sources: list[tuple[Pose, View, np.ndarray, list[dict]]] = []
    candidate_measurements: dict[int, str] = {}
    frame_attitude_prior: np.ndarray | None = None
    frame_attitude_fallback_rotations: tuple[
        tuple[str, np.ndarray], ...
    ] = ()
    frame_attitude_source = "visual_reprojection"
    direct_multitag_gap_frames = 0
    frame_allow_single_tag_imu_translation = False
    authoritative_anchor_frame: int | None = None

    def direct_measurement_source(pose: Pose) -> str:
        if pose.attitude_source == "imu_constrained_single_tag_translation":
            return "direct_single_tag_imu_translation"
        return "direct"

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
        solve_diagnostics: dict = {}
        pose = solve_view(
            detections, view, size, args.min_tags, args.max_rmse_px,
            args.pnp_points, args.pnp_solver,
            frame_attitude_prior, frame_attitude_source,
            args.max_attitude_residual_deg, True,
            frame_attitude_fallback_rotations,
            args.max_imu_translation_rmse_px,
            solve_diagnostics,
            frame_allow_single_tag_imu_translation,
            args.max_single_tag_imu_translation_rmse_px,
        )
        if pose is not None:
            candidate_sources.append((pose, view, perspective, detections))
            candidate_measurements[id(pose)] = direct_measurement_source(pose)
        record = {
            "view": asdict(view),
            "size": size,
            "detections": [detection_audit_record(d) for d in detections],
            "pose": None if pose is None else {
                "xyz": pose.xyz.tolist(), "rpy": pose.rpy,
                "inliers": pose.inliers, "rmse": pose.rmse,
                "attitude_source": pose.attitude_source,
                "attitude_residual_deg": pose.attitude_residual_deg,
                "visual_attitude_residual_deg": pose.visual_attitude_residual_deg,
                "attitude_hypothesis": pose.attitude_hypothesis,
                "capture_panel": pose.capture_panel,
                "virtual_instance_ids": list(pose.virtual_instance_ids),
            },
        }
        if solve_diagnostics:
            record["solve_diagnostics"] = solve_diagnostics
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
        solve_diagnostics: dict = {}
        pose = solve_view(
            detections, view, size, args.min_tags, args.max_rmse_px,
            args.pnp_points, args.pnp_solver,
            frame_attitude_prior, frame_attitude_source,
            args.max_attitude_residual_deg, True,
            frame_attitude_fallback_rotations,
            args.max_imu_translation_rmse_px,
            solve_diagnostics,
            frame_allow_single_tag_imu_translation,
            args.max_single_tag_imu_translation_rmse_px,
        )
        record = {
            "view": asdict(view),
            "size": size,
            "detections": [
                detection_audit_record(detection) for detection in detections
            ],
            "pose": None if pose is None else {
                "xyz": pose.xyz.tolist(), "rpy": pose.rpy,
                "inliers": pose.inliers, "rmse": pose.rmse,
                "attitude_source": pose.attitude_source,
                "attitude_residual_deg": pose.attitude_residual_deg,
                "visual_attitude_residual_deg": pose.visual_attitude_residual_deg,
                "attitude_hypothesis": pose.attitude_hypothesis,
                "capture_panel": pose.capture_panel,
                "virtual_instance_ids": list(pose.virtual_instance_ids),
            },
        }
        if solve_diagnostics:
            record["solve_diagnostics"] = solve_diagnostics
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
                candidate_measurements[id(pose)] = direct_measurement_source(pose)
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
        "qx",
        "qy",
        "qz",
        "qw",
        "parent_frame",
        "child_frame",
        "tag_map_sha256",
        "raw_camera_x_m",
        "raw_camera_y_m",
        "raw_camera_z_m",
        "detected_tag_count",
        "inlier_count",
        "reprojection_rmse_px",
        "detected_ids",
        "selected_view",
        "measurement_source",
        "attitude_source",
        "attitude_hypothesis",
        "capture_panel",
        "virtual_instance_ids",
        "authoritative_anchor_updated",
        "authoritative_anchor_frame",
        "attitude_residual_deg",
        "visual_attitude_residual_deg",
        "imu_bridge_status",
        "imu_bridge_observation_count",
        "imu_bridge_inlier_count",
        "imu_bridge_fit_median_deg",
        "imu_bridge_fit_p95_deg",
        "duplicate_tag_ray_conflict_ids",
        "reanchor_supporting_views",
        "reanchor_position_spread_m",
        "reanchor_attitude_spread_deg",
        "reanchor_speed_m_s",
        "reanchor_angular_speed_deg_s",
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
            frame_attitude_prior = None
            frame_attitude_fallback_rotations = ()
            frame_attitude_source = "visual_reprojection"
            calibrated_attitude = (
                imu_bridge_calibrator.predict(current_quaternion)
                if imu_bridge_enabled and current_quaternion is not None
                else None
            )
            if calibrated_attitude is not None:
                # The calibrated absolute model does not accumulate anchor
                # drift and needs no inverse-relative recovery hypothesis.
                frame_attitude_prior = calibrated_attitude
                frame_attitude_source = "imu_calibrated_bridge"
            elif previous_attitude is not None:
                previous_rotation, previous_imu_quaternion = previous_attitude
                if previous_imu_quaternion is not None and current_quaternion is not None:
                    try:
                        attitude_hypotheses = (
                            predict_camera_to_parent_rotation_hypotheses(
                                previous_rotation,
                                previous_imu_quaternion,
                                current_quaternion,
                            )
                        )
                        frame_attitude_prior = attitude_hypotheses[0][1]
                        frame_attitude_fallback_rotations = (
                            attitude_hypotheses[1],
                        )
                        frame_attitude_source = "imu_relative"
                    except ValueError:
                        frame_attitude_prior = previous_rotation.copy()
                        frame_attitude_source = "previous_visual"
                else:
                    frame_attitude_prior = previous_rotation.copy()
                    frame_attitude_source = "previous_visual"
            frame_allow_single_tag_imu_translation = bool(
                args.max_single_tag_imu_gap_frames > 0
                and previous_attitude is not None
                and is_imu_attitude_source(frame_attitude_source)
                and direct_multitag_gap_frames
                < args.max_single_tag_imu_gap_frames
                and direct_multitag_gap_frames / args.sample_fps
                < args.max_single_tag_imu_gap_s
            )
            if (
                tracked_view is not None
                and tracked_quaternion is not None
                and current_quaternion is not None
            ):
                tracked_view = propagate_view_with_imu(
                    tracked_view,
                    tracked_quaternion,
                    current_quaternion,
                    imu_bridge_calibrator.panorama_to_body,
                )
                tracked_quaternion = current_quaternion

            temporal_success = False
            temporal_authoritative_success = False
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
                        frame_attitude_prior,
                        frame_attitude_source,
                        args.max_attitude_residual_deg,
                        True,
                        frame_attitude_fallback_rotations,
                        args.max_imu_translation_rmse_px,
                        None,
                        frame_allow_single_tag_imu_translation,
                        args.max_single_tag_imu_translation_rmse_px,
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
                            frame_attitude_prior,
                            frame_attitude_source,
                            args.max_attitude_residual_deg,
                            False,
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
                        frame_attitude_prior,
                        frame_attitude_source,
                        args.max_attitude_residual_deg,
                        False,
                    )
                all_ids.update(int(d["id"]) for d in dets)
                view_records.append(
                    {
                        "view": asdict(tracked_view),
                        "size": args.view_size,
                        "tracking_mode": tracking_mode,
                        "detections": [
                            detection_audit_record(d) for d in dets
                        ],
                        "pose": None
                        if candidate is None
                        else {
                            "xyz": candidate.xyz.tolist(),
                            "rpy": candidate.rpy,
                            "inliers": candidate.inliers,
                            "rmse": candidate.rmse,
                            "attitude_source": candidate.attitude_source,
                            "attitude_hypothesis": candidate.attitude_hypothesis,
                            "attitude_residual_deg": candidate.attitude_residual_deg,
                            "visual_attitude_residual_deg": (
                                candidate.visual_attitude_residual_deg
                            ),
                            "capture_panel": candidate.capture_panel,
                            "virtual_instance_ids": list(
                                candidate.virtual_instance_ids
                            ),
                        },
                    }
                )
                if candidate is not None:
                    candidates.append(candidate)
                    candidate_sources.append(
                        (candidate, tracked_view, perspective, dets)
                    )
                    candidate_measurements[id(candidate)] = (
                        direct_measurement_source(candidate)
                        if tracking_mode == "redetected"
                        else "optical_flow"
                    )
                    temporal_image = perspective
                    temporal_detections = dets
                    temporal_age = next_temporal_age
                    temporal_success = True
                    temporal_authoritative_success = (
                        pose_updates_authoritative_anchor(candidate)
                    )
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
                    not temporal_authoritative_success
                    and lost_frames % args.recovery_scan_interval == 0
                )
            )
            if run_full_scan:
                coarse: list[tuple[View, list[dict], Pose | None]] = []
                scan_views = views
                if tracked_view is not None and current_quaternion is not None:
                    scan_views = (*views, tracked_view)
                for view, dets, coarse_candidate, record in evaluate_views(
                    pano, scan_views, search_size
                ):
                    view_records.append(record)
                    coarse.append((view, dets, coarse_candidate))
                    all_ids.update(int(d["id"]) for d in dets)
                    if search_size == args.view_size and coarse_candidate:
                        candidates.append(coarse_candidate)
                    else:
                        # Low-resolution PnP only locates a direction; never
                        # report it as the final measurement.
                        record["pose"] = None
                base, scout_detections, _scout_pose = choose_scout_base(
                    coarse, args.max_attitude_residual_deg,
                )
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
                        best_local_candidates = [
                            (candidate, local_view, dets)
                            for local_view, dets, candidate, _record in local_results
                            if candidate is not None
                        ]
                        best_local_pose = choose_pose(
                            [item[0] for item in best_local_candidates],
                            args.max_attitude_residual_deg,
                        )
                        best_local = next(
                            item for item in best_local_candidates
                            if item[0] is best_local_pose
                        )
                        _candidate, local_view, dets = best_local
                        tracked_view = (
                            recentered_view(local_view, dets, args.view_size)
                            if len(dets) >= 2
                            else local_view
                        )
                        tracked_quaternion = current_quaternion
                if candidates and tracked_view is None:
                    selected = choose_pose(
                        candidates, args.max_attitude_residual_deg,
                    )
                    selected_coarse = next(
                        (
                            (view, detections)
                            for view, detections, _coarse_pose in coarse
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
            duplicate_tag_conflicts = find_duplicate_tag_ray_conflicts(
                view_records,
            )
            if imu_bridge_enabled and duplicate_tag_conflicts:
                imu_bridge_duplicate_conflict_frames += 1
                imu_bridge_duplicate_conflict_ids.update(
                    duplicate_tag_conflicts
                )
            pose = choose_pose(candidates, args.max_attitude_residual_deg)
            measurement_source = candidate_measurements.get(id(pose), "") if pose else ""
            reanchor_decision: GuardedVisualReanchor | None = None
            if (
                args.guarded_visual_reanchor
                and pose is not None
                and is_imu_attitude_source(pose.attitude_source)
                and pose.attitude_residual_deg is not None
                and pose.attitude_residual_deg > args.max_attitude_residual_deg
                and not any(
                    candidate.attitude_source.startswith("imu_constrained_")
                    for candidate in candidates
                )
            ):
                candidate_views = {
                    id(candidate): candidate_view
                    for candidate, candidate_view, _image, _detections
                    in candidate_sources
                }
                reanchor_decision = choose_guarded_visual_reanchor(
                    candidates,
                    candidate_measurements,
                    candidate_views,
                    None if previous is None else previous[1],
                    None if previous_attitude is None else previous_attitude[0],
                    None if previous is None else timestamp - previous[0],
                    min_tags=args.min_tags,
                    imu_gate_deg=args.max_attitude_residual_deg,
                    max_rmse_px=min(
                        args.max_rmse_px, args.max_visual_reanchor_rmse_px,
                    ),
                    max_speed_m_s=args.max_speed,
                    max_gap_s=args.max_visual_reanchor_gap_s,
                    max_position_spread_m=(
                        args.max_visual_reanchor_position_spread_m
                    ),
                    max_attitude_spread_deg=(
                        args.max_visual_reanchor_attitude_spread_deg
                    ),
                    max_angular_speed_deg_s=(
                        args.max_visual_reanchor_angular_speed_deg_s
                    ),
                )
                if reanchor_decision is not None:
                    pose = reanchor_decision.pose
                    measurement_source = "direct"
            if args.temporal_flow and pose is not None and not temporal_success:
                source = next(
                    (
                        item for item in candidate_sources
                        if item[0] is pose or (
                            item[0].view == pose.view
                            and np.allclose(item[0].xyz, pose.xyz, atol=1e-12)
                        )
                    ),
                    None,
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
            authoritative_anchor_updated = False
            if pose and np.isfinite(pose.xyz).all() and math.isfinite(pose.rmse):
                quality = "valid"
                filtered = pose.xyz.copy()
                rmses.append(pose.rmse)
                if pose.attitude_residual_deg is not None:
                    attitude_residuals.append(pose.attitude_residual_deg)
                    if (
                        is_imu_attitude_source(pose.attitude_source)
                        and pose.attitude_residual_deg
                        > args.max_attitude_residual_deg
                    ):
                        quality = "attitude_rejected"
                        filtered = None
                        attitude_rejected += 1
                if pose.visual_attitude_residual_deg is not None:
                    visual_attitude_residuals.append(
                        pose.visual_attitude_residual_deg
                    )
                if quality == "valid" and previous:
                    dt = timestamp - previous[0]
                    jump = float(np.linalg.norm(pose.xyz - previous[1]))
                    jumps.append(jump)
                    if dt > 0 and jump / dt > args.max_speed:
                        quality = "jump_rejected"
                        filtered = None
                if quality == "valid":
                    if reanchor_decision is not None:
                        guarded_visual_reanchors += 1
                        guarded_reanchor_position_spreads.append(
                            reanchor_decision.position_spread_m
                        )
                        guarded_reanchor_attitude_spreads.append(
                            reanchor_decision.attitude_spread_deg
                        )
                    if (
                        pose.attitude_source
                        == "imu_constrained_visual_translation"
                    ):
                        imu_constrained_translations += 1
                    attitude_hypotheses_used[pose.attitude_hypothesis] += 1
                    valid += 1
                    if (
                        pose.attitude_source
                        == "imu_constrained_single_tag_translation"
                    ):
                        imu_constrained_single_tag_translations += 1
                    if pose_updates_authoritative_anchor(pose):
                        previous = (timestamp, filtered.copy())
                        previous_attitude = (
                            pose.rotation_camera_to_board.copy(),
                            None
                            if current_quaternion is None
                            else current_quaternion.copy(),
                        )
                        authoritative_anchor_frame = frame_no
                        authoritative_anchor_updated = True
                    if (
                        imu_bridge_enabled
                        and current_quaternion is not None
                        and measurement_source == "direct"
                        and len(set(pose.ids)) >= max(args.min_tags, 3)
                        and pose.rmse
                        <= min(args.max_rmse_px, 3.0)
                        and not pose.attitude_source.startswith(
                            "imu_constrained_"
                        )
                        # A guarded re-anchor is valuable for continuity, but
                        # must not calibrate the prior that helped trigger it.
                        and pose.attitude_source
                        != "visual_multitag_reanchor"
                        and not duplicate_tag_conflicts
                    ):
                        was_calibrated = (
                            imu_bridge_calibrator.estimate is not None
                        )
                        if imu_bridge_calibrator.add_observation(
                            frame_no,
                            pose.rotation_camera_to_board,
                            current_quaternion,
                        ):
                            imu_bridge_updates += 1
                            estimate = imu_bridge_calibrator.estimate
                            if not was_calibrated and estimate is not None:
                                LOG.info(
                                    "calibrated IMU/panorama bridge at frame=%d "
                                    "observations=%d fit_median=%.2fdeg "
                                    "fit_p95=%.2fdeg axis_ratio=%.3f",
                                    frame_no,
                                    estimate.observation_count,
                                    estimate.residual_median_deg,
                                    estimate.residual_p95_deg,
                                    estimate.excitation_axis_ratio,
                                )
            if (
                quality == "valid"
                and pose is not None
                and measurement_source == "direct"
                and len(pose.ids) >= args.min_tags
                and pose.attitude_source
                != "imu_constrained_single_tag_translation"
            ):
                direct_multitag_gap_frames = 0
            else:
                direct_multitag_gap_frames += 1
            # A weak single-Tag observation remains publishable, but it does
            # not end recovery or become the reference for future propagation.
            lost_frames = (
                0 if authoritative_anchor_updated else lost_frames + 1
            )
            row = dict.fromkeys(csv_fields, "")
            bridge_estimate = imu_bridge_calibrator.estimate
            row.update(
                frame=frame_no,
                timestamp=f"{timestamp:.6f}",
                detected_tag_count=len(all_ids),
                detected_ids=" ".join(map(str, sorted(all_ids))),
                measurement_source=measurement_source,
                attitude_source=(pose.attitude_source if pose else ""),
                attitude_hypothesis=(pose.attitude_hypothesis if pose else ""),
                capture_panel=(pose.capture_panel if pose else ""),
                virtual_instance_ids=(
                    " ".join(pose.virtual_instance_ids) if pose else ""
                ),
                authoritative_anchor_updated=(
                    "1" if authoritative_anchor_updated else "0"
                ),
                authoritative_anchor_frame=(
                    "" if authoritative_anchor_frame is None
                    else str(authoritative_anchor_frame)
                ),
                attitude_residual_deg=(
                    ""
                    if pose is None or pose.attitude_residual_deg is None
                    else f"{pose.attitude_residual_deg:.4f}"
                ),
                visual_attitude_residual_deg=(
                    ""
                    if pose is None or pose.visual_attitude_residual_deg is None
                    else f"{pose.visual_attitude_residual_deg:.4f}"
                ),
                imu_bridge_status=(
                    imu_bridge_calibrator.status
                    if imu_bridge_enabled else "disabled"
                ),
                imu_bridge_observation_count=len(
                    imu_bridge_calibrator.frames
                ),
                imu_bridge_inlier_count=(
                    "" if bridge_estimate is None
                    else bridge_estimate.inlier_count
                ),
                imu_bridge_fit_median_deg=(
                    "" if bridge_estimate is None
                    else f"{bridge_estimate.residual_median_deg:.4f}"
                ),
                imu_bridge_fit_p95_deg=(
                    "" if bridge_estimate is None
                    else f"{bridge_estimate.residual_p95_deg:.4f}"
                ),
                duplicate_tag_ray_conflict_ids=" ".join(
                    map(str, duplicate_tag_conflicts)
                ),
                reanchor_supporting_views=(
                    ""
                    if reanchor_decision is None
                    else " ".join(reanchor_decision.supporting_views)
                ),
                reanchor_position_spread_m=(
                    ""
                    if reanchor_decision is None
                    else f"{reanchor_decision.position_spread_m:.7f}"
                ),
                reanchor_attitude_spread_deg=(
                    ""
                    if reanchor_decision is None
                    else f"{reanchor_decision.attitude_spread_deg:.4f}"
                ),
                reanchor_speed_m_s=(
                    ""
                    if reanchor_decision is None
                    else f"{reanchor_decision.speed_m_s:.4f}"
                ),
                reanchor_angular_speed_deg_s=(
                    ""
                    if reanchor_decision is None
                    else f"{reanchor_decision.angular_speed_deg_s:.4f}"
                ),
                quality_status=quality,
            )
            if pose:
                quaternion = Rotation.from_matrix(
                    pose.rotation_camera_to_board
                ).as_quat()
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
                    qx=f"{quaternion[0]:.9f}",
                    qy=f"{quaternion[1]:.9f}",
                    qz=f"{quaternion[2]:.9f}",
                    qw=f"{quaternion[3]:.9f}",
                    parent_frame=(
                        grid.metadata.get("world_frame", "tag_map")
                        if isinstance(
                            grid, (IndependentTagMap, CaptureDuplicateTagMap)
                        ) else "aprilgrid"
                    ),
                    child_frame="panorama_camera",
                    tag_map_sha256=(
                        grid.metadata.get("tag_map_sha256", "")
                        if isinstance(
                            grid, (IndependentTagMap, CaptureDuplicateTagMap)
                        ) else ""
                    ),
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
                        "attitude_source": pose.attitude_source if pose else None,
                        "attitude_hypothesis": (
                            pose.attitude_hypothesis if pose else None
                        ),
                        "capture_panel": pose.capture_panel if pose else None,
                        "virtual_instance_ids": (
                            list(pose.virtual_instance_ids) if pose else []
                        ),
                        "authoritative_anchor_updated": (
                            authoritative_anchor_updated
                        ),
                        "authoritative_anchor_frame": authoritative_anchor_frame,
                        "attitude_residual_deg": (
                            pose.attitude_residual_deg if pose else None
                        ),
                        "visual_attitude_residual_deg": (
                            pose.visual_attitude_residual_deg if pose else None
                        ),
                        "imu_panorama_bridge": {
                            "status": (
                                imu_bridge_calibrator.status
                                if imu_bridge_enabled else "disabled"
                            ),
                            "observation_count": len(
                                imu_bridge_calibrator.frames
                            ),
                            "inlier_count": (
                                None if bridge_estimate is None
                                else bridge_estimate.inlier_count
                            ),
                            "fit_median_deg": (
                                None if bridge_estimate is None
                                else bridge_estimate.residual_median_deg
                            ),
                            "fit_p95_deg": (
                                None if bridge_estimate is None
                                else bridge_estimate.residual_p95_deg
                            ),
                            "duplicate_tag_ray_conflict_ids": (
                                duplicate_tag_conflicts
                            ),
                        },
                        "guarded_visual_reanchor": (
                            None if reanchor_decision is None else {
                                "supporting_views": list(
                                    reanchor_decision.supporting_views
                                ),
                                "position_spread_m": (
                                    reanchor_decision.position_spread_m
                                ),
                                "attitude_spread_deg": (
                                    reanchor_decision.attitude_spread_deg
                                ),
                                "speed_m_s": reanchor_decision.speed_m_s,
                                "angular_speed_deg_s": (
                                    reanchor_decision.angular_speed_deg_s
                                ),
                            }
                        ),
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
    imu_bridge_audit = imu_bridge_calibrator.audit()
    imu_bridge_audit["enabled"] = imu_bridge_enabled
    imu_bridge_audit["update_count"] = imu_bridge_updates
    imu_bridge_audit["duplicate_conflict_frame_count"] = (
        imu_bridge_duplicate_conflict_frames
    )
    imu_bridge_audit["duplicate_conflict_detections_per_id"] = {
        str(tag_id): count
        for tag_id, count in sorted(imu_bridge_duplicate_conflict_ids.items())
    }
    summary = {
        "input": str(args.input.resolve()),
        "total_frames": total,
        "start_frame": args.start_frame,
        "processed_frames": processed,
        "valid_pose_frames": valid,
        "valid_pose_ratio": valid / processed if processed else 0.0,
        "recognized_ids": sorted(seen),
        "missing_ids": [i for i in expected if i not in seen],
        "detections_per_id": {str(k): v for k, v in sorted(seen.items())},
        "tag_coverage_ratio": len(set(expected) & set(seen)) / len(expected),
        "reprojection_rmse_px": _finite_stats(rmses),
        "adjacent_coordinate_jump_m": _finite_stats(jumps),
        "attitude_residual_deg": _finite_stats(attitude_residuals),
        "visual_attitude_residual_deg": _finite_stats(
            visual_attitude_residuals
        ),
        "attitude_rejected_frames": attitude_rejected,
        "imu_constrained_visual_translation_frames": imu_constrained_translations,
        "imu_constrained_single_tag_translation_frames": (
            imu_constrained_single_tag_translations
        ),
        "imu_panorama_bridge": imu_bridge_audit,
        "guarded_visual_reanchor_frames": guarded_visual_reanchors,
        "guarded_visual_reanchor_position_spread_m": _finite_stats(
            guarded_reanchor_position_spreads
        ),
        "guarded_visual_reanchor_attitude_spread_deg": _finite_stats(
            guarded_reanchor_attitude_spreads
        ),
        "guarded_visual_reanchor_enabled": args.guarded_visual_reanchor,
        "max_visual_reanchor_gap_s": args.max_visual_reanchor_gap_s,
        "max_visual_reanchor_rmse_px": args.max_visual_reanchor_rmse_px,
        "max_visual_reanchor_position_spread_m": (
            args.max_visual_reanchor_position_spread_m
        ),
        "max_visual_reanchor_attitude_spread_deg": (
            args.max_visual_reanchor_attitude_spread_deg
        ),
        "max_visual_reanchor_angular_speed_deg_s": (
            args.max_visual_reanchor_angular_speed_deg_s
        ),
        "single_tag_updates_authoritative_anchor": False,
        "final_authoritative_anchor_frame": authoritative_anchor_frame,
        "attitude_hypotheses_used": dict(attitude_hypotheses_used),
        "max_attitude_residual_deg": args.max_attitude_residual_deg,
        "max_imu_translation_rmse_px": args.max_imu_translation_rmse_px,
        "max_single_tag_imu_gap_frames": args.max_single_tag_imu_gap_frames,
        "max_single_tag_imu_gap_s": args.max_single_tag_imu_gap_s,
        "max_single_tag_imu_translation_rmse_px": (
            args.max_single_tag_imu_translation_rmse_px
        ),
        "tag_size_m": (
            grid.metadata.get("tag_size_m")
            if isinstance(
                grid, (IndependentTagMap, CaptureDuplicateTagMap)
            ) else args.tag_size
        ),
        "spacing_ratio": None if uses_explicit_map else args.spacing,
        "rows": None if uses_explicit_map else args.rows,
        "cols": None if uses_explicit_map else args.cols,
        "first_id": None if uses_explicit_map else args.first_id,
        "grid_id_order": None if uses_explicit_map else args.grid_id_order,
        "tag_map": str(args.tag_map.resolve()) if args.tag_map else None,
        "capture_duplicate_tag_map": (
            str(args.capture_duplicate_tag_map.resolve())
            if args.capture_duplicate_tag_map else None
        ),
        "capture_duplicate_resolution": isinstance(
            grid, CaptureDuplicateTagMap
        ),
        "duplicate_raw_ids": (
            grid.duplicate_raw_ids
            if isinstance(grid, CaptureDuplicateTagMap) else []
        ),
        "expected_virtual_instance_ids": (
            grid.expected_virtual_ids
            if isinstance(grid, CaptureDuplicateTagMap) else []
        ),
        "tag_map_sha256": (
            grid.metadata.get("tag_map_sha256")
            if isinstance(
                grid, (IndependentTagMap, CaptureDuplicateTagMap)
            ) else None
        ),
        "parent_frame": (
            grid.metadata.get("world_frame", "tag_map")
            if isinstance(
                grid, (IndependentTagMap, CaptureDuplicateTagMap)
            ) else "aprilgrid"
        ),
        "child_frame": "panorama_camera",
        "pose_convention": "T_parent_child maps child-frame points into parent frame",
        "calibration_status": (
            grid.metadata.get("calibration_status")
            if isinstance(
                grid, (IndependentTagMap, CaptureDuplicateTagMap)
            ) else None
        ),
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
        "imu_guided_attitude": bool(imu_quaternions),
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
    (session / "imu_panorama_bridge.json").write_text(
        json.dumps(imu_bridge_audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
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
