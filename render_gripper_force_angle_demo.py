#!/usr/bin/env python3
"""Render front-lens jaw angle, uncalibrated pad force, and current CAD animation."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from itertools import combinations
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from render_dual_camera_alignment_demo import UrdfWireframe, load_urdf_wireframe
from rig_revision import load_rig_revision, sha256


YELLOW_LOW = np.array([18, 75, 65], dtype=np.uint8)
YELLOW_HIGH = np.array([48, 255, 255], dtype=np.uint8)
DARK_LOW = np.array([0, 0, 0], dtype=np.uint8)
DARK_HIGH = np.array([179, 255, 115], dtype=np.uint8)
BG = (11, 15, 21)
PANEL = (19, 25, 34)
WHITE = (239, 243, 247)
MUTED = (143, 154, 168)
CYAN = (226, 190, 75)
GREEN = (99, 214, 135)
AMBER = (72, 174, 240)
RED = (86, 91, 238)
YELLOW = (54, 214, 242)
APRILTAG_DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
APRILTAG_PARAMETERS = cv2.aruco.DetectorParameters()
APRILTAG_PARAMETERS.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
APRILTAG_PARAMETERS.adaptiveThreshWinSizeMax = 63
APRILTAG_DETECTOR = cv2.aruco.ArucoDetector(
    APRILTAG_DICTIONARY, APRILTAG_PARAMETERS
)


@dataclass
class DotObservation:
    point: np.ndarray
    area_px2: float
    minor_major_ratio: float


@dataclass
class FrameObservation:
    yellow_left: np.ndarray | None
    yellow_right: np.ndarray | None
    dot_left: DotObservation | None
    dot_right: DotObservation | None
    included_angle_deg: float
    base_tag_ids: tuple[int, ...] = ()


@dataclass
class JawFrame:
    origin: np.ndarray
    axis: np.ndarray
    normal: np.ndarray
    scale_px: float


@dataclass
class ForceModel:
    left_local: np.ndarray
    right_local: np.ndarray
    left_shape: np.ndarray
    right_shape: np.ndarray
    baseline: float
    noise_mad: float
    noise_floor: float
    full_scale: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("front_video", type=Path, help="extracted front fisheye track")
    parser.add_argument("--source-osv", type=Path, required=True)
    parser.add_argument("--rig-revision", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-recovery-gap-s", type=float, default=0.25)
    parser.add_argument(
        "--camera-profile",
        choices=("osmo-front", "insta360-x5-front"),
        default="osmo-front",
    )
    parser.add_argument(
        "--contact-interval-s",
        type=float,
        nargs=2,
        action="append",
        default=[],
        metavar=("START", "END"),
        help="user-labeled contact interval; all frames outside supplied intervals are unloaded",
    )
    parser.add_argument(
        "--contact-event-s",
        type=float,
        action="append",
        default=[],
        metavar="TIME",
        help="user-labeled instant when the gripper is clamping an object",
    )
    parser.add_argument(
        "--display-relative-fingertip-force",
        action="store_true",
        help=(
            "display the existing capture-local relative fingertip-force proxy for X5; "
            "this is not a Newton calibration"
        ),
    )
    parser.add_argument("--x5-angle-revision", type=Path)
    parser.add_argument("--base-tag-id", type=int, choices=(2, 3))
    parser.add_argument("--allow-diagnostic-rig", action="store_true")
    parser.add_argument("--relative-force-revision", type=Path)
    return parser.parse_args()


def scaled(value: float, image: np.ndarray) -> int:
    return round(value * image.shape[1] / 1920.0)


def contour_centroid(contour: np.ndarray) -> np.ndarray | None:
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    return np.array(
        [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
        dtype=float,
    )


def detect_x5_yellow_axis(hsv: np.ndarray, side: str) -> np.ndarray | None:
    height, width = hsv.shape[:2]
    scale = width / 1920.0
    mask = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)
    roi = np.zeros((height, width), dtype=np.uint8)
    x0, x1 = ((550, 960) if side == "left" else (960, 1350))
    roi[
        round(1100 * scale):round(1720 * scale),
        round(x0 * scale):round(x1 * scale),
    ] = 255
    mask = cv2.bitwise_and(mask, roi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    candidates: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        centre = contour_centroid(contour)
        if centre is None or not 5000 * scale * scale <= area <= 50000 * scale * scale:
            continue
        if centre[1] < 1200 * scale:
            continue
        if (side == "left" and centre[0] >= 960 * scale) or (
            side == "right" and centre[0] < 960 * scale
        ):
            continue
        points = contour.reshape(-1, 2).astype(np.float64)
        mean, eigenvectors, _ = cv2.PCACompute2(points, mean=None)
        axis = eigenvectors[0]
        if axis[1] > 0:
            axis = -axis
        projections = (points - mean[0]) @ axis
        base_offset, middle_offset, tip_offset = np.percentile(
            projections, [5, 50, 95]
        )
        base = mean[0] + base_offset * axis
        middle = mean[0] + middle_offset * axis
        tip = mean[0] + tip_offset * axis
        span = float(np.linalg.norm(tip - base))
        if 250 * scale <= span <= 550 * scale:
            candidates.append((area, np.asarray([tip, middle, base])))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def detect_x5_yellow_triad_fixed(
    hsv: np.ndarray, side: str
) -> np.ndarray | None:
    height, width = hsv.shape[:2]
    scale = width / 1920.0
    mask = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)
    roi = np.zeros((height, width), dtype=np.uint8)
    x0, x1 = ((600, 900) if side == "left" else (1050, 1320))
    roi[
        round(1250 * scale):round(1660 * scale),
        round(x0 * scale):round(x1 * scale),
    ] = 255
    mask = cv2.bitwise_and(mask, roi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    candidates: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not 50 * scale * scale <= area <= 1200 * scale * scale:
            continue
        _, _, box_width, box_height = cv2.boundingRect(contour)
        if not 0.35 <= box_width / max(box_height, 1) <= 2.8:
            continue
        centre = contour_centroid(contour)
        if centre is not None:
            candidates.append((area, centre))
    points = []
    for y0, y1 in ((1250, 1415), (1415, 1505), (1505, 1660)):
        band = [
            item for item in candidates
            if y0 * scale <= item[1][1] < y1 * scale
        ]
        if not band:
            return None
        points.append(max(band, key=lambda item: item[0])[1])
    result = np.asarray(points)
    span = float(np.linalg.norm(result[0] - result[2]))
    if not 150 * scale <= span <= 320 * scale:
        return None
    return result


def detect_x5_yellow_triad_adaptive(
    hsv: np.ndarray, side: str
) -> np.ndarray | None:
    height, width = hsv.shape[:2]
    scale = width / 1920.0
    mask = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)
    roi = np.zeros((height, width), dtype=np.uint8)
    x0, x1 = ((450, 960) if side == "left" else (960, 1470))
    roi[
        round(950 * scale):round(1800 * scale),
        round(x0 * scale):round(x1 * scale),
    ] = 255
    mask = cv2.bitwise_and(mask, roi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    candidates: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not 50 * scale * scale <= area <= 1200 * scale * scale:
            continue
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if not 0.45 <= box_width / max(box_height, 1) <= 2.2:
            continue
        centre = contour_centroid(contour)
        if centre is None:
            continue
        radius = max(box_width, box_height)
        patch_y0 = max(0, y - radius)
        patch_y1 = min(height, y + box_height + radius)
        patch_x0 = max(0, x - radius)
        patch_x1 = min(width, x + box_width + radius)
        patch = hsv[patch_y0:patch_y1, patch_x0:patch_x1]
        grid_y, grid_x = np.mgrid[patch_y0:patch_y1, patch_x0:patch_x1]
        distance = np.hypot(grid_x - centre[0], grid_y - centre[1])
        annulus = (distance >= 0.8 * radius) & (distance <= 1.8 * radius)
        dark = (patch[..., 1] < 90) | (patch[..., 2] < 90)
        dark_fraction = float(dark[annulus].mean()) if annulus.any() else 0.0
        if dark_fraction < 0.35:
            continue
        candidates.append((area, centre))

    best: tuple[float, np.ndarray] | None = None
    for items in combinations(candidates, 3):
        points = np.asarray(
            sorted((item[1] for item in items), key=lambda point: point[1])
        )
        top, middle, bottom = points
        span = float(np.linalg.norm(top - bottom))
        if not 140 * scale <= span <= 450 * scale:
            continue
        first_gap = float(np.linalg.norm(top - middle))
        second_gap = float(np.linalg.norm(middle - bottom))
        if min(first_gap, second_gap) < 35 * scale:
            continue
        horizontal_delta = float(bottom[0] - top[0])
        if (side == "left" and horizontal_delta > -15 * scale) or (
            side == "right" and horizontal_delta < 15 * scale
        ):
            continue
        centered = points - points.mean(axis=0)
        _, _, axes = np.linalg.svd(centered, full_matrices=False)
        normal = axes[1]
        line_residual = float(np.max(np.abs(centered @ normal)))
        if line_residual > 22 * scale:
            continue
        gap_ratio_penalty = abs(float(np.log(first_gap / second_gap)))
        area_reward = sum(item[0] for item in items) / (3000 * scale * scale)
        score = line_residual / scale + 6.0 * gap_ratio_penalty - area_reward
        if best is None or score < best[0]:
            best = (score, points)
    return None if best is None else best[1]


def detect_yellow_triad(
    hsv: np.ndarray,
    side: str,
    camera_profile: str = "osmo-front",
    x5_angle_mode: str = "pca-axis",
    x5_dot_selection: str = "fixed-bands",
) -> np.ndarray | None:
    if camera_profile == "insta360-x5-front":
        if x5_angle_mode == "pca-axis":
            return detect_x5_yellow_axis(hsv, side)
        return (
            detect_x5_yellow_triad_adaptive(hsv, side)
            if x5_dot_selection == "adaptive-black-pad"
            else detect_x5_yellow_triad_fixed(hsv, side)
        )
    height, width = hsv.shape[:2]
    scale = width / 1920.0
    mask = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)
    roi = np.zeros((height, width), dtype=np.uint8)
    x0, x1 = ((500, 850) if side == "left" else (1070, 1400))
    roi[round(1120 * scale):round(1500 * scale), round(x0 * scale):round(x1 * scale)] = 255
    mask = cv2.bitwise_and(mask, roi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    candidates: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not 120.0 * scale * scale <= area <= 1200.0 * scale * scale:
            continue
        centre = contour_centroid(contour)
        if centre is not None:
            candidates.append((area, centre))

    x_bands = (
        ((620, 850), (590, 810), (520, 730))
        if side == "left"
        else ((1100, 1280), (1130, 1350), (1200, 1400))
    )
    points = []
    for (y0, y1), (band_x0, band_x1) in zip(
        ((1160, 1240), (1240, 1340), (1340, 1480)), x_bands
    ):
        band = [
            candidate
            for candidate in candidates
            if y0 * scale <= candidate[1][1] < y1 * scale
            and band_x0 * scale <= candidate[1][0] < band_x1 * scale
        ]
        if not band:
            return None
        points.append(max(band, key=lambda candidate: candidate[0])[1])
    result = np.asarray(points)
    span = float(np.linalg.norm(result[0] - result[2]))
    if not 150.0 * scale <= span <= 280.0 * scale:
        return None
    return result


def detect_black_dot(
    hsv: np.ndarray, side: str, camera_profile: str = "osmo-front"
) -> DotObservation | None:
    height, width = hsv.shape[:2]
    scale = width / 1920.0
    dark = cv2.inRange(hsv, DARK_LOW, DARK_HIGH)
    roi = np.zeros((height, width), dtype=np.uint8)
    if camera_profile == "insta360-x5-front":
        x0, x1 = ((720, 970) if side == "left" else (970, 1200))
        y0, y1 = 1120, 1510
        area_min, area_max, annulus_min = 35.0, 350.0, 0.55
    else:
        x0, x1 = ((760, 960) if side == "left" else (960, 1160))
        y0, y1 = 1030, 1190
        area_min, area_max, annulus_min = 45.0, 220.0, 0.75
    roi[round(y0 * scale):round(y1 * scale), round(x0 * scale):round(x1 * scale)] = 255
    dark = cv2.bitwise_and(dark, roi)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    candidates: list[tuple[float, DotObservation]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not area_min * scale * scale <= area <= area_max * scale * scale:
            continue
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if not 0.35 <= box_width / max(box_height, 1) <= 2.8:
            continue
        centre = contour_centroid(contour)
        if centre is None:
            continue
        radius = max(box_width, box_height)
        y0 = max(0, y - radius)
        y1 = min(height, y + box_height + radius)
        x0 = max(0, x - radius)
        x1 = min(width, x + box_width + radius)
        patch = hsv[y0:y1, x0:x1]
        grid_y, grid_x = np.mgrid[y0:y1, x0:x1]
        distance = np.hypot(grid_x - centre[0], grid_y - centre[1])
        annulus = (distance >= 0.65 * radius) & (distance <= 1.45 * radius)
        yellow = (
            (patch[..., 0] >= 15)
            & (patch[..., 0] <= 50)
            & (patch[..., 1] >= 60)
            & (patch[..., 2] >= 60)
        )
        yellow_fraction = float(yellow[annulus].mean()) if annulus.any() else 0.0
        if yellow_fraction < annulus_min:
            continue
        axes = cv2.fitEllipse(contour)[1] if len(contour) >= 5 else (box_width, box_height)
        minor, major = sorted(float(value) for value in axes)
        candidates.append(
            (
                yellow_fraction,
                DotObservation(centre, area, minor / max(major, 1e-6)),
            )
        )
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate[0])[1]


def jaw_frame(points: np.ndarray, side: str) -> JawFrame:
    tip, _, base = points
    axis = tip - base
    scale_px = float(np.linalg.norm(axis))
    axis = axis / scale_px
    inward_sign = -1.0 if side == "left" else 1.0
    normal = np.array([-axis[1], axis[0]]) * inward_sign
    return JawFrame(tip, axis, normal, scale_px)


def point_to_local(point: np.ndarray, frame: JawFrame) -> np.ndarray:
    delta = point - frame.origin
    return np.array([delta @ frame.axis, delta @ frame.normal]) / frame.scale_px


def local_to_point(coordinate: np.ndarray, frame: JawFrame) -> np.ndarray:
    return frame.origin + frame.scale_px * (
        coordinate[0] * frame.axis + coordinate[1] * frame.normal
    )


def included_jaw_angle(left: np.ndarray, right: np.ndarray) -> float:
    left_vector = left[0] - left[2]
    right_vector = right[0] - right[2]
    cosine = float(
        left_vector @ right_vector
        / (np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
    )
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def observe_frame(
    image: np.ndarray,
    camera_profile: str = "osmo-front",
    x5_angle_mode: str = "pca-axis",
    x5_included_angle_range: tuple[float, float] = (35.0, 80.0),
    x5_dot_selection: str = "fixed-bands",
) -> FrameObservation:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    left = detect_yellow_triad(
        hsv, "left", camera_profile, x5_angle_mode, x5_dot_selection
    )
    right = detect_yellow_triad(
        hsv, "right", camera_profile, x5_angle_mode, x5_dot_selection
    )
    angle = np.nan
    if left is not None and right is not None:
        candidate = included_jaw_angle(left, right)
        low, high = (
            (5.0, 50.0)
            if camera_profile == "insta360-x5-front" and x5_angle_mode == "pca-axis"
            else x5_included_angle_range
            if camera_profile == "insta360-x5-front"
            else (40.0, 80.0)
        )
        if low <= candidate <= high:
            angle = candidate
        else:
            left = right = None
    dot_left = detect_black_dot(hsv, "left", camera_profile)
    dot_right = detect_black_dot(hsv, "right", camera_profile)
    if (
        camera_profile == "insta360-x5-front"
        and dot_left is not None
        and dot_right is not None
    ):
        gap = float(np.linalg.norm(dot_left.point - dot_right.point))
        pair_valid = (
            65.0 <= gap <= 230.0
            and dot_left.point[0] < dot_right.point[0]
            and abs(dot_left.point[1] - dot_right.point[1]) <= 60.0
        )
        if not pair_valid:
            dot_left = dot_right = None
    _, tag_ids, _ = APRILTAG_DETECTOR.detectMarkers(image)
    observed_tag_ids = (
        () if tag_ids is None else tuple(sorted(int(value) for value in tag_ids.flat))
    )
    return FrameObservation(
        yellow_left=left,
        yellow_right=right,
        dot_left=dot_left,
        dot_right=dot_right,
        included_angle_deg=angle,
        base_tag_ids=observed_tag_ids,
    )


def bounded_interpolate(values: np.ndarray, maximum_gap: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    result = values.copy()
    recovered = np.zeros(len(values), dtype=bool)
    valid = np.flatnonzero(np.isfinite(values))
    for left, right in zip(valid[:-1], valid[1:]):
        gap = right - left - 1
        if gap <= 0 or gap > maximum_gap:
            continue
        alpha = np.arange(1, gap + 1, dtype=float) / (gap + 1)
        result[left + 1:right] = values[left] + alpha * (values[right] - values[left])
        recovered[left + 1:right] = True
    return result, recovered


def nanmedian_filter(values: np.ndarray, radius: int = 2) -> np.ndarray:
    result = np.full_like(values, np.nan, dtype=float)
    for index in range(len(values)):
        window = values[max(0, index - radius):min(len(values), index + radius + 1)]
        finite = window[np.isfinite(window)]
        if len(finite):
            result[index] = float(np.median(finite))
    return result


def analyze_video(
    path: Path,
    camera_profile: str = "osmo-front",
    x5_angle_mode: str = "pca-axis",
    x5_included_angle_range: tuple[float, float] = (35.0, 80.0),
    x5_dot_selection: str = "fixed-bands",
) -> tuple[list[FrameObservation], float, tuple[int, int]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open front-lens video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width != height:
        raise ValueError(f"front fisheye track must be square, got {width}x{height}")
    observations = []
    while True:
        ok, image = capture.read()
        if not ok:
            break
        observations.append(
            observe_frame(
                image,
                camera_profile,
                x5_angle_mode,
                x5_included_angle_range,
                x5_dot_selection,
            )
        )
    capture.release()
    if not observations:
        raise ValueError("front-lens video has no decodable frames")
    return observations, fps, (width, height)


def opening_angles(
    observations: list[FrameObservation],
    closed_reference: float | None = None,
) -> tuple[np.ndarray, float]:
    included = np.asarray([item.included_angle_deg for item in observations], dtype=float)
    valid = included[np.isfinite(included)]
    if len(valid) < 30:
        raise ValueError(f"only {len(valid)} frames contain both measured jaw axes")
    if closed_reference is None:
        closed_reference = float(np.percentile(valid, 97.0))
    return np.clip(float(closed_reference) - included, 0.0, 55.0), float(
        closed_reference
    )


def axis_heading_deg(points: np.ndarray) -> float:
    axis = points[0] - points[2]
    return float(np.degrees(np.arctan2(axis[1], axis[0])))


def apply_one_sided_opening_fallback(
    observations: list[FrameObservation],
    opening: np.ndarray,
    hardware_angle: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    result = np.asarray(opening, dtype=float).copy()
    states = np.full(len(observations), "UNAVAILABLE", dtype=object)
    states[np.isfinite(result)] = "MEASURED"
    fallback = hardware_angle.get("single_side_fallback")
    audit = {
        "enabled": fallback is not None,
        "bilateral_frames": int(np.count_nonzero(np.isfinite(result))),
        "one_sided_left_frames": 0,
        "one_sided_right_frames": 0,
        "continuity_rejected_frames": 0,
        "extrapolation_rejected_frames": 0,
    }
    if fallback is None:
        return result, states, audit

    side = fallback["available_side"]
    center = float(fallback["heading_center_deg"])
    coefficients = np.asarray(fallback["coefficients_high_to_low"], dtype=float)
    low, high = map(float, fallback["validated_output_range_deg"])
    state = str(fallback["measurement_state"])
    maximum_step = float(fallback.get("maximum_step_deg", np.inf))
    maximum_age = int(fallback.get("continuity_max_age_frames", 0))
    previous_index: int | None = None
    previous_value: float | None = None
    for index, observation in enumerate(observations):
        if np.isfinite(result[index]):
            previous_index = index
            previous_value = float(result[index])
            continue
        points = (
            observation.yellow_right
            if side == "right" and observation.yellow_left is None
            else observation.yellow_left
            if side == "left" and observation.yellow_right is None
            else None
        )
        if points is None:
            continue
        heading = axis_heading_deg(points)
        relative = float(
            np.degrees(np.angle(np.exp(1j * np.radians(heading - center))))
        )
        candidate = float(np.polyval(coefficients, relative))
        if not low <= candidate <= high:
            audit["extrapolation_rejected_frames"] += 1
            continue
        if (
            previous_index is not None
            and previous_value is not None
            and index - previous_index <= maximum_age
            and abs(candidate - previous_value) > maximum_step
        ):
            audit["continuity_rejected_frames"] += 1
            continue
        result[index] = candidate
        states[index] = state
        previous_index = index
        previous_value = candidate
        audit[f"one_sided_{side}_frames"] += 1
    audit["holdout"] = fallback.get("blocked_holdout")
    audit["model"] = fallback.get("model")
    return result, states, audit


def apply_fixed_relative_force(
    observations: list[FrameObservation],
    opening: np.ndarray,
    revision: dict,
) -> tuple[np.ndarray, dict]:
    model = revision["model"]
    coefficients = np.asarray(
        model["expected_gap_polynomial_high_to_low"], dtype=float
    )
    opening_low, opening_high = map(float, model["opening_support_deg"])
    noise_floor = float(model["noise_floor_px"])
    full_scale = float(model["full_scale_signal_px"])
    force = np.full(len(observations), np.nan)
    residuals = np.full(len(observations), np.nan)
    measured = 0
    for index, observation in enumerate(observations):
        if (
            not np.isfinite(opening[index])
            or not np.isfinite(observation.included_angle_deg)
            or observation.dot_left is None
            or observation.dot_right is None
            or not opening_low <= opening[index] <= opening_high
        ):
            continue
        gap = float(
            np.linalg.norm(
                observation.dot_right.point - observation.dot_left.point
            )
        )
        expected_gap = float(np.polyval(coefficients, opening[index]))
        residual = abs(gap - expected_gap)
        signal = max(residual - noise_floor, 0.0)
        force[index] = float(np.clip(100.0 * signal / full_scale, 0.0, 100.0))
        residuals[index] = residual
        measured += 1
    finite = np.isfinite(force)
    if not np.any(finite):
        raise ValueError("relative-force revision produced no measurements")
    audit = {
        "revision_id": revision["revision_id"],
        "quantity": revision["quantity"],
        "fixed_scale_across_captures": True,
        "measured_frames": measured,
        "measured_frame_ratio": measured / len(observations),
        "noise_floor_px": noise_floor,
        "full_scale_signal_px": full_scale,
        "opening_support_deg": [opening_low, opening_high],
        "relative_force_range_percent": [
            float(np.nanmin(force)),
            float(np.nanmax(force)),
        ],
        "zero_force_frame_ratio": float(np.mean(force[finite] == 0.0)),
        "residual_p95_px": float(np.nanpercentile(residuals, 95.0)),
    }
    return force, audit


def labeled_contact_gap_audit(
    observations: list[FrameObservation],
    opening: np.ndarray,
    fps: float,
    intervals_s: list[list[float]],
) -> tuple[dict | None, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame_count = len(observations)
    labels = np.full(frame_count, "UNLABELED", dtype=object)
    gaps = np.full(frame_count, np.nan)
    residuals = np.full(frame_count, np.nan)
    baseline_supported = np.zeros(frame_count, dtype=bool)

    duration_s = (frame_count - 1) / fps
    intervals = sorted((float(start), float(end)) for start, end in intervals_s)
    previous_end = -np.inf
    for start, end in intervals:
        if start < 0.0 or end <= start or end > duration_s:
            raise ValueError(
                f"contact interval [{start}, {end}] must satisfy "
                f"0 <= START < END <= {duration_s:.6f}"
            )
        if start <= previous_end:
            raise ValueError("contact intervals must not overlap or share a boundary")
        previous_end = end

    times = np.arange(frame_count, dtype=float) / fps
    contact = np.zeros(frame_count, dtype=bool)
    for start, end in intervals:
        contact |= (times >= start) & (times <= end)
    labels[:] = "UNLOADED"
    labels[contact] = "CONTACT"

    complete = np.zeros(frame_count, dtype=bool)
    for index, item in enumerate(observations):
        if (
            np.isfinite(opening[index])
            and item.yellow_left is not None
            and item.yellow_right is not None
            and item.dot_left is not None
            and item.dot_right is not None
        ):
            gaps[index] = float(np.linalg.norm(item.dot_right.point - item.dot_left.point))
            complete[index] = True
    if not intervals_s:
        labels[:] = "UNLABELED"
        return None, labels, gaps, residuals, baseline_supported

    unloaded = complete & ~contact
    if np.count_nonzero(unloaded) < 30:
        raise ValueError("fewer than 30 complete unloaded frames remain outside contact intervals")
    unloaded_opening = opening[unloaded]
    opening_min = float(np.min(unloaded_opening))
    opening_max = float(np.max(unloaded_opening))
    if opening_max - opening_min <= 1e-6:
        raise ValueError("unloaded frames do not span a usable jaw-opening range")

    coefficients = np.polyfit(unloaded_opening, gaps[unloaded], 1)
    residuals[complete] = gaps[complete] - np.polyval(coefficients, opening[complete])
    baseline_supported = complete & (opening >= opening_min) & (opening <= opening_max)

    def distribution(mask: np.ndarray, values: np.ndarray) -> dict:
        selected = values[mask & np.isfinite(values)]
        return {
            "frames": int(len(selected)),
            "median": float(np.median(selected)),
            "p10": float(np.percentile(selected, 10.0)),
            "p90": float(np.percentile(selected, 90.0)),
        }

    contact_complete = complete & contact
    contact_supported = contact_complete & baseline_supported
    audit = {
        "source": "user-provided interval annotation",
        "intervals_s": [
            {"start_s": start, "end_s": end, "label": "CONTACT"}
            for start, end in intervals
        ],
        "boundary_policy": "closed intervals: START <= t <= END",
        "outside_intervals_label": "UNLOADED",
        "complete_measurement_coverage": {
            "contact": float(np.mean(complete[contact])),
            "unloaded": float(np.mean(complete[~contact])),
        },
        "geometry_check": {
            "metric": "black_dot_gap_px minus linear unloaded prediction at the same jaw opening",
            "unloaded_model": {
                "slope_px_per_deg": float(coefficients[0]),
                "intercept_px": float(coefficients[1]),
                "opening_support_deg": [opening_min, opening_max],
                "residual_mad_px": float(
                    np.median(np.abs(residuals[unloaded] - np.median(residuals[unloaded])))
                ),
            },
            "black_dot_gap_px": {
                "contact": distribution(contact_complete, gaps),
                "unloaded": distribution(unloaded, gaps),
            },
            "opening_conditioned_gap_residual_px": {
                "contact_within_unloaded_opening_support": distribution(
                    contact_supported, residuals
                ),
                "unloaded": distribution(unloaded, residuals),
                "contact_frames_outside_support": int(
                    np.count_nonzero(contact_complete & ~baseline_supported)
                ),
            },
            "interpretation": (
                "Binary contact labels permit a contact-versus-unloaded geometry check only. "
                "No load magnitude was supplied, so this cannot calibrate force or Newtons."
            ),
        },
    }
    return audit, labels, gaps, residuals, baseline_supported


def contact_event_audit(
    opening: np.ndarray,
    gaps: np.ndarray,
    fps: float,
    event_times_s: list[float],
    window_radius_s: float = 0.25,
) -> dict | None:
    if not event_times_s:
        return None
    duration_s = (len(opening) - 1) / fps
    times = np.arange(len(opening), dtype=float) / fps
    events = []
    for event_time in event_times_s:
        event_time = float(event_time)
        if event_time < 0.0 or event_time > duration_s:
            raise ValueError(
                f"contact event {event_time} must be within [0, {duration_s:.6f}]"
            )
        frame = int(np.argmin(np.abs(times - event_time)))
        measured = np.isfinite(opening) & np.isfinite(gaps)
        window = measured & (np.abs(times - event_time) <= window_radius_s)
        event = {
            "requested_time_s": event_time,
            "nearest_frame": frame,
            "nearest_frame_time_s": float(times[frame]),
            "nearest_frame_measured": bool(measured[frame]),
            "nearest_opening_deg": (
                float(opening[frame]) if measured[frame] else None
            ),
            "nearest_black_dot_gap_px": (
                float(gaps[frame]) if measured[frame] else None
            ),
            "window_s": [
                max(0.0, event_time - window_radius_s),
                min(duration_s, event_time + window_radius_s),
            ],
            "window_complete_frames": int(np.count_nonzero(window)),
            "window_opening_median_deg": (
                float(np.median(opening[window])) if np.any(window) else None
            ),
            "window_black_dot_gap_median_px": (
                float(np.median(gaps[window])) if np.any(window) else None
            ),
        }
        events.append(event)
    return {
        "source": "user-provided clamping event times",
        "events": events,
        "interpretation": (
            "These are contact samples, not equal-force labels. Different objects and "
            "contact geometry can produce different deformation at the same force."
        ),
    }


def normalize_contact_intensity(
    raw: np.ndarray, opening: np.ndarray, valid_indices: np.ndarray
) -> tuple[np.ndarray, float, float]:
    valid_opening = opening[valid_indices]
    valid_raw = raw[valid_indices]
    bin_count = min(31, max(6, int(np.sqrt(len(valid_indices)))))
    edges = np.linspace(
        float(np.min(valid_opening)),
        float(np.max(valid_opening)) + 1e-9,
        bin_count + 1,
    )
    centres = []
    floors = []
    for index in range(bin_count):
        selected = (
            (valid_opening >= edges[index])
            & (valid_opening < edges[index + 1])
        )
        if np.count_nonzero(selected) < 5:
            continue
        centres.append(float(np.median(valid_opening[selected])))
        floors.append(float(np.percentile(valid_raw[selected], 10.0)))
    if len(centres) < 2:
        floor_curve = np.full_like(raw, np.percentile(valid_raw, 10.0))
    else:
        floors = np.asarray(floors)
        degree = min(3, len(centres) - 1)
        coefficients = np.polyfit(centres, floors, degree)
        floor_curve = np.polyval(coefficients, opening)
        floor_curve = np.clip(floor_curve, floors.min(), floors.max())
    signal = np.maximum(raw - floor_curve, 0.0)
    full_scale = float(np.nanpercentile(signal[valid_indices], 99.0))
    if not np.isfinite(full_scale) or full_scale <= 1e-9:
        raise ValueError("black-dot deformation did not produce a usable relative-force range")
    force = np.clip(100.0 * signal / full_scale, 0.0, 100.0)
    return force, float(np.nanmedian(floor_curve[valid_indices])), full_scale


def fit_force_model(
    observations: list[FrameObservation], opening: np.ndarray
) -> tuple[ForceModel, np.ndarray, np.ndarray, np.ndarray]:
    complete = np.asarray(
        [
            np.isfinite(item.included_angle_deg)
            and item.dot_left is not None
            and item.dot_right is not None
            for item in observations
        ],
        dtype=bool,
    )
    valid_indices = np.flatnonzero(complete)
    if len(valid_indices) < 30:
        raise ValueError(f"only {len(valid_indices)} frames contain yellow and black markers")
    open_threshold = float(np.percentile(opening[valid_indices], 65.0))
    unloaded = valid_indices[opening[valid_indices] >= open_threshold]
    if len(unloaded) < 15:
        raise ValueError("not enough open-jaw frames to establish an unloaded pad baseline")

    left_local = []
    right_local = []
    left_shape = []
    right_shape = []
    for index in unloaded:
        item = observations[index]
        assert item.yellow_left is not None and item.yellow_right is not None
        assert item.dot_left is not None and item.dot_right is not None
        left_local.append(
            point_to_local(item.dot_left.point, jaw_frame(item.yellow_left, "left"))
        )
        right_local.append(
            point_to_local(item.dot_right.point, jaw_frame(item.yellow_right, "right"))
        )
        left_shape.append([item.dot_left.area_px2, item.dot_left.minor_major_ratio])
        right_shape.append([item.dot_right.area_px2, item.dot_right.minor_major_ratio])
    left_local_median = np.median(left_local, axis=0)
    right_local_median = np.median(right_local, axis=0)
    left_shape_median = np.median(left_shape, axis=0)
    right_shape_median = np.median(right_shape, axis=0)

    raw = np.full(len(observations), np.nan)
    gap_component = np.full(len(observations), np.nan)
    shape_component = np.full(len(observations), np.nan)
    for index in valid_indices:
        item = observations[index]
        assert item.yellow_left is not None and item.yellow_right is not None
        assert item.dot_left is not None and item.dot_right is not None
        left_frame = jaw_frame(item.yellow_left, "left")
        right_frame = jaw_frame(item.yellow_right, "right")
        predicted_left = local_to_point(left_local_median, left_frame)
        predicted_right = local_to_point(right_local_median, right_frame)
        predicted_gap = predicted_right - predicted_left
        gap_direction = predicted_gap / np.linalg.norm(predicted_gap)
        actual_gap = item.dot_right.point - item.dot_left.point
        gap_excess = max(0.0, float((actual_gap - predicted_gap) @ gap_direction))
        marker_scale = (left_frame.scale_px + right_frame.scale_px) / 2.0
        gap_strain = gap_excess / marker_scale
        local_residual = (
            np.linalg.norm(point_to_local(item.dot_left.point, left_frame) - left_local_median)
            + np.linalg.norm(point_to_local(item.dot_right.point, right_frame) - right_local_median)
        ) / 2.0
        shape_change = (
            abs(np.log(item.dot_left.area_px2 / left_shape_median[0]))
            + abs(np.log(item.dot_right.area_px2 / right_shape_median[0]))
            + abs(np.log(item.dot_left.minor_major_ratio / left_shape_median[1]))
            + abs(np.log(item.dot_right.minor_major_ratio / right_shape_median[1]))
        ) / 4.0
        gap_component[index] = gap_strain
        shape_component[index] = shape_change
        raw[index] = 0.70 * gap_strain + 0.20 * local_residual + 0.10 * shape_change

    baseline = float(np.nanmedian(raw[unloaded]))
    noise_mad = float(np.nanmedian(np.abs(raw[unloaded] - baseline)))
    force, noise_floor, full_scale = normalize_contact_intensity(
        raw, opening, valid_indices
    )
    model = ForceModel(
        left_local_median,
        right_local_median,
        left_shape_median,
        right_shape_median,
        baseline,
        noise_mad,
        noise_floor,
        full_scale,
    )
    return model, force, gap_component, shape_component


def measurement_state(
    observation: FrameObservation,
    angle_recovered: bool,
    force_recovered: bool,
    angle_source: str | None = None,
) -> str:
    if angle_source is not None:
        return angle_source
    if (
        np.isfinite(observation.included_angle_deg)
        and observation.dot_left is not None
        and observation.dot_right is not None
    ):
        return "MEASURED"
    if angle_recovered or force_recovered:
        return "RECOVERED <= 0.25 s"
    return "UNAVAILABLE"


def draw_text(
    image: np.ndarray,
    value: str,
    point: tuple[int, int],
    scale: float,
    color: tuple[int, int, int] = WHITE,
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        value,
        point,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_detection_overlay(
    image: np.ndarray,
    observation: FrameObservation,
    model: ForceModel,
    opening_angle_deg: float = np.nan,
    fingertip_force_percent: float = np.nan,
) -> None:
    for points, color in (
        (observation.yellow_left, CYAN),
        (observation.yellow_right, GREEN),
    ):
        if points is None:
            continue
        pixels = np.round(points).astype(int)
        cv2.polylines(image, [pixels], False, color, 4, cv2.LINE_AA)
        for point in pixels:
            cv2.circle(image, tuple(point), 10, color, 3, cv2.LINE_AA)
    if observation.dot_left is not None and observation.dot_right is not None:
        left = tuple(np.round(observation.dot_left.point).astype(int))
        right = tuple(np.round(observation.dot_right.point).astype(int))
        cv2.line(image, left, right, RED, 3, cv2.LINE_AA)
        cv2.circle(image, left, 12, RED, 3, cv2.LINE_AA)
        cv2.circle(image, right, 12, RED, 3, cv2.LINE_AA)
        if np.isfinite(fingertip_force_percent):
            left_point = observation.dot_left.point
            right_point = observation.dot_right.point
            inward = right_point - left_point
            inward /= np.linalg.norm(inward)
            arrow_length = scaled(
                20.0 + 0.8 * float(np.clip(fingertip_force_percent, 0.0, 100.0)),
                image,
            )
            cv2.arrowedLine(
                image,
                tuple(np.round(left_point).astype(int)),
                tuple(np.round(left_point + arrow_length * inward).astype(int)),
                AMBER,
                6,
                cv2.LINE_AA,
                tipLength=0.28,
            )
            cv2.arrowedLine(
                image,
                tuple(np.round(right_point).astype(int)),
                tuple(np.round(right_point - arrow_length * inward).astype(int)),
                AMBER,
                6,
                cv2.LINE_AA,
                tipLength=0.28,
            )
    if observation.yellow_left is not None and observation.yellow_right is not None:
        predicted_left = local_to_point(
            model.left_local, jaw_frame(observation.yellow_left, "left")
        )
        predicted_right = local_to_point(
            model.right_local, jaw_frame(observation.yellow_right, "right")
        )
        for point in (predicted_left, predicted_right):
            cv2.drawMarker(
                image,
                tuple(np.round(point).astype(int)),
                WHITE,
                cv2.MARKER_CROSS,
                20,
                2,
                cv2.LINE_AA,
            )

        left_axis = observation.yellow_left[0] - observation.yellow_left[2]
        right_axis = observation.yellow_right[0] - observation.yellow_right[2]
        left_axis /= np.linalg.norm(left_axis)
        right_axis /= np.linalg.norm(right_axis)
        vertex = (
            observation.yellow_left[2] + observation.yellow_right[2]
        ) / 2.0
        ray_length = 0.95 * min(
            np.linalg.norm(observation.yellow_left[0] - observation.yellow_left[2]),
            np.linalg.norm(observation.yellow_right[0] - observation.yellow_right[2]),
        )
        left_ray = vertex + ray_length * left_axis
        right_ray = vertex + ray_length * right_axis
        cv2.line(
            image,
            tuple(np.round(vertex).astype(int)),
            tuple(np.round(left_ray).astype(int)),
            CYAN,
            6,
            cv2.LINE_AA,
        )
        cv2.line(
            image,
            tuple(np.round(vertex).astype(int)),
            tuple(np.round(right_ray).astype(int)),
            GREEN,
            6,
            cv2.LINE_AA,
        )
        left_heading = float(np.arctan2(left_axis[1], left_axis[0]))
        right_heading = float(np.arctan2(right_axis[1], right_axis[0]))
        sweep = (right_heading - left_heading + np.pi) % (2.0 * np.pi) - np.pi
        headings = left_heading + np.linspace(0.0, sweep, 40)
        radius = 0.30 * ray_length
        arc = vertex + radius * np.column_stack(
            [np.cos(headings), np.sin(headings)]
        )
        cv2.polylines(
            image,
            [np.round(arc).astype(int)],
            False,
            AMBER,
            5,
            cv2.LINE_AA,
        )

    scale = image.shape[1] / 1920.0
    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (round(600 * scale), round(65 * scale)),
        (round(1390 * scale), round(315 * scale)),
        (8, 12, 18),
        -1,
    )
    cv2.addWeighted(overlay, 0.78, image, 0.22, 0.0, image)
    opening_text = (
        f"{float(opening_angle_deg):.1f} deg"
        if np.isfinite(opening_angle_deg)
        else "N/A"
    )
    included_text = (
        f"{observation.included_angle_deg:.1f} deg"
        if np.isfinite(observation.included_angle_deg)
        else "N/A"
    )
    force_text = (
        f"{float(fingertip_force_percent):.1f} %"
        if np.isfinite(fingertip_force_percent)
        else "N/A"
    )
    draw_text(
        image,
        f"JAW OPENING  {opening_text}",
        (round(640 * scale), round(135 * scale)),
        1.25 * scale,
        CYAN,
        max(2, round(3 * scale)),
    )
    draw_text(
        image,
        f"INCLUDED AXIS ANGLE  {included_text}",
        (round(640 * scale), round(205 * scale)),
        0.78 * scale,
        WHITE,
        max(1, round(2 * scale)),
    )
    draw_text(
        image,
        f"REL FINGERTIP FORCE  {force_text}",
        (round(640 * scale), round(275 * scale)),
        0.78 * scale,
        AMBER,
        max(1, round(2 * scale)),
    )


def project_cad(edges: np.ndarray, origin: tuple[int, int], size: tuple[int, int]) -> np.ndarray:
    points = edges.reshape(-1, 3)
    rotation = Rotation.from_euler("xyz", [58.0, 0.0, 90.0], degrees=True)
    view = rotation.apply(points)
    low = np.percentile(view[:, :2], 1.0, axis=0)
    high = np.percentile(view[:, :2], 99.0, axis=0)
    center = (low + high) / 2.0
    span = np.maximum(high - low, 1e-9)
    scale = 0.88 * min(size[0] / span[0], size[1] / span[1])
    pixels = (view[:, :2] - center) * scale
    pixels[:, 1] *= -1.0
    pixels += np.array([origin[0] + size[0] / 2.0, origin[1] + size[1] / 2.0])
    return pixels.reshape(edges.shape[:-1] + (2,))


def draw_cad(
    canvas: np.ndarray,
    model: UrdfWireframe,
    opening_angle_deg: float,
    origin: tuple[int, int],
    size: tuple[int, int],
) -> None:
    value = 0.0 if not np.isfinite(opening_angle_deg) else float(opening_angle_deg)
    edges = model.articulate(value / 2.0, -value / 2.0)
    pixels = np.round(project_cad(edges, origin, size)).astype(int)
    for first, second in pixels:
        cv2.line(canvas, tuple(first), tuple(second), (128, 194, 234), 1, cv2.LINE_AA)
    cv2.rectangle(
        canvas,
        origin,
        (origin[0] + size[0], origin[1] + size[1]),
        (53, 65, 80),
        1,
    )


def force_color(force_percent: float) -> tuple[int, int, int]:
    if force_percent < 35.0:
        return GREEN
    if force_percent < 70.0:
        return AMBER
    return RED


def transcode_h264(intermediate: Path, output: Path) -> None:
    gst_launch = shutil.which("gst-launch-1.0")
    if gst_launch is None:
        raise ValueError("gst-launch-1.0 is required for the H.264 demo output")
    subprocess.run(
        [
            gst_launch,
            "-e",
            "filesrc",
            f"location={intermediate}",
            "!",
            "qtdemux",
            "!",
            "avdec_mpeg4",
            "!",
            "videoconvert",
            "!",
            "x264enc",
            "speed-preset=veryfast",
            "pass=quant",
            "quantizer=18",
            "key-int-max=60",
            "!",
            "video/x-h264,profile=high",
            "!",
            "mp4mux",
            "faststart=true",
            "!",
            "filesink",
            f"location={output}",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def render_measurement_overlay(
    video: Path,
    output: Path,
    observations: list[FrameObservation],
    fps: float,
    opening: np.ndarray,
    model: ForceModel,
    force: np.ndarray,
    force_display_enabled: bool,
) -> None:
    capture = cv2.VideoCapture(str(video))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    intermediate = output.with_name(output.stem + "_mp4v.mp4")
    writer = cv2.VideoWriter(
        str(intermediate),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise ValueError(f"cannot create measurement overlay video: {output}")
    for index, observation in enumerate(observations):
        ok, frame = capture.read()
        if not ok:
            break
        draw_detection_overlay(
            frame,
            observation,
            model,
            opening[index],
            force[index] if force_display_enabled else np.nan,
        )
        draw_text(
            frame,
            f"t={index / fps:05.2f}s   frame={index}",
            (scaled(640, frame), scaled(350, frame)),
            0.62 * frame.shape[1] / 1920.0,
            MUTED,
            max(1, round(2 * frame.shape[1] / 1920.0)),
        )
        writer.write(frame)
    capture.release()
    writer.release()
    transcode_h264(intermediate, output)
    intermediate.unlink()


def render_demo(
    video: Path,
    output: Path,
    observations: list[FrameObservation],
    fps: float,
    opening: np.ndarray,
    opening_recovered: np.ndarray,
    force: np.ndarray,
    force_recovered: np.ndarray,
    model: ForceModel,
    urdf_model: UrdfWireframe,
    rig_id: str,
    force_validated: bool = True,
    contact_labels: np.ndarray | None = None,
    angle_sources: np.ndarray | None = None,
) -> None:
    capture = cv2.VideoCapture(str(video))
    intermediate = output.with_name(output.stem + "_mp4v.mp4")
    writer = cv2.VideoWriter(
        str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1600, 900)
    )
    if not writer.isOpened():
        raise ValueError(f"cannot create demo video: {output}")
    last_angle = 0.0
    last_force = 0.0
    for index, observation in enumerate(observations):
        ok, frame = capture.read()
        if not ok:
            break
        draw_detection_overlay(
            frame,
            observation,
            model,
            opening[index],
            force[index] if force_validated else np.nan,
        )
        crop = frame[scaled(850, frame):scaled(1620, frame), scaled(450, frame):scaled(1470, frame)]
        crop = cv2.resize(crop, (940, 710), interpolation=cv2.INTER_AREA)
        canvas = np.full((900, 1600, 3), BG, dtype=np.uint8)
        canvas[105:815, 25:965] = crop
        cv2.rectangle(canvas, (25, 105), (965, 815), (60, 73, 89), 2)
        draw_text(canvas, "FRONT LENS / MARKER MEASUREMENT", (30, 78), 0.68, WHITE, 2)
        draw_text(canvas, "yellow: 3 points per jaw   red: black-dot gap   white: unloaded prediction", (30, 852), 0.45, MUTED)

        if np.isfinite(opening[index]):
            last_angle = float(opening[index])
        if np.isfinite(force[index]):
            last_force = float(force[index])
        state = measurement_state(
            observation,
            bool(opening_recovered[index]),
            bool(force_recovered[index]),
            None if angle_sources is None else str(angle_sources[index]),
        )
        state_color = (
            GREEN if state == "MEASURED"
            else AMBER if state.startswith("RECOVERED") or "ONE_SIDED" in state
            else RED
        )

        cv2.rectangle(canvas, (995, 25), (1575, 875), PANEL, -1)
        draw_text(canvas, "CURRENT HARDWARE", (1020, 65), 0.72, WHITE, 2)
        draw_text(canvas, rig_id, (1020, 94), 0.40, MUTED)
        draw_cad(canvas, urdf_model, last_angle, (1020, 115), (530, 410))
        draw_text(canvas, "v52 CAD + current joint origins", (1035, 510), 0.43, MUTED)

        draw_text(canvas, "JAW OPENING", (1020, 570), 0.48, MUTED)
        draw_text(canvas, f"{last_angle:5.1f} deg", (1020, 625), 1.30, CYAN, 3)
        draw_text(
            canvas,
            "RELATIVE FORCE" if force_validated else "RAW DEFORMATION SCORE",
            (1270, 570), 0.48, MUTED,
        )
        color = force_color(last_force) if force_validated else AMBER
        draw_text(canvas, f"{last_force:5.1f} %", (1270, 625), 1.30, color, 3)
        cv2.rectangle(canvas, (1020, 660), (1545, 694), (49, 57, 68), -1)
        cv2.rectangle(canvas, (1020, 660), (1020 + round(5.25 * last_force), 694), color, -1)
        draw_text(
            canvas,
            "UNCALIBRATED: not Newtons" if force_validated else "REJECTED AS FORCE / UNVALIDATED",
            (1020, 725), 0.50, AMBER, 2,
        )
        ground_truth = (
            str(contact_labels[index])
            if contact_labels is not None and contact_labels[index] != "UNLABELED"
            else None
        )
        display_state = (
            state
            if force_validated
            else f"LABEL {ground_truth}" if ground_truth else "RAW SCORE ONLY"
        )
        display_state_color = state_color if force_validated else AMBER if ground_truth else RED
        draw_text(canvas, display_state, (1020, 770), 0.58, display_state_color, 2)
        yellow_count = int(observation.yellow_left is not None) * 3 + int(observation.yellow_right is not None) * 3
        black_count = int(observation.dot_left is not None) + int(observation.dot_right is not None)
        draw_text(
            canvas,
            f"markers {yellow_count}/6 yellow   {black_count}/2 black",
            (1020, 810),
            0.47,
            WHITE,
        )
        draw_text(
            canvas,
            f"t={index / fps:05.2f}s   frame={index}",
            (1020, 850),
            0.45,
            MUTED,
        )
        writer.write(canvas)
    capture.release()
    writer.release()
    transcode_h264(intermediate, output)
    intermediate.unlink()


def write_csv(
    path: Path,
    observations: list[FrameObservation],
    fps: float,
    opening: np.ndarray,
    opening_recovered: np.ndarray,
    force: np.ndarray,
    force_recovered: np.ndarray,
    gap_component: np.ndarray,
    shape_component: np.ndarray,
    contact_labels: np.ndarray,
    black_dot_gap_px: np.ndarray,
    gap_residual_px: np.ndarray,
    baseline_supported: np.ndarray,
    angle_sources: np.ndarray | None = None,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame",
                "time_s",
                "included_jaw_angle_deg",
                "opening_angle_deg",
                "relative_force_percent",
                "measurement_state",
                "contact_ground_truth",
                "black_dot_gap_px",
                "opening_conditioned_gap_residual_px",
                "unloaded_opening_support",
                "yellow_measured",
                "left_yellow_axis_measured",
                "right_yellow_axis_measured",
                "black_measured",
                "gap_strain_component",
                "shape_change_component",
                "left_black_x",
                "left_black_y",
                "right_black_x",
                "right_black_y",
            ]
        )
        for index, item in enumerate(observations):
            state = measurement_state(
                item,
                bool(opening_recovered[index]),
                bool(force_recovered[index]),
                None if angle_sources is None else str(angle_sources[index]),
            )
            left = item.dot_left.point if item.dot_left is not None else [np.nan, np.nan]
            right = item.dot_right.point if item.dot_right is not None else [np.nan, np.nan]
            writer.writerow(
                [
                    index,
                    f"{index / fps:.9f}",
                    item.included_angle_deg,
                    opening[index],
                    force[index],
                    state,
                    contact_labels[index],
                    black_dot_gap_px[index],
                    gap_residual_px[index],
                    int(baseline_supported[index]),
                    int(item.yellow_left is not None and item.yellow_right is not None),
                    int(item.yellow_left is not None),
                    int(item.yellow_right is not None),
                    int(item.dot_left is not None and item.dot_right is not None),
                    gap_component[index],
                    shape_component[index],
                    *left,
                    *right,
                ]
            )


def main() -> int:
    args = parse_args()
    front_video = args.front_video.resolve(strict=True)
    source_osv = args.source_osv.resolve(strict=True)
    rig = load_rig_revision(
        args.rig_revision,
        allow_diagnostic_world=args.allow_diagnostic_rig,
    )
    cad = rig["cad_revision"]
    if cad is None:
        raise ValueError("rig revision does not reference a current CAD revision")
    urdf_path = (Path(__file__).resolve().parent / cad["urdf"]["path"]).resolve()
    urdf_model = load_urdf_wireframe(urdf_path, max_triangles_per_mesh=180)
    angle_revision = None
    angle_revision_path = None
    closed_reference_override = None
    hardware_angle: dict = {}
    hardware_role = None
    camera_serial = None
    x5_angle_mode = "pca-axis"
    x5_included_angle_range = (35.0, 80.0)
    x5_dot_selection = "fixed-bands"
    force_revision = None
    force_revision_path = None
    fixed_force_audit = None
    if args.camera_profile == "insta360-x5-front":
        if args.x5_angle_revision is None or args.base_tag_id is None:
            raise ValueError(
                "X5 processing requires --x5-angle-revision and --base-tag-id"
            )
        angle_revision_path = args.x5_angle_revision.resolve(strict=True)
        angle_revision = json.loads(angle_revision_path.read_text(encoding="utf-8"))
        detector = angle_revision.get("detector", {})
        algorithm = detector.get("algorithm")
        angle_modes = {
            "yellow_jaw_contour_pca_axis": "pca-axis",
            "three_yellow_pad_dot_centroids": "physical-marker-triad",
        }
        if algorithm not in angle_modes:
            raise ValueError(f"unsupported X5 angle algorithm: {algorithm!r}")
        x5_angle_mode = angle_modes[algorithm]
        x5_included_angle_range = tuple(
            map(float, detector.get("included_angle_range_deg", [35.0, 80.0]))
        )
        x5_dot_selection = str(detector.get("dot_selection", "fixed-bands"))
        role = "physical_left" if args.base_tag_id == 2 else "physical_right"
        hardware_angle = angle_revision.get("hardware", {}).get(role, {})
        if hardware_angle.get("base_tag_id") != args.base_tag_id:
            raise ValueError("X5 angle revision BaseTag binding mismatch")
        hardware_matches = [
            (name, robot)
            for name, robot in rig["hardware"]["robots"].items()
            if robot.get("base_tag_id") == args.base_tag_id
        ]
        if len(hardware_matches) != 1:
            raise ValueError("rig must bind the requested BaseTag to exactly one camera")
        hardware_role, bound_robot = hardware_matches[0]
        camera_serial = bound_robot["camera_serial"]
        if "evidence_audit" in hardware_angle:
            evidence_path = Path(hardware_angle["evidence_audit"])
            if sha256(evidence_path) != hardware_angle["evidence_audit_sha256"]:
                raise ValueError("X5 angle revision evidence hash mismatch")
        elif "zero_evidence" in hardware_angle:
            evidence_path = Path(hardware_angle["zero_evidence"]["source_video"])
            if sha256(evidence_path) != hardware_angle["zero_evidence"]["source_sha256"]:
                raise ValueError("X5 angle zero-evidence hash mismatch")
        else:
            raise ValueError("X5 angle revision has no immutable zero evidence")
        closed_reference_override = float(
            hardware_angle["closed_reference_included_angle_deg"]
        )
        if abs(args.maximum_recovery_gap_s - 0.25) > 1e-9:
            raise ValueError("accepted X5 angle revision requires a 0.25 s recovery gap")
    elif args.x5_angle_revision is not None or args.base_tag_id is not None:
        raise ValueError("X5 angle revision arguments require insta360-x5-front")
    if args.relative_force_revision is not None:
        if args.camera_profile != "insta360-x5-front":
            raise ValueError("relative force revisions currently require Insta360 X5")
        force_revision_path = args.relative_force_revision.resolve(strict=True)
        force_revision = json.loads(
            force_revision_path.read_text(encoding="utf-8")
        )
        if force_revision.get("schema_version") != "x5-relative-force-revision/1.0":
            raise ValueError("invalid X5 relative-force revision schema")
        if force_revision.get("source", {}).get("video_sha256") != sha256(source_osv):
            raise ValueError("relative-force revision source-video mismatch")
        force_hardware = force_revision.get("hardware", {})
        if (
            force_hardware.get("camera_serial") != camera_serial
            or force_hardware.get("base_tag_id") != args.base_tag_id
        ):
            raise ValueError("relative-force revision hardware binding mismatch")
        if (
            angle_revision_path is None
            or force_hardware.get("angle_revision", {}).get("sha256")
            != sha256(angle_revision_path)
        ):
            raise ValueError("relative-force revision angle binding mismatch")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    observations, fps, size = analyze_video(
        front_video,
        args.camera_profile,
        x5_angle_mode,
        x5_included_angle_range,
        x5_dot_selection,
    )
    raw_opening, closed_reference = opening_angles(
        observations, closed_reference_override
    )
    raw_opening, angle_sources, one_sided_audit = (
        apply_one_sided_opening_fallback(
            observations, raw_opening, hardware_angle
        )
    )
    base_tag_counts = {
        tag_id: sum(tag_id in item.base_tag_ids for item in observations)
        for tag_id in (2, 3)
    }
    if args.camera_profile == "insta360-x5-front":
        assert args.base_tag_id is not None
        expected_count = base_tag_counts[args.base_tag_id]
        if expected_count < max(10, round(0.10 * len(observations))):
            raise ValueError(
                f"expected BaseTag{args.base_tag_id} is visible in only "
                f"{expected_count}/{len(observations)} frames"
            )
        other_tag_id = 3 if args.base_tag_id == 2 else 2
        if base_tag_counts[other_tag_id] > base_tag_counts[args.base_tag_id]:
            raise ValueError(
                f"BaseTag binding mismatch: ID{other_tag_id} is more visible than "
                f"ID{args.base_tag_id}"
            )
    force_model, legacy_raw_force, gap_component, shape_component = fit_force_model(
        observations, raw_opening
    )
    if force_revision is not None:
        raw_force, fixed_force_audit = apply_fixed_relative_force(
            observations, raw_opening, force_revision
        )
    else:
        raw_force = legacy_raw_force
    maximum_gap = round(args.maximum_recovery_gap_s * fps)
    opening, opening_recovered = bounded_interpolate(raw_opening, maximum_gap)
    angle_sources[opening_recovered] = "RECOVERED <= 0.25 s"
    force, force_recovered = bounded_interpolate(raw_force, maximum_gap)
    opening = np.where(np.isfinite(opening), nanmedian_filter(opening), np.nan)
    force = np.where(np.isfinite(force), nanmedian_filter(force), np.nan)
    (
        contact_ground_truth,
        contact_labels,
        black_dot_gap_px,
        gap_residual_px,
        baseline_supported,
    ) = labeled_contact_gap_audit(
        observations,
        raw_opening,
        fps,
        args.contact_interval_s,
    )
    contact_events = contact_event_audit(
        raw_opening,
        black_dot_gap_px,
        fps,
        args.contact_event_s,
    )
    x5_force_rejected = (
        args.camera_profile == "insta360-x5-front"
        and not args.display_relative_fingertip_force
        and force_revision is None
    )

    csv_path = output_dir / "force_angle_observations.csv"
    video_path = output_dir / "gripper_force_angle_demo.mp4"
    overlay_video_path = output_dir / "jaw_angle_marker_overlay.mp4"
    write_csv(
        csv_path,
        observations,
        fps,
        opening,
        opening_recovered,
        force,
        force_recovered,
        gap_component,
        shape_component,
        contact_labels,
        black_dot_gap_px,
        gap_residual_px,
        baseline_supported,
        angle_sources,
    )
    render_measurement_overlay(
        front_video,
        overlay_video_path,
        observations,
        fps,
        opening,
        force_model,
        force,
        not x5_force_rejected,
    )
    render_demo(
        front_video,
        video_path,
        observations,
        fps,
        opening,
        opening_recovered,
        force,
        force_recovered,
        force_model,
        urdf_model,
        rig["revision"]["revision_id"],
        not x5_force_rejected,
        contact_labels,
        angle_sources,
    )

    yellow_measured = np.asarray(
        [item.yellow_left is not None and item.yellow_right is not None for item in observations]
    )
    black_measured = np.asarray(
        [item.dot_left is not None and item.dot_right is not None for item in observations]
    )
    angle_source_counts = {
        str(state): int(np.count_nonzero(angle_sources == state))
        for state in np.unique(angle_sources)
    }
    audit = {
        "schema_version": "gripper-force-angle-demo/1.0",
        "status": (
            "REJECTED_X5_FORCE_MODEL_UNVALIDATED"
            if x5_force_rejected
            else "DIAGNOSTIC_FIXED_SCALE_RELATIVE_FINGERTIP_FORCE"
            if force_revision is not None
            else "DIAGNOSTIC_UNCALIBRATED_RELATIVE_FINGERTIP_FORCE"
            if args.camera_profile == "insta360-x5-front"
            else "DIAGNOSTIC_UNCALIBRATED_RELATIVE_FORCE"
        ),
        "source": {
            "osv": str(source_osv),
            "osv_sha256": sha256(source_osv),
            "front_lens_video": str(front_video),
            "front_lens_video_sha256": sha256(front_video),
            "front_lens_track": 1,
            "frame_size": list(size),
            "fps": fps,
            "camera_profile": args.camera_profile,
            "frame_count": len(observations),
            "base_tag_id": args.base_tag_id,
            "base_tag_frame_counts": base_tag_counts,
            "hardware_role": hardware_role,
            "camera_serial": camera_serial,
        },
        "rig_revision": {
            "path": str(rig["revision_path"]),
            "id": rig["revision"]["revision_id"],
            "sha256": rig["revision_sha256"],
            "geometry_path": str(rig["geometry_path"]),
            "cad_revision_path": str(rig["cad_revision_path"]),
            "cad_revision_id": cad["revision_id"],
        },
        "angle": {
            "method": (
                {
                    "yellow_jaw_contour_pca_axis": (
                        "yellow jaw-contour PCA axes; included-line angle relative "
                        "to frozen hardware closed reference"
                    ),
                    "three_yellow_pad_dot_centroids": (
                        "three highlighted yellow pad-dot centroids per jaw; "
                        "included-line angle relative to frozen hardware closed reference"
                    ),
                }[angle_revision["detector"]["algorithm"]]
                if angle_revision is not None
                else "three yellow centroids per jaw; included-line angle relative to capture closed reference"
            ),
            "revision": (
                {
                    "path": str(angle_revision_path),
                    "id": angle_revision["revision_id"],
                    "sha256": sha256(angle_revision_path),
                }
                if angle_revision is not None and angle_revision_path is not None
                else None
            ),
            "closed_reference_source": (
                "frozen hardware revision"
                if closed_reference_override is not None
                else "capture 97th percentile"
            ),
            "closed_reference_included_angle_deg": closed_reference,
            "measured_frame_ratio": float(yellow_measured.mean()),
            "opening_range_deg": [float(np.nanmin(opening)), float(np.nanmax(opening))],
            "maximum_recovery_gap_s": args.maximum_recovery_gap_s,
            "available_frame_ratio": float(np.isfinite(opening).mean()),
            "measurement_source_counts": angle_source_counts,
            "one_sided_fallback": one_sided_audit,
        },
        "force": {
            "unit": "relative_percent",
            "newtons_calibrated": False,
            "validated_for_display": not x5_force_rejected,
            "quantity": (
                force_revision["quantity"]
                if force_revision is not None
                else "capture_local_relative_fingertip_force_proxy"
            ),
            "application_point": "two distal fingertip contact points",
            "direction": "equal and opposite inward vectors along the jaw-closing axis",
            "rejection_reason": (
                "Raw pad geometry depends on jaw opening, object shape, and contact "
                "location; no load ground truth is available, so force remains unvalidated."
                if x5_force_rejected else None
            ),
            "method": (
                "fixed opening-conditioned absolute black-dot gap residual"
                if force_revision is not None
                else "black-dot gap residual versus rigid yellow-marker prediction plus dot centroid and ellipse-shape change"
            ),
            "revision": (
                {
                    "path": str(force_revision_path),
                    "id": force_revision["revision_id"],
                    "sha256": sha256(force_revision_path),
                }
                if force_revision is not None and force_revision_path is not None
                else None
            ),
            "fixed_scale_across_captures": force_revision is not None,
            "capture_local_normalization": force_revision is None,
            "unloaded_selection": (
                force_revision["source"]["free_intervals_s"]
                if force_revision is not None
                else "top 35 percent of measured opening angles in this capture"
            ),
            "baseline": None if force_revision is not None else force_model.baseline,
            "baseline_noise_mad": (
                None if force_revision is not None else force_model.noise_mad
            ),
            "onset_noise_floor": (
                fixed_force_audit["noise_floor_px"]
                if fixed_force_audit is not None else force_model.noise_floor
            ),
            "onset_policy": (
                "fixed free-motion P99 absolute residual"
                if force_revision is not None
                else "opening-conditioned 10th-percentile lower envelope"
            ),
            "full_scale_99th_percentile": (
                fixed_force_audit["full_scale_signal_px"]
                if fixed_force_audit is not None else force_model.full_scale
            ),
            "measured_frame_ratio": (
                fixed_force_audit["measured_frame_ratio"]
                if fixed_force_audit is not None
                else float((yellow_measured & black_measured).mean())
            ),
            "fixed_scale_audit": fixed_force_audit,
            "relative_force_range_percent": [
                float(np.nanmin(force)),
                float(np.nanmax(force)),
            ],
            "warning": (
                "Relative force uses a fixed hardware-revision scale; it is not Newtons "
                "and is not valid across different gripper hardware revisions."
                if force_revision is not None
                else "Relative fingertip force is capture-local and must not be interpreted "
                "as Newtons or compared across objects without load calibration."
            ),
        },
        "contact_ground_truth": contact_ground_truth,
        "contact_events": contact_events,
        "outputs": {
            "video": str(video_path),
            "video_sha256": sha256(video_path),
            "jaw_angle_marker_overlay_video": str(overlay_video_path),
            "jaw_angle_marker_overlay_video_sha256": sha256(overlay_video_path),
            "csv": str(csv_path),
            "csv_sha256": sha256(csv_path),
        },
        "training_ready": False,
    }
    audit_path = output_dir / "audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
