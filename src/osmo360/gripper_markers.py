"""Fast fixed-ROI gripper marker detection on full-resolution camera frames."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np


MARKER_ALGORITHM_REVISION = "x5-fixed-roi-dual-colour-gripper-markers-v3"
MARKER_ROI_PX_1920 = (450, 950, 1470, 1800)
YELLOW_ON_BLACK_FAMILY = "black_gripper_yellow_triads"
BLACK_ON_YELLOW_FAMILY = "yellow_gripper_black_pair"
HSV_THRESHOLD = {
    "h_min": 18,
    "h_max": 48,
    "s_min": 75,
    "v_min": 65,
}
DARK_THRESHOLD = {"v_max": 115}


@dataclass(frozen=True)
class GripperMarkerCandidates:
    """Per-frame candidates for both supported physical marker layouts."""

    yellow_left: np.ndarray | None
    yellow_right: np.ndarray | None
    black_left: np.ndarray | None
    black_right: np.ndarray | None

    @property
    def yellow_included_angle_deg(self) -> float:
        if self.yellow_left is None or self.yellow_right is None:
            return float("nan")
        return included_angle_deg(self.yellow_left, self.yellow_right)

    @property
    def black_pair_gap_px(self) -> float:
        if self.black_left is None or self.black_right is None:
            return float("nan")
        return float(np.linalg.norm(self.black_left - self.black_right))


def marker_signature() -> dict[str, object]:
    return {
        "algorithm_revision": MARKER_ALGORITHM_REVISION,
        "source_pixel_format": "yuv420p/yuvj420p with ROI range normalization",
        "source_size": [1920, 1920],
        "roi_xyxy_px": list(MARKER_ROI_PX_1920),
        "families": {
            YELLOW_ON_BLACK_FAMILY: {
                "threshold": {"colour_space": "opencv_hsv", **HSV_THRESHOLD},
                "selection": "dark-annulus side-specific collinear triad",
            },
            BLACK_ON_YELLOW_FAMILY: {
                "threshold": {
                    "colour_space": "opencv_hsv",
                    **DARK_THRESHOLD,
                },
                "selection": "yellow-annulus bilateral dot pair",
            },
        },
    }


def _centroid(contour: np.ndarray) -> np.ndarray | None:
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    return np.asarray(
        [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
        dtype=np.float64,
    )


def _select_triad(
    candidates: list[tuple[float, np.ndarray]], side: str
) -> np.ndarray | None:
    best: tuple[float, np.ndarray] | None = None
    for items in combinations(candidates, 3):
        points = np.asarray(sorted((item[1] for item in items), key=lambda point: point[1]))
        top, middle, bottom = points
        span = float(np.linalg.norm(top - bottom))
        first_gap = float(np.linalg.norm(top - middle))
        second_gap = float(np.linalg.norm(middle - bottom))
        if not 140.0 <= span <= 450.0 or min(first_gap, second_gap) < 35.0:
            continue
        horizontal_delta = float(bottom[0] - top[0])
        if (side == "left" and horizontal_delta > -15.0) or (
            side == "right" and horizontal_delta < 15.0
        ):
            continue
        centered = points - points.mean(axis=0)
        _, _, axes = np.linalg.svd(centered, full_matrices=False)
        residual = float(np.max(np.abs(centered @ axes[1])))
        if residual > 22.0:
            continue
        gap_penalty = abs(float(np.log(first_gap / second_gap)))
        area_reward = sum(item[0] for item in items) / 3000.0
        score = residual + 6.0 * gap_penalty - area_reward
        if best is None or score < best[0]:
            best = (score, points)
    return None if best is None else best[1]


def _detect_candidates(hsv_roi: np.ndarray) -> GripperMarkerCandidates:
    x0, y0, _, _ = MARKER_ROI_PX_1920
    threshold = HSV_THRESHOLD
    yellow_mask = cv2.inRange(
        hsv_roi,
        np.asarray(
            [threshold["h_min"], threshold["s_min"], threshold["v_min"]],
            dtype=np.uint8,
        ),
        np.asarray([threshold["h_max"], 255, 255], dtype=np.uint8),
    )
    yellow_mask = cv2.morphologyEx(
        yellow_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    contours, _ = cv2.findContours(
        yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    by_side: dict[str, list[tuple[float, np.ndarray]]] = {"left": [], "right": []}
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not 50.0 <= area <= 1200.0:
            continue
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if not 0.45 <= box_width / max(box_height, 1) <= 2.2:
            continue
        centre = _centroid(contour)
        if centre is None:
            continue
        radius = max(box_width, box_height)
        patch_y0 = max(0, y - radius)
        patch_y1 = min(hsv_roi.shape[0], y + box_height + radius)
        patch_x0 = max(0, x - radius)
        patch_x1 = min(hsv_roi.shape[1], x + box_width + radius)
        grid_y, grid_x = np.mgrid[patch_y0:patch_y1, patch_x0:patch_x1]
        distance = np.hypot(grid_x - centre[0], grid_y - centre[1])
        annulus = (distance >= 0.8 * radius) & (distance <= 1.8 * radius)
        patch = hsv_roi[patch_y0:patch_y1, patch_x0:patch_x1]
        dark = (patch[..., 1] < 90) | (patch[..., 2] < 90)
        if not annulus.any() or float(dark[annulus].mean()) < 0.35:
            continue
        global_centre = centre + np.asarray([x0, y0], dtype=np.float64)
        side = "left" if global_centre[0] < 960.0 else "right"
        by_side[side].append((area, global_centre))
    yellow_left = _select_triad(by_side["left"], "left")
    yellow_right = _select_triad(by_side["right"], "right")

    dark_mask = cv2.inRange(
        hsv_roi,
        np.asarray([0, 0, 0], dtype=np.uint8),
        np.asarray([179, 255, DARK_THRESHOLD["v_max"]], dtype=np.uint8),
    )
    dark_mask = cv2.morphologyEx(
        dark_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    dark_contours, _ = cv2.findContours(
        dark_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )
    dark_by_side: dict[str, list[tuple[float, float, np.ndarray]]] = {
        "left": [],
        "right": [],
    }
    for contour in dark_contours:
        area = float(cv2.contourArea(contour))
        if not 35.0 <= area <= 350.0:
            continue
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if not 0.35 <= box_width / max(box_height, 1) <= 2.8:
            continue
        centre = _centroid(contour)
        if centre is None:
            continue
        global_centre = centre + np.asarray([x0, y0], dtype=np.float64)
        side = "left" if global_centre[0] < 960.0 else "right"
        side_x0, side_x1 = (720.0, 970.0) if side == "left" else (970.0, 1200.0)
        if not (
            side_x0 <= global_centre[0] < side_x1
            and 1120.0 <= global_centre[1] < 1510.0
        ):
            continue
        radius = max(box_width, box_height)
        patch_y0 = max(0, y - radius)
        patch_y1 = min(hsv_roi.shape[0], y + box_height + radius)
        patch_x0 = max(0, x - radius)
        patch_x1 = min(hsv_roi.shape[1], x + box_width + radius)
        grid_y, grid_x = np.mgrid[patch_y0:patch_y1, patch_x0:patch_x1]
        distance = np.hypot(grid_x - centre[0], grid_y - centre[1])
        annulus = (distance >= 0.65 * radius) & (distance <= 1.45 * radius)
        patch = hsv_roi[patch_y0:patch_y1, patch_x0:patch_x1]
        yellow = (
            (patch[..., 0] >= 15)
            & (patch[..., 0] <= 50)
            & (patch[..., 1] >= 60)
            & (patch[..., 2] >= 60)
        )
        yellow_fraction = float(yellow[annulus].mean()) if annulus.any() else 0.0
        if yellow_fraction < 0.55:
            continue
        axes = (
            cv2.fitEllipse(contour)[1]
            if len(contour) >= 5
            else (box_width, box_height)
        )
        minor, major = sorted(float(value) for value in axes)
        circularity = minor / max(major, 1e-6)
        dark_by_side[side].append(
            (yellow_fraction, circularity, global_centre)
        )
    black_left = black_right = None
    if dark_by_side["left"] and dark_by_side["right"]:
        left = max(dark_by_side["left"], key=lambda item: (item[0], item[1]))[2]
        right = max(dark_by_side["right"], key=lambda item: (item[0], item[1]))[2]
        gap = float(np.linalg.norm(left - right))
        if (
            65.0 <= gap <= 230.0
            and left[0] < right[0]
            and abs(float(left[1] - right[1])) <= 60.0
        ):
            black_left, black_right = left, right
    return GripperMarkerCandidates(
        yellow_left=yellow_left,
        yellow_right=yellow_right,
        black_left=black_left,
        black_right=black_right,
    )


def detect_bgr_gripper_markers(image: np.ndarray) -> GripperMarkerCandidates:
    """Return candidates for both marker layouts from a 1920-square BGR frame."""
    if image.shape[:2] != (1920, 1920) or image.ndim != 3:
        raise ValueError(
            "gripper marker detector requires a 1920x1920 BGR frame, "
            f"got {image.shape}"
        )
    x0, y0, x1, y1 = MARKER_ROI_PX_1920
    hsv_roi = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    return _detect_candidates(hsv_roi)


def detect_yuv420_gripper_markers(
    luma: np.ndarray,
    chroma_u: np.ndarray,
    chroma_v: np.ndarray,
    *,
    full_range: bool = True,
) -> GripperMarkerCandidates:
    """Return both layouts while converting only the fixed gripper ROI."""
    height, width = luma.shape
    if (height, width) != (1920, 1920):
        raise ValueError(
            f"gripper marker detector requires 1920x1920 luma, got {width}x{height}"
        )
    if chroma_u.shape != (height // 2, width // 2) or chroma_v.shape != chroma_u.shape:
        raise ValueError("YUV420 chroma planes do not match the luma geometry")
    x0, y0, x1, y1 = MARKER_ROI_PX_1920
    y_roi = luma[y0:y1, x0:x1]
    u_roi = cv2.resize(
        chroma_u[y0 // 2 : y1 // 2, x0 // 2 : x1 // 2],
        (x1 - x0, y1 - y0),
        interpolation=cv2.INTER_NEAREST,
    )
    v_roi = cv2.resize(
        chroma_v[y0 // 2 : y1 // 2, x0 // 2 : x1 // 2],
        (x1 - x0, y1 - y0),
        interpolation=cv2.INTER_NEAREST,
    )
    if not full_range:
        # Preserve the original limited-range luma for AprilTag tracking, but
        # normalize only the small colour ROI before applying calibrated HSV
        # thresholds.  This supports older H.265 exports tagged yuv420p/tv
        # without adding a second full-frame conversion.
        y_roi = np.clip(
            (y_roi.astype(np.float32) - 16.0) * (255.0 / 219.0), 0.0, 255.0
        ).astype(np.uint8)
        u_roi = np.clip(
            (u_roi.astype(np.float32) - 128.0) * (255.0 / 224.0) + 128.0,
            0.0,
            255.0,
        ).astype(np.uint8)
        v_roi = np.clip(
            (v_roi.astype(np.float32) - 128.0) * (255.0 / 224.0) + 128.0,
            0.0,
            255.0,
        ).astype(np.uint8)
    yuv_roi = np.dstack((y_roi, u_roi, v_roi))
    bgr_roi = cv2.cvtColor(yuv_roi, cv2.COLOR_YUV2BGR)
    return _detect_candidates(cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV))


def detect_yuv420_gripper_triads(
    luma: np.ndarray,
    chroma_u: np.ndarray,
    chroma_v: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Backward-compatible access to the black-gripper yellow triads."""
    markers = detect_yuv420_gripper_markers(luma, chroma_u, chroma_v)
    return markers.yellow_left, markers.yellow_right


def included_angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    left_vector = left[0] - left[2]
    right_vector = right[0] - right[2]
    cosine = float(
        left_vector @ right_vector
        / (np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
    )
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
