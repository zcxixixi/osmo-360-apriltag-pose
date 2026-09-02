"""Fast fixed-ROI gripper marker detection on full-resolution YUV420 frames."""

from __future__ import annotations

from itertools import combinations

import cv2
import numpy as np


MARKER_ALGORITHM_REVISION = "x5-yuv420-fixed-roi-yellow-triads-v1"
MARKER_ROI_PX_1920 = (450, 950, 1470, 1800)
HSV_THRESHOLD = {
    "h_min": 18,
    "h_max": 48,
    "s_min": 75,
    "v_min": 65,
}


def marker_signature() -> dict[str, object]:
    return {
        "algorithm_revision": MARKER_ALGORITHM_REVISION,
        "source_pixel_format": "yuvj420p",
        "source_size": [1920, 1920],
        "roi_xyxy_px": list(MARKER_ROI_PX_1920),
        "threshold": {"colour_space": "opencv_hsv", **HSV_THRESHOLD},
        "selection": "fixed ROI, black-annulus side-specific collinear triad",
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


def detect_yuv420_gripper_triads(
    luma: np.ndarray,
    chroma_u: np.ndarray,
    chroma_v: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return left/right three-dot jaw axes without constructing a full BGR frame."""
    height, width = luma.shape
    if (height, width) != (1920, 1920):
        raise ValueError(f"gripper marker detector requires 1920x1920 luma, got {width}x{height}")
    if chroma_u.shape != (height // 2, width // 2) or chroma_v.shape != chroma_u.shape:
        raise ValueError("YUV420 chroma planes do not match the luma geometry")
    x0, y0, x1, y1 = MARKER_ROI_PX_1920
    # The trajectory consumer receives the exact full-resolution Y plane.  For
    # gripper colour we upsample only the fixed ROI chroma planes; no full-frame
    # BGR/HSV image is ever constructed.
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
    # Convert only the quarter-resolution ROI.  This reproduces the proven
    # HSV yellow gate without paying for a 1920x1920 BGR frame.
    yuv_roi = np.dstack((y_roi, u_roi, v_roi))
    bgr_roi = cv2.cvtColor(yuv_roi, cv2.COLOR_YUV2BGR)
    hsv_roi = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    threshold = HSV_THRESHOLD
    mask = cv2.inRange(
        hsv_roi,
        np.asarray(
            [threshold["h_min"], threshold["s_min"], threshold["v_min"]],
            dtype=np.uint8,
        ),
        np.asarray([threshold["h_max"], 255, 255], dtype=np.uint8),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
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
        patch_y1 = min(y_roi.shape[0], y + box_height + radius)
        patch_x0 = max(0, x - radius)
        patch_x1 = min(y_roi.shape[1], x + box_width + radius)
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
    return _select_triad(by_side["left"], "left"), _select_triad(
        by_side["right"], "right"
    )


def included_angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    left_vector = left[0] - left[2]
    right_vector = right[0] - right[2]
    cosine = float(
        left_vector @ right_vector
        / (np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
    )
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
