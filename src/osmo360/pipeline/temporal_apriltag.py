"""Grayscale temporal AprilTag tracking primitives for raw fisheye video."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FlowAudit:
    attempted_tags: int
    accepted_tags: int
    rejected_status: int
    rejected_forward_backward: int
    rejected_geometry: int


def detect_gray(
    gray: np.ndarray,
    detector: cv2.aruco.ArucoDetector,
) -> dict[int, list[np.ndarray]]:
    """Decode grayscale AprilTags while preserving duplicate-ID candidates."""
    if gray.ndim != 2:
        raise ValueError("AprilTag detection input must be grayscale")
    quads, ids, _ = detector.detectMarkers(gray)
    result: dict[int, list[np.ndarray]] = {}
    if ids is None:
        return result
    for quad, tag_id in zip(quads, ids.flatten()):
        result.setdefault(int(tag_id), []).append(
            np.asarray(quad, dtype=np.float32).reshape(4, 2)
        )
    return result


def _quad_area(quad: np.ndarray) -> float:
    return abs(float(cv2.contourArea(np.asarray(quad, dtype=np.float32))))


def choose_candidate(
    candidates: list[np.ndarray],
    predicted: np.ndarray | None = None,
) -> np.ndarray | None:
    if not candidates:
        return None
    if predicted is None:
        return max(candidates, key=_quad_area)
    center = np.asarray(predicted, dtype=np.float32).mean(axis=0)
    return min(candidates, key=lambda quad: float(np.linalg.norm(quad.mean(axis=0) - center)))


def quad_geometry_valid(
    previous: np.ndarray,
    current: np.ndarray,
    image_shape: tuple[int, int],
    *,
    minimum_area_px2: float = 100.0,
) -> bool:
    current = np.asarray(current, dtype=np.float32).reshape(4, 2)
    previous = np.asarray(previous, dtype=np.float32).reshape(4, 2)
    if not np.isfinite(current).all() or not cv2.isContourConvex(current):
        return False
    height, width = image_shape
    if (
        np.any(current[:, 0] < 0)
        or np.any(current[:, 0] >= width)
        or np.any(current[:, 1] < 0)
        or np.any(current[:, 1] >= height)
    ):
        return False
    previous_area = _quad_area(previous)
    current_area = _quad_area(current)
    if previous_area < minimum_area_px2 or current_area < minimum_area_px2:
        return False
    if not 0.55 <= current_area / previous_area <= 1.8:
        return False
    edges = np.linalg.norm(current - np.roll(current, 1, axis=0), axis=1)
    if float(edges.min()) < 3.0 or float(edges.max() / edges.min()) > 12.0:
        return False
    return True


def track_quads_forward_backward(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    quads: dict[int, np.ndarray],
    *,
    max_forward_backward_error_px: float = 1.5,
    minimum_area_px2: float = 100.0,
    verify_forward_backward: bool = True,
    window_size: int = 31,
    max_level: int = 4,
    max_iterations: int = 30,
) -> tuple[dict[int, np.ndarray], FlowAudit]:
    """Track all known corners in one vectorized pyramidal-LK call."""
    if previous_gray.ndim != 2 or current_gray.ndim != 2:
        raise ValueError("LK tracking input must be grayscale")
    if not quads:
        return {}, FlowAudit(0, 0, 0, 0, 0)
    tag_ids = list(quads)
    points = np.concatenate([quads[tag_id] for tag_id in tag_ids]).astype(
        np.float32
    ).reshape(-1, 1, 2)
    options = {
        "winSize": (window_size, window_size),
        "maxLevel": max_level,
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            max_iterations,
            0.01,
        ),
    }
    forward, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray, current_gray, points, None, **options
    )
    if forward is None or forward_status is None:
        return {}, FlowAudit(len(tag_ids), 0, len(tag_ids), 0, 0)
    forward_points = forward.reshape(-1, 2)
    status = forward_status.ravel().astype(bool)
    finite = np.isfinite(forward_points).all(axis=1)
    if verify_forward_backward:
        backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            current_gray, previous_gray, forward, None, **options
        )
        if backward is None or backward_status is None:
            return {}, FlowAudit(len(tag_ids), 0, len(tag_ids), 0, 0)
        status &= backward_status.ravel().astype(bool)
        error = np.linalg.norm(
            backward.reshape(-1, 2) - points.reshape(-1, 2), axis=1
        )
    else:
        error = np.zeros(len(points), dtype=np.float32)
    tracked: dict[int, np.ndarray] = {}
    rejected_status = 0
    rejected_fb = 0
    rejected_geometry = 0
    for index, tag_id in enumerate(tag_ids):
        selected = slice(index * 4, index * 4 + 4)
        if not np.all(status[selected] & finite[selected]):
            rejected_status += 1
            continue
        if float(np.max(error[selected])) > max_forward_backward_error_px:
            rejected_fb += 1
            continue
        quad = forward_points[selected].astype(np.float32)
        if not quad_geometry_valid(
            quads[tag_id],
            quad,
            current_gray.shape[:2],
            minimum_area_px2=minimum_area_px2,
        ):
            rejected_geometry += 1
            continue
        tracked[tag_id] = quad
    return tracked, FlowAudit(
        attempted_tags=len(tag_ids),
        accepted_tags=len(tracked),
        rejected_status=rejected_status,
        rejected_forward_backward=rejected_fb,
        rejected_geometry=rejected_geometry,
    )


def redetect_rois(
    gray: np.ndarray,
    detector: cv2.aruco.ArucoDetector,
    predicted: dict[int, np.ndarray],
    *,
    margin_ratio: float = 0.75,
    minimum_margin_px: int = 32,
) -> dict[int, np.ndarray]:
    """Decode known IDs in merged LK-predicted grayscale ROIs."""
    height, width = gray.shape[:2]
    result: dict[int, np.ndarray] = {}
    rois: list[tuple[tuple[int, int, int, int], set[int]]] = []
    for tag_id, quad in predicted.items():
        quad = np.asarray(quad, dtype=np.float32).reshape(4, 2)
        span = np.ptp(quad, axis=0)
        margin = max(minimum_margin_px, int(math_ceil(float(span.max()) * margin_ratio)))
        x0 = max(0, int(np.floor(quad[:, 0].min())) - margin)
        y0 = max(0, int(np.floor(quad[:, 1].min())) - margin)
        x1 = min(width, int(np.ceil(quad[:, 0].max())) + margin + 1)
        y1 = min(height, int(np.ceil(quad[:, 1].max())) + margin + 1)
        if x1 - x0 < 16 or y1 - y0 < 16:
            continue
        box = (x0, y0, x1, y1)
        overlapping = [
            index
            for index, (other, _) in enumerate(rois)
            if box[0] < other[2]
            and other[0] < box[2]
            and box[1] < other[3]
            and other[1] < box[3]
        ]
        if not overlapping:
            rois.append((box, {tag_id}))
            continue
        merged_boxes = [box] + [rois[index][0] for index in overlapping]
        merged_ids = {tag_id}
        for index in overlapping:
            merged_ids.update(rois[index][1])
        merged = (
            min(value[0] for value in merged_boxes),
            min(value[1] for value in merged_boxes),
            max(value[2] for value in merged_boxes),
            max(value[3] for value in merged_boxes),
        )
        for index in reversed(overlapping):
            rois.pop(index)
        rois.append((merged, merged_ids))

    # A newly merged ROI can bridge two earlier groups, so close overlaps once
    # more.  Tag counts are small and this avoids a broad full-frame union.
    changed = True
    while changed:
        changed = False
        for first in range(len(rois)):
            a, ids_a = rois[first]
            match = next((
                second
                for second in range(first + 1, len(rois))
                if a[0] < rois[second][0][2]
                and rois[second][0][0] < a[2]
                and a[1] < rois[second][0][3]
                and rois[second][0][1] < a[3]
            ), None)
            if match is None:
                continue
            b, ids_b = rois.pop(match)
            rois[first] = (
                (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])),
                ids_a | ids_b,
            )
            changed = True
            break

    for (x0, y0, x1, y1), expected_ids in rois:
        decoded = detect_gray(gray[y0:y1, x0:x1], detector)
        offset = np.asarray([x0, y0], dtype=np.float32)
        for tag_id in expected_ids:
            candidates = [candidate + offset for candidate in decoded.get(tag_id, [])]
            candidate = choose_candidate(candidates, predicted[tag_id])
            if candidate is not None and quad_geometry_valid(
                predicted[tag_id], candidate, gray.shape[:2], minimum_area_px2=64.0
            ):
                result[tag_id] = candidate
    return result


def math_ceil(value: float) -> int:
    # Kept local to avoid importing the broad math namespace in tight call sites.
    return int(np.ceil(value))


def grayscale_scout_and_refine(
    gray: np.ndarray,
    detector: cv2.aruco.ArucoDetector,
    *,
    scale: float = 0.35,
    predicted: dict[int, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """Locate tags cheaply on a small grayscale frame, then decode full-res ROIs."""
    if not 0.1 <= scale <= 1.0:
        raise ValueError("global grayscale scout scale must be between 0.1 and 1.0")
    if scale == 1.0:
        coarse = detect_gray(gray, detector)
    else:
        small = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
        coarse = {
            tag_id: [quad / scale for quad in quads]
            for tag_id, quads in detect_gray(small, detector).items()
        }
    seeds = {
        tag_id: choose_candidate(quads, None if predicted is None else predicted.get(tag_id))
        for tag_id, quads in coarse.items()
    }
    seeds = {tag_id: quad for tag_id, quad in seeds.items() if quad is not None}
    refined = redetect_rois(gray, detector, seeds, margin_ratio=0.5)
    return refined
