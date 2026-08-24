#!/usr/bin/env python3
"""Calibrate a fixed transform between two AprilTag wall panels.

This utility is intentionally separate from the frame-by-frame visual solver.
It consumes two pose CSV files produced from the *same camera frames*, where
each CSV used a different planar panel as its reference.  Every overlapping
direct pose gives one estimate of the fixed panel-to-panel SE(3) transform.

The estimate is rigid: scale is fixed to exactly one.  A RANSAC-like medoid
initialisation followed by robust SO(3) averaging and translation refinement
rejects bad PnP branches.  Block bootstrap intervals account (approximately)
for the strong temporal correlation between adjacent video frames.

The a9abc654 capture has duplicate tag IDs on the two walls.  Such a capture
can only be salvaged when the two input pose streams were produced from
spatially isolated views.  The emitted configuration records that limitation
and is always marked PROVISIONAL.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation


POSITION_KEYS = ("camera_x_m", "camera_y_m", "camera_z_m")
EULER_KEYS = ("roll_deg", "pitch_deg", "yaw_deg")


@dataclass(frozen=True)
class PoseSample:
    frame: int
    timestamp_s: float
    position_m: np.ndarray
    orientation: Rotation
    rmse_px: float
    detected_ids: tuple[int, ...]
    measurement_source: str


@dataclass(frozen=True)
class TagMap:
    path: Path
    payload: dict
    corners_by_id: dict[int, np.ndarray]
    tag_outer_size_m: float
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_vector(row: dict[str, str], keys: Iterable[str]) -> np.ndarray | None:
    try:
        value = np.asarray([float(row[key]) for key in keys], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None
    return value if np.isfinite(value).all() else None


def read_visual_poses(path: Path, minimum_tags: int = 2) -> dict[int, PoseSample]:
    """Read valid corner-measured poses keyed by source video frame.

    ``optical_flow`` in this pipeline means the AprilTag corners themselves
    were tracked and fed back through PnP.  It is therefore a visual
    measurement, unlike an IMU prediction or a temporal interpolation.  The
    output audit still separates direct/direct pairs from flow-assisted pairs.
    """
    poses: dict[int, PoseSample] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"frame", "timestamp", "quality_status", *POSITION_KEYS, *EULER_KEYS}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"pose CSV {path} is missing columns: {sorted(missing)}")
        for row in reader:
            if row.get("quality_status") != "valid":
                continue
            source = row.get("measurement_source", "")
            # Predictions and interpolation are useful for display, but they
            # must not constrain a room calibration.  Tracked AprilTag corners
            # remain acceptable geometric observations.
            if source and not (
                source.startswith("direct") or source == "optical_flow"
            ):
                continue
            position = _finite_vector(row, POSITION_KEYS)
            euler = _finite_vector(row, EULER_KEYS)
            if position is None or euler is None:
                continue
            try:
                count = int(row.get("detected_tag_count", minimum_tags) or 0)
            except ValueError:
                count = 0
            if count < minimum_tags:
                continue
            ids = tuple(int(item) for item in row.get("detected_ids", "").split() if item)
            try:
                rmse = float(row.get("reprojection_rmse_px", "nan"))
            except ValueError:
                rmse = float("nan")
            if not np.isfinite(rmse) or rmse <= 0:
                rmse = 3.0
            frame = int(row["frame"])
            poses[frame] = PoseSample(
                frame=frame,
                timestamp_s=float(row["timestamp"]),
                position_m=position,
                orientation=Rotation.from_euler("xyz", euler, degrees=True),
                rmse_px=rmse,
                detected_ids=ids,
                measurement_source=source,
            )
    if not poses:
        raise ValueError(f"pose CSV has no valid corner-measured poses: {path}")
    return poses


def load_tag_map(path: Path, expected_tag_size_m: float = 0.2) -> TagMap:
    payload = json.loads(path.read_text(encoding="utf-8"))
    units = payload.get("units")
    if units != "m":
        raise ValueError(f"tag map must use metres, got {units!r}: {path}")
    size = float(payload.get("tag_outer_size_m", "nan"))
    if not np.isfinite(size) or abs(size - expected_tag_size_m) > 1e-6:
        raise ValueError(
            f"tag map {path} has tag_outer_size_m={size}; expected {expected_tag_size_m}"
        )
    corners: dict[int, np.ndarray] = {}
    for tag in payload.get("tags", []):
        tag_id = int(tag["id"])
        values = np.asarray(tag["corners_m"], dtype=np.float64)
        if values.shape != (4, 3) or not np.isfinite(values).all():
            raise ValueError(f"tag {tag_id} in {path} does not contain four finite 3D corners")
        if tag_id in corners:
            raise ValueError(f"duplicate tag ID {tag_id} inside map {path}")
        edges = np.linalg.norm(np.roll(values, -1, axis=0) - values, axis=1)
        if not np.allclose(edges, expected_tag_size_m, atol=1e-6):
            raise ValueError(f"tag {tag_id} corners are not a {expected_tag_size_m} m square")
        corners[tag_id] = values
    if not corners:
        raise ValueError(f"tag map contains no tags: {path}")
    return TagMap(path.resolve(), payload, corners, size, _sha256(path))


def rigid_alignment(source: np.ndarray, target: np.ndarray) -> tuple[Rotation, np.ndarray, float, float]:
    """Return the no-scale transform mapping ``source`` points to ``target``."""
    source = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    if source.shape != target.shape or len(source) < 3:
        raise ValueError("rigid alignment needs at least three corresponding 3D points")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    u, singular, vt = np.linalg.svd(source_zero.T @ target_zero)
    matrix = vt.T @ u.T
    if np.linalg.det(matrix) < 0:
        vt[-1] *= -1
        matrix = vt.T @ u.T
    rotation = Rotation.from_matrix(matrix)
    translation = target_center - rotation.apply(source_center)
    residual = target - (rotation.apply(source) + translation)
    rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    denominator = float(np.sum(source_zero * source_zero))
    diagnostic_scale = float(np.sum(singular) / denominator) if denominator > 0 else float("nan")
    return rotation, translation, rms, diagnostic_scale


def align_map_frames(source: TagMap, target: TagMap) -> tuple[Rotation, np.ndarray, dict]:
    """Estimate ``T_target_source`` from common tag corners in two map files."""
    common = sorted(set(source.corners_by_id) & set(target.corners_by_id))
    if not common:
        raise ValueError(f"maps {source.path} and {target.path} have no common tag IDs")
    source_points = np.concatenate([source.corners_by_id[tag_id] for tag_id in common])
    target_points = np.concatenate([target.corners_by_id[tag_id] for tag_id in common])
    rotation, translation, rms, diagnostic_scale = rigid_alignment(source_points, target_points)
    if rms > 1e-6 or abs(diagnostic_scale - 1.0) > 1e-6:
        raise ValueError(
            "observation and output panel maps are not the same rigid metric geometry: "
            f"rms={rms:.9g} m, diagnostic_scale={diagnostic_scale:.9g}"
        )
    return rotation, translation, {
        "common_ids": common,
        "corner_count": int(len(source_points)),
        "rms_m": rms,
        "diagnostic_unconstrained_scale": diagnostic_scale,
        "enforced_scale": 1.0,
    }


def _weighted_rotation_mean(rotations: list[Rotation], weights: np.ndarray) -> Rotation:
    values = Rotation.concatenate(rotations)
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.clip(weights, 1e-12, None)
    return values.mean(weights=weights)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    result = []
    for column in values.T:
        order = np.argsort(column)
        sorted_values = column[order]
        cumulative = np.cumsum(weights[order])
        result.append(sorted_values[np.searchsorted(cumulative, cumulative[-1] / 2.0)])
    return np.asarray(result)


def _quality_weights(primary: list[PoseSample], secondary: list[PoseSample]) -> np.ndarray:
    combined = np.hypot(
        np.asarray([sample.rmse_px for sample in primary]),
        np.asarray([sample.rmse_px for sample in secondary]),
    )
    weights = 1.0 / np.square(np.clip(combined, 0.5, 10.0))
    # Prevent one suspiciously low reported RMSE from dominating the estimate.
    return np.clip(weights, np.percentile(weights, 10), np.percentile(weights, 90))


def _candidate_transforms(
    primary: list[PoseSample], secondary: list[PoseSample]
) -> tuple[list[Rotation], np.ndarray]:
    rotations: list[Rotation] = []
    translations: list[np.ndarray] = []
    for first, second in zip(primary, secondary):
        rotation = first.orientation * second.orientation.inv()
        rotations.append(rotation)
        translations.append(first.position_m - rotation.apply(second.position_m))
    return rotations, np.asarray(translations)


def _residuals(
    rotation: Rotation,
    translation: np.ndarray,
    primary: list[PoseSample],
    secondary: list[PoseSample],
) -> tuple[np.ndarray, np.ndarray]:
    positional, angular = [], []
    for first, second in zip(primary, secondary):
        predicted_position = rotation.apply(second.position_m) + translation
        predicted_orientation = rotation * second.orientation
        positional.append(np.linalg.norm(first.position_m - predicted_position))
        angular.append(np.degrees((first.orientation.inv() * predicted_orientation).magnitude()))
    return np.asarray(positional), np.asarray(angular)


def robust_panel_transform(
    primary: list[PoseSample],
    secondary: list[PoseSample],
    *,
    max_position_residual_m: float = 0.05,
    max_orientation_residual_deg: float = 3.0,
    minimum_inliers: int = 20,
) -> tuple[Rotation, np.ndarray, np.ndarray, dict]:
    """Estimate ``T_primary_secondary`` without a similarity scale."""
    if len(primary) != len(secondary):
        raise ValueError("paired pose sequences have different lengths")
    if len(primary) < minimum_inliers:
        raise ValueError(f"need at least {minimum_inliers} paired poses, found {len(primary)}")
    candidates_r, candidates_t = _candidate_transforms(primary, secondary)
    quality = _quality_weights(primary, secondary)

    # Candidate medoid / deterministic RANSAC.  Each candidate is scored by
    # how many other candidates agree with both its rotation and translation.
    best_score = -1.0
    best_keep: np.ndarray | None = None
    for center_r, center_t in zip(candidates_r, candidates_t):
        angle = np.degrees((center_r.inv() * Rotation.concatenate(candidates_r)).magnitude())
        distance = np.linalg.norm(candidates_t - center_t, axis=1)
        keep = (angle <= max_orientation_residual_deg) & (
            distance <= max_position_residual_m
        )
        score = float(quality[keep].sum())
        if score > best_score:
            best_score = score
            best_keep = keep
    assert best_keep is not None
    if int(best_keep.sum()) < minimum_inliers:
        raise ValueError(
            "no rigid panel transform has enough support: "
            f"{int(best_keep.sum())} < {minimum_inliers}"
        )

    keep = best_keep
    rotation = _weighted_rotation_mean(
        [value for value, accepted in zip(candidates_r, keep) if accepted], quality[keep]
    )
    translation = _weighted_median(
        np.asarray([
            first.position_m - rotation.apply(second.position_m)
            for first, second, accepted in zip(primary, secondary, keep)
            if accepted
        ]),
        quality[keep],
    )

    # Recompute inliers against the model and refine with Huber weights.
    for _ in range(8):
        position_residual, orientation_residual = _residuals(
            rotation, translation, primary, secondary
        )
        keep = (position_residual <= max_position_residual_m) & (
            orientation_residual <= max_orientation_residual_deg
        )
        if int(keep.sum()) < minimum_inliers:
            raise ValueError(
                "robust refinement rejected too many paired poses: "
                f"{int(keep.sum())} < {minimum_inliers}"
            )
        robust = quality.copy()
        robust *= np.minimum(1.0, max_position_residual_m / np.maximum(position_residual, 1e-9))
        robust *= np.minimum(
            1.0, max_orientation_residual_deg / np.maximum(orientation_residual, 1e-9)
        )
        accepted_weights = robust[keep]
        updated_rotation = _weighted_rotation_mean(
            [value for value, accepted in zip(candidates_r, keep) if accepted],
            accepted_weights,
        )
        translations = np.asarray([
            first.position_m - updated_rotation.apply(second.position_m)
            for first, second, accepted in zip(primary, secondary, keep)
            if accepted
        ])
        updated_translation = _weighted_median(translations, accepted_weights)
        delta_angle = np.degrees((rotation.inv() * updated_rotation).magnitude())
        delta_position = np.linalg.norm(updated_translation - translation)
        rotation, translation = updated_rotation, updated_translation
        if delta_angle < 1e-8 and delta_position < 1e-10:
            break

    position_residual, orientation_residual = _residuals(rotation, translation, primary, secondary)
    keep = (position_residual <= max_position_residual_m) & (
        orientation_residual <= max_orientation_residual_deg
    )
    audit = {
        "paired_frames": len(primary),
        "inlier_frames": int(keep.sum()),
        "inlier_ratio": float(keep.mean()),
        "thresholds": {
            "position_m": max_position_residual_m,
            "orientation_deg": max_orientation_residual_deg,
        },
        "position_residual_m": _distribution(position_residual[keep]),
        "orientation_residual_deg": _distribution(orientation_residual[keep]),
        "all_position_residual_m": _distribution(position_residual),
        "all_orientation_residual_deg": _distribution(orientation_residual),
    }
    return rotation, translation, keep, audit


def _distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _compose(
    parent_middle_r: Rotation,
    parent_middle_t: np.ndarray,
    middle_child_r: Rotation,
    middle_child_t: np.ndarray,
) -> tuple[Rotation, np.ndarray]:
    return (
        parent_middle_r * middle_child_r,
        parent_middle_t + parent_middle_r.apply(middle_child_t),
    )


def _inverse(rotation: Rotation, translation: np.ndarray) -> tuple[Rotation, np.ndarray]:
    inverse_rotation = rotation.inv()
    return inverse_rotation, -inverse_rotation.apply(translation)


def bootstrap_uncertainty(
    primary: list[PoseSample],
    secondary: list[PoseSample],
    keep: np.ndarray,
    reference_rotation: Rotation,
    reference_translation: np.ndarray,
    *,
    block_length: int = 12,
    samples: int = 500,
    random_seed: int = 20260824,
) -> dict:
    indices = np.flatnonzero(keep)
    if len(indices) < 2:
        raise ValueError("bootstrap needs at least two inlier poses")
    block_length = max(1, min(block_length, len(indices)))
    blocks = [indices[start : start + block_length] for start in range(0, len(indices), block_length)]
    rng = np.random.default_rng(random_seed)
    translation_samples, rotation_vector_samples, wall_angles = [], [], []
    for _ in range(samples):
        chosen: list[int] = []
        while len(chosen) < len(indices):
            chosen.extend(blocks[int(rng.integers(0, len(blocks)))].tolist())
        selected = np.asarray(chosen[: len(indices)], dtype=int)
        first = [primary[index] for index in selected]
        second = [secondary[index] for index in selected]
        candidate_r, _ = _candidate_transforms(first, second)
        weights = _quality_weights(first, second)
        rotation = _weighted_rotation_mean(candidate_r, weights)
        translations = np.asarray([
            a.position_m - rotation.apply(b.position_m) for a, b in zip(first, second)
        ])
        translation = _weighted_median(translations, weights)
        relative = reference_rotation.inv() * rotation
        rotation_vector_samples.append(np.degrees(relative.as_rotvec()))
        translation_samples.append(translation)
        normal = rotation.apply([0.0, 0.0, 1.0])
        wall_angles.append(np.degrees(np.arccos(np.clip(abs(normal[2]), -1.0, 1.0))))
    translations = np.asarray(translation_samples)
    rotation_vectors = np.asarray(rotation_vector_samples)
    wall_angles = np.asarray(wall_angles)
    return {
        "method": "contiguous-block-bootstrap",
        "bootstrap_samples": samples,
        "block_length_frames": block_length,
        "effective_input_blocks": len(blocks),
        "translation_m_ci95": {
            "lower": np.percentile(translations, 2.5, axis=0).tolist(),
            "upper": np.percentile(translations, 97.5, axis=0).tolist(),
        },
        "translation_norm_error_m_p95": float(
            np.percentile(np.linalg.norm(translations - reference_translation, axis=1), 95)
        ),
        "rotation_tangent_deg_ci95": {
            "lower": np.percentile(rotation_vectors, 2.5, axis=0).tolist(),
            "upper": np.percentile(rotation_vectors, 97.5, axis=0).tolist(),
        },
        "rotation_angle_error_deg_p95": float(
            np.percentile(np.linalg.norm(rotation_vectors, axis=1), 95)
        ),
        "wall_plane_angle_deg_ci95": [
            float(np.percentile(wall_angles, 2.5)),
            float(np.percentile(wall_angles, 97.5)),
        ],
    }


def _transform_json(parent: str, child: str, rotation: Rotation, translation: np.ndarray) -> dict:
    return {
        "parent_frame": parent,
        "child_frame": child,
        "translation_m": np.asarray(translation).tolist(),
        "quaternion_xyzw": rotation.as_quat().tolist(),
        "scale": 1.0,
    }


def calibrate(
    primary_pose_path: Path,
    secondary_pose_path: Path,
    primary_map_path: Path,
    secondary_observation_map_path: Path,
    secondary_output_map_path: Path,
    output_path: Path,
    *,
    capture_pair_path: Path | None = None,
    primary_frame: str = "left_wall_duplicate_panel_map",
    secondary_frame: str = "right_wall_six_tag_panel_map",
    minimum_inliers: int = 20,
    bootstrap_samples: int = 500,
) -> dict:
    primary_map = load_tag_map(primary_map_path)
    secondary_observation_map = load_tag_map(secondary_observation_map_path)
    secondary_output_map = load_tag_map(secondary_output_map_path)
    primary_rows = read_visual_poses(primary_pose_path)
    secondary_rows = read_visual_poses(secondary_pose_path)
    frames = sorted(set(primary_rows) & set(secondary_rows))
    if len(frames) < minimum_inliers:
        raise ValueError(
            f"need {minimum_inliers} overlapping visual frames, found {len(frames)}"
        )
    primary = [primary_rows[frame] for frame in frames]
    secondary = [secondary_rows[frame] for frame in frames]

    observation_r, observation_t, keep, robust_audit = robust_panel_transform(
        primary, secondary, minimum_inliers=minimum_inliers
    )
    # Map alignment is T_output_observation.  We need T_observation_output
    # before composing T_primary_observation * T_observation_output.
    output_observation_r, output_observation_t, map_alignment = align_map_frames(
        secondary_observation_map, secondary_output_map
    )
    observation_output_r, observation_output_t = _inverse(
        output_observation_r, output_observation_t
    )
    panel_r, panel_t = _compose(
        observation_r, observation_t, observation_output_r, observation_output_t
    )
    inverse_r, inverse_t = _inverse(panel_r, panel_t)

    normal = panel_r.apply([0.0, 0.0, 1.0])
    wall_angle = float(np.degrees(np.arccos(np.clip(abs(normal[2]), -1.0, 1.0))))
    uncertainty = bootstrap_uncertainty(
        primary,
        secondary,
        keep,
        observation_r,
        observation_t,
        samples=bootstrap_samples,
    )
    capture_pair_id = None
    capture_sha = None
    if capture_pair_path is not None:
        capture = json.loads(capture_pair_path.read_text(encoding="utf-8"))
        capture_pair_id = capture.get("capture_pair_id")
        capture_sha = _sha256(capture_pair_path)

    inlier_frames = [frame for frame, accepted in zip(frames, keep) if accepted]
    rejected_frames = [frame for frame, accepted in zip(frames, keep) if not accepted]
    direct_direct_frames = [
        frame for frame in frames
        if primary_rows[frame].measurement_source.startswith("direct")
        and secondary_rows[frame].measurement_source.startswith("direct")
    ]
    duplicate_ids = sorted(
        set(primary_map.corners_by_id) & set(secondary_output_map.corners_by_id)
    )
    report = {
        "schema_version": "capture-panel-pair-calibration/1.0",
        "calibration_status": "PROVISIONAL_DUPLICATE_ID_SALVAGE",
        "capture_pair_id": capture_pair_id,
        "method": {
            "name": "paired-direct-camera-pose-robust-se3",
            "scale_policy": "FIXED_TO_ONE_NO_SIM3",
            "enforced_scale": 1.0,
            "rotation": "RANSAC-medoid plus weighted SO(3) mean",
            "translation": "weighted component median",
            "uncertainty": uncertainty,
        },
        "inputs": {
            "primary_pose_csv": str(primary_pose_path.resolve()),
            "secondary_pose_csv": str(secondary_pose_path.resolve()),
            "primary_tag_map": {
                "path": str(primary_map.path),
                "sha256": primary_map.sha256,
                "tag_outer_size_m": primary_map.tag_outer_size_m,
            },
            "secondary_observation_tag_map": {
                "path": str(secondary_observation_map.path),
                "sha256": secondary_observation_map.sha256,
                "tag_outer_size_m": secondary_observation_map.tag_outer_size_m,
            },
            "secondary_output_tag_map": {
                "path": str(secondary_output_map.path),
                "sha256": secondary_output_map.sha256,
                "tag_outer_size_m": secondary_output_map.tag_outer_size_m,
            },
            "capture_pair_json": None if capture_pair_path is None else str(capture_pair_path.resolve()),
            "capture_pair_sha256": capture_sha,
        },
        "frames": {
            "overlapping_corner_measured": len(frames),
            "overlapping_direct_direct": len(direct_direct_frames),
            "overlapping_flow_assisted": len(frames) - len(direct_direct_frames),
            "inlier": len(inlier_frames),
            "inlier_ratio": len(inlier_frames) / len(frames),
            "first": frames[0],
            "last": frames[-1],
            "inlier_frame_ids": inlier_frames,
            "rejected_frame_ids": rejected_frames,
        },
        "T_primary_secondary_observation": _transform_json(
            primary_frame,
            "right_wall_top_row_observation_map",
            observation_r,
            observation_t,
        ),
        "secondary_map_origin_alignment": {
            "T_output_observation": _transform_json(
                secondary_frame,
                "right_wall_top_row_observation_map",
                output_observation_r,
                output_observation_t,
            ),
            **map_alignment,
        },
        "T_primary_secondary": _transform_json(
            primary_frame, secondary_frame, panel_r, panel_t
        ),
        "T_secondary_primary": _transform_json(
            secondary_frame, primary_frame, inverse_r, inverse_t
        ),
        "geometry_audit": {
            "duplicate_tag_ids_across_panels": duplicate_ids,
            "estimated_wall_plane_angle_deg": wall_angle,
            "expected_wall_plane_angle_deg": 90.0,
            "wall_plane_angle_error_deg": abs(90.0 - wall_angle),
            "determinant": float(np.linalg.det(panel_r.as_matrix())),
            "orthonormal_error_frobenius": float(
                np.linalg.norm(panel_r.as_matrix().T @ panel_r.as_matrix() - np.eye(3))
            ),
            "similarity_scale_estimated": False,
            "enforced_scale": 1.0,
        },
        "residual_audit": robust_audit,
        "selection_contract": {
            "required": True,
            "reason": "the two physical panels reuse tag IDs",
            "rule": "poses must first be spatially isolated into the named primary and secondary panel views; tag ID alone cannot select a wall",
        },
        "training_ready": False,
        "warnings": [
            "PROVISIONAL: derived from one capture and approximate panorama projections",
            "DUPLICATE IDS: this transform cannot make a single global ID-only AprilTag map",
            "Use only for capture-specific salvage and diagnostics until a unique-ID cross-wall calibration is recorded",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-pose", type=Path, required=True)
    parser.add_argument("--secondary-pose", type=Path, required=True)
    parser.add_argument("--primary-map", type=Path, required=True)
    parser.add_argument("--secondary-observation-map", type=Path, required=True)
    parser.add_argument("--secondary-output-map", type=Path, required=True)
    parser.add_argument("--capture-pair", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--primary-frame", default="left_wall_duplicate_panel_map")
    parser.add_argument("--secondary-frame", default="right_wall_six_tag_panel_map")
    parser.add_argument("--minimum-inliers", type=int, default=20)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    args = parser.parse_args()
    report = calibrate(
        args.primary_pose,
        args.secondary_pose,
        args.primary_map,
        args.secondary_observation_map,
        args.secondary_output_map,
        args.output,
        capture_pair_path=args.capture_pair,
        primary_frame=args.primary_frame,
        secondary_frame=args.secondary_frame,
        minimum_inliers=args.minimum_inliers,
        bootstrap_samples=args.bootstrap_samples,
    )
    summary = {
        "output": str(args.output.resolve()),
        "status": report["calibration_status"],
        "paired_frames": report["frames"]["overlapping_corner_measured"],
        "inlier_frames": report["frames"]["inlier"],
        "wall_plane_angle_deg": report["geometry_audit"]["estimated_wall_plane_angle_deg"],
        "position_residual_p95_m": report["residual_audit"]["position_residual_m"]["p95"],
        "orientation_residual_p95_deg": report["residual_audit"]["orientation_residual_deg"]["p95"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
