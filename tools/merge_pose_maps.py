#!/usr/bin/env python3
"""Merge camera poses measured against two fixed AprilTag wall maps.

The rigid transform between maps is estimated only from frames where both maps
produce a valid camera pose. Primary-map measurements are preferred; a
secondary-map measurement is used only when the primary map has no valid pose.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


POSITION_KEYS = ("camera_x_m", "camera_y_m", "camera_z_m")
RAW_POSITION_KEYS = ("raw_camera_x_m", "raw_camera_y_m", "raw_camera_z_m")
EULER_KEYS = ("roll_deg", "pitch_deg", "yaw_deg")


def vector(row: dict[str, str], keys: tuple[str, ...]) -> np.ndarray | None:
    try:
        value = np.asarray([float(row[key]) for key in keys], dtype=float)
    except (KeyError, TypeError, ValueError):
        return None
    return value if np.isfinite(value).all() else None


def valid_pose(row: dict[str, str]) -> bool:
    return (
        row.get("quality_status") == "valid"
        and vector(row, POSITION_KEYS) is not None
        and vector(row, EULER_KEYS) is not None
    )


def read_rows(path: Path) -> tuple[list[str], dict[int, dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = {int(row["frame"]): row for row in reader}
    if not rows:
        raise ValueError(f"pose CSV is empty: {path}")
    return fields, rows


def pose(row: dict[str, str]) -> tuple[np.ndarray, Rotation]:
    position = vector(row, POSITION_KEYS)
    euler = vector(row, EULER_KEYS)
    if position is None or euler is None:
        raise ValueError("row has no finite pose")
    return position, Rotation.from_euler("xyz", euler, degrees=True)


def mad_limit(values: np.ndarray, floor: float) -> float:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return max(floor, median + 3.5 * 1.4826 * mad)


def estimate_transform(
    primary: dict[int, dict[str, str]], secondary: dict[int, dict[str, str]],
    minimum_overlap: int = 10,
) -> tuple[Rotation, np.ndarray, dict]:
    frames = sorted(
        frame for frame in set(primary) & set(secondary)
        if valid_pose(primary[frame]) and valid_pose(secondary[frame])
    )
    if len(frames) < minimum_overlap:
        raise ValueError(
            f"need {minimum_overlap} overlapping valid frames, found {len(frames)}"
        )
    rotations, translations = [], []
    for frame in frames:
        p_primary, r_primary = pose(primary[frame])
        p_secondary, r_secondary = pose(secondary[frame])
        rotation = r_primary * r_secondary.inv()
        rotations.append(rotation)
        translations.append(p_primary - rotation.apply(p_secondary))
    rotations = Rotation.concatenate(rotations)
    translations = np.asarray(translations)

    mean_rotation = rotations.mean()
    median_translation = np.median(translations, axis=0)
    angular = np.degrees((mean_rotation.inv() * rotations).magnitude())
    positional = np.linalg.norm(translations - median_translation, axis=1)
    keep = (angular <= mad_limit(angular, 2.0)) & (positional <= mad_limit(positional, 0.03))
    if int(keep.sum()) < minimum_overlap:
        order = np.argsort(angular / max(np.median(angular), 1e-6) + positional / max(np.median(positional), 1e-6))
        keep = np.zeros(len(frames), dtype=bool)
        keep[order[:minimum_overlap]] = True

    rotation = rotations[keep].mean()
    transformed_translations = []
    for index, frame in enumerate(frames):
        if not keep[index]:
            continue
        p_primary, _ = pose(primary[frame])
        p_secondary, _ = pose(secondary[frame])
        transformed_translations.append(p_primary - rotation.apply(p_secondary))
    translation = np.median(np.asarray(transformed_translations), axis=0)

    position_residuals, angle_residuals = [], []
    for index, frame in enumerate(frames):
        if not keep[index]:
            continue
        p_primary, r_primary = pose(primary[frame])
        p_secondary, r_secondary = pose(secondary[frame])
        position_residuals.append(np.linalg.norm(p_primary - (rotation.apply(p_secondary) + translation)))
        angle_residuals.append(np.degrees((r_primary.inv() * rotation * r_secondary).magnitude()))
    audit = {
        "overlap_frames": len(frames),
        "inlier_overlap_frames": int(keep.sum()),
        "overlap_frame_ids": [frame for frame, accepted in zip(frames, keep) if accepted],
        "secondary_to_primary_quaternion_xyzw": rotation.as_quat().tolist(),
        "secondary_to_primary_translation_m": translation.tolist(),
        "position_residual_m": {
            "median": float(np.median(position_residuals)),
            "p95": float(np.percentile(position_residuals, 95)),
        },
        "orientation_residual_deg": {
            "median": float(np.median(angle_residuals)),
            "p95": float(np.percentile(angle_residuals, 95)),
        },
    }
    return rotation, translation, audit


def transform_row(
    row: dict[str, str], rotation: Rotation, translation: np.ndarray,
    trusted_secondary: bool = False,
) -> dict[str, str]:
    updated = dict(row)
    position, orientation = pose(row)
    position = rotation.apply(position) + translation
    orientation = rotation * orientation
    for key, value in zip(POSITION_KEYS, position):
        updated[key] = f"{value:.9f}"
    for key, value in zip(EULER_KEYS, orientation.as_euler("xyz", degrees=True)):
        updated[key] = f"{value:.9f}"
    raw = vector(row, RAW_POSITION_KEYS)
    if raw is not None:
        raw = rotation.apply(raw) + translation
        for key, value in zip(RAW_POSITION_KEYS, raw):
            updated[key] = f"{value:.9f}"
    prefix = "same_map_relaxed:" if trusted_secondary else "secondary_map:"
    updated["measurement_source"] = prefix + updated.get("measurement_source", "direct")
    return updated


def merge(
    primary_path: Path, secondary_path: Path, output_path: Path,
    minimum_overlap: int = 10, trusted_secondary: bool = False,
) -> dict:
    fields, primary = read_rows(primary_path)
    secondary_fields, secondary = read_rows(secondary_path)
    if fields != secondary_fields:
        raise ValueError("pose CSV schemas differ")
    rotation, translation, audit = estimate_transform(primary, secondary, minimum_overlap)
    merged, primary_count, secondary_count = [], 0, 0
    for frame in sorted(set(primary) | set(secondary)):
        first = primary.get(frame)
        second = secondary.get(frame)
        if first is not None and valid_pose(first):
            merged.append(first); primary_count += 1
        elif second is not None and valid_pose(second):
            merged.append(transform_row(second, rotation, translation, trusted_secondary)); secondary_count += 1
        elif first is not None:
            merged.append(first)
        elif second is not None:
            merged.append(second)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(merged)
    valid_count = primary_count + secondary_count
    audit.update({
        "format": "overlapping-pose-map-merge/1.0",
        "primary": str(primary_path.resolve()),
        "secondary": str(secondary_path.resolve()),
        "output": str(output_path.resolve()),
        "primary_measurements": primary_count,
        "secondary_recovery_measurements": secondary_count,
        "secondary_measurements_trusted": trusted_secondary,
        "merged_frames": len(merged),
        "merged_valid_frames": valid_count,
        "merged_valid_ratio": valid_count / len(merged),
    })
    output_path.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("primary", type=Path)
    parser.add_argument("secondary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--minimum-overlap", type=int, default=10)
    parser.add_argument(
        "--trusted-secondary", action="store_true",
        help="secondary rows use the same physical map with a relaxed quality gate",
    )
    args = parser.parse_args()
    print(json.dumps(merge(
        args.primary, args.secondary, args.output, args.minimum_overlap,
        args.trusted_secondary,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
