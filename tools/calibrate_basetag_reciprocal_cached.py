#!/usr/bin/env python3
"""Fit reciprocal camera-to-BaseTag calibration from a single observation cache.

No video decoding and no inlier-radius tuning are performed here. Two global
IPPE branch hypotheses are fitted on blocked training intervals. The branch is
selected only by closure on held-out temporal blocks, then refitted on all data.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from osmo360.calibration.calibrate_basetag_reciprocal import (
    Transform,
    interpolate_pose,
    load_pose,
    rotation_distance_deg,
)
from osmo360.calibration.estimate_gripper_extrinsic import solve_bearing_ippe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observer-cache", type=Path, required=True)
    parser.add_argument("--observer-pose", type=Path, required=True)
    parser.add_argument("--target-pose", type=Path, required=True)
    parser.add_argument("--observer-name", required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--tag-id", type=int, required=True)
    parser.add_argument("--tag-size-m", type=float, default=0.020)
    parser.add_argument("--tag-corner-quarter-turns", type=int, choices=range(4),
                        required=True)
    parser.add_argument("--start-common-s", type=float)
    parser.add_argument("--end-common-s", type=float)
    parser.add_argument("--temporal-blocks", type=int, default=5)
    parser.add_argument("--holdout-blocks", default="1,3")
    parser.add_argument(
        "--require-both-wall-support", action="store_true",
        help=("accept a calibration observation only when both camera pose rows "
              "directly contain IDs from both perpendicular wall panels"),
    )
    parser.add_argument("--left-wall-ids", default="134,135,136,137")
    parser.add_argument("--right-wall-ids", default="128,129,130,131,132,133")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_ids(value: str) -> set[int]:
    return {int(item) for item in value.split(",") if item.strip()}


def load_both_wall_support_times(
    path: Path, left_ids: set[int], right_ids: set[int],
) -> np.ndarray:
    """Return direct-pose timestamps constrained by both perpendicular walls."""
    result: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("quality_status") != "valid":
                continue
            detected = {
                int(value) for value in row.get("detected_ids", "").split()
                if value
            }
            if detected & left_ids and detected & right_ids:
                result.append(float(row["timestamp"]))
    return np.asarray(result, dtype=float)


def timestamp_supported(times: np.ndarray, value: float,
                        tolerance_s: float = 0.020) -> bool:
    if not len(times):
        return False
    index = int(np.searchsorted(times, value))
    choices = [candidate for candidate in (index - 1, index)
               if 0 <= candidate < len(times)]
    return min(abs(float(times[candidate]) - value) for candidate in choices) <= tolerance_s


def load_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    observer_series = load_pose(args.observer_pose)
    target_series = load_pose(args.target_pose)
    half = args.tag_size_m / 2.0
    tag_points = np.asarray([
        [-half, -half, 0.0], [half, -half, 0.0],
        [half, half, 0.0], [-half, half, 0.0],
    ])
    cache = np.load(args.observer_cache)
    observer_support = target_support = None
    if args.require_both_wall_support:
        left_ids = parse_ids(args.left_wall_ids)
        right_ids = parse_ids(args.right_wall_ids)
        observer_support = load_both_wall_support_times(
            args.observer_pose, left_ids, right_ids)
        target_support = load_both_wall_support_times(
            args.target_pose, left_ids, right_ids)
        if not len(observer_support) or not len(target_support):
            raise RuntimeError("no direct two-wall pose support in observer or target")
    selected = np.flatnonzero(cache["tag_id"] == args.tag_id)
    candidates: list[dict[str, Any]] = []
    for index in selected:
        common_time = float(cache["common_time_s"][index])
        if args.start_common_s is not None and common_time < args.start_common_s:
            continue
        if args.end_common_s is not None and common_time > args.end_common_s:
            continue
        if (observer_support is not None
                and (not timestamp_supported(observer_support, common_time)
                     or not timestamp_supported(target_support, common_time))):
            continue
        world_observer = interpolate_pose(observer_series, common_time)
        world_target = interpolate_pose(target_series, common_time)
        if world_observer is None or world_target is None:
            continue
        rays = np.roll(
            np.asarray(cache["rays_camera"][index], dtype=float),
            -args.tag_corner_quarter_turns,
            axis=0,
        )
        for solution in solve_bearing_ippe(tag_points, rays):
            observer_tag = Transform(
                np.asarray(solution["translation_tag_origin_in_panorama_m"]),
                Rotation.from_matrix(solution["rotation_tag_to_panorama"]),
            )
            target_tag = world_target.inverse().compose(
                world_observer.compose(observer_tag)
            )
            candidates.append({
                "frame": int(cache["frame_index"][index]),
                "common_time_s": common_time,
                "branch": int(solution["branch"]),
                "angular_rmse_deg": float(solution["angular_rmse_deg"]),
                "transform": target_tag,
            })
    return candidates


def robust_fit(items: list[dict[str, Any]]) -> Transform:
    if len(items) < 3:
        raise RuntimeError(f"only {len(items)} observations available for fit")
    positions = np.asarray([item["transform"].p for item in items])
    rotations = Rotation.from_quat([
        item["transform"].r.as_quat() for item in items
    ])
    initial = np.r_[np.median(positions, axis=0), rotations.mean().as_rotvec()]

    def residual(parameters: np.ndarray) -> np.ndarray:
        transform = Transform(parameters[:3], Rotation.from_rotvec(parameters[3:]))
        result = []
        for item in items:
            # Scales normalize SE(3) units for the robust loss; they are not
            # acceptance thresholds and are never tuned per recording.
            result.extend((item["transform"].p - transform.p) / 0.010)
            result.extend(
                (transform.r.inv() * item["transform"].r).as_rotvec()
                / np.radians(5.0)
            )
        return np.asarray(result)

    fit = least_squares(
        residual,
        initial,
        loss="huber",
        f_scale=1.0,
        max_nfev=1000,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )
    return Transform(fit.x[:3], Rotation.from_rotvec(fit.x[3:]))


def metrics(model: Transform, items: list[dict[str, Any]]) -> dict[str, Any]:
    position = np.asarray([
        np.linalg.norm(item["transform"].p - model.p) * 1000.0 for item in items
    ])
    rotation = np.asarray([
        rotation_distance_deg(model.r, item["transform"].r) for item in items
    ])
    angular = np.asarray([item["angular_rmse_deg"] for item in items])

    def summary(values: np.ndarray) -> dict[str, float]:
        return {
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(values.max()),
        }

    return {
        "count": len(items),
        "position_error_mm": summary(position),
        "rotation_error_deg": summary(rotation),
        "tag_bearing_fit_deg": summary(angular),
    }


def main() -> int:
    args = parse_args()
    holdout_blocks = {int(value) for value in args.holdout_blocks.split(",")}
    if args.temporal_blocks < 3:
        raise ValueError("--temporal-blocks must be at least 3")
    if not holdout_blocks or min(holdout_blocks) < 0 or max(holdout_blocks) >= args.temporal_blocks:
        raise ValueError("invalid --holdout-blocks")
    candidates = load_candidates(args)
    frames = sorted({item["frame"] for item in candidates})
    if len(frames) < args.temporal_blocks * 2:
        raise RuntimeError("not enough observed frames for blocked holdout")
    frame_block = {
        frame: min(args.temporal_blocks - 1, rank * args.temporal_blocks // len(frames))
        for rank, frame in enumerate(frames)
    }
    hypotheses = []
    for branch in (0, 1):
        branch_items = [item for item in candidates if item["branch"] == branch]
        train = [item for item in branch_items if frame_block[item["frame"]] not in holdout_blocks]
        holdout = [item for item in branch_items if frame_block[item["frame"]] in holdout_blocks]
        train_model = robust_fit(train)
        holdout_metrics = metrics(train_model, holdout)
        position = holdout_metrics["position_error_mm"]
        rotation = holdout_metrics["rotation_error_deg"]
        score = (
            position["median"] / 3.0
            + position["p95"] / 5.0
            + rotation["median"] / 2.0
            + rotation["p95"] / 3.0
        )
        hypotheses.append({
            "branch": branch,
            "score": float(score),
            "train_count": len(train),
            "holdout_blocks": sorted(holdout_blocks),
            "holdout": holdout_metrics,
            "train_model": train_model,
        })
    selected = min(hypotheses, key=lambda item: item["score"])
    selected_branch = int(selected["branch"])
    all_selected = [item for item in candidates if item["branch"] == selected_branch]
    final_model = robust_fit(all_selected)
    holdout = selected["holdout"]
    position = holdout["position_error_mm"]
    rotation = holdout["rotation_error_deg"]
    passed = (
        position["median"] <= 3.0
        and position["p95"] <= 5.0
        and rotation["median"] <= 2.0
        and rotation["p95"] <= 3.0
    )
    payload = {
        "schema_version": "cached-reciprocal-basetag-calibration/1.0",
        "calibration_status": "HOLDOUT_PASS" if passed else "HOLDOUT_FAILED",
        "observer": args.observer_name,
        "target": args.target_name,
        "tag_id": args.tag_id,
        "tag_size_m": args.tag_size_m,
        "tag_corner_quarter_turns": args.tag_corner_quarter_turns,
        "selected_branch": selected_branch,
        "camera_to_basetag": {
            "parent_frame": "fisheye1_camera_panorama_axes",
            "child_frame": f"{args.target_name}_mount_tag{args.tag_id}",
            "translation_m": final_model.p.tolist(),
            "quaternion_xyzw": final_model.r.as_quat().tolist(),
        },
        "blocked_holdout": {
            "temporal_blocks": args.temporal_blocks,
            "holdout_blocks": sorted(holdout_blocks),
            "quality_gate": {
                "position_median_max_mm": 3.0,
                "position_p95_max_mm": 5.0,
                "rotation_median_max_deg": 2.0,
                "rotation_p95_max_deg": 3.0,
            },
            "selected_result": holdout,
            "hypotheses": [{
                "branch": item["branch"],
                "score": item["score"],
                "train_count": item["train_count"],
                "holdout": item["holdout"],
            } for item in hypotheses],
        },
        "all_data_diagnostic": metrics(final_model, all_selected),
        "source": {
            "observer_cache": str(args.observer_cache.resolve()),
            "observer_pose": str(args.observer_pose.resolve()),
            "target_pose": str(args.target_pose.resolve()),
            "require_both_wall_support": args.require_both_wall_support,
            "left_wall_ids": sorted(parse_ids(args.left_wall_ids)),
            "right_wall_ids": sorted(parse_ids(args.right_wall_ids)),
        },
        "training_ready": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
