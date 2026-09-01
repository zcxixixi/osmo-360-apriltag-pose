#!/usr/bin/env python3
"""Self-calibrate two A3 AprilGrids and track both four-MP4 cameras."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path

import cv2

from osmo360.localization.cached_a3_bootstrap import (
    build_world_map,
    calibrate_panel_pair,
    load_direct_tag_map,
    track_cache,
    write_joint_pose_csv,
    write_pose_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-cache", type=Path, required=True)
    parser.add_argument("--right-cache", type=Path, required=True)
    parser.add_argument("--panel-a-map", type=Path, required=True)
    parser.add_argument("--panel-b-map", type=Path, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--minimum-tags", type=int, default=2)
    parser.add_argument("--minimum-calibration-inliers", type=int, default=20)
    parser.add_argument("--max-angular-rmse-deg", type=float, default=2.0)
    parser.add_argument("--maximum-interpolation-gap-s", type=float, default=0.25)
    parser.add_argument("--opencv-threads", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.opencv_threads <= 0:
        raise ValueError("--opencv-threads must be positive")
    if not 0 < args.maximum_interpolation_gap_s <= 0.25:
        raise ValueError(
            "--maximum-interpolation-gap-s must be positive and no greater than 0.25"
        )
    cv2.setNumThreads(args.opencv_threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel_a_payload, panel_a = load_direct_tag_map(args.panel_a_map)
    panel_b_payload, panel_b = load_direct_tag_map(args.panel_b_map)
    caches = {"left": args.left_cache, "right": args.right_cache}
    transform, calibration = calibrate_panel_pair(
        caches,
        panel_a,
        panel_b,
        minimum_tags=args.minimum_tags,
        minimum_inliers=args.minimum_calibration_inliers,
    )
    world_map = build_world_map(
        args.pair_id, panel_a_payload, panel_b_payload, transform
    )
    world_map_path = args.output_dir / "session_world_map.json"
    world_map_path.write_text(json.dumps(world_map, indent=2) + "\n", encoding="utf-8")
    trajectories = {}
    trajectory_rows = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            side: executor.submit(
                track_cache,
                cache,
                panel_a,
                panel_b,
                transform,
                minimum_tags=args.minimum_tags,
                max_angular_rmse_deg=args.max_angular_rmse_deg,
            )
            for side, cache in caches.items()
        }
        results = {side: future.result() for side, future in futures.items()}
    for side in ("left", "right"):
        rows, summary = results[side]
        write_pose_csv(args.output_dir / f"{side}_pose.csv", rows)
        trajectory_rows[side] = rows
        trajectories[side] = summary
    trajectories["joint"] = write_joint_pose_csv(
        args.output_dir / "joint_trajectory.csv",
        trajectory_rows["left"],
        trajectory_rows["right"],
        map_id=world_map["map_id"],
        maximum_interpolation_gap_s=args.maximum_interpolation_gap_s,
    )
    gates = {
        "calibration_inliers_at_least_minimum": (
            calibration["inliers"] >= calibration["minimum_inliers"]
        ),
        "panel_position_residual_p95_m_at_most_0_080": (
            calibration["fit"]["position_residual_m"]["p95"] <= 0.080
        ),
        "panel_orientation_residual_p95_deg_at_most_5": (
            calibration["fit"]["orientation_residual_deg"]["p95"] <= 5.0
        ),
        "left_valid_ratio_at_least_0_85": trajectories["left"]["valid_ratio"] >= 0.85,
        "right_valid_ratio_at_least_0_85": trajectories["right"]["valid_ratio"] >= 0.85,
        "joint_valid_ratio_at_least_0_85": (
            trajectories["joint"]["joint_valid_ratio"] >= 0.85
        ),
        "maximum_trusted_interpolation_gap_s_at_most_0_25": all(
            value <= 0.25
            for value in trajectories["joint"]["maximum_interpolation_gap_s"].values()
        ),
        "left_angular_rmse_p95_deg_at_most_2": (
            trajectories["left"]["angular_rmse_deg"]["p95"] is not None
            and trajectories["left"]["angular_rmse_deg"]["p95"] <= 2.0
        ),
        "right_angular_rmse_p95_deg_at_most_2": (
            trajectories["right"]["angular_rmse_deg"]["p95"] is not None
            and trajectories["right"]["angular_rmse_deg"]["p95"] <= 2.0
        ),
    }
    passed = all(gates.values())
    report = {
        "schema_version": "cached-a3-self-calibrated-trajectories/1.1",
        "pair_id": args.pair_id,
        "status": "SELF_CALIBRATED_PASS" if passed else "SELF_CALIBRATED_GATE_FAILED",
        "claims": {
            "capture_local_metric_trajectory": True,
            "same_capture_self_calibration": True,
            "external_ground_truth": False,
            "fixed_left_right_camera_extrinsic_used": False,
            "stitching_used": False,
            "joint_timeline_interpolation_used": True,
            "maximum_trusted_interpolation_gap_s": args.maximum_interpolation_gap_s,
            "long_gap_policy": (
                "INTERPOLATED_UNTRUSTED; pose hidden and trajectory trail segmented"
            ),
        },
        "calibration": calibration,
        "trajectories": trajectories,
        "gates": gates,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
