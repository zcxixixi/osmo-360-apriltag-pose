#!/usr/bin/env python3
"""Self-calibrate two A3 AprilGrids and track both four-MP4 cameras."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path

import cv2

from osmo360.localization.cached_a3_bootstrap import (
    MAXIMUM_ABSOLUTE_ANGULAR_SPEED_DEG_S,
    MAXIMUM_ABSOLUTE_SPEED_M_S,
    MAXIMUM_IMU_VISUAL_ROTATION_RESIDUAL_DEG,
    MINIMUM_TEMPORAL_RECOVERY_INLIER_TAGS,
    build_world_map,
    calibrate_panel_pair,
    load_direct_tag_map,
    track_cache,
    write_joint_pose_csv,
    write_pose_csv,
)
from osmo360.localization.instaumi_imu import (
    ImuAssistanceUnavailable,
    calibrate_instaumi_imu_from_visual,
    load_instaumi_imu,
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
    parser.add_argument(
        "--imu-h5",
        type=Path,
        help="optional InstaUMI H5 with independently calibrated per-side IMU streams",
    )
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
    imu_streams = {}
    imu_audit = {
        "status": "NOT_REQUESTED",
        "translation_source": "visual_endpoint_interpolation",
        "orientation_source": "visual_slerp",
    }
    if args.imu_h5 is not None:
        try:
            imu_bundle = load_instaumi_imu(args.imu_h5)
        except (ImuAssistanceUnavailable, OSError, ValueError, KeyError) as exc:
            imu_audit = {
                "status": "UNAVAILABLE_INVALID_H5",
                "h5": str(args.imu_h5.resolve()),
                "reason": str(exc),
                "translation_source": "visual_endpoint_interpolation",
                "orientation_source": "visual_slerp_fallback",
            }
        else:
            imu_streams = imu_bundle.streams
            imu_audit = imu_bundle.audit
    explicit_imu_audit = imu_audit

    def run_tracking() -> dict:
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
                    imu_stream=imu_streams.get(side),
                )
                for side, cache in caches.items()
            }
            return {side: future.result() for side, future in futures.items()}

    results = run_tracking()
    visual_tracking_passes = 1
    if args.imu_h5 is not None and len(imu_streams) != 2:
        try:
            self_calibrated = calibrate_instaumi_imu_from_visual(
                args.imu_h5,
                {side: results[side][0] for side in ("left", "right")},
            )
        except (ImuAssistanceUnavailable, OSError, ValueError, KeyError) as exc:
            imu_audit = {
                "status": "UNAVAILABLE_VISUAL_SELF_CALIBRATION_FAILED",
                "h5": str(args.imu_h5.resolve()),
                "reason": str(exc),
                "explicit_calibration_attempt": explicit_imu_audit,
                "translation_source": "visual_endpoint_interpolation",
                "orientation_source": "visual_slerp_fallback",
            }
        else:
            imu_streams = self_calibrated.streams
            imu_audit = {
                **self_calibrated.audit,
                "explicit_calibration_attempt": explicit_imu_audit,
            }
            if len(imu_streams) == 2:
                results = run_tracking()
                visual_tracking_passes = 2
    trajectories = {}
    trajectory_rows = {}
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
        imu_streams=imu_streams,
        imu_audit=imu_audit,
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
        "joint_pose_ratio_is_one": (
            trajectories["joint"]["joint_pose_ratio"] == 1.0
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
        "schema_version": "cached-a3-self-calibrated-trajectories/1.5",
        "pair_id": args.pair_id,
        "status": "SELF_CALIBRATED_PASS" if passed else "SELF_CALIBRATED_GATE_FAILED",
        "claims": {
            "capture_local_metric_trajectory": True,
            "same_capture_self_calibration": True,
            "external_ground_truth": False,
            "fixed_left_right_camera_extrinsic_used": False,
            "stitching_used": False,
            "joint_timeline_interpolation_used": True,
            "calibrated_per_side_imu_assistance_enabled": bool(imu_streams),
            "calibrated_per_side_imu_visual_attitude_gate_enabled": bool(
                imu_streams
            ),
            "accelerometer_translation_integration_used": (
                trajectories["joint"]["imu_assistance"]
                ["accelerometer_assisted_frames"] > 0
            ),
            "accelerometer_translation_policy": (
                "timestamp-aligned specific force shapes only internal visual gaps; "
                "mean world specific force is removed; both visual positions remain "
                "exact metric anchors; deviation is capped at 0.15 m"
            ),
            "visual_tracking_passes": visual_tracking_passes,
            "maximum_trusted_interpolation_gap_s": args.maximum_interpolation_gap_s,
            "absolute_visual_motion_limits": {
                "speed_m_s": MAXIMUM_ABSOLUTE_SPEED_M_S,
                "angular_speed_deg_s": MAXIMUM_ABSOLUTE_ANGULAR_SPEED_DEG_S,
            },
            "weak_visual_imu_residual_limit_deg": (
                MAXIMUM_IMU_VISUAL_ROTATION_RESIDUAL_DEG
            ),
            "temporal_recovery_minimum_inlier_tags": (
                MINIMUM_TEMPORAL_RECOVERY_INLIER_TAGS
            ),
            "short_gap_policy": (
                "visual endpoint-anchored accelerometer translation plus calibrated "
                "per-side gyro bridge when available; otherwise visual position "
                "interpolation and SLERP"
            ),
            "long_gap_policy": (
                "endpoint-anchored accelerometer/gyro bridge remains explicitly "
                "untrusted; otherwise visual INTERPOLATED_UNTRUSTED; numeric pose "
                "retained with explicit confidence"
            ),
        },
        "calibration": calibration,
        "trajectories": trajectories,
        "imu_assistance": trajectories["joint"]["imu_assistance"],
        "gates": gates,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
