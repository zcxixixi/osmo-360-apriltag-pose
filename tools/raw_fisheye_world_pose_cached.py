#!/usr/bin/env python3
"""Estimate world camera poses from cached raw-fisheye AprilTag bearings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from osmo360.localization.raw_fisheye_world_pose import (
    interpolate_initial,
    load_initial,
    solve_pose,
)
from osmo360.localization.world_frames import compile_world_tag_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation-cache", type=Path, required=True)
    parser.add_argument("--tag-map", type=Path, required=True)
    parser.add_argument("--initial-pose", type=Path, required=True)
    parser.add_argument("--sample-stride", type=int, default=3)
    parser.add_argument("--min-tags", type=int, default=2)
    parser.add_argument("--max-angular-rmse-deg", type=float, default=1.0)
    parser.add_argument("--prior-policy", choices=("initial-first", "previous-first"),
                        default="initial-first",
                        help="use per-frame external priors for calibration; previous-first is tracking-only")
    parser.add_argument("--regularize-prior", action="store_true",
                        help="include a weak prior residual; disabled for metric calibration")
    parser.add_argument("--start-common-s", type=float)
    parser.add_argument("--end-common-s", type=float)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def group_by_frame(cache: np.lib.npyio.NpzFile) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    for index, frame in enumerate(cache["frame_index"]):
        result.setdefault(int(frame), []).append(index)
    return result


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = np.load(args.observation_cache)
    by_frame = group_by_frame(cache)
    compiled = compile_world_tag_map(args.tag_map)
    map_corners = {
        int(tag["id"]): np.asarray(tag["corners_m"], dtype=float)
        for tag in compiled["tags"]
    }
    initial_series = load_initial(args.initial_pose)
    timeline_frames = cache["timeline_frame_index"][::args.sample_stride]
    timeline_times = cache["timeline_common_time_s"][::args.sample_stride]
    rows = []
    previous = None
    for frame, common_time in zip(timeline_frames, timeline_times):
        frame = int(frame)
        common_time = float(common_time)
        if args.start_common_s is not None and common_time < args.start_common_s:
            continue
        if args.end_common_s is not None and common_time > args.end_common_s:
            continue
        selected: dict[int, int] = {}
        for index in by_frame.get(frame, []):
            tag_id = int(cache["tag_id"][index])
            if tag_id not in map_corners:
                continue
            if tag_id not in selected or cache["area_px2"][index] > cache["area_px2"][selected[tag_id]]:
                selected[tag_id] = index
        world_points = []
        rays = []
        for tag_id, index in selected.items():
            world_points.extend(map_corners[tag_id])
            rays.extend(np.asarray(cache["rays_camera"][index], dtype=float))
        external_initial = interpolate_initial(initial_series, common_time)
        if args.prior_policy == "previous-first":
            initial = previous or external_initial
        else:
            # Calibration sequences already carry an independent per-frame
            # pose prior (multi-wall visual/IMU).  Prefer it so one accidental
            # planar mirror branch cannot contaminate the rest of the clip.
            initial = external_initial or previous
        row = {
            "frame": frame,
            "timestamp": f"{common_time:.6f}",
            "parent_frame": compiled.get("world_frame", "tag_map"),
            "child_frame": "fisheye1_camera_panorama_axes",
            "tag_map_sha256": compiled["tag_map_sha256"],
            "detected_tag_count": len(selected),
            "detected_ids": " ".join(map(str, sorted(selected))),
            "measurement_source": "direct_cached_raw_fisheye_unit_bearing",
            "quality_status": "invalid",
            "edge_rectified": "false",
        }
        if len(selected) >= args.min_tags and initial is not None:
            position, rotation, angular_errors = solve_pose(
                np.asarray(world_points), np.asarray(rays), initial,
                regularize=args.regularize_prior,
            )
            rmse = float(np.sqrt(np.mean(angular_errors ** 2)))
            row["inlier_count"] = len(world_points)
            row["angular_rmse_deg"] = f"{rmse:.6f}"
            if rmse <= args.max_angular_rmse_deg:
                previous = (position, rotation)
                quaternion = rotation.as_quat()
                roll, pitch, yaw = rotation.as_euler("xyz", degrees=True)
                row.update({
                    "camera_x_m": f"{position[0]:.9f}",
                    "camera_y_m": f"{position[1]:.9f}",
                    "camera_z_m": f"{position[2]:.9f}",
                    "qx": f"{quaternion[0]:.12f}",
                    "qy": f"{quaternion[1]:.12f}",
                    "qz": f"{quaternion[2]:.12f}",
                    "qw": f"{quaternion[3]:.12f}",
                    "roll_deg": f"{roll:.9f}",
                    "pitch_deg": f"{pitch:.9f}",
                    "yaw_deg": f"{yaw:.9f}",
                    "quality_status": "valid",
                })
            else:
                row["quality_status"] = "angular_rmse_rejected"
        rows.append(row)
    fields = [
        "frame", "timestamp", "camera_x_m", "camera_y_m", "camera_z_m",
        "qx", "qy", "qz", "qw", "roll_deg", "pitch_deg", "yaw_deg",
        "parent_frame", "child_frame",
        "tag_map_sha256", "detected_tag_count", "inlier_count",
        "reprojection_rmse_px", "angular_rmse_deg", "detected_ids",
        "measurement_source", "quality_status", "edge_rectified",
    ]
    with (args.output_dir / "pose.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    valid = [row for row in rows if row["quality_status"] == "valid"]
    residuals = np.asarray([float(row["angular_rmse_deg"]) for row in valid])
    summary = {
        "schema_version": "cached-raw-fisheye-world-pose/1.0",
        "observation_cache": str(args.observation_cache.resolve()),
        "tag_map": str(args.tag_map.resolve()),
        "tag_outer_size_m": compiled.get("tag_outer_size_m"),
        "common_frames": len(rows),
        "valid_frames": len(valid),
        "valid_ratio": len(valid) / len(rows) if rows else 0.0,
        "angular_rmse_deg": {
            "median": float(np.median(residuals)) if len(residuals) else None,
            "p95": float(np.percentile(residuals, 95)) if len(residuals) else None,
        },
        "tag_map_sha256": compiled["tag_map_sha256"],
        "prior_policy": args.prior_policy,
        "prior_regularized": args.regularize_prior,
        "video_decoded": False,
        "training_ready": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
