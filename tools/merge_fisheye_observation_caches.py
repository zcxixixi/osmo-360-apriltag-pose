#!/usr/bin/env python3
"""Merge synchronized raw-fisheye lens caches into one camera-rig cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DETECTION_KEYS = (
    "frame_index",
    "local_time_s",
    "common_time_s",
    "tag_id",
    "corners_px",
    "rays_camera",
    "area_px2",
    "center_px",
    "detection_source",
)
TIMELINE_KEYS = (
    "timeline_frame_index",
    "timeline_local_time_s",
    "timeline_common_time_s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path, nargs="+", help="one cache per lens")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sidecar(path: Path) -> Path:
    return path.with_suffix(".json") if path.suffix else Path(f"{path}.json")


def main() -> int:
    args = parse_args()
    if len(args.cache) < 2:
        raise ValueError("at least two lens caches are required")
    paths = [path.resolve(strict=True) for path in args.cache]
    metadata = [json.loads(sidecar(path).read_text(encoding="utf-8")) for path in paths]
    streams = [int(item["stream"]) for item in metadata]
    if len(set(streams)) != len(streams):
        raise ValueError("lens cache stream IDs must be unique")
    invariant_keys = ("camera_serial", "source_size", "fps", "frame_count", "clock_mapping", "ray_frame")
    for key in invariant_keys:
        values = [item.get(key) for item in metadata]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"lens caches disagree on {key}: {values}")
    caches = [np.load(path) for path in paths]
    for key in TIMELINE_KEYS:
        if any(not np.array_equal(cache[key], caches[0][key]) for cache in caches[1:]):
            raise ValueError(f"lens cache timelines disagree on {key}")
    merged = {key: np.concatenate([cache[key] for cache in caches]) for key in DETECTION_KEYS}
    lens_stream = np.concatenate([
        np.full(len(cache["frame_index"]), stream, dtype=np.int8)
        for cache, stream in zip(caches, streams)
    ])
    order = np.lexsort((merged["tag_id"], merged["frame_index"]))
    merged = {key: value[order] for key, value in merged.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            **{key: caches[0][key] for key in TIMELINE_KEYS},
            **merged,
            lens_stream=lens_stream[order],
        )
    temporary.replace(args.output)
    report = {
        "schema_version": "fisheye-apriltag-observation-cache/1.2-dual-lens",
        "camera_serial": metadata[0]["camera_serial"],
        "streams": sorted(streams),
        "source_videos": [item["video"] for item in metadata],
        "source_size": metadata[0]["source_size"],
        "fps": metadata[0]["fps"],
        "frame_count": metadata[0]["frame_count"],
        "clock_mapping": metadata[0]["clock_mapping"],
        "ray_frame": metadata[0]["ray_frame"],
        "calibration": "embedded_x5_offset",
        "calibration_sha256": sorted({item.get("calibration_sha256") for item in metadata}),
        "x5_offset": metadata[0].get("x5_offset"),
        "detection_count": int(len(merged["tag_id"])),
        "detected_ids": sorted(set(map(int, merged["tag_id"]))),
        "corner_order": "opencv_aruco_apriltag_canonical",
        "cached_measurement": "two raw X5 fisheye tracks mapped into one rig ray frame",
        "stitching_used": False,
        "synthetic_frames_used": False,
        "cache": str(args.output.resolve()),
    }
    sidecar(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
