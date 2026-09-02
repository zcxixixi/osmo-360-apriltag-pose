#!/usr/bin/env python3
"""Merge non-overlapping resumable chunks from one raw fisheye MP4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.merge_fisheye_observation_caches import DETECTION_KEYS, TIMELINE_KEYS


TRACKING_COUNTERS = (
    "forward_backward_check_frame_count",
    "flow_attempted_tag_count",
    "flow_accepted_tag_count",
    "flow_rejected_status_count",
    "flow_rejected_forward_backward_count",
    "flow_rejected_geometry_count",
    "global_scout_frame_count",
    "global_scout_decoded_count",
    "local_redetect_frame_count",
    "local_redetect_decoded_count",
    "tracked_output_observation_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chunk", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sidecar(path: Path) -> Path:
    return path.with_suffix(".json") if path.suffix else Path(f"{path}.json")


def _metadata(path: Path) -> dict[str, Any]:
    metadata_path = sidecar(path)
    if not metadata_path.is_file():
        raise ValueError(f"chunk sidecar is missing: {metadata_path}")
    value = json.loads(metadata_path.read_text(encoding="utf-8"))
    if value.get("schema_version") not in {
        "fisheye-apriltag-observation-cache/1.0",
        "fisheye-apriltag-observation-cache/1.3-temporal",
    }:
        raise ValueError(f"unsupported chunk schema: {metadata_path}")
    decoded_range = value.get("decoded_frame_range")
    if not (
        isinstance(decoded_range, list)
        and len(decoded_range) == 2
        and all(isinstance(item, int) for item in decoded_range)
    ):
        raise ValueError(f"invalid decoded frame range: {metadata_path}")
    return value


def merge_chunks(paths: list[Path], output: Path) -> dict[str, Any]:
    resolved = [path.resolve(strict=True) for path in paths]
    pairs = sorted(
        ((path, _metadata(path)) for path in resolved),
        key=lambda item: item[1]["decoded_frame_range"][0],
    )
    invariant_keys = (
        "video",
        "video_sha256",
        "calibration_sha256",
        "x5_offset",
        "camera_serial",
        "stream",
        "source_size",
        "fps",
        "frame_count",
        "timeline_frame_count",
        "ignored_trailing_video_frames",
        "clock_mapping",
        "radial_model",
        "rectified_detection",
        "rectified_detection_policy",
        "rectified_min_direct_tags",
        "rectified_required_ids",
        "rectified_view_size",
        "rectification_radial_model",
        "frame_stride",
        "processing_signature",
        "temporal_tracking",
        "ray_frame",
    )
    for key in invariant_keys:
        values = [metadata.get(key) for _, metadata in pairs]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"chunks disagree on {key}: {values}")
    tracking_values = [metadata.get("tracking") for _, metadata in pairs]
    if any(tracking_values):
        if not all(isinstance(value, dict) for value in tracking_values):
            raise ValueError("chunks mix temporal and non-temporal caches")
        tracking_configs = [
            {key: value for key, value in tracking.items() if key not in TRACKING_COUNTERS}
            for tracking in tracking_values
        ]
        if any(value != tracking_configs[0] for value in tracking_configs[1:]):
            raise ValueError("chunks disagree on temporal tracking configuration")
    previous_end = -1
    for path, metadata in pairs:
        start, end = metadata["decoded_frame_range"]
        if start != previous_end + 1:
            raise ValueError(
                f"chunk ranges are not contiguous before {path}: "
                f"expected {previous_end + 1}, got {start}"
            )
        if end < start:
            raise ValueError(f"empty/reversed chunk range: {path}")
        previous_end = end
    expected_frame_count = int(
        pairs[0][1].get("timeline_frame_count") or pairs[0][1]["frame_count"]
    )
    expected_last = expected_frame_count - 1
    if previous_end != expected_last:
        raise ValueError(
            f"chunks stop at frame {previous_end}; processing timeline ends at "
            f"{expected_last}"
        )

    caches = [np.load(path) for path, _ in pairs]
    try:
        timeline = {
            key: np.concatenate([cache[key] for cache in caches])
            for key in TIMELINE_KEYS
        }
        detections = {
            key: np.concatenate([cache[key] for cache in caches])
            for key in DETECTION_KEYS
        }
        if len(timeline["timeline_frame_index"]):
            actual = timeline["timeline_frame_index"]
            expected = np.arange(len(actual), dtype=actual.dtype)
            if not np.array_equal(actual, expected):
                raise ValueError("merged chunk timeline is not a complete zero-based sequence")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **timeline, **detections)
        temporary.replace(output)
    finally:
        for cache in caches:
            cache.close()

    first = pairs[0][1]
    detection_sources = detections["detection_source"].tolist()
    report = {
        key: first.get(key)
        for key in (
            "schema_version",
            "video",
            "video_sha256",
            "calibration",
            "calibration_sha256",
            "x5_offset",
            "camera_serial",
            "stream",
            "source_size",
            "fps",
            "frame_count",
            "timeline_frame_count",
            "ignored_trailing_video_frames",
            "clock_mapping",
            "radial_model",
            "rectified_detection",
            "rectified_detection_policy",
            "rectified_min_direct_tags",
            "rectified_required_ids",
            "rectified_view_size",
            "rectification_radial_model",
            "frame_stride",
            "processing_signature",
            "temporal_tracking",
            "corner_order",
            "ray_frame",
        )
    }
    report.update(
        {
            "decoded_frame_count": int(len(timeline["timeline_frame_index"])),
            "decoded_frame_range": [0, expected_last],
            "detection_frame_range": [0, expected_last],
            "detection_count": int(len(detections["tag_id"])),
            "detected_ids": sorted(set(map(int, detections["tag_id"]))),
            "detection_source_counts": {
                source: detection_sources.count(source)
                for source in sorted(set(detection_sources))
            },
            "detector_frame_count": sum(
                int(metadata.get("detector_frame_count", 0)) for _, metadata in pairs
            ),
            "rectified_detector_frame_count": sum(
                int(metadata.get("rectified_detector_frame_count", 0))
                for _, metadata in pairs
            ),
            "adaptive_rectification_skipped_frames": sum(
                int(metadata.get("adaptive_rectification_skipped_frames", 0))
                for _, metadata in pairs
            ),
            "chunk_count": len(pairs),
            "chunks": [str(path) for path, _ in pairs],
            "scan_policy": (
                "resumable contiguous target-frame chunks; decoder keyframe seek overhead "
                "may overlap at chunk boundaries"
            ),
            "cache": str(output.resolve()),
        }
    )
    if tracking_values[0] is not None:
        tracking = dict(tracking_configs[0])
        tracking.update({
            key: sum(int(value.get(key, 0)) for value in tracking_values)
            for key in TRACKING_COUNTERS
        })
        report["tracking"] = tracking
    metadata_path = sidecar(output)
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    temporary_metadata.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    temporary_metadata.replace(metadata_path)
    return report


def main() -> int:
    args = parse_args()
    report = merge_chunks(args.chunk, args.output)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
