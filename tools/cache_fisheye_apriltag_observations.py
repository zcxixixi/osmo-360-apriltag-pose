#!/usr/bin/env python3
"""Scan a raw fisheye stream once and cache AprilTag corners and unit bearings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from osmo360.localization.raw_fisheye_world_pose import (
    detect_rectified_tags,
    make_ray_converter,
    make_rectified_maps,
    make_x5_offset_ray_converter,
    make_x5_rectified_maps,
)

CACHE_PRODUCER_REVISION = "raw-fisheye-cache-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--calibration", type=Path)
    source.add_argument("--x5-offset")
    parser.add_argument("--camera-serial")
    parser.add_argument("--panoforge-root", type=Path)
    parser.add_argument("--stream", type=int, default=1)
    parser.add_argument("--source-width", type=int, required=True)
    parser.add_argument("--source-height", type=int, required=True)
    parser.add_argument("--clock-intercept-s", type=float, default=0.0,
                        help="local_time = intercept + slope * common_time")
    parser.add_argument("--clock-slope", type=float, default=1.0,
                        help="local_time = intercept + slope * common_time")
    parser.add_argument("--radial-model", choices=("stitch", "factory-polynomial"),
                        default="stitch")
    parser.add_argument("--rectified-detection", action="store_true",
                        help="detect in overlapping tangent views and map corners back to raw pixels")
    parser.add_argument("--rectified-view-size", type=int, default=960)
    parser.add_argument("--rectification-radial-model", choices=("metric", "stitch"),
                        default="stitch")
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="decode every frame but run the detector only on every Nth frame")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int,
                        help="inclusive final frame for detection; later frames remain on the timeline")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if args.clock_slope <= 0:
        raise ValueError("--clock-slope must be positive")
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")
    if args.start_frame < 0 or (args.end_frame is not None and args.end_frame < args.start_frame):
        raise ValueError("invalid detection frame range")
    video = args.video.resolve(strict=True)
    x5_offset_record = None
    if args.calibration is not None:
        if args.panoforge_root is None:
            raise ValueError("--panoforge-root is required with --calibration")
        calibration = args.calibration.resolve(strict=True)
        converter, scaled = make_ray_converter(SimpleNamespace(
            calibration=calibration,
            panoforge_root=args.panoforge_root,
            source_width=args.source_width,
            source_height=args.source_height,
            stream=args.stream,
            radial_model=args.radial_model,
        ))
        rectified_maps = make_rectified_maps(SimpleNamespace(
            edge_rectification=args.rectified_detection,
            panoforge_root=args.panoforge_root,
            stream=args.stream,
            rectified_view_size=args.rectified_view_size,
            rectification_radial_model=args.rectification_radial_model,
        ), scaled)
        calibration_source = str(calibration)
        calibration_sha256 = sha256(calibration)
    else:
        converter, scaled = make_x5_offset_ray_converter(
            args.x5_offset,
            stream=args.stream,
            source_width=args.source_width,
            source_height=args.source_height,
        )
        rectified_maps = (
            make_x5_rectified_maps(
                args.x5_offset,
                stream=args.stream,
                source_width=args.source_width,
                source_height=args.source_height,
                view_size=args.rectified_view_size,
            )
            if args.rectified_detection
            else []
        )
        x5_offset_record = args.x5_offset
        calibration_source = "embedded_x5_offset"
        calibration_sha256 = hashlib.sha256(args.x5_offset.encode()).hexdigest()
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.adaptiveThreshWinSizeMax = 63
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    timeline_frame = []
    timeline_local = []
    timeline_common = []
    frames = []
    local_times = []
    common_times = []
    tag_ids = []
    corners = []
    rays = []
    areas = []
    centers = []
    detection_sources = []
    frame = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        local_time = frame / fps
        common_time = (local_time - args.clock_intercept_s) / args.clock_slope
        timeline_frame.append(frame)
        timeline_local.append(local_time)
        timeline_common.append(common_time)
        detections: list[tuple[int, np.ndarray, str]] = []
        in_detection_range = frame >= args.start_frame and (
            args.end_frame is None or frame <= args.end_frame
        )
        if in_detection_range and (frame - args.start_frame) % args.frame_stride == 0:
            quads, ids, _ = detector.detectMarkers(image)
            direct = {} if ids is None else {
                int(tag_id): np.asarray(quad, dtype=np.float32).reshape(4, 2)
                for quad, tag_id in zip(quads, ids.flatten())
            }
            if rectified_maps:
                # A raw fisheye image bends the physical edges of a large Tag.
                # Fitting a straight quadrilateral directly to those curves
                # biases the corners and, in turn, planar-PnP depth.  Tangent
                # views restore straight edges; their corners are then mapped
                # back to raw pixels before the factory ray conversion.
                rectified = dict(detect_rectified_tags(image, detector, rectified_maps))
                direct.update(rectified)
                rectified_ids = set(rectified)
            else:
                rectified_ids = set()
            detections = [
                (tag_id, quad, "rectified_tangent" if tag_id in rectified_ids else "direct_raw")
                for tag_id, quad in direct.items()
            ]
        for tag_id, quad, detection_source in detections:
            frames.append(frame)
            local_times.append(local_time)
            common_times.append(common_time)
            tag_ids.append(int(tag_id))
            corners.append(quad)
            rays.append(converter(quad).astype(np.float32))
            areas.append(abs(float(cv2.contourArea(quad))))
            centers.append(quad.mean(axis=0))
            detection_sources.append(detection_source)
        frame += 1
    capture.release()
    if frame != total:
        total = frame
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            timeline_frame_index=np.asarray(timeline_frame, dtype=np.int32),
            timeline_local_time_s=np.asarray(timeline_local, dtype=np.float64),
            timeline_common_time_s=np.asarray(timeline_common, dtype=np.float64),
            frame_index=np.asarray(frames, dtype=np.int32),
            local_time_s=np.asarray(local_times, dtype=np.float64),
            common_time_s=np.asarray(common_times, dtype=np.float64),
            tag_id=np.asarray(tag_ids, dtype=np.int32),
            corners_px=np.asarray(corners, dtype=np.float32).reshape(-1, 4, 2),
            rays_camera=np.asarray(rays, dtype=np.float32).reshape(-1, 4, 3),
            area_px2=np.asarray(areas, dtype=np.float32),
            center_px=np.asarray(centers, dtype=np.float32).reshape(-1, 2),
            detection_source=np.asarray(detection_sources, dtype="U24"),
        )
    temporary.replace(args.output)
    metadata = {
        "schema_version": "fisheye-apriltag-observation-cache/1.0",
        "producer_revision": CACHE_PRODUCER_REVISION,
        "video": str(video),
        "video_sha256": sha256(video),
        "calibration": calibration_source,
        "calibration_sha256": calibration_sha256,
        "x5_offset": x5_offset_record,
        "camera_serial": args.camera_serial or scaled.get("serial"),
        "stream": args.stream,
        "source_size": [args.source_width, args.source_height],
        "fps": fps,
        "frame_count": total,
        "detection_count": len(tag_ids),
        "detected_ids": sorted(set(tag_ids)),
        "clock_mapping": {
            "formula": "local_time = intercept_s + slope * common_time",
            "intercept_s": args.clock_intercept_s,
            "slope": args.clock_slope,
        },
        "radial_model": (
            args.radial_model if args.calibration is not None else "x5-offset-equidistant"
        ),
        "rectified_detection": args.rectified_detection,
        "rectified_view_size": args.rectified_view_size if args.rectified_detection else None,
        "rectification_radial_model": (
            args.rectification_radial_model if args.rectified_detection else None
        ),
        "frame_stride": args.frame_stride,
        "detection_frame_range": [args.start_frame, args.end_frame],
        "detection_source_counts": {
            source: detection_sources.count(source) for source in sorted(set(detection_sources))
        },
        "corner_order": "opencv_aruco_apriltag_canonical",
        "ray_frame": scaled.get(
            "ray_frame", "fisheye optical centre, panorama OpenCV axes"
        ),
        "scan_policy": "every decoded frame exactly once; direct detector only",
        "cache": str(args.output.resolve()),
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
