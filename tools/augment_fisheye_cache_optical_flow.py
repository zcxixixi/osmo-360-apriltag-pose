#!/usr/bin/env python3
"""Fill direct AprilTag gaps with audited forward/backward LK corner tracks.

The detector cache remains authoritative.  A track starts only from a direct
detection, is reset whenever that Tag is decoded again, and is discarded as
soon as any of its four corners fails the forward/backward or geometry gate.
Tracked corners are converted back to factory-calibrated raw-fisheye bearings;
they are image measurements, never temporal pose interpolation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from osmo360.localization.raw_fisheye_world_pose import make_ray_converter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--input-cache", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--panoforge-root", type=Path, required=True)
    parser.add_argument("--stream", type=int, default=1)
    parser.add_argument("--source-width", type=int, default=3840)
    parser.add_argument("--source-height", type=int, default=3840)
    parser.add_argument("--radial-model", default="factory-polynomial")
    parser.add_argument("--minimum-id", type=int, default=128)
    parser.add_argument("--maximum-id", type=int, default=137)
    parser.add_argument("--max-track-age", type=int, default=12)
    parser.add_argument("--max-fb-error-px", type=float, default=1.5)
    parser.add_argument(
        "--instance-track-id", type=int, action="append", default=[],
        help=(
            "Track this physical Tag instance from its first decoded quad and "
            "do not jump to a far same-ID detection (for example a screen copy)."
        ),
    )
    parser.add_argument(
        "--max-instance-reacquire-distance-px", type=float, default=160.0,
        help="Maximum centre displacement for a decoded same-ID instance to replace its LK track.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def track_quad(previous: np.ndarray, current: np.ndarray, quad: np.ndarray,
               max_fb_error: float) -> np.ndarray | None:
    points = np.asarray(quad, np.float32).reshape(-1, 1, 2)
    options = dict(winSize=(31, 31), maxLevel=4,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    forward, ok_f, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None, **options)
    if forward is None or ok_f is None or not np.all(ok_f):
        return None
    backward, ok_b, _ = cv2.calcOpticalFlowPyrLK(current, previous, forward, None, **options)
    if backward is None or ok_b is None or not np.all(ok_b):
        return None
    if float(np.max(np.linalg.norm(backward - points, axis=2))) > max_fb_error:
        return None
    result = forward.reshape(4, 2)
    if not np.isfinite(result).all() or not cv2.isContourConvex(result.astype(np.float32)):
        return None
    old_area = abs(float(cv2.contourArea(np.asarray(quad, np.float32))))
    new_area = abs(float(cv2.contourArea(result.astype(np.float32))))
    if old_area < 100.0 or new_area < 100.0 or not 0.55 <= new_area / old_area <= 1.8:
        return None
    return result


def main() -> int:
    args = parse_args()
    cache = np.load(args.input_cache)
    converter, scaled = make_ray_converter(SimpleNamespace(
        calibration=args.calibration, panoforge_root=args.panoforge_root,
        source_width=args.source_width, source_height=args.source_height,
        stream=args.stream, radial_model=args.radial_model,
    ))
    direct_by_frame: dict[int, dict[int, int]] = {}
    for index, (frame, tag_id) in enumerate(zip(cache["frame_index"], cache["tag_id"])):
        direct_by_frame.setdefault(int(frame), {})[int(tag_id)] = index
    output = {key: [] for key in (
        "frame_index", "local_time_s", "common_time_s", "tag_id", "corners_px",
        "rays_camera", "area_px2", "center_px", "detection_source",
    )}
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {args.video}")
    active: dict[int, tuple[np.ndarray, int]] = {}
    previous_gray = None
    frame = 0
    tracked_count = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        direct = direct_by_frame.get(frame, {})
        direct_wall_count = sum(
            args.minimum_id <= tag_id <= args.maximum_id for tag_id in direct
        )
        candidates: dict[int, tuple[np.ndarray, str]] = {}
        locked_ids = set(args.instance_track_id)
        for tag_id, index in direct.items():
            if tag_id not in locked_ids or tag_id not in active:
                candidates[tag_id] = (
                    np.asarray(cache["corners_px"][index], np.float32),
                    str(cache["detection_source"][index]),
                )
        # Never let a stale track perturb a frame that already has enough
        # decoded wall Tags for metric PnP.  LK exists only to bridge detector
        # gaps; it is not an additional vote on authoritative direct frames.
        if previous_gray is not None:
            for tag_id, (quad, age) in list(active.items()):
                instance_locked = tag_id in locked_ids
                if not instance_locked and direct_wall_count >= 2:
                    continue
                if (tag_id in direct and not instance_locked) or age >= args.max_track_age:
                    continue
                tracked = track_quad(previous_gray, gray, quad, args.max_fb_error_px)
                if tracked is not None:
                    source = "lk_instance_track" if instance_locked else "lk_forward_backward"
                    if instance_locked and tag_id in direct:
                        decoded = np.asarray(
                            cache["corners_px"][direct[tag_id]], np.float32
                        )
                        distance = float(np.linalg.norm(
                            decoded.mean(axis=0) - tracked.mean(axis=0)
                        ))
                        if distance <= args.max_instance_reacquire_distance_px:
                            tracked = decoded
                            source = str(cache["detection_source"][direct[tag_id]])
                    candidates[tag_id] = (tracked, source)
                    tracked_count += 1
        next_active = {}
        for tag_id, (quad, source) in candidates.items():
            decoded_same_instance = (
                tag_id in direct and source != "lk_instance_track"
            )
            age = 0 if decoded_same_instance else active.get(tag_id, (quad, -1))[1] + 1
            if (args.minimum_id <= tag_id <= args.maximum_id
                    or tag_id in locked_ids):
                next_active[tag_id] = (quad, age)
            # Preserve every direct observation (including BaseTags); only wall
            # IDs are synthesized by LK.
            local_time = float(cache["timeline_local_time_s"][frame])
            common_time = float(cache["timeline_common_time_s"][frame])
            output["frame_index"].append(frame)
            output["local_time_s"].append(local_time)
            output["common_time_s"].append(common_time)
            output["tag_id"].append(tag_id)
            output["corners_px"].append(quad)
            output["rays_camera"].append(converter(quad).astype(np.float32))
            output["area_px2"].append(abs(float(cv2.contourArea(quad))))
            output["center_px"].append(quad.mean(axis=0))
            output["detection_source"].append(source)
        active = next_active
        previous_gray = gray
        frame += 1
    capture.release()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        np.savez_compressed(
            handle,
            timeline_frame_index=cache["timeline_frame_index"],
            timeline_local_time_s=cache["timeline_local_time_s"],
            timeline_common_time_s=cache["timeline_common_time_s"],
            frame_index=np.asarray(output["frame_index"], np.int32),
            local_time_s=np.asarray(output["local_time_s"], np.float64),
            common_time_s=np.asarray(output["common_time_s"], np.float64),
            tag_id=np.asarray(output["tag_id"], np.int32),
            corners_px=np.asarray(output["corners_px"], np.float32).reshape(-1, 4, 2),
            rays_camera=np.asarray(output["rays_camera"], np.float32).reshape(-1, 4, 3),
            area_px2=np.asarray(output["area_px2"], np.float32),
            center_px=np.asarray(output["center_px"], np.float32).reshape(-1, 2),
            detection_source=np.asarray(output["detection_source"], dtype="U24"),
        )
    input_meta = json.loads(args.input_cache.with_suffix(".json").read_text())
    metadata = dict(input_meta)
    metadata.update({
        "schema_version": "fisheye-apriltag-observation-cache/1.1-lk",
        "cache": str(args.output.resolve()),
        "parent_cache": str(args.input_cache.resolve()),
        "parent_cache_sha256": sha256(args.input_cache),
        "tracking": {
            "method": "pyramidal LK forward/backward on raw fisheye pixels",
            "max_track_age_frames": args.max_track_age,
            "max_fb_error_px": args.max_fb_error_px,
            "tracked_observation_count": tracked_count,
            "pose_interpolation_used": False,
            "instance_track_ids": sorted(set(args.instance_track_id)),
            "max_instance_reacquire_distance_px": args.max_instance_reacquire_distance_px,
        },
        "detection_count": len(output["tag_id"]),
        "scan_policy": "direct detector plus bounded forward/backward LK image measurements",
    })
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
