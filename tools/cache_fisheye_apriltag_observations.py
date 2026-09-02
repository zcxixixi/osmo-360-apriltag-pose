#!/usr/bin/env python3
"""Scan a raw fisheye stream once and cache AprilTag corners and unit bearings."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import cv2
import h5py
import numpy as np

from osmo360.pipeline.temporal_apriltag import (
    grayscale_scout_and_refine,
    redetect_rois,
    track_quads_forward_backward,
)
from osmo360.pipeline.ffmpeg_gray_pipe import (
    FFmpegGrayPipe,
    probe_video_stream,
)
from osmo360.localization.raw_fisheye_world_pose import (
    detect_rectified_tags,
    make_kannala_brandt_ray_converter,
    make_kannala_brandt_rectified_maps,
    make_ray_converter,
    make_rectified_maps,
    make_x5_offset_ray_converter,
    make_x5_rectified_maps,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--calibration", type=Path)
    source.add_argument("--x5-offset")
    parser.add_argument("--camera-serial")
    parser.add_argument(
        "--calibration-bundle-sha256",
        help="identity of the complete dual-lens calibration used by the caller",
    )
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
    parser.add_argument(
        "--rectified-detection-policy",
        choices=("always", "adaptive"),
        default="always",
        help="adaptive skips tangent views only when direct detections satisfy the configured gate",
    )
    parser.add_argument(
        "--rectified-min-direct-tags",
        type=int,
        default=0,
        help="adaptive gate: minimum unique direct Tag count",
    )
    parser.add_argument(
        "--rectified-required-id",
        type=int,
        action="append",
        default=[],
        help="adaptive gate: direct detection must contain every repeated required ID",
    )
    parser.add_argument("--rectified-view-size", type=int, default=960)
    parser.add_argument("--rectification-radial-model", choices=("metric", "stitch"),
                        default="stitch")
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="legacy detector stride, or temporal-cache output stride")
    parser.add_argument(
        "--decode-stride",
        type=int,
        default=1,
        help="retrieve grayscale images only every N source frames; intermediate frames are decoder-grabbed",
    )
    decoder = parser.add_mutually_exclusive_group()
    decoder.add_argument(
        "--native-grayscale-decode",
        action="store_true",
        help="request the decoder luma plane directly instead of converting a BGR frame",
    )
    decoder.add_argument(
        "--ffmpeg-gray-pipe",
        action="store_true",
        help="stream selected luma planes from the verified FFmpeg runtime into Python",
    )
    parser.add_argument(
        "--optical-flow-scale",
        type=float,
        default=1.0,
        help="run pyramidal LK on this image scale while retaining full-resolution corners",
    )
    parser.add_argument("--optical-flow-window-size", type=int, default=31)
    parser.add_argument("--optical-flow-max-level", type=int, default=4)
    parser.add_argument("--optical-flow-max-iterations", type=int, default=30)
    parser.add_argument(
        "--timeline-h5",
        type=Path,
        help="optional InstaUMI H5 supplying exact aligned frame timestamps",
    )
    parser.add_argument(
        "--timeline-camera",
        choices=("left", "right"),
        help="camera timeline to read from --timeline-h5",
    )
    parser.add_argument(
        "--instaumi-rear-calibration",
        action="store_true",
        help="for stream 0, use the explicit rear-lens KB calibration in --timeline-h5",
    )
    parser.add_argument(
        "--temporal-tracking",
        action="store_true",
        help="track decoded grayscale corners on every frame and redetect only when scheduled",
    )
    parser.add_argument("--global-scout-interval-frames", type=int, default=30)
    parser.add_argument("--global-scout-scale", type=float, default=0.35)
    parser.add_argument("--local-redetect-interval-frames", type=int, default=12)
    parser.add_argument("--rectified-recovery-interval-frames", type=int, default=30)
    parser.add_argument("--max-track-age-frames", type=int, default=60)
    parser.add_argument("--max-flow-forward-backward-error-px", type=float, default=1.5)
    parser.add_argument(
        "--forward-backward-check-interval-frames",
        type=int,
        default=1,
        help="run the backward LK validation at this source-frame interval",
    )
    parser.add_argument("--max-reacquire-distance-px", type=float, default=160.0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int,
                        help="inclusive final frame for detection; later frames remain on the timeline")
    parser.add_argument(
        "--seek-to-start",
        action="store_true",
        help="seek directly to --start-frame before decoding (for resumable chunks)",
    )
    parser.add_argument(
        "--stop-after-end-frame",
        action="store_true",
        help="stop decoding after --end-frame instead of retaining the remaining timeline",
    )
    parser.add_argument(
        "--video-sha256",
        help="precomputed source hash; avoids hashing a large MP4 once per chunk",
    )
    parser.add_argument(
        "--opencv-threads",
        type=int,
        default=0,
        help="limit OpenCV worker threads; zero keeps the OpenCV default",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def should_run_rectified(
    policy: str,
    direct_ids: set[int],
    *,
    minimum_direct_tags: int,
    required_ids: set[int],
) -> bool:
    if policy == "always":
        return True
    return len(direct_ids) < minimum_direct_tags or not required_ids.issubset(direct_ids)


def ignored_trailing_video_frames(
    *,
    timestamp_count: int,
    video_frame_count: int,
    end_frame: int | None,
    stop_after_end_frame: bool,
) -> int:
    """Validate H5 coverage and return safely ignored trailing video frames.

    An aligned InstaUMI export can retain a short encoded tail in its raw lens
    MP4 while the H5 timeline and preview stop at the intended common range.
    Chunked processing is safe in that case only when its explicit inclusive
    end frame is covered by the H5 timestamps. Missing video frames and any
    unbounded decode remain hard failures.
    """
    if timestamp_count == video_frame_count:
        return 0
    if video_frame_count < timestamp_count:
        raise RuntimeError(
            "InstaUMI timestamp/video frame count mismatch: "
            f"{timestamp_count} != {video_frame_count}"
        )
    if (
        not stop_after_end_frame
        or end_frame is None
        or end_frame >= timestamp_count
    ):
        raise RuntimeError(
            "InstaUMI timestamp/video frame count mismatch outside the bounded "
            f"H5 timeline: {timestamp_count} != {video_frame_count}"
        )
    return video_frame_count - timestamp_count


def merge_temporal_detections(
    current: dict[int, tuple[np.ndarray, int, str]],
    decoded: dict[int, np.ndarray],
    source: str,
    *,
    max_reacquire_distance_px: float,
) -> None:
    """Refresh tracks without jumping a healthy ID to a far duplicate instance."""
    for tag_id, quad in decoded.items():
        if tag_id in current:
            predicted = current[tag_id][0]
            distance = float(np.linalg.norm(quad.mean(axis=0) - predicted.mean(axis=0)))
            if distance > max_reacquire_distance_px:
                continue
        current[tag_id] = (np.asarray(quad, np.float32), 0, source)


def main() -> int:
    args = parse_args()
    if args.clock_slope <= 0:
        raise ValueError("--clock-slope must be positive")
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")
    if args.decode_stride <= 0:
        raise ValueError("--decode-stride must be positive")
    if args.frame_stride % args.decode_stride:
        raise ValueError("--frame-stride must be a multiple of --decode-stride")
    if args.ffmpeg_gray_pipe and not args.temporal_tracking:
        raise ValueError("--ffmpeg-gray-pipe requires --temporal-tracking")
    if not 0.25 <= args.optical_flow_scale <= 1.0:
        raise ValueError("--optical-flow-scale must be between 0.25 and 1.0")
    if args.optical_flow_window_size < 9 or not args.optical_flow_window_size % 2:
        raise ValueError("--optical-flow-window-size must be odd and >= 9")
    if args.optical_flow_max_level < 0 or args.optical_flow_max_iterations <= 0:
        raise ValueError("invalid optical-flow pyramid/iteration limit")
    if bool(args.timeline_h5) != bool(args.timeline_camera):
        raise ValueError("--timeline-h5 and --timeline-camera must be supplied together")
    if args.start_frame < 0 or (args.end_frame is not None and args.end_frame < args.start_frame):
        raise ValueError("invalid detection frame range")
    if args.stop_after_end_frame and args.end_frame is None:
        raise ValueError("--stop-after-end-frame requires --end-frame")
    if args.seek_to_start and args.start_frame == 0:
        args.seek_to_start = False
    if args.opencv_threads < 0:
        raise ValueError("--opencv-threads cannot be negative")
    if args.rectified_min_direct_tags < 0:
        raise ValueError("--rectified-min-direct-tags cannot be negative")
    if args.rectified_detection_policy == "adaptive" and not args.rectified_detection:
        raise ValueError("adaptive rectification requires --rectified-detection")
    if args.video_sha256 is not None and not re.fullmatch(
        r"[0-9a-f]{64}", args.video_sha256
    ):
        raise ValueError("--video-sha256 must be a lowercase SHA-256 digest")
    for name in (
        "global_scout_interval_frames",
        "local_redetect_interval_frames",
        "rectified_recovery_interval_frames",
        "max_track_age_frames",
        "forward_backward_check_interval_frames",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not 0.1 <= args.global_scout_scale <= 1.0:
        raise ValueError("--global-scout-scale must be between 0.1 and 1.0")
    if args.max_flow_forward_backward_error_px <= 0 or args.max_reacquire_distance_px <= 0:
        raise ValueError("temporal tracking error/distance gates must be positive")
    if args.opencv_threads:
        cv2.setNumThreads(args.opencv_threads)
    if hasattr(cv2, "setLogLevel"):
        cv2.setLogLevel(2)
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
    elif args.instaumi_rear_calibration:
        if args.timeline_h5 is None or args.timeline_camera is None or args.stream != 0:
            raise ValueError(
                "--instaumi-rear-calibration requires --timeline-h5, "
                "--timeline-camera and --stream 0"
            )
        with h5py.File(args.timeline_h5, "r") as handle:
            raw_calibration = handle["/calib/calibration_full.json"][()]
        if isinstance(raw_calibration, bytes):
            raw_calibration = raw_calibration.decode("utf-8")
        payload = json.loads(str(raw_calibration))
        camera = payload["cameras"][args.timeline_camera]
        distortion = camera["distortion"]
        explicit_intrinsics = {
            **camera["intrinsics"],
            "coefficients": distortion["coefficients"],
        }
        rig_from_camera = np.asarray(
            payload["extrinsics"][f"T_rig_camera_{args.timeline_camera}"], dtype=float
        )
        calibration_width, calibration_height = map(int, camera["image_size"])
        converter, scaled = make_kannala_brandt_ray_converter(
            explicit_intrinsics,
            rig_from_camera,
            calibration_width=calibration_width,
            calibration_height=calibration_height,
            source_width=args.source_width,
            source_height=args.source_height,
        )
        rectified_maps = (
            make_kannala_brandt_rectified_maps(
                explicit_intrinsics,
                rig_from_camera,
                calibration_width=calibration_width,
                calibration_height=calibration_height,
                source_width=args.source_width,
                source_height=args.source_height,
                view_size=args.rectified_view_size,
            )
            if args.rectified_detection
            else []
        )
        x5_offset_record = args.x5_offset
        calibration_source = (
            f"instaumi_h5:{args.timeline_h5.resolve()}"
            f"#/calib/calibration_full.json/cameras/{args.timeline_camera}"
        )
        calibration_sha256 = hashlib.sha256(
            str(raw_calibration).encode("utf-8")
        ).hexdigest()
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
    capture: cv2.VideoCapture | None = None
    gray_pipe: FFmpegGrayPipe | None = None
    decoder_provenance: dict[str, object]
    if args.ffmpeg_gray_pipe:
        stream_info = probe_video_stream(video)
        fps = stream_info.fps
        source_frame_count = stream_info.frame_count
        if (stream_info.width, stream_info.height) != (
            args.source_width, args.source_height
        ):
            raise RuntimeError(
                "ffprobe/video geometry does not match the manifest: "
                f"{stream_info.width}x{stream_info.height} != "
                f"{args.source_width}x{args.source_height}"
            )
        decoder_provenance = {
            "decoder_transport": "ffmpeg_rawvideo_pipe",
            "pixel_format": "gray8_luma",
        }
    else:
        if args.opencv_threads and hasattr(cv2, "CAP_PROP_N_THREADS"):
            capture = cv2.VideoCapture(
                str(video),
                cv2.CAP_FFMPEG,
                [cv2.CAP_PROP_N_THREADS, args.opencv_threads],
            )
        else:
            capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open {video}")
        if args.native_grayscale_decode and not capture.set(cv2.CAP_PROP_CONVERT_RGB, 0):
            raise RuntimeError("video backend does not support native grayscale decode")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        decoder_provenance = {
            "decoder_transport": "opencv_videocapture",
            "pixel_format": (
                "native_decoder_luma" if args.native_grayscale_decode else "bgr_to_gray"
            ),
        }
    if fps <= 0 or source_frame_count <= 0:
        raise RuntimeError(f"invalid video timing metadata: fps={fps}, frames={source_frame_count}")
    exact_timestamp_s: np.ndarray | None = None
    ignored_video_tail_frames = 0
    if args.timeline_h5 is not None:
        timeline_h5 = args.timeline_h5.resolve(strict=True)
        with h5py.File(timeline_h5, "r") as handle:
            key = f"/sensor/camera/{args.timeline_camera}/timestamp_ns"
            if key not in handle:
                raise RuntimeError(f"InstaUMI timestamp dataset is missing: {key}")
            exact_timestamp_s = np.asarray(handle[key], dtype=np.float64) / 1e9
        ignored_video_tail_frames = ignored_trailing_video_frames(
            timestamp_count=len(exact_timestamp_s),
            video_frame_count=source_frame_count,
            end_frame=args.end_frame,
            stop_after_end_frame=args.stop_after_end_frame,
        )
        if exact_timestamp_s[0] != 0 or np.any(np.diff(exact_timestamp_s) <= 0):
            raise RuntimeError("InstaUMI timestamps must start at zero and increase")
    frame = 0
    if args.seek_to_start:
        if args.ffmpeg_gray_pipe:
            frame = args.start_frame
        else:
            assert capture is not None
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame):
                raise RuntimeError(f"decoder cannot seek to frame {args.start_frame}: {video}")
            frame = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
            if frame != args.start_frame:
                raise RuntimeError(
                    f"decoder seek was not exact: requested {args.start_frame}, got {frame}"
                )
    if args.ffmpeg_gray_pipe:
        image_start_frame = args.start_frame
        image_end_frame = (
            source_frame_count - 1
            if args.end_frame is None
            else min(args.end_frame, source_frame_count - 1)
        )
        image_stride = args.decode_stride if args.temporal_tracking else 1
        gray_pipe = FFmpegGrayPipe(
            video,
            width=args.source_width,
            height=args.source_height,
            fps=fps,
            start_frame=image_start_frame,
            end_frame=image_end_frame,
            frame_stride=image_stride,
            decoder_threads=args.opencv_threads,
        )
        decoder_provenance = gray_pipe.provenance
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
    detector_frame_count = 0
    rectified_detector_frame_count = 0
    adaptive_rectification_skipped_frames = 0
    global_scout_frame_count = 0
    global_scout_decoded_count = 0
    local_redetect_frame_count = 0
    local_redetect_decoded_count = 0
    flow_attempted_tag_count = 0
    flow_accepted_tag_count = 0
    flow_rejected_status_count = 0
    flow_rejected_forward_backward_count = 0
    flow_rejected_geometry_count = 0
    forward_backward_check_frame_count = 0
    temporal_output_observation_count = 0
    active_tracks: dict[int, tuple[np.ndarray, int, str]] = {}
    previous_gray: np.ndarray | None = None
    previous_image_frame: int | None = None
    first_decoded_frame: int | None = None
    last_decoded_frame: int | None = None
    retrieved_frame_count = 0
    while True:
        if gray_pipe is not None and frame >= source_frame_count:
            break
        if args.stop_after_end_frame and frame > args.end_frame:
            break
        in_detection_range = frame >= args.start_frame and (
            args.end_frame is None or frame <= args.end_frame
        )
        retrieve_image = not args.temporal_tracking or (
            in_detection_range and (frame - args.start_frame) % args.decode_stride == 0
        )
        if retrieve_image and gray_pipe is not None:
            image = gray_pipe.read()
            ok = True
        elif retrieve_image:
            assert capture is not None
            ok, image = capture.read()
        else:
            ok = True if gray_pipe is not None else capture.grab()
            image = None
        if not ok:
            break
        if first_decoded_frame is None:
            first_decoded_frame = frame
        last_decoded_frame = frame
        local_time = (
            float(exact_timestamp_s[frame])
            if exact_timestamp_s is not None
            else frame / fps
        )
        common_time = (local_time - args.clock_intercept_s) / args.clock_slope
        timeline_frame.append(frame)
        timeline_local.append(local_time)
        timeline_common.append(common_time)
        if not retrieve_image:
            frame += 1
            continue
        retrieved_frame_count += 1
        assert image is not None
        if image.ndim == 2:
            if image.shape != (args.source_height, args.source_width):
                raise RuntimeError(
                    f"native decoder returned unexpected luma shape {image.shape}"
                )
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        flow_gray = (
            gray
            if args.optical_flow_scale == 1.0
            else cv2.resize(
                gray,
                None,
                fx=args.optical_flow_scale,
                fy=args.optical_flow_scale,
                interpolation=cv2.INTER_AREA,
            )
        )
        detections: list[tuple[int, np.ndarray, str]] = []
        if args.temporal_tracking and in_detection_range:
            relative_frame = frame - args.start_frame
            current: dict[int, tuple[np.ndarray, int, str]] = {}
            if previous_gray is not None and active_tracks:
                verify_forward_backward = (
                    relative_frame % args.forward_backward_check_interval_frames == 0
                )
                if verify_forward_backward:
                    forward_backward_check_frame_count += 1
                tracked, flow_audit = track_quads_forward_backward(
                    previous_gray,
                    flow_gray,
                    {
                        tag_id: value[0] * args.optical_flow_scale
                        for tag_id, value in active_tracks.items()
                    },
                    max_forward_backward_error_px=(
                        args.max_flow_forward_backward_error_px * args.optical_flow_scale
                    ),
                    minimum_area_px2=100.0 * args.optical_flow_scale**2,
                    verify_forward_backward=verify_forward_backward,
                    window_size=args.optical_flow_window_size,
                    max_level=args.optical_flow_max_level,
                    max_iterations=args.optical_flow_max_iterations,
                )
                tracked = {
                    tag_id: quad / args.optical_flow_scale
                    for tag_id, quad in tracked.items()
                }
                flow_attempted_tag_count += flow_audit.attempted_tags
                flow_accepted_tag_count += flow_audit.accepted_tags
                flow_rejected_status_count += flow_audit.rejected_status
                flow_rejected_forward_backward_count += flow_audit.rejected_forward_backward
                flow_rejected_geometry_count += flow_audit.rejected_geometry
                for tag_id, quad in tracked.items():
                    age = active_tracks[tag_id][1] + (
                        frame - previous_image_frame
                        if previous_image_frame is not None
                        else args.decode_stride
                    )
                    if age <= args.max_track_age_frames:
                        current[tag_id] = (quad, age, "lk_forward_backward")

            if current and relative_frame % args.local_redetect_interval_frames == 0:
                local_redetect_frame_count += 1
                local = redetect_rois(
                    gray,
                    detector,
                    {tag_id: value[0] for tag_id, value in current.items()},
                )
                local_redetect_decoded_count += len(local)
                merge_temporal_detections(
                    current,
                    local,
                    "local_roi_gray",
                    max_reacquire_distance_px=args.max_reacquire_distance_px,
                )

            run_global_scout = (
                relative_frame == 0
                or relative_frame % args.global_scout_interval_frames == 0
            )
            if run_global_scout:
                global_scout_frame_count += 1
                global_detections = grayscale_scout_and_refine(
                    gray,
                    detector,
                    scale=args.global_scout_scale,
                    predicted={tag_id: value[0] for tag_id, value in current.items()},
                )
                global_scout_decoded_count += len(global_detections)
                merge_temporal_detections(
                    current,
                    global_detections,
                    "global_scout_roi_gray",
                    max_reacquire_distance_px=args.max_reacquire_distance_px,
                )

            need_rectified = bool(rectified_maps) and should_run_rectified(
                args.rectified_detection_policy,
                set(current),
                minimum_direct_tags=args.rectified_min_direct_tags,
                required_ids=set(args.rectified_required_id),
            )
            rectified_due = (
                relative_frame % args.rectified_recovery_interval_frames == 0
            )
            if need_rectified and rectified_due:
                detector_frame_count += 1
                rectified_detector_frame_count += 1
                rectified = dict(detect_rectified_tags(gray, detector, rectified_maps))
                merge_temporal_detections(
                    current,
                    rectified,
                    "rectified_tangent_gray",
                    max_reacquire_distance_px=args.max_reacquire_distance_px,
                )
            elif rectified_due and rectified_maps:
                adaptive_rectification_skipped_frames += 1

            active_tracks = current
            if relative_frame % args.frame_stride == 0:
                detections = [
                    (tag_id, value[0], value[2])
                    for tag_id, value in current.items()
                ]
                temporal_output_observation_count += len(detections)
        elif in_detection_range and (frame - args.start_frame) % args.frame_stride == 0:
            detector_frame_count += 1
            quads, ids, _ = detector.detectMarkers(gray)
            direct = {} if ids is None else {
                int(tag_id): np.asarray(quad, dtype=np.float32).reshape(4, 2)
                for quad, tag_id in zip(quads, ids.flatten())
            }
            run_rectified = bool(rectified_maps) and should_run_rectified(
                args.rectified_detection_policy,
                set(direct),
                minimum_direct_tags=args.rectified_min_direct_tags,
                required_ids=set(args.rectified_required_id),
            )
            if run_rectified:
                rectified_detector_frame_count += 1
                # A raw fisheye image bends the physical edges of a large Tag.
                # Fitting a straight quadrilateral directly to those curves
                # biases the corners and, in turn, planar-PnP depth.  Tangent
                # views restore straight edges; their corners are then mapped
                # back to raw pixels before the factory ray conversion.
                rectified = dict(detect_rectified_tags(gray, detector, rectified_maps))
                direct.update(rectified)
                rectified_ids = set(rectified)
            else:
                rectified_ids = set()
                if rectified_maps:
                    adaptive_rectification_skipped_frames += 1
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
        previous_gray = flow_gray if args.temporal_tracking else None
        previous_image_frame = frame if args.temporal_tracking else None
        frame += 1
    if gray_pipe is not None:
        gray_pipe.close()
    elif capture is not None:
        capture.release()
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
    video_digest = args.video_sha256 or sha256(video)
    processing_signature = {
        "temporal_tracking": args.temporal_tracking,
        "output_stride_frames": args.frame_stride,
        "decode_stride_frames": args.decode_stride,
        "timestamp_source": (
            f"instaumi_h5:/sensor/camera/{args.timeline_camera}/timestamp_ns"
            if exact_timestamp_s is not None
            else "video_nominal_fps"
        ),
        "timeline_h5_sha256": (
            sha256(args.timeline_h5.resolve()) if args.timeline_h5 is not None else None
        ),
        "calibration_bundle_sha256": args.calibration_bundle_sha256,
        "native_grayscale_decode": args.native_grayscale_decode,
        "ffmpeg_gray_pipe": args.ffmpeg_gray_pipe,
        "decoder_transport": decoder_provenance["decoder_transport"],
        "decoder_pixel_format": decoder_provenance["pixel_format"],
        "optical_flow_scale": args.optical_flow_scale,
        "forward_backward_check_interval_frames": (
            args.forward_backward_check_interval_frames
        ),
        "optical_flow_window_size": args.optical_flow_window_size,
        "optical_flow_max_level": args.optical_flow_max_level,
        "optical_flow_max_iterations": args.optical_flow_max_iterations,
        "rectified_detection": args.rectified_detection,
        "rectified_detection_policy": args.rectified_detection_policy,
        "rectified_min_direct_tags": args.rectified_min_direct_tags,
        "rectified_required_ids": sorted(set(args.rectified_required_id)),
        "rectified_view_size": (
            args.rectified_view_size if args.rectified_detection else None
        ),
        "global_scout_interval_frames": args.global_scout_interval_frames,
        "global_scout_scale": args.global_scout_scale,
        "local_redetect_interval_frames": args.local_redetect_interval_frames,
        "rectified_recovery_interval_frames": args.rectified_recovery_interval_frames,
        "max_track_age_frames": args.max_track_age_frames,
        "max_flow_forward_backward_error_px": args.max_flow_forward_backward_error_px,
        "max_reacquire_distance_px": args.max_reacquire_distance_px,
    }
    metadata = {
        "schema_version": (
            "fisheye-apriltag-observation-cache/1.3-temporal"
            if args.temporal_tracking
            else "fisheye-apriltag-observation-cache/1.0"
        ),
        "video": str(video),
        "video_sha256": video_digest,
        "calibration": calibration_source,
        "calibration_sha256": calibration_sha256,
        "x5_offset": x5_offset_record,
        "camera_serial": args.camera_serial or scaled.get("serial"),
        "stream": args.stream,
        "source_size": [args.source_width, args.source_height],
        "fps": fps,
        "frame_count": source_frame_count,
        "timeline_frame_count": (
            len(exact_timestamp_s) if exact_timestamp_s is not None else source_frame_count
        ),
        "ignored_trailing_video_frames": ignored_video_tail_frames,
        "decoded_frame_count": len(timeline_frame),
        "retrieved_image_frame_count": retrieved_frame_count,
        "decoded_frame_range": [first_decoded_frame, last_decoded_frame],
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
        "rectified_detection_policy": args.rectified_detection_policy,
        "rectified_min_direct_tags": args.rectified_min_direct_tags,
        "rectified_required_ids": sorted(set(args.rectified_required_id)),
        "rectified_view_size": args.rectified_view_size if args.rectified_detection else None,
        "rectification_radial_model": (
            args.rectification_radial_model if args.rectified_detection else None
        ),
        "frame_stride": args.frame_stride,
        "decode_stride": args.decode_stride,
        "native_grayscale_decode": args.native_grayscale_decode,
        "ffmpeg_gray_pipe": args.ffmpeg_gray_pipe,
        "decoder": decoder_provenance,
        "optical_flow_scale": args.optical_flow_scale,
        "forward_backward_check_interval_frames": (
            args.forward_backward_check_interval_frames
        ),
        "optical_flow_window_size": args.optical_flow_window_size,
        "optical_flow_max_level": args.optical_flow_max_level,
        "optical_flow_max_iterations": args.optical_flow_max_iterations,
        "timestamp_source": (
            f"instaumi_h5:/sensor/camera/{args.timeline_camera}/timestamp_ns"
            if exact_timestamp_s is not None
            else "video_nominal_fps"
        ),
        "timeline_h5_sha256": processing_signature["timeline_h5_sha256"],
        "processing_signature": processing_signature,
        "temporal_tracking": args.temporal_tracking,
        "tracking": (
            {
                "method": "pyramidal LK forward/backward on raw fisheye pixels",
                "integrated_one_pass": True,
                "pose_interpolation_used": False,
                "grayscale_detection": True,
                "output_stride_frames": args.frame_stride,
                "decode_stride_frames": args.decode_stride,
                "native_grayscale_decode": args.native_grayscale_decode,
                "ffmpeg_gray_pipe": args.ffmpeg_gray_pipe,
                "decoder_transport": decoder_provenance["decoder_transport"],
                "optical_flow_scale": args.optical_flow_scale,
                "forward_backward_check_interval_frames": (
                    args.forward_backward_check_interval_frames
                ),
                "forward_backward_check_frame_count": forward_backward_check_frame_count,
                "optical_flow_window_size": args.optical_flow_window_size,
                "optical_flow_max_level": args.optical_flow_max_level,
                "optical_flow_max_iterations": args.optical_flow_max_iterations,
                "global_scout_interval_frames": args.global_scout_interval_frames,
                "global_scout_scale": args.global_scout_scale,
                "local_redetect_interval_frames": args.local_redetect_interval_frames,
                "rectified_recovery_interval_frames": args.rectified_recovery_interval_frames,
                "max_track_age_frames": args.max_track_age_frames,
                "max_forward_backward_error_px": args.max_flow_forward_backward_error_px,
                "max_reacquire_distance_px": args.max_reacquire_distance_px,
                "flow_attempted_tag_count": flow_attempted_tag_count,
                "flow_accepted_tag_count": flow_accepted_tag_count,
                "flow_rejected_status_count": flow_rejected_status_count,
                "flow_rejected_forward_backward_count": flow_rejected_forward_backward_count,
                "flow_rejected_geometry_count": flow_rejected_geometry_count,
                "global_scout_frame_count": global_scout_frame_count,
                "global_scout_decoded_count": global_scout_decoded_count,
                "local_redetect_frame_count": local_redetect_frame_count,
                "local_redetect_decoded_count": local_redetect_decoded_count,
                "tracked_output_observation_count": temporal_output_observation_count,
            }
            if args.temporal_tracking
            else None
        ),
        "detection_frame_range": [args.start_frame, args.end_frame],
        "detection_source_counts": {
            source: detection_sources.count(source) for source in sorted(set(detection_sources))
        },
        "detector_frame_count": detector_frame_count,
        "rectified_detector_frame_count": rectified_detector_frame_count,
        "adaptive_rectification_skipped_frames": adaptive_rectification_skipped_frames,
        "corner_order": "opencv_aruco_apriltag_canonical",
        "ray_frame": scaled.get(
            "ray_frame", "fisheye optical centre, panorama OpenCV axes"
        ),
        "scan_policy": (
            "single decoder pass: sampled grayscale LK, local ROI redetection, low-resolution global scout, bounded rectified recovery"
            if args.temporal_tracking
            else "bounded resumable chunk; every frame in the chunk decoded once"
            if args.stop_after_end_frame
            else "every decoded frame exactly once; direct detector only"
        ),
        "opencv_threads": args.opencv_threads or None,
        "cache": str(args.output.resolve()),
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_temporary = metadata_path.with_suffix(".json.tmp")
    metadata_temporary.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    metadata_temporary.replace(metadata_path)
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
