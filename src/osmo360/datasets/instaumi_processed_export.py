"""Export synchronized InstaUMI camera trajectories and gripper opening CSVs."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from osmo360.datasets.world_flu import (
    WORLD_FLU_REVISION,
    derive_world_flu_transform,
    transform_trajectory_rows,
)
from osmo360.gripper_markers import (
    BLACK_ON_YELLOW_FAMILY,
    YELLOW_ON_BLACK_FAMILY,
    detect_bgr_gripper_markers,
    marker_signature,
)
from osmo360.paths import ROOT
from osmo360.pipeline.four_mp4 import PIPELINE_REVISION
from osmo360.pipeline.instaumi import is_instaumi_dataset, load_instaumi_config
from osmo360.pipeline.manifest import ManifestError, confined_path, sha256
from osmo360.visualization.render_gripper_force_angle_demo import (
    FrameObservation,
    apply_one_sided_opening_fallback,
    bounded_interpolate,
    included_jaw_angle,
    nanmedian_filter,
    opening_angles,
)


EXPORT_REVISION = "instaumi-csv-v6-dual-colour-gripper"
LEGACY_EXPORT_REVISION = "instaumi-csv-v1"
CSV_NAMES = ("trajectory.csv", "gripper.csv", "processed.csv", "metadata.csv")
PIPELINE_FINAL_ENTRIES = frozenset({"manifest.lock.json", "status.json", "pairs"})
PROFILE_PATH = (
    ROOT
    / "config/rig_revisions/instaumi_gripper_signal_20260903_r6.json"
)
SIDES = ("left", "right")


@dataclass(frozen=True)
class SideProfile:
    side: str
    calibration_source_camera_serial: str
    base_tag_id: int
    angle_revision_id: str
    angle_revision_sha256: str
    detector_mode: str
    included_angle_range: tuple[float, float]
    dot_selection: str
    closed_reference_deg: float
    angle_hardware: dict[str, Any]
    black_gap_range_px: tuple[float, float]
    black_gap_slope_deg_per_px: float
    black_gap_intercept_deg: float
    black_pair_family_min_bilateral_ratio: float | None
    width_angle_deg: np.ndarray
    width_m: np.ndarray


@dataclass(frozen=True)
class SideInput:
    side: str
    video: Path
    video_kind: str
    timestamp_s: np.ndarray
    marker_cache: Path | None = None


@dataclass(frozen=True)
class SideSignal:
    opening_deg: np.ndarray
    width_m: np.ndarray
    state: np.ndarray
    timestamp_s: np.ndarray
    source_frame: np.ndarray
    measured_ratio: float
    available_ratio: float
    marker_family: str
    yellow_bilateral_ratio: float
    black_bilateral_ratio: float


def _reference_payload(reference: dict[str, Any], *, field: str) -> tuple[Path, dict[str, Any], str]:
    if not isinstance(reference, dict) or not {"path", "sha256"}.issubset(reference):
        raise ManifestError(f"{field} must contain path and sha256")
    path = (ROOT / str(reference["path"])).resolve()
    if not path.is_file():
        raise ManifestError(f"{field} is missing: {path}")
    actual = sha256(path)
    if actual != reference["sha256"]:
        raise ManifestError(
            f"{field} SHA-256 mismatch: expected {reference['sha256']}, got {actual}"
        )
    return path, json.loads(path.read_text(encoding="utf-8")), actual


def load_profile(path: Path = PROFILE_PATH) -> tuple[dict[str, Any], dict[str, SideProfile]]:
    source = path.resolve(strict=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "instaumi-gripper-signal-revision/1.0":
        raise ManifestError("unsupported InstaUMI gripper signal profile")
    if payload.get("training_ready") is not False:
        raise ManifestError("gripper signal profile must retain diagnostic training_ready=false")
    maximum_gap_s = float(payload.get("maximum_recovery_gap_s", -1))
    if not math.isclose(maximum_gap_s, 0.25, abs_tol=1e-12):
        raise ManifestError("gripper signal profile must use a 0.25 s recovery limit")
    if set(payload.get("sides", {})) != set(SIDES):
        raise ManifestError("gripper signal profile must define exactly left and right")
    marker_detector = payload.get("marker_detector", {})
    if (
        payload.get("prefer_fused_trajectory_marker_cache", False)
        and "black_pair_family_min_bilateral_ratio" in marker_detector
    ):
        if marker_detector.get("cache_signature") != marker_signature():
            raise ManifestError(
                "gripper signal profile cache signature does not match the implementation"
            )
    detector_range_override = marker_detector.get("included_angle_range_deg")
    dot_selection_override = marker_detector.get("dot_selection")
    raw_black_pair_gate = marker_detector.get(
        "black_pair_family_min_bilateral_ratio"
    )
    black_pair_gate = (
        None if raw_black_pair_gate is None else float(raw_black_pair_gate)
    )
    if black_pair_gate is not None and not 0.5 <= black_pair_gate <= 1.0:
        raise ManifestError(
            "black-pair marker-family gate must require at least 50% bilateral coverage"
        )

    profiles: dict[str, SideProfile] = {}
    for side in SIDES:
        item = payload["sides"][side]
        _, angle_revision, angle_sha = _reference_payload(
            item.get("angle_revision"), field=f"sides.{side}.angle_revision"
        )
        if angle_revision.get("schema_version") != "x5-jaw-angle-revision/1.0":
            raise ManifestError(f"{side} uses an invalid jaw-angle revision")
        detector = angle_revision.get("detector", {})
        if detector.get("algorithm") != "three_yellow_pad_dot_centroids":
            raise ManifestError(f"{side} jaw-angle revision must use physical marker triads")
        hardware_key = str(item.get("angle_hardware_key", ""))
        angle_hardware = angle_revision.get("hardware", {}).get(hardware_key)
        if not isinstance(angle_hardware, dict):
            raise ManifestError(f"{side} jaw-angle hardware binding is missing")
        if int(angle_hardware.get("base_tag_id", -1)) != int(item.get("base_tag_id", -2)):
            raise ManifestError(f"{side} jaw-angle BaseTag binding mismatch")

        _, width_hardware, _ = _reference_payload(
            item.get("width_calibration"), field=f"sides.{side}.width_calibration"
        )
        robot_role = str(item["width_calibration"].get("robot_role", ""))
        robot = width_hardware.get("robots", {}).get(robot_role)
        if not isinstance(robot, dict):
            raise ManifestError(f"{side} width calibration robot binding is missing")
        width = robot.get("gripper_width_calibration", {})
        angle_values = np.asarray(width.get("angle_deg"), dtype=np.float64)
        width_values = np.asarray(width.get("width_m"), dtype=np.float64)
        if (
            angle_values.ndim != 1
            or len(angle_values) < 2
            or width_values.shape != angle_values.shape
            or not np.isfinite(angle_values).all()
            or not np.isfinite(width_values).all()
            or np.any(np.diff(angle_values) <= 0)
            or np.any(np.diff(width_values) < 0)
            or robot.get("gripper_width_verified") is not True
        ):
            raise ManifestError(f"{side} has an invalid gripper width calibration")
        included = tuple(
            map(
                float,
                detector_range_override
                if detector_range_override is not None
                else detector.get("included_angle_range_deg", []),
            )
        )
        if len(included) != 2 or included[0] >= included[1]:
            raise ManifestError(f"{side} has an invalid included-angle range")
        black_gap_model = item.get("black_on_yellow_gap_model")
        if black_pair_gate is not None and not isinstance(black_gap_model, dict):
            raise ManifestError(f"{side} black-on-yellow gap model is missing")
        black_gap_model = black_gap_model or {
            "validated_gap_range_px": [0.0, 1.0],
            "opening_deg_coefficients_high_to_low": [1.0, 0.0],
        }
        black_gap_range = tuple(
            map(float, black_gap_model.get("validated_gap_range_px", []))
        )
        black_gap_coefficients = tuple(
            map(
                float,
                black_gap_model.get("opening_deg_coefficients_high_to_low", []),
            )
        )
        if (
            len(black_gap_range) != 2
            or black_gap_range[0] >= black_gap_range[1]
            or len(black_gap_coefficients) != 2
            or black_gap_coefficients[0] <= 0.0
        ):
            raise ManifestError(f"{side} has an invalid black-on-yellow gap model")
        profiles[side] = SideProfile(
            side=side,
            calibration_source_camera_serial=str(item["camera_serial"]),
            base_tag_id=int(item["base_tag_id"]),
            angle_revision_id=str(angle_revision["revision_id"]),
            angle_revision_sha256=angle_sha,
            detector_mode="physical-marker-triad",
            included_angle_range=included,
            dot_selection=str(
                dot_selection_override
                if dot_selection_override is not None
                else detector.get("dot_selection", "fixed-bands")
            ),
            closed_reference_deg=float(
                angle_hardware["closed_reference_included_angle_deg"]
            ),
            angle_hardware=angle_hardware,
            black_gap_range_px=black_gap_range,
            black_gap_slope_deg_per_px=black_gap_coefficients[0],
            black_gap_intercept_deg=black_gap_coefficients[1],
            black_pair_family_min_bilateral_ratio=black_pair_gate,
            width_angle_deg=angle_values,
            width_m=width_values,
        )
    return payload, profiles


def _decode_text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _safe_dataset_file(root: Path, relative: str, *, field: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ManifestError(f"{field} must be a relative path inside the dataset")
    path = confined_path(root, *value.parts, field=field)
    if not path.is_file():
        raise ManifestError(f"{field} is missing: {path}")
    return path


def _marker_cache_paths(
    root: Path,
    pair_id: str,
    profile: dict[str, Any],
) -> dict[str, Path]:
    if not profile.get("prefer_fused_trajectory_marker_cache", False):
        return {}
    index_path = confined_path(
        root,
        "final",
        PIPELINE_REVISION,
        "pairs",
        pair_id,
        "cache-index.json",
        field="four-MP4 cache index",
    )
    if not index_path.is_file():
        return {}
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"invalid four-MP4 cache index: {index_path}") from exc
    mapping = index.get("gripper_marker_cache")
    if not isinstance(mapping, dict) or not mapping:
        return {}
    expected_signature = marker_signature()
    if index.get("gripper_marker_signature") != expected_signature:
        raise ManifestError("four-MP4 gripper marker cache signature mismatch")
    h5_digest = sha256(root / "dataset.h5")
    result: dict[str, Path] = {}
    for side in SIDES:
        raw = mapping.get(side)
        if not isinstance(raw, str) or not raw:
            raise ManifestError(f"four-MP4 gripper marker cache is missing {side}")
        raw_candidate = Path(raw)
        if not raw_candidate.is_absolute():
            raw_candidate = root / raw_candidate
        try:
            relative = raw_candidate.absolute().relative_to(root)
        except ValueError:
            # Dataset directories are sometimes moved after processing.  The
            # cache layout itself is deterministic, so relocate only to the
            # exact in-dataset lens-0 path rather than trusting stale absolutes.
            candidate = confined_path(
                root,
                ".osmo-cache",
                pair_id,
                PIPELINE_REVISION,
                pair_id,
                "observations",
                side,
                "lens-0-corners.npz",
                field=f"{side} relocated gripper marker cache",
            )
        else:
            candidate = confined_path(
                root,
                *relative.parts,
                field=f"{side} gripper marker cache",
            )
        if not candidate.is_file() or candidate.is_symlink():
            raise ManifestError(f"{side} gripper marker cache is missing: {candidate}")
        metadata_path = candidate.with_suffix(".json")
        if not metadata_path.is_file() or metadata_path.is_symlink():
            raise ManifestError(f"{side} gripper marker cache sidecar is missing")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("gripper_marker_signature") != expected_signature:
            raise ManifestError(f"{side} gripper marker cache signature mismatch")
        metadata_h5_digest = metadata.get("timeline_h5_sha256")
        if metadata_h5_digest is None:
            metadata_h5_digest = metadata.get("processing_signature", {}).get(
                "timeline_h5_sha256"
            )
        if metadata_h5_digest != h5_digest:
            raise ManifestError(f"{side} gripper marker cache H5 identity mismatch")
        result[side] = candidate
    return result


def load_side_inputs(root: Path, profile: dict[str, Any]) -> dict[str, SideInput]:
    h5_path = root / "dataset.h5"
    result: dict[str, SideInput] = {}
    with h5py.File(h5_path, "r") as handle:
        metadata = json.loads(_decode_text(handle["/metadata/dataset.json"][()]))
        pair_id = str(metadata.get("dataset_id", ""))
        if not pair_id:
            raise ManifestError("InstaUMI metadata.dataset_id is missing")
        for side in SIDES:
            timestamp_ns = np.asarray(
                handle[f"/sensor/camera/{side}/timestamp_ns"], dtype=np.int64
            )
            if len(timestamp_ns) == 0 or np.any(np.diff(timestamp_ns) <= 0):
                raise ManifestError(f"{side} H5 timeline must be non-empty and increasing")
            preview_relative = _decode_text(
                handle[f"/sensor/camera/{side}/video_path"][()]
            )
            preview = None
            if bool(profile.get("prefer_h5_preview", True)):
                preview_value = Path(preview_relative)
                if preview_value.is_absolute() or ".." in preview_value.parts:
                    raise ManifestError(
                        f"{side} H5 preview must be a relative path inside the dataset"
                    )
                preview_candidate = root.joinpath(*preview_value.parts)
                if preview_candidate.exists() or preview_candidate.is_symlink():
                    preview = _safe_dataset_file(
                        root, preview_relative, field=f"{side} H5 preview"
                    )
            if preview is not None:
                video_meta = metadata.get("video", {}).get(side, {})
                expected_hash = str(video_meta.get("sha256", ""))
                if video_meta.get("path") != preview_relative or len(expected_hash) != 64:
                    raise ManifestError(f"{side} H5 preview metadata is incomplete")
                actual_hash = sha256(preview)
                if actual_hash != expected_hash:
                    raise ManifestError(
                        f"{side} H5 preview SHA-256 mismatch: expected {expected_hash}, "
                        f"got {actual_hash}"
                    )
                video = preview
                kind = "h5_sha256_verified_preview"
            else:
                title = side.title()
                video = _safe_dataset_file(
                    root,
                    f"video/{title}_{profile.get('input_lens', 'back')}.mp4",
                    field=f"{side} gripper video",
                )
                kind = "four_mp4_back_fallback"
            result[side] = SideInput(
                side=side,
                video=video,
                video_kind=kind,
                timestamp_s=timestamp_ns.astype(np.float64) / 1e9,
            )
    marker_caches = _marker_cache_paths(root, pair_id, profile)
    for side, cache in marker_caches.items():
        source = result[side]
        result[side] = SideInput(
            side=source.side,
            video=source.video,
            video_kind="fused_trajectory_yuv420_roi_cache",
            timestamp_s=source.timestamp_s,
            marker_cache=cache,
        )
    return result


def _observations_to_signal(
    source: SideInput,
    profile: SideProfile,
    observations: list[FrameObservation],
    source_frame: np.ndarray,
    *,
    fps: float,
    maximum_gap_s: float,
    yellow_bilateral_ratio: float,
    black_bilateral_ratio: float,
) -> SideSignal:
    opening, _ = opening_angles(observations, profile.closed_reference_deg)
    opening, states, _ = apply_one_sided_opening_fallback(
        observations, opening, profile.angle_hardware
    )
    direct = np.isfinite(opening)
    maximum_gap_frames = max(1, round(maximum_gap_s * fps))
    opening, recovered = bounded_interpolate(opening, maximum_gap_frames)
    opening = np.where(np.isfinite(opening), nanmedian_filter(opening), np.nan)
    states = states.astype(object)
    states[recovered] = "RECOVERED_SHORT_GAP"
    states[~np.isfinite(opening)] = "UNAVAILABLE"
    width_values = np.full(len(opening), np.nan, dtype=np.float64)
    available = np.isfinite(opening)
    width_values[available] = np.interp(
        opening[available], profile.width_angle_deg, profile.width_m
    )
    return SideSignal(
        opening_deg=opening,
        width_m=width_values,
        state=states,
        timestamp_s=source.timestamp_s,
        source_frame=source_frame,
        measured_ratio=float(direct.mean()),
        available_ratio=float(available.mean()),
        marker_family=YELLOW_ON_BLACK_FAMILY,
        yellow_bilateral_ratio=yellow_bilateral_ratio,
        black_bilateral_ratio=black_bilateral_ratio,
    )


def _black_gap_to_signal(
    source: SideInput,
    profile: SideProfile,
    gaps_px: np.ndarray,
    source_frame: np.ndarray,
    *,
    fps: float,
    maximum_gap_s: float,
    yellow_bilateral_ratio: float,
    black_bilateral_ratio: float,
) -> SideSignal:
    low, high = profile.black_gap_range_px
    valid = np.isfinite(gaps_px) & (gaps_px >= low) & (gaps_px <= high)
    opening = np.full(len(gaps_px), np.nan, dtype=np.float64)
    opening[valid] = np.clip(
        profile.black_gap_slope_deg_per_px * gaps_px[valid]
        + profile.black_gap_intercept_deg,
        0.0,
        55.0,
    )
    direct = np.isfinite(opening)
    maximum_gap_frames = max(1, round(maximum_gap_s * fps))
    opening, recovered = bounded_interpolate(opening, maximum_gap_frames)
    opening = np.where(np.isfinite(opening), nanmedian_filter(opening), np.nan)
    states = np.full(len(opening), "UNAVAILABLE", dtype=object)
    states[direct] = "MEASURED_BLACK_ON_YELLOW_PAIR"
    states[recovered] = "RECOVERED_SHORT_GAP"
    width_values = np.full(len(opening), np.nan, dtype=np.float64)
    available = np.isfinite(opening)
    width_values[available] = np.interp(
        opening[available], profile.width_angle_deg, profile.width_m
    )
    return SideSignal(
        opening_deg=opening,
        width_m=width_values,
        state=states,
        timestamp_s=source.timestamp_s,
        source_frame=source_frame,
        measured_ratio=float(direct.mean()),
        available_ratio=float(available.mean()),
        marker_family=BLACK_ON_YELLOW_FAMILY,
        yellow_bilateral_ratio=yellow_bilateral_ratio,
        black_bilateral_ratio=black_bilateral_ratio,
    )


def _cached_observations(
    source: SideInput,
    profile: SideProfile,
) -> tuple[list[FrameObservation], np.ndarray, np.ndarray, float, float]:
    assert source.marker_cache is not None
    required = {
        "gripper_frame_index",
        "gripper_left_points_px",
        "gripper_right_points_px",
        "gripper_included_angle_deg",
        "gripper_black_left_point_px",
        "gripper_black_right_point_px",
        "gripper_black_pair_gap_px",
    }
    with np.load(source.marker_cache) as cache:
        missing = sorted(required - set(cache.files))
        if missing:
            raise ManifestError(
                f"{source.side} gripper marker cache arrays are missing: {missing}"
            )
        frames = np.asarray(cache["gripper_frame_index"], dtype=np.int64)
        left_points = np.asarray(cache["gripper_left_points_px"], dtype=np.float64)
        right_points = np.asarray(cache["gripper_right_points_px"], dtype=np.float64)
        angles = np.asarray(cache["gripper_included_angle_deg"], dtype=np.float64)
        black_left = np.asarray(
            cache["gripper_black_left_point_px"], dtype=np.float64
        )
        black_right = np.asarray(
            cache["gripper_black_right_point_px"], dtype=np.float64
        )
        black_gaps = np.asarray(
            cache["gripper_black_pair_gap_px"], dtype=np.float64
        )
    count = len(source.timestamp_s)
    if (
        not np.array_equal(frames, np.arange(count, dtype=np.int64))
        or left_points.shape != (count, 3, 2)
        or right_points.shape != (count, 3, 2)
        or angles.shape != (count,)
        or black_left.shape != (count, 2)
        or black_right.shape != (count, 2)
        or black_gaps.shape != (count,)
    ):
        raise ManifestError(f"{source.side} gripper marker cache timeline mismatch")
    observations: list[FrameObservation] = []
    low, high = profile.included_angle_range
    for index in range(count):
        left = left_points[index] if np.isfinite(left_points[index]).all() else None
        right = right_points[index] if np.isfinite(right_points[index]).all() else None
        included_angle = math.nan
        if left is not None and right is not None:
            recalculated = included_jaw_angle(left, right)
            cached = float(angles[index])
            if not math.isfinite(cached) or not math.isclose(
                recalculated, cached, rel_tol=1e-4, abs_tol=1e-3
            ):
                raise ManifestError(
                    f"{source.side} gripper marker cache angle mismatch at frame {index}"
                )
            if low <= cached <= high:
                included_angle = cached
            else:
                left = right = None
        elif math.isfinite(float(angles[index])):
            raise ManifestError(
                f"{source.side} cache has an angle without bilateral markers at frame {index}"
            )
        observations.append(
            FrameObservation(
                yellow_left=left,
                yellow_right=right,
                dot_left=None,
                dot_right=None,
                included_angle_deg=included_angle,
            )
        )
        has_black_left = np.isfinite(black_left[index]).all()
        has_black_right = np.isfinite(black_right[index]).all()
        cached_gap = float(black_gaps[index])
        if has_black_left and has_black_right:
            recalculated_gap = float(
                np.linalg.norm(black_left[index] - black_right[index])
            )
            if not math.isfinite(cached_gap) or not math.isclose(
                recalculated_gap, cached_gap, rel_tol=1e-4, abs_tol=1e-3
            ):
                raise ManifestError(
                    f"{source.side} black marker cache gap mismatch at frame {index}"
                )
        elif has_black_left or has_black_right or math.isfinite(cached_gap):
            raise ManifestError(
                f"{source.side} cache has an incomplete black marker pair at frame {index}"
            )
    yellow_ratio = float(np.isfinite(angles).mean())
    black_ratio = float(np.isfinite(black_gaps).mean())
    return observations, frames, black_gaps, yellow_ratio, black_ratio


def _analyze_side(
    source: SideInput,
    profile: SideProfile,
    *,
    processing_width: int,
    maximum_gap_s: float,
) -> SideSignal:
    expected_fps = 1.0 / float(np.median(np.diff(source.timestamp_s)))
    if source.marker_cache is not None:
        (
            observations,
            source_frame,
            black_gaps,
            yellow_ratio,
            black_ratio,
        ) = _cached_observations(source, profile)
        if (
            profile.black_pair_family_min_bilateral_ratio is not None
            and black_ratio >= profile.black_pair_family_min_bilateral_ratio
        ):
            return _black_gap_to_signal(
                source,
                profile,
                black_gaps,
                source_frame,
                fps=expected_fps,
                maximum_gap_s=maximum_gap_s,
                yellow_bilateral_ratio=yellow_ratio,
                black_bilateral_ratio=black_ratio,
            )
        return _observations_to_signal(
            source,
            profile,
            observations,
            source_frame,
            fps=expected_fps,
            maximum_gap_s=maximum_gap_s,
            yellow_bilateral_ratio=yellow_ratio,
            black_bilateral_ratio=black_ratio,
        )

    capture = cv2.VideoCapture(str(source.video), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise ManifestError(f"cannot open {source.side} gripper video: {source.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width != height or fps <= 0:
        capture.release()
        raise ManifestError(
            f"{source.side} gripper video must be a timed square fisheye track"
        )
    if not math.isclose(fps, expected_fps, rel_tol=1e-3, abs_tol=1e-2):
        capture.release()
        raise ManifestError(
            f"{source.side} gripper video/H5 rate mismatch: {fps} != {expected_fps}"
        )
    observations: list[FrameObservation] = []
    black_gaps: list[float] = []
    while len(observations) < len(source.timestamp_s):
        ok, image = capture.read()
        if not ok:
            break
        if image.shape[1] != processing_width:
            image = cv2.resize(
                image,
                (processing_width, processing_width),
                interpolation=cv2.INTER_AREA,
            )
        markers = detect_bgr_gripper_markers(image)
        left = markers.yellow_left
        right = markers.yellow_right
        included_angle = math.nan
        if left is not None and right is not None:
            candidate = included_jaw_angle(left, right)
            low, high = profile.included_angle_range
            if low <= candidate <= high:
                included_angle = candidate
            else:
                left = right = None
        observations.append(
            FrameObservation(
                yellow_left=left,
                yellow_right=right,
                dot_left=None,
                dot_right=None,
                included_angle_deg=included_angle,
            )
        )
        black_gaps.append(markers.black_pair_gap_px)
    capture.release()
    if len(observations) != len(source.timestamp_s):
        raise ManifestError(
            f"{source.side} gripper video/H5 frame mismatch: "
            f"{len(observations)} != {len(source.timestamp_s)}"
        )
    black_gap_values = np.asarray(black_gaps, dtype=np.float64)
    yellow_ratio = float(
        np.mean([math.isfinite(item.included_angle_deg) for item in observations])
    )
    black_ratio = float(np.isfinite(black_gap_values).mean())
    source_frame = np.arange(len(observations), dtype=np.int64)
    if (
        profile.black_pair_family_min_bilateral_ratio is not None
        and black_ratio >= profile.black_pair_family_min_bilateral_ratio
    ):
        return _black_gap_to_signal(
            source,
            profile,
            black_gap_values,
            source_frame,
            fps=fps,
            maximum_gap_s=maximum_gap_s,
            yellow_bilateral_ratio=yellow_ratio,
            black_bilateral_ratio=black_ratio,
        )
    return _observations_to_signal(
        source,
        profile,
        observations,
        source_frame,
        fps=fps,
        maximum_gap_s=maximum_gap_s,
        yellow_bilateral_ratio=yellow_ratio,
        black_bilateral_ratio=black_ratio,
    )


def _nearest_indices(source_time: np.ndarray, query_time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    upper = np.searchsorted(source_time, query_time, side="left")
    lower = np.clip(upper - 1, 0, len(source_time) - 1)
    upper = np.clip(upper, 0, len(source_time) - 1)
    choose_upper = np.abs(source_time[upper] - query_time) < np.abs(
        source_time[lower] - query_time
    )
    indices = np.where(choose_upper, upper, lower)
    return indices, np.abs(source_time[indices] - query_time)


def _read_trajectory(root: Path, pair_id: str) -> tuple[Path, list[str], list[dict[str, str]], dict[str, Any]]:
    tracking = confined_path(
        root,
        "final",
        PIPELINE_REVISION,
        "pairs",
        pair_id,
        "tracking",
        field="trajectory output",
    )
    trajectory = tracking / "joint_trajectory.csv"
    report_path = tracking / "report.json"
    if not trajectory.is_file() or not report_path.is_file():
        raise ManifestError(
            f"completed {PIPELINE_REVISION} tracking output is missing under {tracking}"
        )
    with trajectory.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if len(rows) < 2 or "timestamp_s" not in fields:
        raise ManifestError("joint trajectory CSV needs at least two timestamped rows")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") not in {"SELF_CALIBRATED_PASS", "HOLDOUT_PASS"}:
        raise ManifestError(f"trajectory quality gate did not pass: {report.get('status')}")
    return trajectory, fields, rows, report


def _read_world_flu_trajectory(
    root: Path,
    pair_id: str,
) -> tuple[list[str], list[dict[str, str]], dict[str, Any], Any, np.ndarray]:
    trajectory_path, fields, rows, report = _read_trajectory(root, pair_id)
    world_map_path = trajectory_path.parent / "session_world_map.json"
    if not world_map_path.is_file() or world_map_path.is_symlink():
        raise ManifestError(f"trajectory world map is missing: {world_map_path}")
    world_map = json.loads(world_map_path.read_text(encoding="utf-8"))
    world_transform = derive_world_flu_transform(world_map)
    for side in SIDES:
        for name in (f"{side}_parent_frame", f"{side}_child_frame"):
            if name not in fields:
                fields.append(name)
    rows = transform_trajectory_rows(rows, world_transform)
    query_time = np.asarray(
        [float(row["timestamp_s"]) for row in rows], dtype=np.float64
    )
    if np.any(~np.isfinite(query_time)) or np.any(np.diff(query_time) <= 0):
        raise ManifestError("trajectory timestamps must be finite and increasing")
    return fields, rows, report, world_transform, query_time


def _cell(value: float, digits: int = 9) -> str:
    return "" if not math.isfinite(float(value)) else f"{float(value):.{digits}f}"


def _write_rows(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _publish_csv_files(build_dir: Path, processed_root: Path) -> None:
    for name in CSV_NAMES:
        staged = build_dir / name
        destination = processed_root / name
        if not staged.is_file() or staged.is_symlink():
            raise ManifestError(f"staged processed CSV is invalid: {staged}")
        if destination.is_symlink() or (destination.exists() and not destination.is_file()):
            raise ManifestError(f"processed CSV destination is unsafe: {destination}")
    backup = Path(
        tempfile.mkdtemp(prefix=f".{EXPORT_REVISION}-backup-", dir=processed_root)
    )
    published: list[str] = []
    try:
        for name in CSV_NAMES:
            destination = processed_root / name
            if destination.exists():
                destination.replace(backup / name)
        for name in CSV_NAMES:
            (build_dir / name).replace(processed_root / name)
            published.append(name)
    except Exception:
        for name in reversed(published):
            destination = processed_root / name
            if destination.exists():
                destination.replace(build_dir / name)
        for name in CSV_NAMES:
            saved = backup / name
            if saved.exists():
                saved.replace(processed_root / name)
        raise
    finally:
        if backup.exists():
            shutil.rmtree(backup)

    legacy = processed_root / LEGACY_EXPORT_REVISION
    if legacy.is_dir() and not legacy.is_symlink():
        entries = list(legacy.iterdir())
        if all(
            entry.name in CSV_NAMES and entry.is_file() and not entry.is_symlink()
            for entry in entries
        ):
            shutil.rmtree(legacy)


def remove_pipeline_final(root: Path) -> bool:
    """Remove only this pipeline revision after its CSV export is safely published."""
    final_root = confined_path(root, "final", field="pipeline final root")
    revision_root = confined_path(
        root,
        "final",
        PIPELINE_REVISION,
        field="pipeline final revision",
    )
    removed = False
    if revision_root.exists():
        if revision_root.is_symlink() or not revision_root.is_dir():
            raise ManifestError(
                f"pipeline final revision must be a real directory: {revision_root}"
            )
        manifest_path = revision_root / "manifest.lock.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ManifestError(
                f"refusing to remove pipeline final without its manifest: {manifest_path}"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ManifestError(
                f"refusing to remove pipeline final with an invalid manifest: {manifest_path}"
            ) from error
        if manifest.get("pipeline_revision") != PIPELINE_REVISION:
            raise ManifestError(
                "refusing to remove pipeline final whose manifest revision does not match "
                f"{PIPELINE_REVISION}"
            )
        unexpected = sorted(
            entry.name
            for entry in revision_root.iterdir()
            if entry.name not in PIPELINE_FINAL_ENTRIES
        )
        if unexpected:
            raise ManifestError(
                "refusing to remove pipeline final with unexpected top-level entries: "
                + ", ".join(unexpected)
            )
        shutil.rmtree(revision_root)
        removed = True

    if final_root.exists():
        if final_root.is_symlink() or not final_root.is_dir():
            raise ManifestError(f"pipeline final root must be a real directory: {final_root}")
        if not any(final_root.iterdir()):
            final_root.rmdir()
    return removed


def export_trajectory_only(
    dataset_root: Path,
    *,
    remove_final: bool = False,
) -> dict[str, Any]:
    """Publish only the joint world-FLU trajectory without gripper identity gates."""
    root = dataset_root.expanduser().resolve(strict=True)
    if not is_instaumi_dataset(root):
        raise ManifestError(
            "dataset must contain dataset.h5 and video/{Left,Right}_{back,forward}.mp4"
        )
    processed_root = confined_path(root, "processed", field="processed output root")
    processed_root.mkdir(parents=True, exist_ok=True)
    if processed_root.is_symlink():
        raise ManifestError("processed output root must not be a symlink")
    pair_id = str(load_instaumi_config(root)["pair_id"])
    fields, rows, report, world_transform, query_time = _read_world_flu_trajectory(
        root, pair_id
    )
    destination = processed_root / "trajectory.csv"
    if destination.is_symlink() or (
        destination.exists() and not destination.is_file()
    ):
        raise ManifestError(f"processed trajectory destination is unsafe: {destination}")
    build_dir = Path(
        tempfile.mkdtemp(prefix=f".{EXPORT_REVISION}-trajectory-build-", dir=processed_root)
    )
    backup = build_dir / "previous-trajectory.csv"
    try:
        staged = build_dir / "trajectory.csv"
        _write_rows(staged, fields, rows)
        if destination.exists():
            destination.replace(backup)
        try:
            staged.replace(destination)
        except Exception:
            if backup.exists():
                backup.replace(destination)
            raise
    finally:
        if build_dir.exists():
            shutil.rmtree(build_dir)
    pipeline_final_removed = remove_pipeline_final(root) if remove_final else False
    return {
        "status": "COMPLETE",
        "mode": "trajectory_only",
        "output": str(destination),
        "rows": len(rows),
        "trajectory_rate_hz": 1.0 / float(np.median(np.diff(query_time))),
        "trajectory_status": report["status"],
        "source_world_frame": world_transform.source_frame,
        "world_frame": world_transform.target_frame,
        "camera_child_frame": "hand_camera_flu_back_x",
        "pipeline_final_removed": pipeline_final_removed,
    }


def export_processed_dataset(
    dataset_root: Path,
    *,
    profile_path: Path = PROFILE_PATH,
    remove_final: bool = False,
) -> dict[str, Any]:
    root = dataset_root.expanduser().resolve(strict=True)
    if not is_instaumi_dataset(root):
        raise ManifestError(
            "dataset must contain dataset.h5 and video/{Left,Right}_{back,forward}.mp4"
        )
    processed_root = confined_path(root, "processed", field="processed output root")
    processed_root.mkdir(parents=True, exist_ok=True)
    if processed_root.is_symlink():
        raise ManifestError("processed output root must not be a symlink")
    profile, side_profiles = load_profile(profile_path)
    config = load_instaumi_config(root)
    pair_id = str(config["pair_id"])
    camera_identity_policy = profile.get("camera_identity_policy", {})
    camera_identity_mode = str(camera_identity_policy.get("mode", "exact_serial"))
    if camera_identity_mode not in {"exact_serial", "provenance_only"}:
        raise ManifestError(
            f"unsupported gripper camera identity policy {camera_identity_mode!r}"
        )
    camera_identity: dict[str, dict[str, str]] = {}
    for side in SIDES:
        actual = str(config["cameras"][side]["serial"])
        calibration_source = side_profiles[side].calibration_source_camera_serial
        if not actual:
            raise ManifestError(f"{side} camera serial is missing from dataset metadata")
        if camera_identity_mode == "exact_serial" and actual != calibration_source:
            raise ManifestError(
                f"{side} camera serial {actual!r} does not match gripper profile "
                f"{calibration_source!r}"
            )
        camera_identity[side] = {
            "actual": actual,
            "calibration_source": calibration_source,
            "transfer_status": (
                "EXACT_CAMERA_SERIAL"
                if actual == calibration_source
                else "ROLE_BOUND_FIXED_ROI_SERIAL_TRANSFER"
            ),
        }
    (
        trajectory_fields,
        trajectory_rows,
        report,
        world_transform,
        query_time,
    ) = _read_world_flu_trajectory(root, pair_id)
    side_inputs = load_side_inputs(root, profile)
    processing_width = int(profile.get("processing_width_px", 1024))
    if not 480 <= processing_width <= 1920:
        raise ManifestError("processing_width_px must be between 480 and 1920")
    cv2.setNumThreads(1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            side: executor.submit(
                _analyze_side,
                side_inputs[side],
                side_profiles[side],
                processing_width=processing_width,
                maximum_gap_s=float(profile["maximum_recovery_gap_s"]),
            )
            for side in SIDES
        }
        signals = {side: futures[side].result() for side in SIDES}

    sampled: dict[str, dict[str, np.ndarray]] = {}
    for side in SIDES:
        signal = signals[side]
        expected_shape = signal.timestamp_s.shape
        if any(
            value.shape != expected_shape
            for value in (
                signal.opening_deg,
                signal.width_m,
                signal.state,
                signal.source_frame,
            )
        ):
            raise ManifestError(f"{side} gripper signal arrays have inconsistent lengths")
        index, error_s = _nearest_indices(signal.timestamp_s, query_time)
        median_step = float(np.median(np.diff(signal.timestamp_s)))
        aligned = error_s <= 0.51 * median_step
        opening = signal.opening_deg[index].copy()
        width = signal.width_m[index].copy()
        state = signal.state[index].astype(object)
        opening[~aligned] = np.nan
        width[~aligned] = np.nan
        state[~aligned] = "UNAVAILABLE_TIMELINE_MISMATCH"
        sampled[side] = {
            "index": signal.source_frame[index],
            "time": signal.timestamp_s[index],
            "error": error_s,
            "opening": opening,
            "width": width,
            "state": state,
            "available": np.isfinite(opening),
            "measured": np.char.startswith(state.astype(str), "MEASURED"),
        }

    gripper_fields = ["frame", "timestamp_s"]
    for side in SIDES:
        gripper_fields.extend(
            [
                f"{side}_opening_angle_deg",
                f"{side}_opening_width_m",
                f"{side}_opening_state",
                f"{side}_opening_measured",
                f"{side}_opening_available",
                f"{side}_source_frame",
                f"{side}_source_timestamp_s",
                f"{side}_timestamp_error_s",
                f"{side}_angle_revision",
                f"{side}_marker_family",
            ]
        )
    gripper_rows: list[dict[str, Any]] = []
    processed_fields = trajectory_fields + gripper_fields[2:]
    processed_rows: list[dict[str, Any]] = []
    for row_index, trajectory_row in enumerate(trajectory_rows):
        gripper_row: dict[str, Any] = {
            "frame": trajectory_row.get("frame", str(row_index)),
            "timestamp_s": trajectory_row["timestamp_s"],
        }
        for side in SIDES:
            values = sampled[side]
            profile_side = side_profiles[side]
            gripper_row.update(
                {
                    f"{side}_opening_angle_deg": _cell(values["opening"][row_index], 6),
                    f"{side}_opening_width_m": _cell(values["width"][row_index], 9),
                    f"{side}_opening_state": str(values["state"][row_index]),
                    f"{side}_opening_measured": int(values["measured"][row_index]),
                    f"{side}_opening_available": int(values["available"][row_index]),
                    f"{side}_source_frame": int(values["index"][row_index]),
                    f"{side}_source_timestamp_s": _cell(values["time"][row_index], 9),
                    f"{side}_timestamp_error_s": _cell(values["error"][row_index], 9),
                    f"{side}_angle_revision": profile_side.angle_revision_id,
                    f"{side}_marker_family": signals[side].marker_family,
                }
            )
        gripper_rows.append(gripper_row)
        processed_rows.append({**trajectory_row, **{key: value for key, value in gripper_row.items() if key not in {"frame", "timestamp_s"}}})

    output_dir = processed_root
    build_dir = Path(
        tempfile.mkdtemp(prefix=f".{EXPORT_REVISION}-build-", dir=processed_root)
    )
    try:
        _write_rows(
            build_dir / "trajectory.csv", trajectory_fields, trajectory_rows
        )
        _write_rows(build_dir / "gripper.csv", gripper_fields, gripper_rows)
        _write_rows(build_dir / "processed.csv", processed_fields, processed_rows)
        metadata_fields = [
            "schema_version",
            "dataset_id",
            "dataset_directory",
            "pair_id",
            "pipeline_revision",
            "trajectory_status",
            "trajectory_rows",
            "trajectory_rate_hz",
            "source_world_frame",
            "world_frame",
            "world_frame_convention",
            "world_reframe_revision",
            "world_origin_definition",
            "world_x_positive_definition",
            "world_y_positive_definition",
            "world_z_positive_definition",
            "world_origin_source_x_m",
            "world_origin_source_y_m",
            "world_origin_source_z_m",
            "world_qx_from_source",
            "world_qy_from_source",
            "world_qz_from_source",
            "world_qw_from_source",
            "camera_child_frame",
            "gripper_signal_revision",
            "gripper_camera_identity_policy",
            "training_ready",
            "left_camera_serial",
            "right_camera_serial",
            "left_calibration_source_camera_serial",
            "right_calibration_source_camera_serial",
            "left_calibration_transfer_status",
            "right_calibration_transfer_status",
            "left_base_tag_id",
            "right_base_tag_id",
            "left_video_source",
            "right_video_source",
            "left_measured_ratio",
            "right_measured_ratio",
            "left_available_ratio",
            "right_available_ratio",
            "left_marker_family",
            "right_marker_family",
            "left_yellow_bilateral_ratio",
            "right_yellow_bilateral_ratio",
            "left_black_bilateral_ratio",
            "right_black_bilateral_ratio",
        ]
        frequency = 1.0 / float(np.median(np.diff(query_time)))
        world_frames = sorted({row.get("world_frame", "") for row in trajectory_rows})
        child_frames = sorted(
            {
                row.get(key, "")
                for row in trajectory_rows
                for key in ("left_child_frame", "right_child_frame", "child_frame")
                if row.get(key)
            }
        )
        if not child_frames:
            child_frames = ["hand_camera_flu_back_x"]
        metadata_row = {
            "schema_version": "instaumi-processed-csv/6.0-dual-colour-gripper",
            "dataset_id": pair_id,
            "dataset_directory": root.name,
            "pair_id": pair_id,
            "pipeline_revision": PIPELINE_REVISION,
            "trajectory_status": report["status"],
            "trajectory_rows": len(trajectory_rows),
            "trajectory_rate_hz": f"{frequency:.9f}",
            "source_world_frame": world_transform.source_frame,
            "world_frame": ";".join(world_frames),
            "world_frame_convention": "FLU",
            "world_reframe_revision": WORLD_FLU_REVISION,
            "world_origin_definition": (
                "midpoint_of_grid_A_and_grid_B_geometric_centers"
            ),
            "world_x_positive_definition": "AprilGrid_back",
            "world_y_positive_definition": "left_when_looking_along_positive_x",
            "world_z_positive_definition": "physical_up",
            "world_origin_source_x_m": f"{world_transform.origin_source_m[0]:.12f}",
            "world_origin_source_y_m": f"{world_transform.origin_source_m[1]:.12f}",
            "world_origin_source_z_m": f"{world_transform.origin_source_m[2]:.12f}",
            **{
                f"world_q{axis}_from_source": f"{value:.12f}"
                for axis, value in zip(
                    "xyzw",
                    Rotation.from_matrix(
                        world_transform.rotation_target_from_source
                    ).as_quat(),
                    strict=True,
                )
            },
            "camera_child_frame": ";".join(child_frames),
            "gripper_signal_revision": profile["revision_id"],
            "gripper_camera_identity_policy": camera_identity_mode,
            "training_ready": 0,
            "left_camera_serial": camera_identity["left"]["actual"],
            "right_camera_serial": camera_identity["right"]["actual"],
            "left_calibration_source_camera_serial": camera_identity["left"][
                "calibration_source"
            ],
            "right_calibration_source_camera_serial": camera_identity["right"][
                "calibration_source"
            ],
            "left_calibration_transfer_status": camera_identity["left"][
                "transfer_status"
            ],
            "right_calibration_transfer_status": camera_identity["right"][
                "transfer_status"
            ],
            "left_base_tag_id": side_profiles["left"].base_tag_id,
            "right_base_tag_id": side_profiles["right"].base_tag_id,
            "left_video_source": side_inputs["left"].video_kind,
            "right_video_source": side_inputs["right"].video_kind,
            "left_measured_ratio": f"{signals['left'].measured_ratio:.9f}",
            "right_measured_ratio": f"{signals['right'].measured_ratio:.9f}",
            "left_available_ratio": f"{signals['left'].available_ratio:.9f}",
            "right_available_ratio": f"{signals['right'].available_ratio:.9f}",
            "left_marker_family": signals["left"].marker_family,
            "right_marker_family": signals["right"].marker_family,
            "left_yellow_bilateral_ratio": (
                f"{signals['left'].yellow_bilateral_ratio:.9f}"
            ),
            "right_yellow_bilateral_ratio": (
                f"{signals['right'].yellow_bilateral_ratio:.9f}"
            ),
            "left_black_bilateral_ratio": (
                f"{signals['left'].black_bilateral_ratio:.9f}"
            ),
            "right_black_bilateral_ratio": (
                f"{signals['right'].black_bilateral_ratio:.9f}"
            ),
        }
        _write_rows(build_dir / "metadata.csv", metadata_fields, [metadata_row])
        _publish_csv_files(build_dir, processed_root)
    finally:
        if build_dir.exists():
            shutil.rmtree(build_dir)

    pipeline_final_removed = remove_pipeline_final(root) if remove_final else False

    result = {
        "status": "COMPLETE",
        "output_dir": str(output_dir),
        "files": [
            str(output_dir / name)
            for name in CSV_NAMES
        ],
        "rows": len(trajectory_rows),
        "trajectory_rate_hz": 1.0 / float(np.median(np.diff(query_time))),
        "left_opening_available": int(np.count_nonzero(sampled["left"]["available"])),
        "right_opening_available": int(np.count_nonzero(sampled["right"]["available"])),
        "camera_identity_policy": camera_identity_mode,
        "camera_identity": camera_identity,
        "marker_family": {
            side: signals[side].marker_family for side in SIDES
        },
        "pipeline_final_removed": pipeline_final_removed,
        "training_ready": False,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--profile", type=Path, default=PROFILE_PATH)
    parser.add_argument(
        "--remove-pipeline-final",
        action="store_true",
        help="remove this pipeline revision after processed CSVs are published",
    )
    parser.add_argument(
        "--trajectory-only",
        action="store_true",
        help="publish processed/trajectory.csv without running gripper identity gates",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.trajectory_only:
        result = export_trajectory_only(
            args.dataset_root,
            remove_final=args.remove_pipeline_final,
        )
    else:
        result = export_processed_dataset(
            args.dataset_root,
            profile_path=args.profile,
            remove_final=args.remove_pipeline_final,
        )
        for side, identity in result["camera_identity"].items():
            if identity["transfer_status"] != "EXACT_CAMERA_SERIAL":
                print(
                    "[夹爪] "
                    f"{side} 数据相机 {identity['actual']}；固定 ROI/物理夹爪标定来源相机 "
                    f"{identity['calibration_source']}。序列号仅作溯源，结果保持 "
                    "training_ready=0；两者已写入 metadata.csv。",
                    file=sys.stderr,
                )
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
