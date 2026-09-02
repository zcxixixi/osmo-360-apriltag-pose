"""Manifest discovery and orchestration for four raw X5 fisheye MP4s."""

from __future__ import annotations

import contextlib
import fcntl
import json
import math
import os
import re
import stat
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from osmo360.ffmpeg_runtime import project_ffmpeg_runtime

from .devices import load_device_pairs, load_inventory
from .manifest import ManifestError, ROOT, confined_path, validate_path_component


PIPELINE_REVISION = "dual-x5-four-mp4-cpu-v12"
INPUT_SCHEMA = "dual-x5-four-mp4-input/1.0"
LOCK_SCHEMA = "dual-x5-four-mp4-dataset-lock/1.0"

SERIAL_PATTERN = re.compile(rb"IAHE[A-Z0-9]{10}")
OFFSET_PATTERN = re.compile(rb"[mn]2(?:_-?\d+(?:\.\d+)?){15}")
STREAM_PATTERN = re.compile(r"(?:lens|stream)[-_]?([01])", re.IGNORECASE)
TIME_PATTERN = re.compile(r"VID_(\d{8})_(\d{6})_")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _input_config(root: Path) -> dict[str, Any]:
    path = root / "raw/four-mp4.json"
    if not path.is_file():
        from .instaumi import is_instaumi_dataset, load_instaumi_config

        return load_instaumi_config(root) if is_instaumi_dataset(root) else {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ManifestError(f"invalid four-MP4 input descriptor: {error}") from error
    if value.get("schema_version") != INPUT_SCHEMA:
        raise ManifestError(f"raw/four-mp4.json must use {INPUT_SCHEMA}")
    return value


def is_four_mp4_dataset(root: Path) -> bool:
    from .instaumi import is_instaumi_dataset

    raw = root / "raw"
    return (
        (raw / "four-mp4.json").is_file()
        or any(raw.glob("*/*.mp4"))
        or is_instaumi_dataset(root)
    )


def _safe_input_path(root: Path, value: str) -> Path:
    path = (root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ManifestError(f"four-MP4 input must stay inside the dataset root: {path}") from error
    if not path.is_file():
        raise ManifestError(f"four-MP4 input is missing: {path}")
    return path


def _lens_paths(root: Path, config: dict[str, Any], side: str) -> dict[int, Path]:
    camera = config.get("cameras", {}).get(side, {})
    configured = camera.get("lenses")
    if configured is not None:
        if not isinstance(configured, list) or len(configured) != 2:
            raise ManifestError(f"cameras.{side}.lenses must contain two paths")
        paths = [_safe_input_path(root, str(value)) for value in configured]
    else:
        directory = root / "raw" / side
        if not directory.is_dir():
            raise ManifestError(f"required input directory is missing: {directory}")
        paths = sorted(directory.glob("*.mp4"))
        if len(paths) != 2:
            raise ManifestError(
                f"raw/{side} must contain exactly two MP4 files, found {len(paths)}"
            )
    result: dict[int, Path] = {}
    unnamed: list[Path] = []
    for path in paths:
        match = STREAM_PATTERN.search(path.stem)
        if match is None:
            unnamed.append(path)
            continue
        stream = int(match.group(1))
        if stream in result:
            raise ManifestError(f"duplicate lens-{stream} for {side}: {paths}")
        result[stream] = path
    if unnamed:
        if len(unnamed) == 2 and not result:
            result = {index: path for index, path in enumerate(sorted(unnamed))}
        else:
            raise ManifestError(
                f"cannot infer lens streams for {side}; name files lens-0.mp4/lens-1.mp4 "
                "or list them in raw/four-mp4.json"
            )
    if set(result) != {0, 1}:
        raise ManifestError(f"{side} inputs must resolve to lens streams 0 and 1")
    return result


def _fraction(value: str) -> float:
    numerator, denominator = map(float, value.split("/"))
    if denominator == 0:
        raise ManifestError(f"invalid frame rate: {value}")
    return numerator / denominator


def _probe_mp4(path: Path) -> dict[str, Any]:
    runtime = project_ffmpeg_runtime()
    process = subprocess.run(
        [
            str(runtime.ffprobe), "-v", "error", "-show_entries",
            "stream=codec_type,width,height,r_frame_rate,avg_frame_rate,duration,nb_frames:format=duration",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    payload = json.loads(process.stdout)
    video_streams = [
        stream for stream in payload.get("streams", [])
        if stream.get("codec_type") == "video"
    ]
    if len(video_streams) != 1:
        raise ManifestError(f"each raw lens MP4 must contain one video stream: {path}")
    stream = video_streams[0]
    fps_value = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    fps = _fraction(str(fps_value))
    duration = float(stream.get("duration") or payload.get("format", {}).get("duration"))
    frame_count_value = stream.get("nb_frames")
    frame_count = int(frame_count_value) if frame_count_value not in (None, "N/A") else int(round(duration * fps))
    if fps <= 0 or duration <= 0 or frame_count <= 0:
        raise ManifestError(f"invalid timing metadata in {path}")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "duration_s": duration,
        "frame_count": frame_count,
        "has_audio": any(
            item.get("codec_type") == "audio" for item in payload.get("streams", [])
        ),
        "probe_runtime": runtime.provenance(),
    }


def _embedded_identity(path: Path) -> tuple[str | None, str | None]:
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 16 * 1024 * 1024))
        tail = handle.read()
    serial = SERIAL_PATTERN.search(tail)
    offset = OFFSET_PATTERN.search(tail)
    return (
        None if serial is None else serial.group().decode("ascii"),
        None if offset is None else offset.group().decode("ascii"),
    )


def _registered_binding(
    side: str,
    requested_serial: str | None = None,
) -> tuple[str, int]:
    if requested_serial:
        device = load_inventory()["devices"].get(requested_serial)
        assignment = None if device is None else device.get("assignment")
        expected_role = f"physical_{side}"
        if (
            not isinstance(assignment, dict)
            or assignment.get("role") != expected_role
            or int(assignment.get("base_tag_id", -1)) not in {2, 3}
        ):
            raise ManifestError(
                f"{side} camera {requested_serial} has no verified {expected_role} "
                "and BaseTag assignment in the X5 inventory"
            )
        return requested_serial, int(assignment["base_tag_id"])
    candidates = []
    for pair in load_device_pairs()["pairs"].values():
        item = pair[side]
        candidates.append((str(item["serial"]), int(item["base_tag_id"])))
    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise ManifestError(
            f"cannot infer {side} camera identity from {len(candidates)} registered bindings; "
            "set it in raw/four-mp4.json"
        )
    return candidates[0]


def _camera_record(
    root: Path,
    config: dict[str, Any],
    side: str,
) -> dict[str, Any]:
    paths = _lens_paths(root, config, side)
    camera_config = config.get("cameras", {}).get(side, {})
    embedded = {stream: _embedded_identity(path) for stream, path in paths.items()}
    configured_serial = camera_config.get("serial")
    registered_serial, registered_tag = _registered_binding(
        side,
        None if configured_serial is None else str(configured_serial),
    )
    embedded_serials = {value[0] for value in embedded.values() if value[0]}
    embedded_offsets = {value[1] for value in embedded.values() if value[1]}
    if len(embedded_serials) > 1 or len(embedded_offsets) > 1:
        raise ManifestError(f"{side} lens MP4s disagree on embedded camera metadata")
    serial = str(camera_config.get("serial") or next(iter(embedded_serials), registered_serial))
    base_tag_id = int(camera_config.get("base_tag_id", registered_tag))
    if serial != registered_serial or base_tag_id != registered_tag:
        raise ManifestError(
            f"{side} binding {serial}/BaseTag{base_tag_id} does not match the registered "
            f"{registered_serial}/BaseTag{registered_tag}"
        )
    x5_offset = camera_config.get("x5_offset") or next(iter(embedded_offsets), None)
    if not x5_offset:
        raise ManifestError(
            f"{side} MP4s do not carry the X5 lens offset; add cameras.{side}.x5_offset "
            "to raw/four-mp4.json"
        )
    if OFFSET_PATTERN.fullmatch(str(x5_offset).encode("ascii", errors="ignore")) is None:
        raise ManifestError(f"cameras.{side}.x5_offset is not a valid X5 offset record")
    probes = {stream: _probe_mp4(path) for stream, path in paths.items()}
    first = probes[0]
    for stream in (1,):
        current = probes[stream]
        for key in ("width", "height", "frame_count"):
            if current[key] != first[key]:
                raise ManifestError(f"{side} lens MP4s disagree on {key}: {probes}")
        if not math.isclose(current["fps"], first["fps"], rel_tol=1e-6, abs_tol=1e-3):
            raise ManifestError(f"{side} lens MP4s disagree on fps: {probes}")
        if not math.isclose(current["duration_s"], first["duration_s"], abs_tol=0.05):
            raise ManifestError(f"{side} lens MP4s disagree on duration: {probes}")
    return {
        "side": side,
        "serial": serial,
        "serial_source": "descriptor" if camera_config.get("serial") else (
            "embedded_mp4" if embedded_serials else "registered_device_pair"
        ),
        "base_tag_id": base_tag_id,
        "x5_offset": str(x5_offset),
        "lens_size": [first["width"], first["height"]],
        "fps": first["fps"],
        "duration_s": min(value["duration_s"] for value in probes.values()),
        "frame_count": min(value["frame_count"] for value in probes.values()),
        **(
            {"probe_runtime": first["probe_runtime"]}
            if first.get("probe_runtime") else {}
        ),
        "lenses": [
            {
                "stream": stream,
                "path": paths[stream].relative_to(root).as_posix(),
                "size_bytes": paths[stream].stat().st_size,
                **{
                    key: value for key, value in probes[stream].items()
                    if key != "probe_runtime"
                },
            }
            for stream in (0, 1)
        ],
        **(
            {
                "timeline_h5": str(camera_config["timeline_h5"]),
                "timeline_camera": str(camera_config["timeline_camera"]),
                "timeline": camera_config.get("timeline"),
            }
            if camera_config.get("timeline_h5")
            else {}
        ),
        **(
            {"lens_mapping": camera_config["lens_mapping"]}
            if camera_config.get("lens_mapping")
            else {}
        ),
        **(
            {"factory_offset_source": camera_config["factory_offset_source"]}
            if camera_config.get("factory_offset_source")
            else {}
        ),
        **(
            {
                "rear_calibration": camera_config["rear_calibration"],
                "rear_calibration_source": camera_config["rear_calibration_source"],
                "rear_calibration_sha256": camera_config["rear_calibration_sha256"],
                "timeline_h5_sha256": camera_config["timeline_h5_sha256"],
            }
            if camera_config.get("rear_calibration")
            else {}
        ),
    }


def _positive_number(
    config: dict[str, Any], environment_name: str, config_name: str, default: float
) -> float:
    environment = os.environ.get(environment_name)
    value = float(environment if environment is not None else config.get(config_name, default))
    if not math.isfinite(value) or value <= 0:
        raise ManifestError(f"{environment_name}/{config_name} must be positive")
    return value


def _boolean_option(
    config: dict[str, Any], environment_name: str, config_name: str, default: bool
) -> bool:
    value: Any = os.environ.get(environment_name, config.get(config_name, default))
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ManifestError(f"{environment_name}/{config_name} must be boolean")


def resource_budget(config: dict[str, Any] | None = None) -> dict[str, Any]:
    processing = (config or {}).get("processing", {})
    cpu_count = os.cpu_count() or 1
    requested_workers = int(_positive_number(
        processing, "OSMO_CPU_WORKERS", "cache_workers", 1
    ))
    requested_threads = int(_positive_number(
        processing, "OSMO_THREADS_PER_WORKER", "threads_per_worker", min(2, cpu_count)
    ))
    per_job_limit = min(cpu_count, 16)
    workers = min(requested_workers, min(4, per_job_limit))
    threads = min(requested_threads, min(8, max(1, per_job_limit // workers)))
    trajectory_fps_default = processing.get(
        "trajectory_observation_fps", processing.get("detection_fps", 30.0)
    )
    trajectory_fps = _positive_number(
        {"trajectory_observation_fps": trajectory_fps_default},
        "OSMO_TRAJECTORY_FPS",
        "trajectory_observation_fps",
        30.0,
    )
    decode_fps = _positive_number(
        processing,
        "OSMO_DECODE_FPS",
        "decode_fps",
        trajectory_fps,
    )
    if decode_fps < trajectory_fps:
        raise ManifestError("decode_fps must be at least trajectory_observation_fps")
    chunk_s = _positive_number(
        processing, "OSMO_CACHE_CHUNK_SECONDS", "cache_chunk_duration_s", 120.0
    )
    local_hz = _positive_number(
        processing, "OSMO_LOCAL_REDETECTION_HZ", "local_redetection_hz", 5.0
    )
    global_hz = _positive_number(
        processing, "OSMO_GLOBAL_SCOUT_HZ", "global_scout_hz", 2.0
    )
    global_scale = _positive_number(
        processing, "OSMO_GLOBAL_SCOUT_SCALE", "global_scout_scale", 0.35
    )
    rectified_hz = _positive_number(
        processing, "OSMO_RECTIFIED_RECOVERY_HZ", "rectified_recovery_hz", 2.0
    )
    rectified_view_size = int(_positive_number(
        processing, "OSMO_RECTIFIED_VIEW_SIZE", "rectified_view_size", 720
    ))
    optical_flow_scale = _positive_number(
        processing, "OSMO_OPTICAL_FLOW_SCALE", "optical_flow_scale", 1.0
    )
    forward_backward_hz = _positive_number(
        processing,
        "OSMO_FORWARD_BACKWARD_CHECK_HZ",
        "forward_backward_check_hz",
        decode_fps,
    )
    optical_flow_window_size = int(_positive_number(
        processing,
        "OSMO_OPTICAL_FLOW_WINDOW_SIZE",
        "optical_flow_window_size",
        31,
    ))
    optical_flow_max_level = int(_positive_number(
        processing,
        "OSMO_OPTICAL_FLOW_MAX_LEVEL",
        "optical_flow_max_level",
        4,
    ))
    optical_flow_max_iterations = int(_positive_number(
        processing,
        "OSMO_OPTICAL_FLOW_MAX_ITERATIONS",
        "optical_flow_max_iterations",
        30,
    ))
    native_grayscale = _boolean_option(
        processing,
        "OSMO_NATIVE_GRAYSCALE_DECODE",
        "native_grayscale_decode",
        False,
    )
    if not 0.1 <= global_scale <= 1.0:
        raise ManifestError("global_scout_scale must be between 0.1 and 1.0")
    if not 256 <= rectified_view_size <= 1920:
        raise ManifestError("rectified_view_size must be between 256 and 1920")
    if not 0.25 <= optical_flow_scale <= 1.0:
        raise ManifestError("optical_flow_scale must be between 0.25 and 1.0")
    if optical_flow_window_size < 9 or not optical_flow_window_size % 2:
        raise ManifestError("optical_flow_window_size must be an odd integer >= 9")
    if optical_flow_max_level > 8 or optical_flow_max_iterations > 100:
        raise ManifestError("optical flow level/iteration limits are too large")
    if requested_workers > 4:
        raise ManifestError("OSMO_CPU_WORKERS is capped at 4 for the low-resource profile")
    if requested_threads > 8:
        raise ManifestError("OSMO_THREADS_PER_WORKER is capped at 8")
    if "OSMO_CPU_WORKERS" in os.environ and workers != requested_workers:
        raise ManifestError(
            f"OSMO_CPU_WORKERS={requested_workers} exceeds this host's safe limit {workers}"
        )
    if "OSMO_THREADS_PER_WORKER" in os.environ and threads != requested_threads:
        raise ManifestError(
            f"OSMO_THREADS_PER_WORKER={requested_threads} exceeds this host's safe limit {threads}"
        )
    maximum_active_cpu_threads = workers * threads
    concurrent_jobs = int(_positive_number(
        processing, "OSMO_MAX_CONCURRENT_JOBS", "maximum_concurrent_jobs", 1
    ))
    if concurrent_jobs > 4:
        raise ManifestError("OSMO_MAX_CONCURRENT_JOBS is capped at 4")
    aggregate_threads = maximum_active_cpu_threads * concurrent_jobs
    if aggregate_threads > cpu_count:
        raise ManifestError(
            "maximum_concurrent_jobs * active threads per job must not exceed "
            f"the {cpu_count} logical CPUs"
        )
    slot_timeout_s = _positive_number(
        processing, "OSMO_JOB_SLOT_TIMEOUT_S", "job_slot_timeout_s", 3600.0
    )
    return {
        "profile": str(processing.get("profile", "bounded-cpu")),
        "cache_workers": workers,
        "threads_per_worker": threads,
        "maximum_active_cpu_threads": maximum_active_cpu_threads,
        "maximum_concurrent_jobs": concurrent_jobs,
        "aggregate_maximum_active_cpu_threads": aggregate_threads,
        "job_slot_timeout_s": slot_timeout_s,
        "trajectory_observation_fps": trajectory_fps,
        "decode_fps": decode_fps,
        "native_grayscale_decode": native_grayscale,
        "optical_flow_scale": optical_flow_scale,
        "forward_backward_check_hz": forward_backward_hz,
        "optical_flow_window_size": optical_flow_window_size,
        "optical_flow_max_level": optical_flow_max_level,
        "optical_flow_max_iterations": optical_flow_max_iterations,
        "source_frame_tracking": (
            "sampled grayscale frames" if decode_fps < 59.0 else "every decoded frame"
        ),
        "local_tag_redetection_hz": local_hz,
        "global_grayscale_scout_hz": global_hz,
        "global_scout_scale": global_scale,
        "maximum_rectified_recovery_hz": rectified_hz,
        "rectified_view_size": rectified_view_size,
        "cache_chunk_duration_s": chunk_s,
        "cuda_required": False,
        "stitching_required": False,
    }


@contextlib.contextmanager
def pipeline_job_slot(
    maximum_concurrent_jobs: int,
    *,
    timeout_s: float,
    lock_root: Path | None = None,
):
    """Hold one host-local slot so independent datasets cannot oversubscribe CPUs."""

    if maximum_concurrent_jobs <= 0 or timeout_s <= 0:
        raise ManifestError("pipeline job slot count and timeout must be positive")
    if lock_root is None:
        configured = os.environ.get("OSMO_JOB_LOCK_DIR", "").strip()
        if configured:
            raw_directory = Path(configured).expanduser()
        else:
            runtime = os.environ.get("XDG_RUNTIME_DIR", "").strip()
            raw_directory = (
                Path(runtime) / "osmo360-pipeline-job-slots"
                if runtime
                # This fallback is user-scoped and validated below as a real,
                # 0700, same-owner directory before any lock file is opened.
                else Path(f"/tmp/osmo360-pipeline-job-slots-{os.getuid()}")  # nosec B108
            )
    else:
        raw_directory = Path(lock_root)
    if not raw_directory.is_absolute():
        raise ManifestError(f"pipeline lock root must be absolute: {raw_directory}")
    if raw_directory.is_symlink():
        raise ManifestError(f"pipeline lock root must not be a symlink: {raw_directory}")
    try:
        raw_directory.mkdir(mode=0o700)
    except FileExistsError:
        pass
    metadata = raw_directory.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid():
        raise ManifestError(
            f"pipeline lock root must be a real directory owned by this user: {raw_directory}"
        )
    directory = raw_directory.resolve()
    directory.chmod(0o700)
    deadline = time.monotonic() + timeout_s
    started = time.monotonic()
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    while True:
        for slot in range(maximum_concurrent_jobs):
            path = directory / f"slot-{slot}.lock"
            try:
                descriptor = os.open(path, flags, 0o600)
            except OSError as exc:
                raise ManifestError(f"cannot open pipeline job slot {path}: {exc}") from exc
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
                    raise ManifestError(
                        f"pipeline job slot must be a regular file owned by this user: {path}"
                    )
                os.fchmod(descriptor, 0o600)
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    os.close(descriptor)
                    continue
                try:
                    yield {
                        "slot": slot,
                        "maximum_concurrent_jobs": maximum_concurrent_jobs,
                        "waited_s": time.monotonic() - started,
                    }
                finally:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                    os.close(descriptor)
                return
            except BaseException:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
                raise
        if time.monotonic() >= deadline:
            raise ManifestError(
                f"timed out after {timeout_s:.1f}s waiting for one of "
                f"{maximum_concurrent_jobs} pipeline job slots"
            )
        time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))


def chunk_ranges(
    frame_count: int,
    fps: float,
    chunk_duration_s: float,
    *,
    alignment: int = 1,
) -> list[tuple[int, int]]:
    if frame_count <= 0 or fps <= 0 or chunk_duration_s <= 0 or alignment <= 0:
        raise ValueError("frame_count, fps, chunk duration and alignment must be positive")
    chunk_frames = max(1, int(round(fps * chunk_duration_s)))
    chunk_frames = max(alignment, int(round(chunk_frames / alignment)) * alignment)
    return [
        (start, min(frame_count - 1, start + chunk_frames - 1))
        for start in range(0, frame_count, chunk_frames)
    ]


def discover_four_mp4_dataset(dataset_root: Path) -> dict[str, Any]:
    root = dataset_root.resolve(strict=True)
    config = _input_config(root)
    left = _camera_record(root, config, "left")
    right = _camera_record(root, config, "right")
    if not math.isclose(left["fps"], right["fps"], rel_tol=1e-6, abs_tol=1e-3):
        raise ManifestError("left/right cameras use different frame rates")
    common_duration = min(left["duration_s"], right["duration_s"])
    if abs(left["duration_s"] - right["duration_s"]) > 5.0:
        raise ManifestError("left/right MP4 durations differ by more than 5 seconds")
    times = []
    for camera in (left, right):
        for lens in camera["lenses"]:
            match = TIME_PATTERN.search(Path(lens["path"]).name)
            if match:
                times.append(datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S"))
    if config.get("recorded_at"):
        recorded_at = datetime.fromisoformat(str(config["recorded_at"]).replace("Z", "+00:00"))
    else:
        recorded_at = min(times) if times else datetime.fromtimestamp(
            min((root / lens["path"]).stat().st_mtime for camera in (left, right) for lens in camera["lenses"])
        )
    sync_config = config.get("sync", {})
    if "offset_s" in sync_config:
        sync = {
            **sync_config,
            "method": str(sync_config.get("method", "fixed_descriptor")),
            "offset_s": float(sync_config["offset_s"]),
            "mapping": "right_time_s = left_time_s + offset_s",
        }
    else:
        if not left["lenses"][0]["has_audio"] or not right["lenses"][0]["has_audio"]:
            raise ManifestError(
                "lens-0 MP4s have no audio; set sync.offset_s in raw/four-mp4.json"
            )
        sync = {
            "method": "bounded_audio_cross_correlation",
            "window_s": float(sync_config.get("window_s", 120.0)),
            "mapping": "right_time_s = left_time_s + offset_s",
        }
    pair_id = validate_path_component(
        config["pair_id"] if "pair_id" in config else f"pair-01-{recorded_at:%H%M%S}",
        field="pair_id",
    )
    probe_runtime = left.get("probe_runtime") or right.get("probe_runtime")
    return {
        "schema_version": LOCK_SCHEMA,
        "pipeline_revision": PIPELINE_REVISION,
        "input_format": str(config.get("input_format", "four-independent-raw-fisheye-mp4")),
        "dataset_root": ".",
        "pair_count": 1,
        "pairs": [
            {
                "pair_id": pair_id,
                "recorded_at": recorded_at.isoformat(),
                "left": left,
                "right": right,
                "sync": sync,
                "common_duration_upper_bound_s": common_duration,
                "tracking": config.get("tracking"),
                "auto_tracking": config.get("auto_tracking"),
            }
        ],
        "resource_budget": resource_budget(config),
        **({"ffmpeg_runtime": probe_runtime} if probe_runtime else {}),
        **({"instaumi": config["instaumi"]} if config.get("instaumi") else {}),
    }


def process_four_mp4_dataset(dataset_root: Path, *, dry_run: bool = False) -> dict[str, Any]:
    root = dataset_root.resolve(strict=True)
    lock = discover_four_mp4_dataset(root)
    final_root = confined_path(root, "final", PIPELINE_REVISION, field="final output root")
    lock_path = final_root / "manifest.lock.json"
    cache_base = Path(
        os.environ.get("OSMO_PIPELINE_CACHE", str(root / ".osmo-cache"))
    ).expanduser().resolve()
    pair = lock["pairs"][0]
    dataset_component = validate_path_component(root.name, field="dataset directory name")
    cache_root = confined_path(
        cache_base,
        dataset_component,
        PIPELINE_REVISION,
        pair["pair_id"],
        field="pipeline cache root",
    )
    command = [
        str(ROOT / ".venv/bin/python"),
        "-m",
        "osmo360.pipeline.four_mp4_worker",
        str(root),
        "--pair-id",
        pair["pair_id"],
        "--cache-root",
        str(cache_root),
    ]
    if dry_run:
        return {
            "status": "DRY_RUN",
            "pipeline_revision": PIPELINE_REVISION,
            "manifest_lock": str(lock_path),
            "resource_budget": lock["resource_budget"],
            "command": command,
        }
    with pipeline_job_slot(
        int(lock["resource_budget"]["maximum_concurrent_jobs"]),
        timeout_s=float(lock["resource_budget"]["job_slot_timeout_s"]),
    ) as job_slot:
        _atomic_json(lock_path, lock)
        process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        result = {
            "command": command,
            "returncode": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
            "job_slot": job_slot,
        }
        pair_status_path = confined_path(
            final_root, "pairs", pair["pair_id"], "status.json", field="pair status path"
        )
        pair_status = (
            json.loads(pair_status_path.read_text(encoding="utf-8"))
            if pair_status_path.is_file()
            else None
        )
        status = {
            "schema_version": "dual-x5-four-mp4-pipeline-status/1.0",
            "status": "FAILED" if process.returncode else (
                pair_status.get("status", "COMPLETE") if pair_status else "COMPLETE"
            ),
            "pipeline_revision": PIPELINE_REVISION,
            "manifest_lock": str(lock_path.relative_to(root)),
            "resource_budget": lock["resource_budget"],
            "result": result,
        }
        _atomic_json(final_root / "status.json", status)
        if process.returncode:
            raise ManifestError(
                f"four-MP4 worker failed; see {final_root / 'status.json'}"
            )
        return status
