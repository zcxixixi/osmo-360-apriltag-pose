"""Low-resource, resumable four-MP4 observation and trajectory worker."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset_worker import estimate_audio_offset
from .four_mp4 import PIPELINE_REVISION, chunk_ranges
from .manifest import (
    ManifestError,
    ROOT,
    confined_path,
    publish_directory,
    validate_path_component,
)


PYTHON = ROOT / ".venv/bin/python"
FFMPEG = ROOT / "work/tools/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg"
if not FFMPEG.is_file():
    FFMPEG = Path("/usr/bin/ffmpeg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def status_update(path: Path, stage: str, state: str, **details: Any) -> None:
    payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {
        "schema_version": "dual-x5-four-mp4-worker-status/1.0",
        "pipeline_revision": PIPELINE_REVISION,
        "status": "RUNNING",
        "stages": {},
    }
    payload["stages"][stage] = {"state": state, **details}
    atomic_json(path, payload)


def run(
    command: list[str],
    log: Path,
    *,
    environment: dict[str, str] | None = None,
    gate: bool = False,
) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=environment,
    )
    temporary = log.with_suffix(log.suffix + ".tmp")
    temporary.write_text(process.stdout, encoding="utf-8")
    temporary.replace(log)
    if process.returncode and not (gate and process.returncode == 2):
        raise RuntimeError(
            f"command failed ({process.returncode}): {' '.join(command)}; log={log}"
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(root: Path, pair: dict[str, Any], cache_root: Path) -> dict[str, str]:
    index_path = cache_root / "source-index.json"
    previous = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {}
    entries: dict[str, Any] = {}
    result: dict[str, str] = {}
    for side in ("left", "right"):
        for lens in pair[side]["lenses"]:
            relative = str(lens["path"])
            path = root / relative
            stat = path.stat()
            old = previous.get("sources", {}).get(relative, {})
            if old.get("size_bytes") == stat.st_size and old.get("mtime_ns") == stat.st_mtime_ns:
                digest = str(old.get("sha256", ""))
            else:
                digest = sha256(path)
            if len(digest) != 64:
                digest = sha256(path)
            entries[relative] = {
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": digest,
            }
            result[relative] = digest
    atomic_json(
        index_path,
        {
            "schema_version": "four-mp4-source-index/1.0",
            "sources": entries,
        },
    )
    return result


def extract_audio(
    source: Path,
    output: Path,
    duration_s: float,
    log: Path,
    *,
    reuse: bool = True,
) -> None:
    if reuse and output.is_file():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            str(FFMPEG), "-y", "-t", str(duration_s), "-i", str(source),
            "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "2000",
            "-c:a", "pcm_s16le", str(output),
        ],
        log,
    )


def resolve_sync(
    root: Path,
    pair: dict[str, Any],
    cache_root: Path,
    logs: Path,
    hashes: dict[str, str],
) -> dict[str, Any]:
    sync = pair["sync"]
    if "offset_s" in sync:
        return dict(sync)
    duration_s = min(float(sync.get("window_s", 120.0)), pair["common_duration_upper_bound_s"])
    audio = cache_root / "audio"
    left_relative = pair["left"]["lenses"][0]["path"]
    right_relative = pair["right"]["lenses"][0]["path"]
    left_source = root / left_relative
    right_source = root / right_relative
    sync_path = audio / "sync.json"
    expected_identity = {
        "left_video_sha256": hashes[left_relative],
        "right_video_sha256": hashes[right_relative],
        "window_s": duration_s,
        "sample_rate_hz": 2000,
    }
    if sync_path.is_file():
        try:
            cached = json.loads(sync_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = {}
        if all(cached.get(key) == value for key, value in expected_identity.items()):
            return cached
    extract_audio(
        left_source,
        audio / "left.wav",
        duration_s,
        logs / "audio-left.log",
        reuse=False,
    )
    extract_audio(
        right_source,
        audio / "right.wav",
        duration_s,
        logs / "audio-right.log",
        reuse=False,
    )
    result = estimate_audio_offset(audio / "left.wav", audio / "right.wav", 0.0)
    result.update({"method": sync["method"], **expected_identity})
    atomic_json(sync_path, result)
    return result


@dataclass(frozen=True)
class ChunkTask:
    side: str
    stream: int
    start: int
    end: int
    output: Path
    command: list[str]
    log: Path
    expected: dict[str, Any]


def _sidecar(path: Path) -> Path:
    return path.with_suffix(".json")


def _valid_metadata(path: Path, expected: dict[str, Any]) -> bool:
    metadata_path = _sidecar(path)
    if not path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return all(metadata.get(key) == value for key, value in expected.items())


def _worker_environment(threads: int) -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = str(threads)
    environment["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"threads;{threads}"
    return environment


def build_chunk_tasks(
    root: Path,
    pair: dict[str, Any],
    cache_root: Path,
    hashes: dict[str, str],
    sync: dict[str, Any],
    budget: dict[str, Any],
) -> list[ChunkTask]:
    tasks = []
    trajectory_fps = float(budget["trajectory_observation_fps"])
    decode_fps = float(budget.get("decode_fps", trajectory_fps))
    threads = int(budget["threads_per_worker"])

    def aligned_interval(source_fps: float, frequency_hz: float, alignment: int) -> int:
        raw = source_fps / frequency_hz
        return max(alignment, int(round(raw / alignment)) * alignment)

    for side in ("left", "right"):
        camera = pair[side]
        intercept = 0.0 if side == "left" else float(sync["offset_s"])
        stride = max(1, int(round(float(camera["fps"]) / trajectory_fps)))
        decode_stride = max(1, int(round(float(camera["fps"]) / decode_fps)))
        if stride % decode_stride:
            raise ManifestError(
                "trajectory observation cadence must be a multiple of decode cadence"
            )
        global_scout_interval = aligned_interval(
            float(camera["fps"]), float(budget["global_grayscale_scout_hz"]), decode_stride
        )
        local_redetect_interval = aligned_interval(
            float(camera["fps"]), float(budget["local_tag_redetection_hz"]), decode_stride
        )
        rectified_recovery_interval = aligned_interval(
            float(camera["fps"]), float(budget["maximum_rectified_recovery_hz"]), decode_stride
        )
        forward_backward_interval = aligned_interval(
            float(camera["fps"]), float(budget["forward_backward_check_hz"]), decode_stride
        )
        max_track_age = aligned_interval(float(camera["fps"]), 1.0, decode_stride)
        rectified_view_size = int(budget["rectified_view_size"])
        global_scout_scale = float(budget["global_scout_scale"])
        optical_flow_scale = float(budget["optical_flow_scale"])
        optical_flow_window_size = int(budget["optical_flow_window_size"])
        optical_flow_max_level = int(budget["optical_flow_max_level"])
        optical_flow_max_iterations = int(budget["optical_flow_max_iterations"])
        native_grayscale = bool(budget["native_grayscale_decode"])
        timestamp_source = (
            f"instaumi_h5:/sensor/camera/{camera['timeline_camera']}/timestamp_ns"
            if camera.get("timeline_h5")
            else "video_nominal_fps"
        )
        processing_signature = {
            "temporal_tracking": True,
            "output_stride_frames": stride,
            "decode_stride_frames": decode_stride,
            "native_grayscale_decode": native_grayscale,
            "optical_flow_scale": optical_flow_scale,
            "forward_backward_check_interval_frames": forward_backward_interval,
            "optical_flow_window_size": optical_flow_window_size,
            "optical_flow_max_level": optical_flow_max_level,
            "optical_flow_max_iterations": optical_flow_max_iterations,
            "timestamp_source": timestamp_source,
            "rectified_detection": True,
            "rectified_detection_policy": "adaptive",
            "rectified_min_direct_tags": 4,
            "rectified_required_ids": [2, 3],
            "rectified_view_size": rectified_view_size,
            "global_scout_interval_frames": global_scout_interval,
            "global_scout_scale": global_scout_scale,
            "local_redetect_interval_frames": local_redetect_interval,
            "rectified_recovery_interval_frames": rectified_recovery_interval,
            "max_track_age_frames": max_track_age,
            "max_flow_forward_backward_error_px": 1.5,
            "max_reacquire_distance_px": 160.0,
        }
        for lens in camera["lenses"]:
            stream = int(lens["stream"])
            video = root / lens["path"]
            use_h5_rear_calibration = bool(
                stream == 0 and camera.get("rear_calibration")
            )
            calibration_sha256 = (
                str(camera["rear_calibration_sha256"])
                if use_h5_rear_calibration
                else hashlib.sha256(camera["x5_offset"].encode()).hexdigest()
            )
            calibration_bundle_sha256 = hashlib.sha256(
                "|".join((
                    str(camera.get("rear_calibration_sha256", "")),
                    hashlib.sha256(camera["x5_offset"].encode()).hexdigest(),
                    str(camera.get("timeline_h5_sha256", "")),
                )).encode()
            ).hexdigest()
            lens_signature = {
                **processing_signature,
                "timeline_h5_sha256": camera.get("timeline_h5_sha256"),
                "calibration_bundle_sha256": calibration_bundle_sha256,
            }
            chunk_root = cache_root / "observations" / side / f"lens-{stream}" / "chunks"
            for index, (start, end) in enumerate(
                chunk_ranges(
                    int(lens["frame_count"]),
                    float(lens["fps"]),
                    float(budget["cache_chunk_duration_s"]),
                    alignment=stride,
                )
            ):
                output = chunk_root / f"chunk-{index:05d}-{start:09d}-{end:09d}.npz"
                command = [
                    str(PYTHON), "-m", "tools.cache_fisheye_apriltag_observations",
                    str(video), "--x5-offset", camera["x5_offset"],
                    "--camera-serial", camera["serial"], "--stream", str(stream),
                    "--calibration-bundle-sha256", calibration_bundle_sha256,
                    "--source-width", str(lens["width"]),
                    "--source-height", str(lens["height"]),
                    "--clock-intercept-s", str(intercept),
                    "--frame-stride", str(stride), "--start-frame", str(start),
                    "--decode-stride", str(decode_stride),
                    "--end-frame", str(end), "--stop-after-end-frame",
                    "--video-sha256", hashes[lens["path"]],
                    "--opencv-threads", str(threads),
                    "--rectified-detection", "--rectified-view-size", str(rectified_view_size),
                    "--rectified-detection-policy", "adaptive",
                    "--rectified-min-direct-tags", "4",
                    "--rectified-required-id", "2",
                    "--rectified-required-id", "3",
                    "--temporal-tracking",
                    "--global-scout-interval-frames", str(global_scout_interval),
                    "--global-scout-scale", str(global_scout_scale),
                    "--local-redetect-interval-frames", str(local_redetect_interval),
                    "--rectified-recovery-interval-frames", str(rectified_recovery_interval),
                    "--max-track-age-frames", str(max_track_age),
                    "--max-flow-forward-backward-error-px", "1.5",
                    "--max-reacquire-distance-px", "160",
                    "--optical-flow-scale", str(optical_flow_scale),
                    "--forward-backward-check-interval-frames", str(forward_backward_interval),
                    "--optical-flow-window-size", str(optical_flow_window_size),
                    "--optical-flow-max-level", str(optical_flow_max_level),
                    "--optical-flow-max-iterations", str(optical_flow_max_iterations),
                    "--output", str(output),
                ]
                if native_grayscale:
                    command.append("--native-grayscale-decode")
                if camera.get("timeline_h5"):
                    command.extend([
                        "--timeline-h5", str(root / camera["timeline_h5"]),
                        "--timeline-camera", str(camera["timeline_camera"]),
                    ])
                if use_h5_rear_calibration:
                    command.append("--instaumi-rear-calibration")
                if start:
                    command.append("--seek-to-start")
                tasks.append(
                    ChunkTask(
                        side=side,
                        stream=stream,
                        start=start,
                        end=end,
                        output=output,
                        command=command,
                        log=cache_root / "logs" / f"cache-{side}-{stream}-{index:05d}.log",
                        expected={
                            "video_sha256": hashes[lens["path"]],
                            "calibration_sha256": calibration_sha256,
                            "timeline_h5_sha256": camera.get("timeline_h5_sha256"),
                            "stream": stream,
                            "detection_frame_range": [start, end],
                            "decoded_frame_range": [start, end],
                            "frame_stride": stride,
                            "decode_stride": decode_stride,
                            "native_grayscale_decode": native_grayscale,
                            "optical_flow_scale": optical_flow_scale,
                            "forward_backward_check_interval_frames": forward_backward_interval,
                            "optical_flow_window_size": optical_flow_window_size,
                            "optical_flow_max_level": optical_flow_max_level,
                            "optical_flow_max_iterations": optical_flow_max_iterations,
                            "timestamp_source": timestamp_source,
                            "rectified_view_size": rectified_view_size,
                            "rectified_detection_policy": "adaptive",
                            "rectified_min_direct_tags": 4,
                            "rectified_required_ids": [2, 3],
                            "temporal_tracking": True,
                            "processing_signature": lens_signature,
                            "opencv_threads": threads,
                            "clock_mapping": {
                                "formula": "local_time = intercept_s + slope * common_time",
                                "intercept_s": intercept,
                                "slope": 1.0,
                            },
                        },
                    )
                )
    return sorted(tasks, key=lambda task: (task.start, task.side, task.stream))


def run_chunks(tasks: list[ChunkTask], budget: dict[str, Any], status: Path) -> None:
    pending = [task for task in tasks if not _valid_metadata(task.output, task.expected)]
    completed = len(tasks) - len(pending)
    status_update(
        status,
        "observation_chunks",
        "RUNNING" if pending else "REUSED",
        total=len(tasks),
        completed=completed,
        pending=len(pending),
    )
    environment = _worker_environment(int(budget["threads_per_worker"]))
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=int(budget["cache_workers"])
    ) as executor:
        futures = {
            executor.submit(run, task.command, task.log, environment=environment): task
            for task in pending
        }
        for future in concurrent.futures.as_completed(futures):
            future.result()
            completed += 1
            status_update(
                status,
                "observation_chunks",
                "RUNNING" if completed < len(tasks) else "PASS",
                total=len(tasks),
                completed=completed,
                pending=len(tasks) - completed,
            )


def merge_observations(
    pair: dict[str, Any],
    tasks: list[ChunkTask],
    cache_root: Path,
    hashes: dict[str, str],
    logs: Path,
) -> dict[str, Path]:
    dual: dict[str, Path] = {}
    for side in ("left", "right"):
        lens_outputs = []
        for lens in pair[side]["lenses"]:
            stream = int(lens["stream"])
            chunks = [
                task.output for task in tasks
                if task.side == side and task.stream == stream
            ]
            output = cache_root / "observations" / side / f"lens-{stream}-corners.npz"
            expected = {
                "video_sha256": hashes[lens["path"]],
                "stream": stream,
                "chunk_count": len(chunks),
                "processing_signature": next(
                    task.expected["processing_signature"]
                    for task in tasks
                    if task.side == side and task.stream == stream
                ),
            }
            if not _valid_metadata(output, expected):
                run(
                    [
                        str(PYTHON), "-m", "tools.merge_fisheye_observation_chunks",
                        *map(str, chunks), "--output", str(output),
                    ],
                    logs / f"merge-{side}-{stream}-chunks.log",
                )
            lens_outputs.append(output)
        output = cache_root / "observations" / side / "dual-lens-corners.npz"
        expected_hashes = [hashes[lens["path"]] for lens in pair[side]["lenses"]]
        expected_signature = json.loads(
            _sidecar(lens_outputs[0]).read_text(encoding="utf-8")
        ).get("processing_signature")
        if not _valid_metadata(output, {
            "source_video_sha256": expected_hashes,
            "processing_signature": expected_signature,
        }):
            run(
                [
                    str(PYTHON), "-m", "tools.merge_fisheye_observation_caches",
                    *map(str, lens_outputs), "--output", str(output),
                ],
                logs / f"merge-{side}-dual.log",
            )
        dual[side] = output
    return dual


def _tracking_path(root: Path, value: str) -> Path:
    path = Path(value)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_file():
        raise ManifestError(f"tracking input is missing: {path}")
    return path


def run_tracking(
    root: Path,
    pair: dict[str, Any],
    dual: dict[str, Path],
    cache_root: Path,
    budget: dict[str, Any],
    logs: Path,
) -> Path | None:
    tracking = pair.get("tracking")
    if not tracking or not tracking.get("enabled", False):
        automatic = pair.get("auto_tracking")
        if not automatic or not automatic.get("enabled", False):
            return None
        panel_a = _tracking_path(root, automatic["panel_a_map"])
        panel_b = _tracking_path(root, automatic["panel_b_map"])
        output = cache_root / "tracking"
        report = output / "report.json"
        signature_path = output / "input-signature.json"
        signature = {
            "schema_version": "four-mp4-auto-tracking-input/1.0",
            "algorithm_revision": (
                "cached-a3-shared-map-joint-v6-bounded-interpolation-hand-flu-back-x"
            ),
            "mode": automatic.get("mode"),
            "observation_processing_signature": {
                side: json.loads(_sidecar(path).read_text(encoding="utf-8"))[
                    "processing_signature"
                ]
                for side, path in dual.items()
            },
            "panel_map_sha256": {
                "A": sha256(panel_a),
                "B": sha256(panel_b),
            },
            "pair_id": pair["pair_id"],
        }
        cached_signature = None
        if signature_path.is_file():
            try:
                cached_signature = json.loads(signature_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        if not report.is_file() or cached_signature != signature:
            threads = min(int(budget["maximum_active_cpu_threads"]), 4)
            run(
                [
                    str(PYTHON), "-m", "tools.bootstrap_cached_a3_trajectories",
                    "--left-cache", str(dual["left"]),
                    "--right-cache", str(dual["right"]),
                    "--panel-a-map", str(panel_a),
                    "--panel-b-map", str(panel_b),
                    "--pair-id", pair["pair_id"],
                    "--opencv-threads", str(threads),
                    "--output-dir", str(output),
                ],
                logs / "auto-shared-map-tracking.log",
                environment=_worker_environment(threads),
                gate=True,
            )
            if not report.is_file():
                raise RuntimeError(f"automatic tracking did not produce {report}")
            atomic_json(signature_path, signature)
        return output
    required = {
        "left_initial_pose_common_time",
        "right_initial_pose_common_time",
        "initial_world_map",
    }
    missing = sorted(required - set(tracking))
    if missing:
        raise ManifestError(f"tracking descriptor is missing: {missing}")
    panel_a = _tracking_path(
        root, tracking.get("left_panel_map", str(ROOT / "config/a3_aprilgrid_A_200_205_120mm.json"))
    )
    panel_b = _tracking_path(
        root, tracking.get("right_panel_map", str(ROOT / "config/a3_aprilgrid_B_210_215_120mm.json"))
    )
    output = cache_root / "tracking"
    sample_stride = int(tracking.get(
        "sample_stride",
        max(
            1,
            round(
                float(pair["left"]["fps"])
                / float(budget["trajectory_observation_fps"])
            ),
        ),
    ))
    command = [
        str(PYTHON), "-m", "tools.joint_dual_camera_pose_graph_cached",
        "--left-cache", str(dual["left"]), "--right-cache", str(dual["right"]),
        "--left-initial-pose", str(_tracking_path(root, tracking["left_initial_pose_common_time"])),
        "--right-initial-pose", str(_tracking_path(root, tracking["right_initial_pose_common_time"])),
        "--left-panel-map", str(panel_a), "--right-panel-map", str(panel_b),
        "--initial-world-map", str(_tracking_path(root, tracking["initial_world_map"])),
        "--left-tag-id", str(pair["left"]["base_tag_id"]),
        "--right-tag-id", str(pair["right"]["base_tag_id"]),
        "--start-common-s", str(float(tracking.get("start_common_s", 0.0))),
        "--end-common-s", str(float(tracking.get("end_common_s", pair["common_duration_upper_bound_s"]))),
        "--sample-stride", str(sample_stride),
        "--alternations", str(int(tracking.get("alternations", 4))),
        "--workers", str(min(int(budget["maximum_active_cpu_threads"]), 8)),
        "--anchored-two-pass", "--output-dir", str(output),
    ]
    report = output / "report.json"
    signature_path = output / "input-signature.json"
    signature = {
        "schema_version": "four-mp4-tracking-input/1.0",
        "source_video_sha256": {
            side: json.loads(_sidecar(path).read_text(encoding="utf-8"))[
                "source_video_sha256"
            ]
            for side, path in dual.items()
        },
        "observation_processing_signature": {
            side: json.loads(_sidecar(path).read_text(encoding="utf-8"))[
                "processing_signature"
            ]
            for side, path in dual.items()
        },
        "tracking": tracking,
        "sample_stride": sample_stride,
        "workers": min(int(budget["maximum_active_cpu_threads"]), 8),
    }
    cached_signature = None
    if signature_path.is_file():
        try:
            cached_signature = json.loads(signature_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    if not report.is_file() or cached_signature != signature:
        run(command, logs / "joint-tracking.log", gate=True)
        if not report.is_file():
            raise RuntimeError(f"joint tracking did not produce {report}")
        atomic_json(signature_path, signature)
    return output


def process_pair(root: Path, pair: dict[str, Any], cache_root: Path, budget: dict[str, Any]) -> int:
    pair_id = validate_path_component(pair.get("pair_id"), field="manifest pair_id")
    cache_root.mkdir(parents=True, exist_ok=True)
    logs = cache_root / "logs"
    status = cache_root / "status.json"
    status_update(status, "identity", "RUNNING")
    hashes = source_hashes(root, pair, cache_root)
    status_update(status, "identity", "PASS", source_sha256=hashes)
    sync = resolve_sync(root, pair, cache_root, logs, hashes)
    status_update(status, "sync", "PASS", **sync)
    tasks = build_chunk_tasks(root, pair, cache_root, hashes, sync, budget)
    run_chunks(tasks, budget, status)
    dual = merge_observations(pair, tasks, cache_root, hashes, logs)
    status_update(
        status,
        "dual_lens_observations",
        "PASS",
        left=str(dual["left"]),
        right=str(dual["right"]),
        stitching_used=False,
        full_video_target_frame_scans_per_lens=1,
        chunk_seek_keyframe_overlap_possible=True,
    )
    tracking = run_tracking(root, pair, dual, cache_root, budget, logs)
    final = confined_path(
        root,
        "final",
        PIPELINE_REVISION,
        "pairs",
        pair_id,
        field="pair output directory",
    )
    final.mkdir(parents=True, exist_ok=True)
    atomic_json(
        final / "cache-index.json",
        {
            "schema_version": "four-mp4-cache-index/1.0",
            "source_sha256": hashes,
            "observation_cache": {side: str(path) for side, path in dual.items()},
            "resource_budget": budget,
            "sync": sync,
        },
    )
    payload = json.loads(status.read_text(encoding="utf-8"))
    if tracking is None:
        status_update(
            status,
            "trajectory_tracking",
            "WAITING_FOR_BOOTSTRAP_INPUTS",
            reason="raw/four-mp4.json has no enabled tracking descriptor",
        )
        overall = "OBSERVATIONS_READY"
    else:
        destination = confined_path(final, "tracking", field="tracking output directory")
        publish_directory(tracking, destination, allowed_root=root)
        report = json.loads((tracking / "report.json").read_text(encoding="utf-8"))
        tracking_passed = report.get("status") in {"HOLDOUT_PASS", "SELF_CALIBRATED_PASS"}
        status_update(
            status,
            "trajectory_tracking",
            "PASS" if tracking_passed else "FAIL",
            report_status=report.get("status"),
        )
        overall = (
            "COMPLETE"
            if tracking_passed
            else "TRACKING_GATE_FAILED"
        )
    payload = json.loads(status.read_text(encoding="utf-8"))
    payload["status"] = overall
    atomic_json(status, payload)
    atomic_json(final / "status.json", payload)
    return 0 if overall in {"OBSERVATIONS_READY", "COMPLETE"} else 2


def main() -> int:
    args = parse_args()
    root = args.dataset_root.resolve(strict=True)
    requested_pair_id = validate_path_component(args.pair_id, field="--pair-id")
    lock_path = confined_path(
        root, "final", PIPELINE_REVISION, "manifest.lock.json", field="manifest lock"
    )
    if not lock_path.is_file():
        raise ManifestError(f"internal manifest lock is missing: {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("pipeline_revision") != PIPELINE_REVISION:
        raise ManifestError("internal manifest lock uses a different pipeline revision")
    pairs = lock.get("pairs")
    if not isinstance(pairs, list):
        raise ManifestError("internal manifest lock pairs must be a list")
    for value in pairs:
        if not isinstance(value, dict):
            raise ManifestError("internal manifest lock pair must be an object")
        validate_path_component(value.get("pair_id"), field="manifest pair_id")
    pair = next(
        (value for value in pairs if value["pair_id"] == requested_pair_id),
        None,
    )
    if pair is None:
        raise ManifestError(f"pair not found in internal manifest: {requested_pair_id}")
    return process_pair(root, pair, args.cache_root.resolve(), lock["resource_budget"])


if __name__ == "__main__":
    raise SystemExit(main())
