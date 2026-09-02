from __future__ import annotations

import argparse
import concurrent.futures
import csv
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate, correlation_lags

from .devices import load_inventory
from .insta360_telemetry import extract_x5_imu
from .instaumi_format import common_window, write_dataset_h5
from .manifest import ManifestError, ROOT

AUTOMATION_REVISION = "instaumi-auto-v2"
TARGET_FPS = "30000/1001"
TARGET_FPS_FLOAT = 30000 / 1001
SERIAL_PATTERN = re.compile(rb"IAHE[A-Z0-9]{10}")
OFFSET_PATTERN = re.compile(rb"[mn]2(?:_-?\d+(?:\.\d+)?){15}")
TIME_PATTERN = re.compile(r"VID_(\d{8})_(\d{6})_")
COLLECTOR_PATTERN = re.compile(r"^\d{4}_instaumi_[a-z0-9_]+$")
GRIPPER_PROFILE = (
    ROOT / "config/rig_revisions/instaumi_pair01_gripper_signal_20260902_r3.json"
)
FFMPEG = ROOT / "work/tools/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg"
FFPROBE = FFMPEG.with_name("ffprobe")


@dataclass(frozen=True)
class Source:
    side: str
    path: Path
    relative_name: str
    sha256: str
    recorded_at: datetime


@dataclass(frozen=True)
class Pair:
    collector_root: Path
    left: Source
    right: Source

    @property
    def episode_name(self) -> str:
        return f"instaumi_{self.left.recorded_at:%Y%m%d_%H%M%S}"

    @property
    def key(self) -> str:
        return ":".join((
            self.collector_root.name,
            self.left.path.name,
            self.right.path.name,
        ))


class PipelineFailure(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        return default
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PipelineFailure(f"cannot read automation state {path}: {error}") from error
    if not isinstance(value, dict):
        raise PipelineFailure(f"automation state must be an object: {path}")
    return value


def _sha_registry(raw_root: Path) -> dict[str, str]:
    path = raw_root / "sha256.txt"
    if not path.is_file():
        return {}
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or not re.fullmatch(r"[0-9a-fA-F]{64}", fields[0]):
            raise PipelineFailure(f"malformed SHA-256 line in {path}: {line!r}")
        result[fields[1].lstrip("*./")] = fields[0].lower()
    return result


def _recorded_at(path: Path) -> datetime:
    match = TIME_PATTERN.search(path.name)
    if match is None:
        raise PipelineFailure(f"INSV filename has no recording timestamp: {path}")
    return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")


def _sources_with_sha(collector_root: Path, side: str) -> dict[str, Source]:
    raw = collector_root / "raw"
    registry = _sha_registry(raw)
    directory = raw / side
    if not directory.is_dir():
        return {}
    result = {}
    for path in sorted(directory.glob("*.insv")):
        relative = f"{side}/{path.name}"
        expected = registry.get(relative)
        if expected is None:
            continue
        result[path.name] = Source(
            side=side,
            path=path,
            relative_name=relative,
            sha256=expected,
            recorded_at=_recorded_at(path),
        )
    return result


def _approved_mapping(data_root: Path) -> list[dict[str, Any]]:
    candidates = sorted((data_root / "_review").glob("*/alignment-review-v1/collector_video_mapping.json"))
    items: list[dict[str, Any]] = []
    for path in candidates:
        payload = _load_json(path, {})
        raw_items = payload.get("items", [])
        if isinstance(raw_items, list):
            items.extend(item for item in raw_items if isinstance(item, dict))
    return items


def discover_pairs(data_root: Path, collectors: set[str] | None = None) -> list[Pair]:
    roots = [
        path for path in sorted(data_root.iterdir())
        if path.is_dir()
        and COLLECTOR_PATTERN.fullmatch(path.name)
        and (collectors is None or path.name in collectors)
    ]
    approved = _approved_mapping(data_root)
    result: list[Pair] = []
    for collector_root in roots:
        left = _sources_with_sha(collector_root, "left")
        right = _sources_with_sha(collector_root, "right")
        used_left: set[str] = set()
        used_right: set[str] = set()
        collector = collector_root.name.rsplit("_", 1)[-1]
        for item in approved:
            left_name = item.get("left_source")
            right_name = item.get("right_source")
            if item.get("collector") != collector:
                continue
            left_is_here = left_name in left
            right_is_here = right_name in right
            if left_is_here:
                used_left.add(str(left_name))
            if right_is_here:
                used_right.add(str(right_name))
            if item.get("usable") is True and left_is_here and right_is_here:
                result.append(Pair(collector_root, left[left_name], right[right_name]))

        candidates = sorted(
            (
                abs((right_source.recorded_at - left_source.recorded_at).total_seconds()),
                left_name,
                right_name,
            )
            for left_name, left_source in left.items()
            if left_name not in used_left
            for right_name, right_source in right.items()
            if right_name not in used_right
        )
        for delta_s, left_name, right_name in candidates:
            if delta_s > 60:
                break
            if left_name in used_left or right_name in used_right:
                continue
            result.append(Pair(collector_root, left[left_name], right[right_name]))
            used_left.add(left_name)
            used_right.add(right_name)
    return sorted(result, key=lambda pair: (pair.left.recorded_at, pair.collector_root.name))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_source(source: Source, state: dict[str, Any]) -> None:
    sources = state.setdefault("sources", {})
    stat = source.path.stat()
    key = str(source.path)
    cached = sources.get(key, {})
    if (
        cached.get("size_bytes") == stat.st_size
        and cached.get("mtime_ns") == stat.st_mtime_ns
        and cached.get("sha256") == source.sha256
        and cached.get("verified") is True
    ):
        return
    actual = _sha256(source.path)
    if actual != source.sha256:
        raise PipelineFailure(
            f"SHA-256 mismatch for {source.path}: expected {source.sha256}, found {actual}"
        )
    sources[key] = {
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": actual,
        "verified": True,
        "verified_at_utc": _utc_now(),
    }


def _probe_source(source: Source) -> dict[str, Any]:
    with source.path.open("rb") as handle:
        handle.seek(max(0, source.path.stat().st_size - 16 * 1024 * 1024))
        tail = handle.read()
    serial_match = SERIAL_PATTERN.search(tail)
    offset_match = OFFSET_PATTERN.search(tail)
    if serial_match is None or offset_match is None:
        raise PipelineFailure(f"INSV identity or X5 lens metadata is missing: {source.path}")
    serial = serial_match.group().decode("ascii")
    inventory = load_inventory().get("devices", {})
    assignment = inventory.get(serial, {}).get("assignment")
    expected_role = f"physical_{source.side}"
    if not isinstance(assignment, dict) or assignment.get("role") != expected_role:
        raise PipelineFailure(f"camera {serial} has no {expected_role} assignment")
    process = subprocess.run([
        str(FFPROBE),
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,width,height,avg_frame_rate,duration",
        "-of",
        "json",
        str(source.path),
    ], check=True, capture_output=True, text=True, timeout=120)
    streams = json.loads(process.stdout).get("streams", [])
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    if len(videos) != 2:
        raise PipelineFailure(f"X5 INSV must contain two video streams: {source.path}")
    duration_s = min(float(stream["duration"]) for stream in videos)
    width = int(videos[0]["width"])
    height = int(videos[0]["height"])
    numerator, denominator = map(int, videos[0]["avg_frame_rate"].split("/"))
    return {
        "path": source.relative_name,
        "size_bytes": source.path.stat().st_size,
        "serial": serial,
        "side": source.side,
        "base_tag_id": int(assignment["base_tag_id"]),
        "recorded_at": source.recorded_at.isoformat(),
        "duration_s": duration_s,
        "fps": numerator / denominator,
        "lens_size": [width, height],
        "lens_tracks": 2,
        "x5_offset": offset_match.group().decode("ascii"),
        "sha256": source.sha256,
    }


def _run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write_text(process.stdout, encoding="utf-8")
    if process.returncode:
        raise PipelineFailure(
            f"command failed ({process.returncode}): {' '.join(command)}; log={log}"
        )


def _extract_audio(source: Path, output: Path, log: Path) -> None:
    _run([
        str(FFMPEG), "-v", "error", "-y", "-i", str(source), "-vn",
        "-ac", "1", "-ar", "2000", "-c:a", "pcm_s16le", str(output),
    ], log)


def _audio_offset(left_path: Path, right_path: Path, approximate_s: float) -> dict[str, float]:
    left_rate, left = wavfile.read(left_path)
    right_rate, right = wavfile.read(right_path)
    if left_rate != right_rate:
        raise PipelineFailure("left/right audio sample rates differ")
    left = left.astype(np.float64)
    right = right.astype(np.float64)
    left -= left.mean()
    right -= right.mean()
    correlation = correlate(right, left, mode="full", method="fft")
    lags = correlation_lags(len(right), len(left), mode="full")
    expected = int(round(approximate_s * left_rate))
    keep = np.abs(lags - expected) <= 5 * left_rate
    index = int(np.flatnonzero(keep)[int(np.argmax(correlation[keep]))])
    denominator = math.sqrt(float(np.dot(left, left) * np.dot(right, right)))
    return {
        "mapping": "right_time_s = left_time_s + offset_s",
        "offset_s": int(lags[index]) / left_rate,
        "correlation": float(correlation[index] / denominator),
        "uncertainty_s": 1 / left_rate,
    }


def _encode_lens(
    source: Path,
    output: Path,
    *,
    stream: int,
    start_s: float,
    frame_count: int,
    size: int,
    log: Path,
) -> None:
    temporary = output.with_name(output.stem + ".partial.mp4")
    temporary.unlink(missing_ok=True)
    _run([
        str(FFMPEG), "-v", "error", "-y", "-ss", f"{start_s:.9f}",
        "-i", str(source), "-map", f"0:v:{stream}", "-an", "-vf",
        f"fps={TARGET_FPS},scale={size}:{size}:flags=lanczos", "-frames:v",
        str(frame_count), "-c:v", "libx265", "-preset", "ultrafast",
        "-crf", "24", "-pix_fmt", "yuv420p", "-tag:v", "hvc1",
        "-bf", "0", "-g", "30", "-movflags", "+faststart", str(temporary),
    ], log)
    temporary.replace(output)


def _write_alignment(path: Path, pair: Pair, sync: dict[str, float], left_start: float, right_start: float, duration: float) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "dataset_id", "method", "left_source", "right_source",
            "left_source_start_s", "right_source_start_s", "duration_s",
            "right_minus_left_s", "correlation", "uncertainty_s",
        ))
        writer.writerow((
            pair.episode_name, "audio_cross_correlation", pair.left.path.name,
            pair.right.path.name, f"{left_start:.9f}", f"{right_start:.9f}",
            f"{duration:.9f}", f"{sync['offset_s']:.9f}",
            f"{sync['correlation']:.9f}", f"{sync['uncertainty_s']:.9f}",
        ))


def _format_complete(episode: Path) -> bool:
    return (episode / "dataset.h5").is_file() and (episode / "processed/time_alignment.csv").is_file() and all(
        (episode / "video" / name).is_file()
        for name in (
            "Left.mp4", "Right.mp4", "Left_back.mp4", "Left_forward.mp4",
            "Right_back.mp4", "Right_forward.mp4",
        )
    )


def _full_export_available(episode: Path) -> bool:
    if not _format_complete(episode) or not GRIPPER_PROFILE.is_file():
        return False
    with h5py.File(episode / "dataset.h5", "r") as handle:
        raw = handle["metadata/dataset.json"][()]
    metadata = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
    profile = json.loads(GRIPPER_PROFILE.read_text(encoding="utf-8"))
    return all(
        metadata.get("devices", {}).get(side, {}).get("serial_number")
        == profile.get("sides", {}).get(side, {}).get("camera_serial")
        for side in ("left", "right")
    )


def _process_complete(episode: Path) -> bool:
    processed = episode / "processed"
    if all(
        (processed / name).is_file()
        for name in ("trajectory.csv", "gripper.csv", "processed.csv", "metadata.csv")
    ):
        return True
    status_path = processed / "automation_status.json"
    if not (processed / "trajectory.csv").is_file() or not status_path.is_file():
        return False
    status = _load_json(status_path, {})
    return (
        status.get("status") == "COMPLETE"
        and status.get("mode") == "trajectory_only"
    )


def format_pair(pair: Pair, automation_root: Path) -> Path:
    episode = pair.collector_root / pair.episode_name
    if _format_complete(episode):
        return episode
    if episode.exists():
        raise PipelineFailure(f"incomplete existing episode requires review: {episode}")

    staging = pair.collector_root / f".{pair.episode_name}.formatting"
    shutil.rmtree(staging, ignore_errors=True)
    video = staging / "video"
    processed = staging / "processed"
    video.mkdir(parents=True)
    processed.mkdir()
    scratch = automation_root / "work" / hashlib.sha256(pair.key.encode()).hexdigest()[:16]
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    logs = automation_root / "logs" / pair.collector_root.name / pair.episode_name

    left_record, right_record = _probe_source(pair.left), _probe_source(pair.right)
    audio = scratch / "audio"
    audio.mkdir()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = (
            executor.submit(_extract_audio, pair.left.path, audio / "left.wav", logs / "audio-left.log"),
            executor.submit(_extract_audio, pair.right.path, audio / "right.wav", logs / "audio-right.log"),
        )
        for future in futures:
            future.result()
    approximate = (pair.right.recorded_at - pair.left.recorded_at).total_seconds()
    sync = _audio_offset(audio / "left.wav", audio / "right.wav", approximate)
    left_start, right_start, duration = common_window(
        left_record["duration_s"], right_record["duration_s"], sync["offset_s"]
    )
    frame_count = max(1, int(round(duration * TARGET_FPS_FLOAT)))

    jobs = []
    for side, source, start in (
        ("Left", pair.left.path, left_start),
        ("Right", pair.right.path, right_start),
    ):
        for stream, lens in ((0, "back"), (1, "forward")):
            jobs.append((source, video / f"{side}_{lens}.mp4", stream, start, 1920, f"{side.lower()}-{lens}"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                _encode_lens, source, output, stream=stream, start_s=start,
                frame_count=frame_count, size=size, log=logs / f"video-{name}.log",
            )
            for source, output, stream, start, size, name in jobs
        ]
        for future in futures:
            future.result()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                _encode_lens,
                video / f"{side}_back.mp4",
                video / f"{side}.mp4",
                stream=0,
                start_s=0.0,
                frame_count=frame_count,
                size=1024,
                log=logs / f"video-{side.lower()}-1024.log",
            )
            for side in ("Left", "Right")
        ]
        for future in futures:
            future.result()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        imu_futures = {
            "left": executor.submit(
                extract_x5_imu, pair.left.path, scratch / "imu-left",
                source_start_s=left_start, duration_s=duration,
                expected_serial=left_record["serial"],
            ),
            "right": executor.submit(
                extract_x5_imu, pair.right.path, scratch / "imu-right",
                source_start_s=right_start, duration_s=duration,
                expected_serial=right_record["serial"],
            ),
        }
        imu = {side: future.result() for side, future in imu_futures.items()}

    write_dataset_h5(
        staging / "dataset.h5",
        dataset_id=pair.episode_name,
        left_video=video / "Left.mp4",
        right_video=video / "Right.mp4",
        left_source=pair.left.path,
        right_source=pair.right.path,
        left_start_s=left_start,
        right_start_s=right_start,
        sync=sync,
        ffprobe=FFPROBE,
        source_records={"left": left_record, "right": right_record},
        imu=imu,
    )
    _write_alignment(
        processed / "time_alignment.csv", pair, sync, left_start, right_start, duration
    )
    _atomic_json(processed / "automation_status.json", {
        "schema_version": "instaumi-automation-status/1.0",
        "revision": AUTOMATION_REVISION,
        "status": "FORMATTED",
        "updated_at_utc": _utc_now(),
        "sources": {
            "left": {"path": pair.left.relative_name, "sha256": pair.left.sha256},
            "right": {"path": pair.right.relative_name, "sha256": pair.right.sha256},
        },
        "frame_rate": TARGET_FPS,
        "frame_count": frame_count,
    })
    staging.replace(episode)
    shutil.rmtree(scratch, ignore_errors=True)
    return episode


def _pair_status(state: dict[str, Any], pair: Pair) -> dict[str, Any]:
    return state.setdefault("pairs", {}).setdefault(pair.key, {})


def _record_failure(automation_root: Path, state: dict[str, Any], pair: Pair, error: Exception) -> None:
    status = _pair_status(state, pair)
    attempts = int(status.get("attempts", 0)) + 1
    retry_s = min(3600, 60 * (2 ** min(attempts - 1, 6)))
    status.update({
        "status": "FAILED",
        "attempts": attempts,
        "updated_at_utc": _utc_now(),
        "next_retry_unix_s": time.time() + retry_s,
        "error": f"{type(error).__name__}: {error}",
    })
    safe_name = hashlib.sha256(pair.key.encode()).hexdigest()[:16]
    _atomic_json(automation_root / "failures" / f"{safe_name}.json", {
        "pair": pair.key,
        **status,
    })


def scan_once(
    data_root: Path,
    process_script: Path,
    *,
    collectors: set[str] | None = None,
    episode_name: str | None = None,
    max_pairs: int = 1,
    dry_run: bool = False,
) -> dict[str, Any]:
    data_root = data_root.resolve(strict=True)
    process_script = process_script.resolve(strict=True)
    automation_root = data_root / "_automation"
    automation_root.mkdir(exist_ok=True)
    state_path = automation_root / "state.json"
    lock_path = automation_root / "scan.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "BUSY", "processed": []}
        state = _load_json(state_path, {
            "schema_version": "instaumi-auto-state/1.0",
            "revision": AUTOMATION_REVISION,
            "sources": {},
            "pairs": {},
        })
        pairs = discover_pairs(data_root, collectors)
        if episode_name is not None:
            pairs = [pair for pair in pairs if pair.episode_name == episode_name]
        processed_pairs = []
        skipped = []
        for pair in pairs:
            episode = pair.collector_root / pair.episode_name
            pair_status = _pair_status(state, pair)
            if _process_complete(episode):
                pair_status.update({
                    "status": "COMPLETE",
                    "episode": str(episode),
                    "updated_at_utc": _utc_now(),
                })
                skipped.append({"pair": pair.key, "reason": "complete"})
                continue
            if float(pair_status.get("next_retry_unix_s", 0)) > time.time():
                skipped.append({"pair": pair.key, "reason": "retry_backoff"})
                continue
            if dry_run:
                formatted = _format_complete(episode)
                mode = (
                    "full"
                    if formatted and _full_export_available(episode)
                    else "trajectory_only"
                )
                processed_pairs.append({
                    "pair": pair.key,
                    "episode": str(episode),
                    "action": "process" if formatted else "format_then_process",
                    "mode": mode,
                })
                if len(processed_pairs) >= max_pairs:
                    break
                continue
            try:
                _verify_source(pair.left, state)
                _verify_source(pair.right, state)
                pair_status.update({
                    "status": "RUNNING",
                    "stage": "format",
                    "updated_at_utc": _utc_now(),
                })
                _atomic_json(state_path, state)
                episode = format_pair(pair, automation_root)
                mode = "full" if _full_export_available(episode) else "trajectory_only"
                pair_status.update({
                    "status": "RUNNING",
                    "stage": "trajectory",
                    "mode": mode,
                    "episode": str(episode),
                    "updated_at_utc": _utc_now(),
                })
                _atomic_json(state_path, state)
                log = automation_root / "logs" / pair.collector_root.name / pair.episode_name / "process.log"
                command = [str(process_script)]
                if mode == "trajectory_only":
                    command.append("--trajectory-only")
                command.append(str(episode))
                _run(command, log)
                if not _process_complete(episode) and not (
                    mode == "trajectory_only"
                    and (episode / "processed/trajectory.csv").is_file()
                ):
                    raise PipelineFailure(f"downstream outputs are incomplete: {episode / 'processed'}")
                outputs = (
                    ["trajectory.csv", "gripper.csv", "processed.csv", "metadata.csv"]
                    if mode == "full"
                    else ["trajectory.csv"]
                )
                pair_status.update({
                    "status": "COMPLETE",
                    "stage": "complete",
                    "mode": mode,
                    "episode": str(episode),
                    "updated_at_utc": _utc_now(),
                    "attempts": int(pair_status.get("attempts", 0)),
                })
                pair_status.pop("error", None)
                pair_status.pop("next_retry_unix_s", None)
                _atomic_json(episode / "processed/automation_status.json", {
                    "schema_version": "instaumi-automation-status/1.0",
                    "revision": AUTOMATION_REVISION,
                    "status": "COMPLETE",
                    "mode": mode,
                    "updated_at_utc": _utc_now(),
                    "source_pair": pair.key,
                    "outputs": outputs,
                })
                processed_pairs.append({
                    "pair": pair.key,
                    "episode": str(episode),
                    "action": "complete",
                    "mode": mode,
                })
            except Exception as error:
                _record_failure(automation_root, state, pair, error)
                processed_pairs.append({"pair": pair.key, "action": "failed", "error": str(error)})
            finally:
                _atomic_json(state_path, state)
            if len(processed_pairs) >= max_pairs:
                break
        _atomic_json(state_path, state)
        return {
            "status": "COMPLETE",
            "revision": AUTOMATION_REVISION,
            "discovered_pairs": len(pairs),
            "processed": processed_pairs,
            "skipped_count": len(skipped),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover, format and process uploaded InstaUMI INSV pairs")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/ps/current-robotics-data-2/total_annotation/umi_insta360"),
    )
    parser.add_argument(
        "--process-script",
        type=Path,
        default=ROOT / "bin/process_instaumi_dataset.sh",
    )
    parser.add_argument("--collector", action="append", default=[])
    parser.add_argument("--episode")
    parser.add_argument("--max-pairs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_pairs <= 0:
        raise SystemExit("--max-pairs must be positive")
    result = scan_once(
        args.data_root,
        args.process_script,
        collectors=set(args.collector) or None,
        episode_name=args.episode,
        max_pairs=args.max_pairs,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))
    return 0 if not any(item.get("action") == "failed" for item in result.get("processed", [])) else 1


if __name__ == "__main__":
    raise SystemExit(main())
