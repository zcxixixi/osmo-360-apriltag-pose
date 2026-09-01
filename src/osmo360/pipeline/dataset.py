from __future__ import annotations

import concurrent.futures
import json
import os
import re
import shlex
import socket
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .devices import load_device_pairs
from .manifest import ManifestError, ROOT

FFPROBE = ROOT / "work/tools/ffmpeg-master-latest-linux64-gpl/bin/ffprobe"
SERIAL_PATTERN = re.compile(rb"IAHE[A-Z0-9]{10}")
OFFSET_PATTERN = re.compile(rb"[mn]2(?:_-?\d+(?:\.\d+)?){15}")
TIME_PATTERN = re.compile(r"VID_(\d{8})_(\d{6})_")
PIPELINE_REVISION = "instaumi-align-v1"


@dataclass(frozen=True)
class RawVideo:
    side: str
    path: Path
    serial: str
    base_tag_id: int
    recorded_at: datetime
    duration_s: float
    fps: float
    width: int
    height: int
    x5_offset: str

    def record(self, root: Path) -> dict[str, Any]:
        return {
            "path": self.path.relative_to(root).as_posix(),
            "size_bytes": self.path.stat().st_size,
            "serial": self.serial,
            "side": self.side,
            "base_tag_id": self.base_tag_id,
            "recorded_at": self.recorded_at.isoformat(),
            "duration_s": self.duration_s,
            "fps": self.fps,
            "lens_size": [self.width, self.height],
            "lens_tracks": 2,
            "x5_offset": self.x5_offset,
        }


def _registered_sides() -> dict[str, tuple[str, int]]:
    result: dict[str, tuple[str, int]] = {}
    for pair in load_device_pairs()["pairs"].values():
        for side in ("left", "right"):
            item = pair[side]
            result[str(item["serial"])] = (side, int(item["base_tag_id"]))
    return result


def _probe(path: Path) -> dict[str, float | int]:
    if not FFPROBE.is_file():
        raise ManifestError(f"bundled ffprobe is missing: {FFPROBE}")
    process = subprocess.run(
        [str(FFPROBE), "-v", "error", "-select_streams", "v",
         "-show_entries", "stream=width,height,r_frame_rate,duration",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True, timeout=120,
    )
    streams = json.loads(process.stdout).get("streams", [])
    if len(streams) != 2:
        raise ManifestError(f"X5 INSV must contain two raw lens tracks: {path}")
    first = streams[0]
    numerator, denominator = map(float, first["r_frame_rate"].split("/"))
    return {
        "width": int(first["width"]),
        "height": int(first["height"]),
        "fps": numerator / denominator,
        "duration_s": min(float(stream["duration"]) for stream in streams),
    }


def inspect_video(path: Path, expected_side: str) -> RawVideo:
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 16 * 1024 * 1024))
        tail = handle.read()
    serial_match = SERIAL_PATTERN.search(tail)
    offset_match = OFFSET_PATTERN.search(tail)
    time_match = TIME_PATTERN.search(path.name)
    if serial_match is None or offset_match is None or time_match is None:
        raise ManifestError(f"INSV identity/calibration metadata is incomplete: {path}")
    serial = serial_match.group().decode("ascii")
    binding = _registered_sides().get(serial)
    if binding is None:
        raise ManifestError(f"unregistered X5 serial: {serial}")
    side, base_tag_id = binding
    if side != expected_side:
        raise ManifestError(
            f"{path} is under raw/{expected_side} but serial {serial} is {side}"
        )
    probe = _probe(path)
    return RawVideo(
        side=side,
        path=path,
        serial=serial,
        base_tag_id=base_tag_id,
        recorded_at=datetime.strptime("".join(time_match.groups()), "%Y%m%d%H%M%S"),
        duration_s=float(probe["duration_s"]),
        fps=float(probe["fps"]),
        width=int(probe["width"]),
        height=int(probe["height"]),
        x5_offset=offset_match.group().decode("ascii"),
    )


def _source_checksums(root: Path) -> dict[str, str]:
    path = root / "sha256.txt"
    if not path.is_file():
        return {}
    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) == 2 and re.fullmatch(r"[0-9a-fA-F]{64}", fields[0]):
            checksums[fields[1].lstrip("*./")] = fields[0].lower()
    return checksums


def discover_dataset(dataset_root: Path) -> dict[str, Any]:
    root = dataset_root.resolve(strict=True)
    checksums = _source_checksums(root)
    paths: list[tuple[Path, str]] = []
    for side in ("left", "right"):
        directory = root / "raw" / side
        if not directory.is_dir():
            raise ManifestError(f"required input directory is missing: {directory}")
        paths.extend((path, side) for path in sorted(directory.glob("*.insv")))
    if not paths:
        raise ManifestError("raw/left and raw/right contain no INSV files")
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(paths))) as executor:
        videos = list(executor.map(lambda item: inspect_video(*item), paths))
    long_videos = [video for video in videos if video.duration_s >= 30.0]
    short_videos = [video for video in videos if video.duration_s < 30.0]
    left = sorted((video for video in long_videos if video.side == "left"), key=lambda item: item.recorded_at)
    right = sorted((video for video in long_videos if video.side == "right"), key=lambda item: item.recorded_at)
    if len(left) != len(right):
        raise ManifestError(
            f"usable left/right capture counts differ: left={len(left)} right={len(right)}"
        )
    pairs = []
    for index, (left_video, right_video) in enumerate(zip(left, right), 1):
        start_delta = (right_video.recorded_at - left_video.recorded_at).total_seconds()
        if abs(start_delta) > 15.0:
            raise ManifestError(
                f"pair {index} recording starts differ by {start_delta:.3f}s (>15s)"
            )
        pair_id = f"pair-{index:02d}-{max(left_video.recorded_at, right_video.recorded_at):%H%M%S}"
        left_record = left_video.record(root)
        right_record = right_video.record(root)
        left_record["sha256"] = checksums.get(left_record["path"], "")
        right_record["sha256"] = checksums.get(right_record["path"], "")
        pairs.append({
            "pair_id": pair_id,
            "dataset_id": f"instaumi_{index:06d}",
            "left": left_record,
            "right": right_record,
            "recording_start_delta_s": start_delta,
            "common_duration_upper_bound_s": min(left_video.duration_s, right_video.duration_s),
        })
    return {
        "schema_version": "dual-x5-dataset-lock/1.0",
        "pipeline_revision": PIPELINE_REVISION,
        "dataset_root": ".",
        "pair_count": len(pairs),
        "pairs": pairs,
        "ignored_short_recordings": [video.record(root) for video in short_videos],
    }


def _nodes() -> list[str]:
    configured = os.environ.get(
        "OSMO_PIPELINE_NODES", "local,current@192.168.109.124"
    )
    nodes = [value.strip() for value in configured.split(",") if value.strip()]
    if not nodes or nodes[0] != "local":
        raise ManifestError("OSMO_PIPELINE_NODES must begin with local")
    return nodes


def _worker_command(root: Path, pair_id: str, dataset_id: str, scratch: Path) -> list[str]:
    return [
        "./.venv/bin/python", "-m", "osmo360.pipeline.dataset_worker",
        str(root), "--pair-id", pair_id, "--dataset-id", dataset_id,
        "--scratch-root", str(scratch),
    ]


def _run_assigned(node: str, commands: list[list[str]]) -> list[dict[str, Any]]:
    results = []
    remote_repo = os.environ.get(
        "OSMO_REMOTE_REPO", "/home/current/osmo-360-apriltag-pose"
    )
    for command in commands:
        if node == "local":
            actual = command
        else:
            remote_command = f"cd {shlex.quote(remote_repo)} && {shlex.join(command)}"
            actual = ["ssh", node, remote_command]
        process = subprocess.run(actual, cwd=ROOT, text=True, capture_output=True)
        results.append({
            "node": socket.gethostname() if node == "local" else node,
            "command": command,
            "returncode": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
        })
    return results


def process_dataset(dataset_root: Path, *, dry_run: bool = False) -> dict[str, Any]:
    root = dataset_root.resolve(strict=True)
    lock = discover_dataset(root)
    # Each standardized dataset is self-contained. The worker lock is stored
    # with derived data, beside (not inside) dataset.h5 and video/.
    for pair in lock["pairs"]:
        processed = root / pair["dataset_id"] / "processed"
        processed.mkdir(parents=True, exist_ok=True)
        (processed / "manifest.lock.json").write_text(
            json.dumps({**lock, "pairs": [pair], "pair_count": 1}, indent=2) + "\n",
            encoding="utf-8",
        )
    lock_path = root / lock["pairs"][0]["dataset_id"] / "processed" / "manifest.lock.json" if lock["pairs"] else root / "raw" / "manifest.empty.json"
    nodes = _nodes()
    scratch_base = Path(os.environ.get("OSMO_PIPELINE_SCRATCH", "/tmp/osmo-pipeline"))
    assignments: dict[str, list[list[str]]] = {node: [] for node in nodes}
    for index, pair in enumerate(lock["pairs"]):
        node = nodes[index % len(nodes)]
        scratch = scratch_base / root.name / PIPELINE_REVISION / pair["dataset_id"]
        assignments[node].append(
            _worker_command(root, pair["pair_id"], pair["dataset_id"], scratch)
        )
    if dry_run:
        return {
            "status": "DRY_RUN",
            "manifest_lock": str(lock_path),
            "assignments": assignments,
        }
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes)) as executor:
        futures = [
            executor.submit(_run_assigned, node, commands)
            for node, commands in assignments.items()
        ]
        results = [item for future in futures for item in future.result()]
    failed = [result for result in results if result["returncode"] != 0]
    status = {
        "schema_version": "dual-x5-dataset-pipeline-status/1.0",
        "status": "FAILED" if failed else "COMPLETE",
        "pipeline_revision": PIPELINE_REVISION,
        "manifest_lock": str(lock_path.relative_to(root)),
        "nodes": nodes,
        "results": results,
    }
    for pair in lock["pairs"]:
        status_path = root / pair["dataset_id"] / "processed" / "pipeline_status.json"
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    if failed:
        raise ManifestError(f"{len(failed)} pair worker(s) failed; see instaumi_*/processed/pipeline_status.json")
    return status
