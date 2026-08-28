#!/usr/bin/env python3
"""Measure commands and summarize sequential/phase-parallel pipeline runtime."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import time
from pathlib import Path
from typing import Any


def measure(command: list[str], *, stage: str, media_duration_s: float | None,
            frame_count: int | None, cwd: Path | None) -> dict[str, Any]:
    if not command:
        raise ValueError("measurement command is empty")
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.perf_counter()
    process = subprocess.run(command, cwd=cwd, check=False)
    wall_s = time.perf_counter() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    result: dict[str, Any] = {
        "schema_version": "command-runtime-measurement/1.0",
        "stage": stage,
        "command": command,
        "cwd": str(cwd.resolve()) if cwd else None,
        "returncode": process.returncode,
        "wall_s": wall_s,
        "user_cpu_s": after.ru_utime - before.ru_utime,
        "system_cpu_s": after.ru_stime - before.ru_stime,
        "max_rss_kib": after.ru_maxrss,
        "host": platform.node(),
        "logical_cpu_count": os.cpu_count(),
    }
    if media_duration_s is not None:
        result["media_duration_s"] = media_duration_s
        result["realtime_factor"] = wall_s / media_duration_s
        result["media_seconds_per_wall_second"] = media_duration_s / wall_s
    if frame_count is not None:
        result["frame_count"] = frame_count
        result["throughput_fps"] = frame_count / wall_s
    return result


def summarize_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    media_duration_s = float(payload["media_duration_s"])
    stages = payload.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("runtime manifest needs non-empty stages")
    normalized = []
    phase_order = []
    phase_wall: dict[str, float] = {}
    for item in stages:
        name = str(item["name"])
        phase = str(item["phase"])
        wall_s = float(item["wall_s"])
        if wall_s < 0:
            raise ValueError(f"stage {name} wall_s must be non-negative")
        normalized.append({**item, "name": name, "phase": phase, "wall_s": wall_s})
        if phase not in phase_wall:
            phase_order.append(phase)
            phase_wall[phase] = 0.0
        phase_wall[phase] = max(phase_wall[phase], wall_s)
    sequential_s = sum(item["wall_s"] for item in normalized)
    phase_parallel_s = sum(phase_wall[phase] for phase in phase_order)
    bottleneck = max(normalized, key=lambda item: item["wall_s"])
    return {
        "schema_version": "pipeline-runtime-summary/1.0",
        "label": payload.get("label"),
        "media_duration_s": media_duration_s,
        "stages": normalized,
        "phase_critical_path": [
            {"phase": phase, "wall_s": phase_wall[phase]} for phase in phase_order
        ],
        "sequential": {
            "wall_s": sequential_s,
            "minutes": sequential_s / 60.0,
            "realtime_factor": sequential_s / media_duration_s,
        },
        "phase_parallel": {
            "wall_s": phase_parallel_s,
            "minutes": phase_parallel_s / 60.0,
            "realtime_factor": phase_parallel_s / media_duration_s,
        },
        "bottleneck": {
            "stage": bottleneck["name"],
            "wall_s": bottleneck["wall_s"],
            "share_of_sequential": bottleneck["wall_s"] / sequential_s,
        },
        "assumption": "Stages sharing a phase run concurrently; phases run in listed order.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument("--stage", required=True)
    measure_parser.add_argument("--output", type=Path, required=True)
    measure_parser.add_argument("--media-duration-s", type=float)
    measure_parser.add_argument("--frame-count", type=int)
    measure_parser.add_argument("--cwd", type=Path)
    measure_parser.add_argument("command", nargs=argparse.REMAINDER)
    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("manifest", type=Path)
    summary_parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "measure":
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        result = measure(
            command, stage=args.stage, media_duration_s=args.media_duration_s,
            frame_count=args.frame_count, cwd=args.cwd,
        )
        exit_code = int(result["returncode"])
    else:
        result = summarize_manifest(json.loads(args.manifest.read_text(encoding="utf-8")))
        exit_code = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
