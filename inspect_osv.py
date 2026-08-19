#!/usr/bin/env python3
"""Read-only ffprobe inspection and representative frame extraction for an OSV copy."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, capture_output=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="OSV copy under work/input")
    parser.add_argument("--output-dir", type=Path, default=Path("work/osv_inspection"))
    args = parser.parse_args()
    source = args.input.resolve(strict=True)
    if shutil.which("ffprobe") is None or shutil.which("ffmpeg") is None:
        raise SystemExit(
            "ffmpeg/ffprobe not found; install the Ubuntu ffmpeg package first"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    probe = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-show_programs",
            "-show_chapters",
            "-show_private_data",
            "-of",
            "json",
            str(source),
        ]
    )
    report = json.loads(probe.stdout)
    (args.output_dir / "ffprobe.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    duration = float(report.get("format", {}).get("duration", 0))
    times = {"start": 0.0, "middle": duration / 2, "end": max(0.0, duration - 0.25)}
    frames = args.output_dir / "frames"
    frames.mkdir(exist_ok=True)
    for label, timestamp in times.items():
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{timestamp:.6f}",
                "-i",
                str(source),
                "-frames:v",
                "1",
                "-y",
                str(frames / f"{label}.png"),
            ],
            check=True,
        )
    print(args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
