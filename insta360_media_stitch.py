#!/usr/bin/env python3
"""Stitch an Insta360 raw video with the official Linux MediaSDK."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_SDK_ROOT = ROOT / "work/insta360-sdk/media"
BUNDLED_FFPROBE = ROOT / "work/tools/ffmpeg-master-latest-linux64-gpl/bin/ffprobe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official Insta360 MediaSDK raw-video stitcher")
    parser.add_argument("input", type=Path, nargs="+", help="one or more parts of one Insta360 clip")
    parser.add_argument("output", type=Path)
    parser.add_argument("--sdk-root", type=Path, default=DEFAULT_SDK_ROOT)
    parser.add_argument("--width", type=int, choices=(1920, 3840, 6144, 7680), default=3840)
    parser.add_argument(
        "--stitch-type", choices=("template", "optflow", "dynamicstitch", "aistitch"),
        default="optflow",
    )
    parser.add_argument("--disable-cuda", action="store_true")
    parser.add_argument("--soft-decode", action="store_true")
    parser.add_argument("--soft-encode", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sdk_paths(sdk_root: Path) -> tuple[Path, Path, Path]:
    binary = sdk_root / "usr/bin/MediaSDKTest"
    library_dir = sdk_root / "usr/lib"
    model_dir = sdk_root / "models"
    missing = [path for path in (binary, library_dir, model_dir) if not path.exists()]
    if missing:
        raise SystemExit(
            "Insta360 MediaSDK is not deployed; missing: " + ", ".join(str(path) for path in missing)
        )
    return binary, library_dir, model_dir


def media_sdk_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    binary, library_dir, model_dir = sdk_paths(args.sdk_root.resolve())
    inputs = [path.resolve() for path in args.input]
    for path in inputs:
        if not path.is_file():
            raise SystemExit(f"missing Insta360 input: {path}")
        if path.suffix.lower() not in {".insv", ".lrv", ".mp4"}:
            raise SystemExit(f"unsupported Insta360 input suffix: {path}")
    command = [
        str(binary), "-inputs", *(str(path) for path in inputs),
        "-output", str(args.output.resolve()),
        "-model_root_dir", str(model_dir) + os.sep,
        "-stitch_type", args.stitch_type,
        "-output_size", f"{args.width}x{args.width // 2}",
    ]
    # These switches are deliberately absent: FlowState/direction lock would
    # remove the original camera motion that the downstream 6DoF tracker needs.
    if args.disable_cuda:
        command.append("-disable_cuda")
    if args.soft_decode:
        command.append("-enable_soft_decode")
    if args.soft_encode:
        command.append("-enable_soft_encode")
    environment = os.environ.copy()
    old_path = environment.get("LD_LIBRARY_PATH")
    environment["LD_LIBRARY_PATH"] = str(library_dir) + (os.pathsep + old_path if old_path else "")
    return command, environment


def probe_output(path: Path) -> dict:
    ffprobe = shutil.which("ffprobe") or (str(BUNDLED_FFPROBE) if BUNDLED_FFPROBE.is_file() else None)
    if not ffprobe:
        raise SystemExit("ffprobe is required to validate the MediaSDK output")
    process = subprocess.run(
        [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=60,
    )
    if process.returncode:
        raise SystemExit(f"MediaSDK output validation failed: {process.stderr.strip()[:500]}")
    probe = json.loads(process.stdout)
    videos = [stream for stream in probe.get("streams", []) if stream.get("codec_type") == "video"]
    if not videos:
        raise SystemExit("MediaSDK output has no video stream")
    width = int(videos[0].get("width", 0))
    height = int(videos[0].get("height", 0))
    if height <= 0 or width != 2 * height:
        raise SystemExit(f"MediaSDK output is not a 2:1 panorama: {width}x{height}")
    return {"width": width, "height": height, "codec": videos[0].get("codec_name")}


def camera_model_hint(path: Path) -> str | None:
    """Read the small printable camera marker stored near the INSV/LRV footer."""
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 8 * 1024 * 1024))
        tail = handle.read()
    match = re.search(rb"Insta360 [A-Za-z0-9]+(?: [A-Za-z0-9]+)?", tail)
    if not match:
        return None
    return match.group(0).decode("ascii")


def main() -> int:
    args = parse_args()
    args.sdk_root = args.sdk_root.resolve()
    args.output = args.output.resolve()
    if args.output.exists() and not args.force:
        print(f"INSTA360_STITCH_REUSE {args.output}")
        return 0
    command, environment = media_sdk_command(args)
    print("INSTA360_MEDIA_SDK " + " ".join(command))
    if args.dry_run:
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()
    process = subprocess.Popen(
        command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    output_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        output_lines.append(line)
    return_code = process.wait()
    sdk_output = "".join(output_lines)
    if return_code:
        raise SystemExit(f"Insta360 MediaSDK failed with exit code {return_code}")
    if "Camera's LensType is incorrect" in sdk_output:
        model = camera_model_hint(args.input[0].resolve())
        detail = f" ({model})" if model else ""
        raise SystemExit(
            "this Insta360 MediaSDK build does not recognize the camera lens metadata"
            f"{detail}; install a MediaSDK release that explicitly supports this camera model"
        )
    if not args.output.is_file() or args.output.stat().st_size == 0:
        raise SystemExit("Insta360 MediaSDK completed without producing a video")
    metadata = probe_output(args.output)
    print(f"INSTA360_STITCH_READY {args.output} {metadata['width']}x{metadata['height']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
