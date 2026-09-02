#!/usr/bin/env python3
"""Factory-calibrated DJI Osmo 360 OSV to equirectangular MP4 converter.

This is a small command-line adapter around the local PanoForge engine.  It
never treats an OSV as a renamed MP4: calibration and IMU are extracted from
the proprietary djmd tracks and the two fisheye streams are remapped with the
embedded per-camera calibration.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


from tools._root import ROOT
DEFAULT_PANOFORGE = ROOT.parent / "panoforge-test"
DEFAULT_FFMPEG_BIN = ROOT / "work/tools/ffmpeg-master-latest-linux64-gpl/bin"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PanoForge factory-calibrated DJI OSV stitch")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--panoforge-root", type=Path, default=DEFAULT_PANOFORGE)
    parser.add_argument("--ffmpeg-bin", type=Path, default=DEFAULT_FFMPEG_BIN)
    parser.add_argument("--width", type=int, choices=(3840, 6144, 7680), default=3840)
    parser.add_argument("--codec", choices=("h264", "hevc"), default="h264")
    parser.add_argument("--encoder", choices=("auto", "cpu", "nvenc"), default="auto")
    parser.add_argument("--quality", type=int, default=18)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run_ffmpeg(command: list[str], duration_s: float) -> None:
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    assert process.stdout is not None
    last_percent = -1
    block: dict[str, str] = {}
    for raw in process.stdout:
        line = raw.rstrip()
        if "=" in line:
            key, value = line.split("=", 1)
            block[key] = value
        elif line:
            print(line, flush=True)
        if line.startswith("progress="):
            try:
                out_time = int(block.get("out_time_us", "0")) / 1_000_000.0
            except ValueError:
                out_time = 0.0
            percent = min(100, int(round(100 * out_time / duration_s))) if duration_s else 0
            if percent >= last_percent + 5 or block.get("progress") == "end":
                print(f"DJI_STITCH {percent}%", flush=True)
                last_percent = percent
            block.clear()
    code = process.wait()
    if code:
        raise RuntimeError(f"PanoForge ffmpeg failed with exit code {code}")


def main() -> int:
    args = parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    metadata_dir = args.metadata_dir.resolve()
    panoforge = args.panoforge_root.resolve()
    ffmpeg_bin = args.ffmpeg_bin.resolve()
    if not source.is_file() or source.suffix.lower() != ".osv":
        raise SystemExit(f"expected an existing DJI .OSV file: {source}")
    if output.exists() and not args.force:
        print(f"DJI_STITCH reuse {output}")
        return 0
    for executable in ("ffmpeg", "ffprobe"):
        if not (ffmpeg_bin / executable).is_file():
            raise SystemExit(f"missing bundled {executable}: {ffmpeg_bin / executable}")
    if not (panoforge / "app/core/osv.py").is_file():
        raise SystemExit(f"PanoForge engine not found: {panoforge}")

    os.environ["PATH"] = str(ffmpeg_bin) + os.pathsep + os.environ.get("PATH", "")
    sys.path.insert(0, str(panoforge))
    from app.core.maps import generate_remap_maps, scale_calibration_to_source
    from app.core.osv import extract_metadata, probe
    from app.core.spherical import inject_spherical
    from app.core.stitch import StitchOptions, build_command

    info = probe(str(source))
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata = extract_metadata(str(source), str(metadata_dir))
    calibration = metadata.get("calibration")
    if not calibration:
        raise SystemExit("DJI factory calibration was not found in the djmd track; refusing approximation")
    scaled = scale_calibration_to_source(calibration, info.width, info.height)
    (metadata_dir / "calibration_scaled_to_source.json").write_text(
        json.dumps(scaled, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    source_info = {
        **info.to_dict(),
        "camera_family": "dji_osmo_360",
        "stitch": {
            "engine": "PanoForge",
            "mode": "factory_calibrated",
            "stabilization": "off",
            "output_width": args.width,
            "codec": args.codec,
            "quality": args.quality,
        },
    }
    (metadata_dir / "source_info.json").write_text(
        json.dumps(source_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dji_factory_stitch_", dir=output.parent) as temporary:
        work = Path(temporary)
        maps = generate_remap_maps(scaled, args.width, args.width // 2, str(work / "maps"))
        if not maps.calibrated:
            raise SystemExit("PanoForge returned non-calibrated maps; refusing approximation")
        stitched = work / "stitched.mp4"
        options = StitchOptions(
            out_w=args.width, codec=args.codec, encoder=args.encoder,
            quality=args.quality, mode="calibrated", stabilize=False,
        )
        command = build_command(str(source), str(stitched), options, maps)
        try:
            run_ffmpeg(command, info.duration_s)
        except RuntimeError:
            if args.encoder != "auto":
                raise
            print("DJI_STITCH NVENC unavailable at runtime; falling back to libx264 CPU", flush=True)
            stitched.unlink(missing_ok=True)
            options.encoder = "cpu"
            run_ffmpeg(
                build_command(str(source), str(stitched), options, maps),
                info.duration_s,
            )
        spherical = work / "spherical.mp4"
        inject_spherical(str(stitched), str(spherical))
        final_source = spherical if spherical.is_file() else stitched
        part = output.with_suffix(output.suffix + ".part")
        shutil.copy2(final_source, part)
        os.replace(part, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
