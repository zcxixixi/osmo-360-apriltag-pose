#!/usr/bin/env python3
"""Extract DJI OSV factory calibration and IMU without stitching video."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


from tools._root import ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--panoforge-root", type=Path, default=ROOT.parent / "panoforge-test")
    args = parser.parse_args()
    ffmpeg_bin = ROOT / "work/tools/ffmpeg-master-latest-linux64-gpl/bin"
    os.environ["PATH"] = str(ffmpeg_bin) + os.pathsep + os.environ.get("PATH", "")
    sys.path.insert(0, str(args.panoforge_root.resolve()))
    from app.core.osv import extract_metadata, probe

    source = args.input.resolve(strict=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    info = probe(str(source))
    metadata = extract_metadata(str(source), str(args.output_dir.resolve()))
    result = {**info.to_dict(), "camera_serial": metadata["calibration"].get("serial")}
    (args.output_dir / "source_info.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
