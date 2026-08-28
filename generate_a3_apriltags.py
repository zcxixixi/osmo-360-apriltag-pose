#!/usr/bin/env python3
"""Generate print-ready A3 AprilTag 36h11 sheets for the two-tag experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


A3_MM = (297.0, 420.0)
FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ids", type=int, nargs=2, default=(200, 201))
    parser.add_argument("--tag-size-mm", type=float, default=240.0)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def mm_to_px(value: float, dpi: int) -> int:
    return round(value * dpi / 25.4)


def main() -> int:
    args = parse_args()
    if not 0 < args.tag_size_mm <= A3_MM[0] - 20:
        raise ValueError("tag size must fit A3 width with at least 10 mm margin per side")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    page_size = tuple(mm_to_px(value, args.dpi) for value in A3_MM)
    marker_px = mm_to_px(args.tag_size_mm, args.dpi)
    label_font = ImageFont.truetype(str(FONT), max(28, args.dpi // 6))
    preview_panels = []
    files = []

    for tag_id in args.ids:
        marker = cv2.aruco.generateImageMarker(dictionary, tag_id, marker_px, borderBits=1)
        marker_image = Image.fromarray(marker, mode="L")
        page = Image.new("L", page_size, 255)
        left = (page.width - marker_px) // 2
        top = mm_to_px(28.0, args.dpi)
        page.paste(marker_image, (left, top))
        draw = ImageDraw.Draw(page)
        label = f"AprilTag 36h11 · ID {tag_id} · black outer square {args.tag_size_mm:.1f} mm · print 100%"
        draw.text((mm_to_px(10, args.dpi), top + marker_px + mm_to_px(12, args.dpi)), label, font=label_font, fill=0)
        stem = f"apriltag36h11_id{tag_id}_A3_{args.tag_size_mm:.0f}mm"
        png = args.output_dir / f"{stem}.png"
        pdf = args.output_dir / f"{stem}.pdf"
        raw = args.output_dir / f"apriltag36h11_id{tag_id}_raw.png"
        marker_image.save(raw)
        page.save(png, dpi=(args.dpi, args.dpi))
        page.convert("RGB").save(pdf, "PDF", resolution=float(args.dpi))
        preview_panels.append(page.resize((424, 600), Image.Resampling.LANCZOS))
        files.append({"id": tag_id, "png": png.name, "pdf": pdf.name, "raw": raw.name})

    preview = Image.new("L", (preview_panels[0].width * 2 + 24, preview_panels[0].height), 225)
    preview.paste(preview_panels[0], (0, 0));preview.paste(preview_panels[1], (preview_panels[0].width + 24, 0))
    preview.save(args.output_dir / "two_tag_A3_preview.png")
    manifest = {
        "schema_version": "two-a3-apriltag-print/1.0",
        "family": "AprilTag 36h11",
        "dictionary": "cv2.aruco.DICT_APRILTAG_36h11",
        "ids": list(args.ids),
        "page_mm": list(A3_MM),
        "tag_black_outer_size_mm": args.tag_size_mm,
        "dpi": args.dpi,
        "print_scale": "100%; disable fit-to-page; measure black outer square after printing",
        "files": files,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
