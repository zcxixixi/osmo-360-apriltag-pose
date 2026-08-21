#!/usr/bin/env python3
"""Generate two exact-size 1500 mm AprilTag 36h11 wall panels."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np

PANEL_MM = 1500.0
ROWS = COLS = 8
TAG_MM = 140.0
GAP_MM = 40.0
PITCH_MM = TAG_MM + GAP_MM
MARGIN_MM = (PANEL_MM - (COLS * TAG_MM + (COLS - 1) * GAP_MM)) / 2.0
MODULES = 8  # 6x6 payload plus one black border module on every side.


def marker_bits(tag_id: int) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    return cv2.aruco.generateImageMarker(dictionary, tag_id, MODULES, borderBits=1)


def svg_panel(first_id: int) -> str:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{PANEL_MM:g}mm" height="{PANEL_MM:g}mm" viewBox="0 0 {PANEL_MM:g} {PANEL_MM:g}">',
        f'<rect width="{PANEL_MM:g}" height="{PANEL_MM:g}" fill="white"/>',
    ]
    module = TAG_MM / MODULES
    for row in range(ROWS):
        for col in range(COLS):
            tag_id = first_id + row * COLS + col
            bits = marker_bits(tag_id)
            left = MARGIN_MM + col * PITCH_MM
            top = MARGIN_MM + row * PITCH_MM
            for yy, xx in np.argwhere(bits == 0):
                lines.append(
                    f'<rect x="{left + xx * module:.6f}" y="{top + yy * module:.6f}" '
                    f'width="{module:.6f}" height="{module:.6f}" fill="black"/>'
                )
    lines.append('</svg>')
    return "\n".join(lines) + "\n"


def pdf_panel(path: Path, first_id: int) -> None:
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import mm
    except ImportError as exc:
        raise SystemExit("PDF output requires reportlab: uv run --with reportlab python generate_wall_tag_panels.py") from exc
    page = canvas.Canvas(str(path), pagesize=(PANEL_MM * mm, PANEL_MM * mm), pageCompression=1)
    page.setFillColorRGB(1, 1, 1)
    page.rect(0, 0, PANEL_MM * mm, PANEL_MM * mm, fill=1, stroke=0)
    page.setFillColorRGB(0, 0, 0)
    module = TAG_MM / MODULES
    for row in range(ROWS):
        for col in range(COLS):
            bits = marker_bits(first_id + row * COLS + col)
            left = MARGIN_MM + col * PITCH_MM
            top = MARGIN_MM + row * PITCH_MM
            for yy, xx in np.argwhere(bits == 0):
                x = (left + xx * module) * mm
                # PDF coordinates start at bottom-left.
                y = (PANEL_MM - top - (yy + 1) * module) * mm
                page.rect(x, y, module * mm, module * mm, fill=1, stroke=0)
    page.showPage()
    page.save()


def preview_panel(path: Path, first_id: int) -> None:
    # Preview only: 1 px/mm (25.4 dpi). Use SVG/PDF for printing.
    image = np.full((1500, 1500), 255, np.uint8)
    for row in range(ROWS):
        for col in range(COLS):
            marker = cv2.resize(marker_bits(first_id + row * COLS + col), (140, 140), interpolation=cv2.INTER_NEAREST)
            x = int(MARGIN_MM + col * PITCH_MM)
            y = int(MARGIN_MM + row * PITCH_MM)
            image[y:y + 140, x:x + 140] = marker
    cv2.imwrite(str(path), image)


def panel_map(name: str, first_id: int) -> dict:
    tags = []
    for row in range(ROWS):
        for col in range(COLS):
            left = MARGIN_MM + col * PITCH_MM
            top = MARGIN_MM + row * PITCH_MM
            tags.append({
                "id": first_id + row * COLS + col,
                "row": row,
                "col": col,
                "center_panel_mm": [left + TAG_MM / 2 - PANEL_MM / 2, PANEL_MM / 2 - top - TAG_MM / 2, 0.0],
                "size_mm": TAG_MM,
            })
    return {"name": name, "first_id": first_id, "last_id": first_id + ROWS * COLS - 1, "tags": tags}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("sessions/wall-tag-panels-1500mm"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panels = [("A", 0), ("B", 64)]
    for name, first_id in panels:
        stem = f"wall_{name}_1500x1500mm_ids_{first_id:03d}-{first_id + 63:03d}"
        (args.output_dir / f"{stem}.svg").write_text(svg_panel(first_id), encoding="utf-8")
        pdf_panel(args.output_dir / f"{stem}.pdf", first_id)
        preview_panel(args.output_dir / f"{stem}_preview.png", first_id)
    manifest = {
        "dictionary": "APRILTAG_36h11",
        "panel_size_mm": [PANEL_MM, PANEL_MM],
        "rows": ROWS,
        "cols": COLS,
        "tag_size_mm": TAG_MM,
        "gap_mm": GAP_MM,
        "spacing_ratio": GAP_MM / TAG_MM,
        "pitch_mm": PITCH_MM,
        "margin_mm": MARGIN_MM,
        "print_scale": "100% / actual size; disable fit-to-page",
        "panels": [panel_map(name, first_id) for name, first_id in panels],
    }
    (args.output_dir / "tag_map.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "PRINT_README.txt").write_text(
        "两张墙面均为 1500 x 1500 mm。\n"
        "打印必须选择 100% / Actual size，关闭 Fit to page。\n"
        "优先使用 PDF 或 SVG；PNG 仅用于预览，不可作为大幅打印源。\n"
        "A墙 ID 0-63，B墙 ID 64-127，禁止重复使用。\n"
        "单Tag黑色外框 140 mm，净间距40 mm，四周边距50 mm。\n"
        "安装后请实测Tag尺寸和两墙相对位姿。\n",
        encoding="utf-8",
    )
    print(args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
