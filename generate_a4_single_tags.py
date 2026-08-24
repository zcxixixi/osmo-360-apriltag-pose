#!/usr/bin/env python3
"""Generate exact-size A4 AprilTags and a combined print-ready PDF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

PAGE_W_MM = 210.0
PAGE_H_MM = 297.0
TAG_MM = 200.0
MODULES = 8
DEFAULT_IDS = (128, 129, 130, 131, 132, 133)


def bits(tag_id: int) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    return cv2.aruco.generateImageMarker(dictionary, tag_id, MODULES, borderBits=1)


def geometry() -> tuple[float, float, float]:
    return (PAGE_W_MM - TAG_MM) / 2, (PAGE_H_MM - TAG_MM) / 2, TAG_MM / MODULES


def write_svg(path: Path, tag_id: int) -> None:
    left, top, module = geometry()
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{PAGE_W_MM:g}mm" height="{PAGE_H_MM:g}mm" viewBox="0 0 {PAGE_W_MM:g} {PAGE_H_MM:g}">',
        f'<rect width="{PAGE_W_MM:g}" height="{PAGE_H_MM:g}" fill="white"/>',
    ]
    for yy, xx in np.argwhere(bits(tag_id) == 0):
        lines.append(
            f'<rect x="{left + xx * module:.6f}" y="{top + yy * module:.6f}" '
            f'width="{module:.6f}" height="{module:.6f}" fill="black"/>'
        )
    lines.append('</svg>')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw_pdf_page(page, tag_id: int) -> None:
    from reportlab.lib.units import mm

    left, top, module = geometry()
    page.setFillColorRGB(1, 1, 1)
    page.rect(0, 0, PAGE_W_MM * mm, PAGE_H_MM * mm, fill=1, stroke=0)
    page.setFillColorRGB(0, 0, 0)
    for yy, xx in np.argwhere(bits(tag_id) == 0):
        x = (left + xx * module) * mm
        y = (PAGE_H_MM - top - (yy + 1) * module) * mm
        page.rect(x, y, module * mm, module * mm, fill=1, stroke=0)
    page.setFont("Helvetica", 8)
    page.drawCentredString(
        PAGE_W_MM * mm / 2,
        12 * mm,
        f"tag36h11 ID {tag_id} | outer square 200 mm | print at 100% actual size",
    )


def write_pdf(path: Path, tag_id: int) -> None:
    try:
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas
    except ImportError as exc:
        raise SystemExit("Run with: uv run --with reportlab python generate_a4_single_tags.py") from exc
    page = canvas.Canvas(str(path), pagesize=(PAGE_W_MM * mm, PAGE_H_MM * mm), pageCompression=1)
    draw_pdf_page(page, tag_id)
    page.showPage()
    page.save()


def write_combined_pdf(path: Path, tag_ids: list[int]) -> None:
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas

    page = canvas.Canvas(str(path), pagesize=(PAGE_W_MM * mm, PAGE_H_MM * mm), pageCompression=1)
    for tag_id in tag_ids:
        draw_pdf_page(page, tag_id)
        page.showPage()
    page.save()


def write_preview(path: Path, tag_id: int) -> None:
    scale = 5  # 5 px/mm; preview only.
    image = np.full((int(PAGE_H_MM * scale), int(PAGE_W_MM * scale)), 255, np.uint8)
    marker = cv2.resize(bits(tag_id), (int(TAG_MM * scale), int(TAG_MM * scale)), interpolation=cv2.INTER_NEAREST)
    left = int((PAGE_W_MM - TAG_MM) / 2 * scale)
    top = int((PAGE_H_MM - TAG_MM) / 2 * scale)
    image[top:top + marker.shape[0], left:left + marker.shape[1]] = marker
    cv2.imwrite(str(path), image)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("sessions/a4-single-apriltags"))
    parser.add_argument("--ids", type=int, nargs="+", default=DEFAULT_IDS)
    args = parser.parse_args()
    if not args.ids or len(set(args.ids)) != len(args.ids) or any(not 0 <= value < 587 for value in args.ids):
        raise SystemExit("IDs must be unique APRILTAG_36h11 IDs in 0..586")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for tag_id in args.ids:
        stem = f"a4_apriltag_36h11_id_{tag_id:03d}_200mm"
        write_pdf(args.output_dir / f"{stem}.pdf", tag_id)
        write_svg(args.output_dir / f"{stem}.svg", tag_id)
        write_preview(args.output_dir / f"{stem}_preview.png", tag_id)
    id_range = f"{min(args.ids):03d}-{max(args.ids):03d}"
    combined = args.output_dir / f"a4_apriltag_36h11_ids_{id_range}_200mm.pdf"
    write_combined_pdf(combined, args.ids)
    manifest = {
        "dictionary": "APRILTAG_36h11",
        "ids": list(args.ids),
        "page_mm": [PAGE_W_MM, PAGE_H_MM],
        "tag_outer_size_mm": TAG_MM,
        "horizontal_margin_mm": (PAGE_W_MM - TAG_MM) / 2,
        "vertical_margin_mm": (PAGE_H_MM - TAG_MM) / 2,
        "print_scale": "100% / actual size; disable fit-to-page",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "PRINT_README.txt").write_text(
        f"共{len(args.ids)}张A4，AprilTag 36h11 IDs: {', '.join(map(str, args.ids))}。\n"
        "每个Tag黑色外框精确为200 x 200 mm。\n"
        "打印选择100% / Actual size，关闭Fit to page。\n"
        "部分普通A4打印机无法打印到距纸边5 mm，请使用无边距打印或印刷店。\n"
        "SVG/PDF是打印源，PNG仅供预览。\n",
        encoding="utf-8",
    )
    print(combined.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
