#!/usr/bin/env python3
"""Generate two exact-size A4 wall panels for dual-gripper room calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

PAGE_W_MM = 297.0
PAGE_H_MM = 210.0
TAG_MM = 90.0
GAP_MM = 6.0
MODULES = 8
PANELS = {
    "left_wall": [[136, 134], [137, 135]],
    "right_wall": [[130, 129, 128], [133, 132, 131]],
}


def marker_bits(tag_id: int) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    return cv2.aruco.generateImageMarker(dictionary, tag_id, MODULES, borderBits=1)


def layout(grid: list[list[int]]) -> list[tuple[int, float, float]]:
    rows, cols = len(grid), len(grid[0])
    width = cols * TAG_MM + (cols - 1) * GAP_MM
    height = rows * TAG_MM + (rows - 1) * GAP_MM
    left = (PAGE_W_MM - width) / 2
    top = (PAGE_H_MM - height) / 2
    return [
        (tag_id, left + col * (TAG_MM + GAP_MM), top + row * (TAG_MM + GAP_MM))
        for row, values in enumerate(grid)
        for col, tag_id in enumerate(values)
    ]


def draw_page(page, name: str, grid: list[list[int]]) -> None:
    from reportlab.lib.units import mm

    page.setFillColorRGB(1, 1, 1)
    page.rect(0, 0, PAGE_W_MM * mm, PAGE_H_MM * mm, fill=1, stroke=0)
    module = TAG_MM / MODULES
    for tag_id, left, top in layout(grid):
        page.setFillColorRGB(0, 0, 0)
        for yy, xx in np.argwhere(marker_bits(tag_id) == 0):
            page.rect(
                (left + xx * module) * mm,
                (PAGE_H_MM - top - (yy + 1) * module) * mm,
                module * mm,
                module * mm,
                fill=1,
                stroke=0,
            )
    # Print verification ruler and page identity outside the marker field.
    page.setStrokeColorRGB(0, 0, 0)
    page.setLineWidth(0.25 * mm)
    ruler_left, ruler_y = (PAGE_W_MM - 100.0) / 2, 6.0
    page.line(ruler_left * mm, ruler_y * mm, (ruler_left + 100.0) * mm, ruler_y * mm)
    for value in (0, 50, 100):
        x = ruler_left + value
        page.line(x * mm, (ruler_y - 2) * mm, x * mm, (ruler_y + 2) * mm)
    page.setFont("Helvetica", 7)
    page.drawCentredString(PAGE_W_MM * mm / 2, 9 * mm, "CHECK: line below must measure exactly 100 mm")
    ids = ",".join(str(value) for row in grid for value in row)
    if name == "left_wall":
        heading = (
            f"LEFT | trim 48 mm from BOTH sides -> final 201 x 210 mm | "
            f"IDs {ids} | tags {TAG_MM:g} x {TAG_MM:g} mm"
        )
        # Crop at x=48 and x=249 mm. The marks are outside the retained
        # 7.5 mm quiet margin and disappear when the shop makes the cut.
        page.setLineWidth(0.25 * mm)
        for crop_x in (48.0, 249.0):
            page.line(crop_x * mm, 0, crop_x * mm, 6 * mm)
            page.line(crop_x * mm, (PAGE_H_MM - 6) * mm, crop_x * mm, PAGE_H_MM * mm)
    else:
        heading = (
            f"RIGHT | final A4 297 x 210 mm | NO TRIM | IDs {ids} | "
            f"tags {TAG_MM:g} x {TAG_MM:g} mm"
        )
    page.drawCentredString(PAGE_W_MM * mm / 2, (PAGE_H_MM - 8) * mm, heading)


def write_pdf(path: Path, pages: list[tuple[str, list[list[int]]]]) -> None:
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas

    pdf = canvas.Canvas(str(path), pagesize=(PAGE_W_MM * mm, PAGE_H_MM * mm), pageCompression=1)
    for name, grid in pages:
        draw_page(pdf, name, grid)
        pdf.showPage()
    pdf.save()


def write_preview(path: Path, name: str, grid: list[list[int]]) -> None:
    scale = 4
    image = np.full((round(PAGE_H_MM * scale), round(PAGE_W_MM * scale), 3), 255, np.uint8)
    for tag_id, left, top in layout(grid):
        size = round(TAG_MM * scale)
        marker = cv2.resize(marker_bits(tag_id), (size, size), interpolation=cv2.INTER_NEAREST)
        x, y = round(left * scale), round(top * scale)
        image[y:y + size, x:x + size] = marker[:, :, None]
    cv2.putText(image, name, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
    cv2.imwrite(str(path), image)


def panel_map(name: str, grid: list[list[int]]) -> dict:
    tags = []
    for tag_id, left, top in layout(grid):
        x0 = (left - PAGE_W_MM / 2) / 1000
        x1 = (left + TAG_MM - PAGE_W_MM / 2) / 1000
        y0 = (top - PAGE_H_MM / 2) / 1000
        y1 = (top + TAG_MM - PAGE_H_MM / 2) / 1000
        tags.append({"id": tag_id, "corners_m": [[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]]})
    return {
        "schema_version": "independent-apriltag-map/1.0",
        "dictionary": "APRILTAG_36h11",
        "description": f"One A4 landscape {name} panel, print at 100% actual size",
        "units": "m",
        "tag_outer_size_m": TAG_MM / 1000,
        "sheet_size_m": [PAGE_W_MM / 1000, PAGE_H_MM / 1000],
        "origin": "physical A4 page center",
        "axes": "x right, y down, z out of printed face",
        "tag_gap_m": GAP_MM / 1000,
        "tags": tags,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("sessions/a4-two-wall-panels-90mm"))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pages = list(PANELS.items())
    write_pdf(output / "two_wall_apriltag_panels_A4_90mm.pdf", pages)
    for name, grid in pages:
        write_pdf(output / f"{name}_A4_90mm.pdf", [(name, grid)])
        write_preview(output / f"{name}_90mm_preview.png", name, grid)
        (output / f"{name}_90mm_map.json").write_text(
            json.dumps(panel_map(name, grid), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    manifest = {
        "dictionary": "APRILTAG_36h11",
        "page_orientation": "A4 landscape",
        "page_size_mm": [PAGE_W_MM, PAGE_H_MM],
        "tag_outer_size_mm": TAG_MM,
        "tag_gap_mm": GAP_MM,
        "module_size_mm": TAG_MM / MODULES,
        "panels": PANELS,
        "print": "100% / Actual size; disable Fit/Shrink/Scale",
        "verification": f"Measure each black square as {TAG_MM:g} x {TAG_MM:g} mm and the ruler as 100 mm",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "打印与测量说明.txt").write_text(
        "打印：A4横向，100%/实际尺寸，关闭适应页面、缩小超大页面和无边距缩放。\n"
        "打印后：每个黑色Tag外框必须为90 x 90 mm；底部校验线必须为100 mm。\n"
        "第1页左墙：按裁切标记，左边裁48 mm、右边裁48 mm，成品201 x 210 mm。\n"
        "第2页右墙：不裁切，成品297 x 210 mm。不得裁掉上下边。\n"
        "裁后左墙Tag区186 x 186 mm，右墙Tag区282 x 186 mm；黑框外侧白边不得再裁。\n"
        "两页上边缘对齐贴到两面墙，纸张不得跨墙角折弯。\n"
        "记录：两页上边缘离桌面的高度，以及两页靠墙角纸边离墙角的水平距离。\n",
        encoding="utf-8",
    )
    print(output / "two_wall_apriltag_panels_A4_90mm.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
