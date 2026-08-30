#!/usr/bin/env python3
"""Generate two one-piece, print-shop-size AprilTag wall panels."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

TAG_MM = 200.0
GAP_MM = 20.0
MARGIN_MM = 20.0
MODULES = 8
PANELS = {
    "LEFT_4_TAGS": [[136, 134], [137, 135]],
    "RIGHT_6_TAGS": [[130, 129, 128], [133, 132, 131]],
}


def marker(tag_id: int) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    return cv2.aruco.generateImageMarker(dictionary, tag_id, MODULES, borderBits=1)


def dimensions(grid: list[list[int]]) -> tuple[float, float]:
    rows, cols = len(grid), len(grid[0])
    return (
        2 * MARGIN_MM + cols * TAG_MM + (cols - 1) * GAP_MM,
        2 * MARGIN_MM + rows * TAG_MM + (rows - 1) * GAP_MM,
    )


def draw_panel(pdf, name: str, grid: list[list[int]]) -> None:
    from reportlab.lib.units import mm

    width, height = dimensions(grid)
    pdf.setPageSize((width * mm, height * mm))
    pdf.setFillColorRGB(1, 1, 1)
    pdf.rect(0, 0, width * mm, height * mm, fill=1, stroke=0)
    module = TAG_MM / MODULES
    for row, values in enumerate(grid):
        for col, tag_id in enumerate(values):
            left = MARGIN_MM + col * (TAG_MM + GAP_MM)
            top = MARGIN_MM + row * (TAG_MM + GAP_MM)
            pdf.setFillColorRGB(0, 0, 0)
            for yy, xx in np.argwhere(marker(tag_id) == 0):
                pdf.rect(
                    (left + xx * module) * mm,
                    (height - top - (yy + 1) * module) * mm,
                    module * mm,
                    module * mm,
                    fill=1,
                    stroke=0,
                )
    ids = ",".join(str(value) for row in grid for value in row)
    pdf.setFont("Helvetica", 8)
    pdf.drawCentredString(
        width * mm / 2,
        (height - 10) * mm,
        f"{name} | FINAL {width:g} x {height:g} mm | IDs {ids} | each tag 200 x 200 mm",
    )
    # Exact 100 mm print verification ruler in the lower white margin.
    x0, y = (width - 100) / 2, 7
    pdf.setLineWidth(0.3 * mm)
    pdf.line(x0 * mm, y * mm, (x0 + 100) * mm, y * mm)
    for value in (0, 50, 100):
        x = x0 + value
        pdf.line(x * mm, (y - 2) * mm, x * mm, (y + 2) * mm)
    pdf.setFont("Helvetica", 7)
    pdf.drawCentredString(width * mm / 2, 11 * mm, "CHECK: this line must measure exactly 100 mm")
    pdf.showPage()


def panel_map(name: str, grid: list[list[int]]) -> dict:
    width, height = dimensions(grid)
    tags = []
    for row, values in enumerate(grid):
        for col, tag_id in enumerate(values):
            left = MARGIN_MM + col * (TAG_MM + GAP_MM)
            top = MARGIN_MM + row * (TAG_MM + GAP_MM)
            x0, x1 = (left - width / 2) / 1000, (left + TAG_MM - width / 2) / 1000
            y0, y1 = (top - height / 2) / 1000, (top + TAG_MM - height / 2) / 1000
            tags.append({"id": tag_id, "corners_m": [[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]]})
    return {
        "schema_version": "independent-apriltag-map/1.0",
        "dictionary": "APRILTAG_36h11",
        "description": f"One-piece large-format {name} panel",
        "units": "m",
        "tag_outer_size_m": TAG_MM / 1000,
        "tag_gap_m": GAP_MM / 1000,
        "outer_margin_m": MARGIN_MM / 1000,
        "panel_size_m": [width / 1000, height / 1000],
        "origin": "physical finished panel center",
        "axes": "x right, y down, z out of printed face",
        "tags": tags,
    }


def main() -> int:
    from reportlab.pdfgen import canvas

    output = Path("sessions/large-two-wall-panels-200mm-final")
    output.mkdir(parents=True, exist_ok=True)
    combined = output / "双墙AprilTag标定板_大幅面最终版_200mm.pdf"
    pdf = canvas.Canvas(str(combined), pagesize=(460, 460), pageCompression=1)
    for name, grid in PANELS.items():
        draw_panel(pdf, name, grid)
        (output / f"{name.lower()}_map.json").write_text(
            json.dumps(panel_map(name, grid), indent=2) + "\n", encoding="utf-8"
        )
    pdf.save()
    manifest = {
        "dictionary": "APRILTAG_36h11",
        "tag_outer_size_mm": TAG_MM,
        "gap_mm": GAP_MM,
        "outer_margin_mm": MARGIN_MM,
        "left_finished_size_mm": dimensions(PANELS["LEFT_4_TAGS"]),
        "right_finished_size_mm": dimensions(PANELS["RIGHT_6_TAGS"]),
        "print": "large-format, 100% actual size, no fit/scale/tile",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(combined)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
