#!/usr/bin/env python3
"""Generate two print-ready A3 landscape AprilTag 36h11 grid boards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


A3_LANDSCAPE_MM = (420.0, 297.0)
GRID_SHAPE = (3, 2)
FONT_CANDIDATES = (
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
)


def load_font(size: int):
    for path in FONT_CANDIDATES:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--tag-size-mm", type=float, default=120.0)
    parser.add_argument("--gap-mm", type=float, default=20.0)
    parser.add_argument("--sheet-a-ids", type=int, nargs=6, default=(200, 201, 202, 203, 204, 205))
    parser.add_argument("--sheet-b-ids", type=int, nargs=6, default=(210, 211, 212, 213, 214, 215))
    return parser.parse_args()


def mm_to_px(value: float, dpi: int) -> int:
    return round(value * dpi / 25.4)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def centered_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str,
                  font: ImageFont.FreeTypeFont, fill: int = 0) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1]), text, font=font, fill=fill)


def layout_for(ids: tuple[int, ...], tag_size_mm: float, gap_mm: float) -> list[dict]:
    columns, rows = GRID_SHAPE
    grid_width = columns * tag_size_mm + (columns - 1) * gap_mm
    grid_height = rows * tag_size_mm + (rows - 1) * gap_mm
    start_x = (A3_LANDSCAPE_MM[0] - grid_width) / 2
    start_y = (A3_LANDSCAPE_MM[1] - grid_height) / 2
    tags = []
    for index, tag_id in enumerate(ids):
        column, row = index % columns, index // columns
        left = start_x + column * (tag_size_mm + gap_mm)
        top = start_y + row * (tag_size_mm + gap_mm)
        x0 = round((left - A3_LANDSCAPE_MM[0] / 2) / 1000, 9)
        y0 = round((top - A3_LANDSCAPE_MM[1] / 2) / 1000, 9)
        size = round(tag_size_mm / 1000, 9)
        x1 = round(x0 + size, 9)
        y1 = round(y0 + size, 9)
        tags.append({
            "id": tag_id,
            "row": row,
            "column": column,
            "black_outer_top_left_mm": [left, top],
            "corners_m": [
                [x0, y0, 0.0],
                [x1, y0, 0.0],
                [x1, y1, 0.0],
                [x0, y1, 0.0],
            ],
        })
    return tags


def render_sheet(output_dir: Path, name: str, ids: tuple[int, ...], dpi: int,
                 tag_size_mm: float, gap_mm: float, dictionary,
                 revision_id: str) -> tuple[dict, Image.Image]:
    page_px = tuple(mm_to_px(value, dpi) for value in A3_LANDSCAPE_MM)
    page = Image.new("L", page_px, 255)
    draw = ImageDraw.Draw(page)
    title_font = load_font(max(28, round(dpi * 0.16)))
    marker_px = mm_to_px(tag_size_mm, dpi)
    layout = layout_for(ids, tag_size_mm, gap_mm)

    centered_text(draw, (page.width // 2, mm_to_px(2.5, dpi)),
                  f"APRILGRID {name} · TOP · IDs {ids[0]}–{ids[-1]}", title_font)
    markers_dir = output_dir / "markers"
    markers_dir.mkdir(exist_ok=True)
    raw_files = []
    for tag in layout:
        marker = cv2.aruco.generateImageMarker(dictionary, tag["id"], marker_px, borderBits=1)
        marker_image = Image.fromarray(marker, mode="L")
        left_mm, top_mm = tag["black_outer_top_left_mm"]
        page.paste(marker_image, (mm_to_px(left_mm, dpi), mm_to_px(top_mm, dpi)))
        raw_path = markers_dir / f"tag36h11_id{tag['id']}_{tag_size_mm:.0f}mm.png"
        marker_image.save(raw_path, dpi=(dpi, dpi))
        raw_files.append(raw_path)


    stem = f"A3_aprilgrid_{name}_{ids[0]}-{ids[-1]}_{tag_size_mm:.0f}mm"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    page.save(png_path, dpi=(dpi, dpi))
    page.convert("RGB").save(pdf_path, "PDF", resolution=float(dpi))
    board_path = output_dir / f"{stem}_layout.json"
    board = {
        "schema_version": "apriltag-grid-board/1.0",
        "revision_id": revision_id,
        "board": name,
        "family": "AprilTag 36h11",
        "dictionary": "cv2.aruco.DICT_APRILTAG_36h11",
        "page_mm": list(A3_LANDSCAPE_MM),
        "page_orientation": "landscape",
        "grid_columns": GRID_SHAPE[0],
        "grid_rows": GRID_SHAPE[1],
        "tag_black_outer_size_mm": tag_size_mm,
        "black_square_gap_mm": gap_mm,
        "origin": "A3 sheet center",
        "axes": "x right, y down, z out of printed face",
        "units": "m",
        "tags": layout,
    }
    board_path.write_text(json.dumps(board, indent=2) + "\n", encoding="utf-8")
    return {
        "board": name,
        "ids": list(ids),
        "pdf": pdf_path.name,
        "png": png_path.name,
        "layout": board_path.name,
        "raw_markers": [str(path.relative_to(output_dir)) for path in raw_files],
    }, page


def main() -> int:
    args = parse_args()
    all_ids = tuple(args.sheet_a_ids) + tuple(args.sheet_b_ids)
    if len(set(all_ids)) != 12:
        raise ValueError("all 12 Tag IDs must be unique")
    if any(tag_id < 0 or tag_id >= 587 for tag_id in all_ids):
        raise ValueError("AprilTag 36h11 IDs must be in [0, 586]")
    if args.dpi <= 0 or args.tag_size_mm <= 0 or args.gap_mm <= 0:
        raise ValueError("dpi, tag size, and gap must be positive")
    grid_width = GRID_SHAPE[0] * args.tag_size_mm + (GRID_SHAPE[0] - 1) * args.gap_mm
    grid_height = GRID_SHAPE[1] * args.tag_size_mm + (GRID_SHAPE[1] - 1) * args.gap_mm
    if grid_width > A3_LANDSCAPE_MM[0] - 20 or grid_height > A3_LANDSCAPE_MM[1] - 30:
        raise ValueError("grid does not leave the required A3 printer-safe margins")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    revision_id = (
        f"a3-aprilgrid-pair-{args.sheet_a_ids[0]}-{args.sheet_a_ids[-1]}_"
        f"{args.sheet_b_ids[0]}-{args.sheet_b_ids[-1]}-"
        f"{args.tag_size_mm:g}mm-{args.gap_mm:g}mm-20260828-r1"
    )
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    sheet_a, page_a = render_sheet(
        args.output_dir, "A", tuple(args.sheet_a_ids), args.dpi,
        args.tag_size_mm, args.gap_mm, dictionary, revision_id,
    )
    sheet_b, page_b = render_sheet(
        args.output_dir, "B", tuple(args.sheet_b_ids), args.dpi,
        args.tag_size_mm, args.gap_mm, dictionary, revision_id,
    )
    preview_height = 600
    preview_width = round(preview_height * A3_LANDSCAPE_MM[0] / A3_LANDSCAPE_MM[1])
    preview = Image.new("L", (preview_width * 2 + 24, preview_height), 225)
    preview.paste(page_a.resize((preview_width, preview_height), Image.Resampling.LANCZOS), (0, 0))
    preview.paste(page_b.resize((preview_width, preview_height), Image.Resampling.LANCZOS), (preview_width + 24, 0))
    preview_path = args.output_dir / "A3_aprilgrid_pair_preview.png"
    preview.save(preview_path)

    files = [path for path in args.output_dir.rglob("*") if path.is_file()]
    manifest = {
        "schema_version": "a3-aprilgrid-pair-print/1.0",
        "revision_id": revision_id,
        "status": "PRINT_LAYOUT_DEFINED_NOT_WORLD_CALIBRATED",
        "family": "AprilTag 36h11",
        "page_mm": list(A3_LANDSCAPE_MM),
        "page_orientation": "landscape",
        "grid": {"columns": 3, "rows": 2},
        "tag_black_outer_size_mm": args.tag_size_mm,
        "black_square_gap_mm": args.gap_mm,
        "minimum_white_quiet_zone_per_adjacent_tag_mm": args.gap_mm / 2,
        "dpi": args.dpi,
        "sheets": [sheet_a, sheet_b],
        "mounting": "Keep each complete A3 sheet flat and rigid; do not cut individual Tags.",
        "print_scale": "100%; disable fit-to-page; landscape A3; measure several black squares and center pitches after printing.",
        "acceptance": {
            "black_square_mm": [args.tag_size_mm - 0.5, args.tag_size_mm + 0.5],
            "center_pitch_mm": [
                args.tag_size_mm + args.gap_mm - 0.5,
                args.tag_size_mm + args.gap_mm + 0.5,
            ],
        },
        "pair_transform_status": "UNKNOWN_UNTIL_MOUNTED_AND_CALIBRATED",
        "file_sha256": {str(path.relative_to(args.output_dir)): sha256(path) for path in files},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
