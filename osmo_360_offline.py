#!/usr/bin/env python3
"""Offline AprilGrid pose estimation for stitched 2:1 equirectangular video.

This deliberately does not implement DJI's private OSV stitching.  Feed it a
DJI Studio export, or an independently validated OSV conversion.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import signal
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import py360convert

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from osmo_apriltag_demo import Grid, rotation_to_rpy

LOG = logging.getLogger("osmo360.offline")
STOP = False


@dataclass(frozen=True)
class View:
    name: str
    yaw: float
    pitch: float
    fov: float = 100.0


# Overlap is intentional: a tag near a cardinal-view edge gets another chance.
DEFAULT_VIEWS = tuple(
    [View(f"h{yaw:+04d}", float(yaw), 0.0) for yaw in range(-180, 180, 45)]
    + [View("up", 0.0, 90.0, 110.0), View("down", 0.0, -90.0, 110.0)]
)


@dataclass
class Pose:
    xyz: np.ndarray
    rotation_camera_to_board: np.ndarray
    rpy: tuple[float, float, float]
    inliers: int
    rmse: float
    view: str
    ids: list[int]


def view_to_panorama_rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Rotation from OpenCV perspective axes into panorama camera axes.

    Axes are x right, y down, z forward. Positive yaw looks right and positive
    pitch looks up, matching py360convert's e2p convention.
    """
    yaw, pitch = np.radians([yaw_deg, pitch_deg])
    ry = np.array(
        [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]]
    )
    rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)],
        ]
    )
    return ry @ rx


def pose_view_to_panorama(
    rvec: np.ndarray, tvec: np.ndarray, view: View
) -> tuple[np.ndarray, np.ndarray]:
    """Convert board->view PnP output to camera pose in the board frame."""
    board_to_view, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    view_to_pano = view_to_panorama_rotation(view.yaw, view.pitch)
    board_to_pano = view_to_pano @ board_to_view
    board_origin_pano = view_to_pano @ np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    pano_to_board = board_to_pano.T
    camera_xyz_board = (-pano_to_board @ board_origin_pano).reshape(3)
    return camera_xyz_board, pano_to_board


def perspective_intrinsics(size: int, fov_deg: float) -> np.ndarray:
    focal = size / (2 * math.tan(math.radians(fov_deg) / 2))
    return np.array(
        [[focal, 0, size / 2], [0, focal, size / 2], [0, 0, 1]], dtype=np.float64
    )


def make_detector() -> cv2.aruco.ArucoDetector:
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 63
    params.adaptiveThreshWinSizeStep = 4
    params.minMarkerPerimeterRate = 0.008
    params.maxMarkerPerimeterRate = 4.0
    # Conservative decode: compression damage should become a miss, not a false ID.
    params.errorCorrectionRate = 0.25
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    return cv2.aruco.ArucoDetector(dictionary, params)


def detect_view(
    image: np.ndarray, detector: cv2.aruco.ArucoDetector, grid: Grid
) -> list[dict]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _rejected = detector.detectMarkers(gray)
    found: list[dict] = []
    if ids is None:
        return found
    for marker_corners, raw_id in zip(corners, ids.flatten()):
        tag_id = int(raw_id)
        center = grid.center(tag_id)
        if center is None:
            continue
        px = marker_corners.reshape(4, 2).astype(np.float32)
        found.append(
            {
                "id": tag_id,
                "corners_px": px,
                "center_px": px.mean(axis=0),
                "object_center": center,
                "area_px2": abs(float(cv2.contourArea(px))),
            }
        )
    return found


def solve_view(
    detections: list[dict], view: View, size: int, min_tags: int
) -> Pose | None:
    # A repeated decoded ID in one view is suspicious; keep only the largest.
    best: dict[int, dict] = {}
    for det in detections:
        if det["id"] not in best or det["area_px2"] > best[det["id"]]["area_px2"]:
            best[det["id"]] = det
    detections = list(best.values())
    if len(detections) < min_tags:
        return None
    obj = np.asarray([d["object_center"] for d in detections], np.float32)
    img = np.asarray([d["center_px"] for d in detections], np.float32)
    k = perspective_intrinsics(size, view.fov)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj,
        img,
        k,
        None,
        iterationsCount=200,
        reprojectionError=3.0,
        confidence=0.999,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok or inliers is None or len(inliers) < min_tags:
        return None
    ii = inliers[:, 0]
    rvec, tvec = cv2.solvePnPRefineLM(obj[ii], img[ii], k, None, rvec, tvec)
    projected, _ = cv2.projectPoints(obj[ii], rvec, tvec, k, None)
    residual = projected.reshape(-1, 2) - img[ii]
    rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    xyz, camera_to_board = pose_view_to_panorama(rvec, tvec, view)
    return Pose(
        xyz,
        camera_to_board,
        rotation_to_rpy(camera_to_board),
        len(ii),
        rmse,
        view.name,
        sorted(int(detections[i]["id"]) for i in ii),
    )


def choose_pose(candidates: list[Pose]) -> Pose | None:
    if not candidates:
        return None
    # Inlier support dominates; RMSE breaks ties. This avoids blending incompatible
    # planar PnP solutions while preserving all candidates in detections.jsonl.
    return min(candidates, key=lambda p: (-p.inliers, p.rmse))


def _finite_stats(values: Iterable[float]) -> dict[str, float | None]:
    data = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
    if not len(data):
        return {"mean": None, "median": None, "p95": None, "max": None}
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "p95": float(np.percentile(data, 95)),
        "max": float(np.max(data)),
    }


def generate_plot(session: Path, summary: dict) -> None:
    rows = list(
        csv.DictReader((session / "pose.csv").open(encoding="utf-8", newline=""))
    )
    points, errors = [], []
    for row in rows:
        if row["quality_status"] in {"valid", "jump_rejected"} and row["camera_x_m"]:
            points.append([float(row[f"camera_{a}_m"]) for a in "xyz"])
            errors.append(float(row["reprojection_rmse_px"]))
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 10), facecolor="#0b0d10")
    ratio = summary["valid_pose_ratio"] * 100
    fig.suptitle(
        f"Osmo 360 AprilGrid trajectory · valid {ratio:.1f}% · coverage {summary['tag_coverage_ratio'] * 100:.1f}%",
        fontsize=17,
    )
    if points:
        p = np.asarray(points)
        c = np.linspace(0, 1, len(p))
        ax = fig.add_subplot(221, projection="3d")
        ax.plot(*p.T, color="#57b9ff")
        ax.scatter(*p.T, c=c, cmap="viridis", s=9)
        ax.scatter(*p[0], c="#4ade80", s=90, label="Start")
        ax.scatter(*p[-1], c="#fb5b5b", s=90, label="End")
        ax.scatter(0, 0, 0, c="white", marker="s", s=80, label="AprilGrid origin")
        ax.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)", title="3D trajectory")
        ax.legend()
        for slot, ai, bi, title in [
            (222, 0, 1, "XY"),
            (223, 0, 2, "XZ"),
            (224, 1, 2, "YZ"),
        ]:
            a = fig.add_subplot(slot)
            a.plot(p[:, ai], p[:, bi], color="#57b9ff")
            a.scatter(p[:, ai], p[:, bi], c=c, cmap="viridis", s=8)
            a.scatter(0, 0, c="white", marker="s")
            a.scatter(*p[0, [ai, bi]], c="#4ade80", s=60)
            a.scatter(*p[-1, [ai, bi]], c="#fb5b5b", s=60)
            a.set_title(title)
            a.grid(alpha=0.2)
            a.set_aspect("equal", adjustable="datalim")
    else:
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, "No valid AprilGrid pose", ha="center", fontsize=24)
    rmse = summary["reprojection_rmse_px"]["median"]
    fig.text(
        0.02,
        0.02,
        f"IDs {summary['recognized_ids']} · median RMSE {rmse if rmse is not None else 'n/a'} px",
    )
    fig.text(
        0.98,
        0.02,
        "APPROXIMATE / DEMO-GRADE — UNCALIBRATED PANORAMA MODEL",
        ha="right",
        color="#ffc857",
        weight="bold",
    )
    fig.savefig(
        session / "relative_coordinates.png",
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline AprilGrid pose from a stitched 2:1 panorama"
    )
    p.add_argument("input", type=Path)
    p.add_argument("--tag-size", type=float, default=0.088)
    p.add_argument("--spacing", type=float, default=0.30)
    p.add_argument("--rows", type=int, default=6)
    p.add_argument("--cols", type=int, default=6)
    p.add_argument("--first-id", type=int, default=0)
    p.add_argument("--sample-fps", type=float, default=5.0)
    p.add_argument("--output-dir", type=Path, default=Path("sessions"))
    p.add_argument("--min-tags", type=int, default=6)
    p.add_argument("--view-size", type=int, default=960)
    p.add_argument(
        "--max-speed",
        type=float,
        default=5.0,
        help="reject filtered jumps faster than m/s",
    )
    p.add_argument(
        "--official-stitched",
        action="store_true",
        help="input was exported by DJI Studio",
    )
    p.add_argument("--session-name")
    p.add_argument("--status-file", type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if (
        args.tag_size <= 0
        or args.spacing < 0
        or args.sample_fps <= 0
        or args.min_tags < 4
    ):
        raise SystemExit("invalid grid/sampling parameters")
    session = args.output_dir / (
        args.session_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    # The web controller pre-creates the directory for its launcher log.
    session.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(session / "processor.log"),
            logging.StreamHandler(),
        ],
    )
    signal.signal(signal.SIGINT, lambda *_: globals().__setitem__("STOP", True))
    cap = cv2.VideoCapture(str(args.input), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        LOG.error("cannot open input: %s", args.input)
        return 2
    width, height = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if width != 2 * height:
        LOG.error("expected 2:1 equirectangular input, got %dx%d", width, height)
        return 3
    step = max(1, round(source_fps / args.sample_fps)) if source_fps > 0 else 1
    grid = Grid(args.rows, args.cols, args.tag_size, args.spacing, args.first_id)
    detector = make_detector()
    seen = Counter()
    rmses: list[float] = []
    jumps: list[float] = []
    processed = valid = frame_no = 0
    previous: tuple[float, np.ndarray] | None = None
    csv_fields = [
        "frame",
        "timestamp",
        "camera_x_m",
        "camera_y_m",
        "camera_z_m",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
        "raw_camera_x_m",
        "raw_camera_y_m",
        "raw_camera_z_m",
        "detected_tag_count",
        "inlier_count",
        "reprojection_rmse_px",
        "detected_ids",
        "selected_view",
        "quality_status",
    ]
    with (
        (session / "pose.csv").open("w", newline="", encoding="utf-8") as cf,
        (session / "detections.jsonl").open("w", encoding="utf-8") as jf,
    ):
        writer = csv.DictWriter(cf, fieldnames=csv_fields)
        writer.writeheader()
        while not STOP:
            ok, pano = cap.read()
            if not ok:
                break
            if frame_no % step:
                frame_no += 1
                continue
            timestamp = frame_no / source_fps if source_fps > 0 else float(processed)
            processed += 1
            all_ids: set[int] = set()
            view_records = []
            candidates = []
            for view in DEFAULT_VIEWS:
                perspective = py360convert.e2p(
                    pano,
                    fov_deg=view.fov,
                    u_deg=view.yaw,
                    v_deg=view.pitch,
                    out_hw=(args.view_size, args.view_size),
                    mode="bilinear",
                )
                detections = detect_view(perspective, detector, grid)
                ids = sorted({int(d["id"]) for d in detections})
                all_ids.update(ids)
                pose = solve_view(detections, view, args.view_size, args.min_tags)
                if pose:
                    candidates.append(pose)
                view_records.append(
                    {
                        "view": asdict(view),
                        "detections": [
                            {
                                "id": d["id"],
                                "corners_px": d["corners_px"].round(2).tolist(),
                            }
                            for d in detections
                        ],
                        "pose": None
                        if pose is None
                        else {
                            "xyz": pose.xyz.tolist(),
                            "rpy": pose.rpy,
                            "inliers": pose.inliers,
                            "rmse": pose.rmse,
                        },
                    }
                )
            for tag_id in all_ids:
                seen[tag_id] += 1
            pose = choose_pose(candidates)
            quality = "insufficient_tags"
            filtered = None
            jump = None
            if pose:
                quality = "valid"
                filtered = pose.xyz.copy()
                rmses.append(pose.rmse)
                if previous:
                    dt = timestamp - previous[0]
                    jump = float(np.linalg.norm(pose.xyz - previous[1]))
                    jumps.append(jump)
                    if dt > 0 and jump / dt > args.max_speed:
                        quality = "jump_rejected"
                        filtered = previous[1].copy()
                if quality == "valid":
                    previous = (timestamp, filtered.copy())
                    valid += 1
            row = dict.fromkeys(csv_fields, "")
            row.update(
                frame=frame_no,
                timestamp=f"{timestamp:.6f}",
                detected_tag_count=len(all_ids),
                detected_ids=" ".join(map(str, sorted(all_ids))),
                quality_status=quality,
            )
            if pose:
                row.update(
                    raw_camera_x_m=f"{pose.xyz[0]:.7f}",
                    raw_camera_y_m=f"{pose.xyz[1]:.7f}",
                    raw_camera_z_m=f"{pose.xyz[2]:.7f}",
                    inlier_count=pose.inliers,
                    reprojection_rmse_px=f"{pose.rmse:.4f}",
                    selected_view=pose.view,
                    roll_deg=f"{pose.rpy[0]:.4f}",
                    pitch_deg=f"{pose.rpy[1]:.4f}",
                    yaw_deg=f"{pose.rpy[2]:.4f}",
                )
                if filtered is not None:
                    row.update(
                        camera_x_m=f"{filtered[0]:.7f}",
                        camera_y_m=f"{filtered[1]:.7f}",
                        camera_z_m=f"{filtered[2]:.7f}",
                    )
            writer.writerow(row)
            cf.flush()
            jf.write(
                json.dumps(
                    {
                        "frame": frame_no,
                        "timestamp": timestamp,
                        "detected_ids": sorted(all_ids),
                        "views": view_records,
                        "selected_view": pose.view if pose else None,
                        "quality_status": quality,
                        "jump_m": jump,
                    }
                )
                + "\n"
            )
            jf.flush()
            status = {
                "running": True,
                "frame": frame_no,
                "processed_frames": processed,
                "pose_frames": valid,
                "seen_ids": sorted(seen),
            }
            if args.status_file:
                args.status_file.write_text(json.dumps(status), encoding="utf-8")
            if processed % 5 == 0:
                LOG.info(
                    "frame=%d processed=%d ids=%s valid=%d",
                    frame_no,
                    processed,
                    sorted(all_ids),
                    valid,
                )
            frame_no += 1
    cap.release()
    expected = list(range(args.first_id, args.first_id + args.rows * args.cols))
    summary = {
        "input": str(args.input.resolve()),
        "total_frames": total,
        "processed_frames": processed,
        "valid_pose_frames": valid,
        "valid_pose_ratio": valid / processed if processed else 0.0,
        "recognized_ids": sorted(seen),
        "missing_ids": [i for i in expected if i not in seen],
        "detections_per_id": {str(k): v for k, v in sorted(seen.items())},
        "tag_coverage_ratio": len(set(expected) & set(seen)) / len(expected),
        "reprojection_rmse_px": _finite_stats(rmses),
        "adjacent_coordinate_jump_m": _finite_stats(jumps),
        "tag_size_m": args.tag_size,
        "spacing_ratio": args.spacing,
        "rows": args.rows,
        "cols": args.cols,
        "first_id": args.first_id,
        "official_stitched_panorama": args.official_stitched,
        "measurement_grade_camera_model": False,
        "accuracy_label": "APPROXIMATE / DEMO-GRADE",
        "stopped": STOP,
    }
    (session / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    generate_plot(session, summary)
    if args.status_file:
        args.status_file.write_text(
            json.dumps({"running": False, **summary}), encoding="utf-8"
        )
    LOG.info("complete: %s", json.dumps(summary, ensure_ascii=False))
    print(session.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
