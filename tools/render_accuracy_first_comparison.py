#!/usr/bin/env python3
"""Visualize continuous v51b display poses against gated raw v52 measurements."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from rig_revision import load_rig_revision


LEFT = "#27b7e7"
RIGHT = "#45ce87"
TRUSTED = "#f4d35e"
RAW = "#8b98a8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rig-revision", type=Path, required=True)
    parser.add_argument("--pair-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def rows(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(newline="", encoding="utf-8")))


def base_to_tcp(path: Path, offset: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = rows(path)
    time = np.asarray([float(row["timestamp"]) for row in data])
    position = np.asarray([
        [float(row[f"base_{axis}_m"]) for axis in "xyz"] for row in data
    ])
    rotation = Rotation.from_quat([
        [float(row[key]) for key in ("qx", "qy", "qz", "qw")] for row in data
    ])
    return time, position + rotation.apply(np.tile(offset, (len(data), 1))), rotation.as_quat()


def raw_track(data: list[dict[str, str]], role: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [row for row in data if row.get(f"{role}_tcp_x_m") not in (None, "")]
    time = np.asarray([float(row["timestamp"]) for row in selected])
    position = np.asarray([
        [float(row[f"{role}_tcp_{axis}_m"]) for axis in "xyz"] for row in selected
    ])
    trusted = np.asarray([row["quality_status"] == "direct_trusted" for row in selected])
    return time, position, trusted


def status_family(status: str) -> str:
    if status == "direct_trusted":
        return "trusted"
    if "single_panel" in status:
        return "single wall"
    if "angular_rmse" in status:
        return "RMSE gate"
    if "cross_bearing" in status:
        return "cross-Tag gate"
    if "jump" in status:
        return "jump gate"
    return "pose unavailable"


def main() -> int:
    args = parse_args()
    bundle = load_rig_revision(args.rig_revision)
    pair = args.pair_dir.resolve()
    v52_dir = pair / "accuracy-first-v52-r1"
    v51_dir = pair / "fused-independent-world-v51b-relaxed-left"
    v52_rows = rows(v52_dir / "accuracy_first_raw_trajectory.csv")
    report = json.loads((v52_dir / "report.json").read_text(encoding="utf-8"))
    if report["revision_sha256"] != bundle["revision_sha256"]:
        raise ValueError("v52 report uses a different rig revision")
    offset = np.asarray(bundle["geometry"]["base_to_tcp"]["translation_m"], dtype=float)
    lt, lp, _ = base_to_tcp(v51_dir / "left_base_pose.csv", offset)
    rt, rp, _ = base_to_tcp(v51_dir / "right_base_pose.csv", offset)
    lrt, lrp, ltrust = raw_track(v52_rows, "left")
    rrt, rrp, rtrust = raw_track(v52_rows, "right")
    common_time = np.asarray([float(row["timestamp"]) for row in v52_rows])
    raw_separation = np.asarray([
        float(row["raw_tcp_separation_m"])
        if row.get("raw_tcp_separation_m") not in (None, "") else np.nan
        for row in v52_rows
    ])
    trusted_mask = np.asarray([
        row["quality_status"] == "direct_trusted" for row in v52_rows
    ])
    display_separation = np.linalg.norm(rp - lp, axis=1)
    time_origin = min(float(lt[0]), float(common_time[0]))

    plt.style.use("dark_background")
    figure, axes = plt.subplots(2, 2, figsize=(18, 11), constrained_layout=True)
    figure.patch.set_facecolor("#09111a")
    for axis in axes.flat:
        axis.set_facecolor("#0e1823")
        axis.grid(color="#314052", alpha=0.35, linewidth=0.7)

    top = axes[0, 0]
    top.plot(lp[:, 0], lp[:, 2], color=LEFT, alpha=0.45, linewidth=2,
             label="v51b left filtered")
    top.plot(rp[:, 0], rp[:, 2], color=RIGHT, alpha=0.45, linewidth=2,
             label="v51b right filtered")
    top.scatter(lrp[:, 0], lrp[:, 2], color=RAW, s=8, alpha=0.20,
                label="v52 raw candidates")
    top.scatter(rrp[:, 0], rrp[:, 2], color=RAW, s=8, alpha=0.20)
    top.scatter(lrp[ltrust, 0], lrp[ltrust, 2], color=LEFT, edgecolor=TRUSTED,
                s=42, linewidth=1.1, label="v52 left trusted", zorder=5)
    top.scatter(rrp[rtrust, 0], rrp[rtrust, 2], color=RIGHT, edgecolor=TRUSTED,
                s=42, linewidth=1.1, label="v52 right trusted", zorder=5)
    top.set_title("Top view: X-Z world plane")
    top.set_xlabel("world X [m]")
    top.set_ylabel("world Z [m]")
    top.axis("equal")
    top.legend(fontsize=8, loc="best")

    height = axes[0, 1]
    height.plot(lt - time_origin, -lp[:, 1], color=LEFT, alpha=0.45,
                linewidth=2, label="v51b left filtered")
    height.plot(rt - time_origin, -rp[:, 1], color=RIGHT, alpha=0.45,
                linewidth=2, label="v51b right filtered")
    height.scatter(lrt[ltrust] - time_origin, -lrp[ltrust, 1], color=LEFT,
                   edgecolor=TRUSTED, s=38, linewidth=1.0, label="v52 left trusted")
    height.scatter(rrt[rtrust] - time_origin, -rrp[rtrust, 1], color=RIGHT,
                   edgecolor=TRUSTED, s=38, linewidth=1.0, label="v52 right trusted")
    height.set_title("Physical height (-world Y)")
    height.set_xlabel("time [s]")
    height.set_ylabel("height [m]")
    height.legend(fontsize=8, loc="best")

    separation = axes[1, 0]
    separation.plot(lt - time_origin, display_separation * 1000.0, color="#b08cff",
                    alpha=0.65, linewidth=2, label="v51b filtered TCP distance")
    finite = np.isfinite(raw_separation)
    separation.scatter(common_time[finite] - time_origin,
                       raw_separation[finite] * 1000.0, color=RAW, alpha=0.22,
                       s=8, label="v52 raw candidates")
    separation.scatter(common_time[trusted_mask] - time_origin,
                       raw_separation[trusted_mask] * 1000.0, color=TRUSTED,
                       edgecolor="#fff1a8", linewidth=0.8, s=45,
                       label="v52 trusted raw distance", zorder=5)
    separation.set_title("TCP separation: display-smoothed vs raw trusted")
    separation.set_xlabel("time [s]")
    separation.set_ylabel("TCP distance [mm]")
    separation.legend(fontsize=8, loc="best")

    status = axes[1, 1]
    palette = {
        "trusted": TRUSTED,
        "single wall": "#f08a5d",
        "RMSE gate": "#b57edc",
        "cross-Tag gate": "#e85d75",
        "jump gate": "#ff3b30",
        "pose unavailable": "#5c6775",
    }
    families = [status_family(row["quality_status"]) for row in v52_rows]
    for family, color in palette.items():
        mask = np.asarray([value == family for value in families])
        if mask.any():
            status.scatter(common_time[mask] - time_origin, np.zeros(mask.sum()),
                           marker="s", s=28, color=color, label=family)
    status.set_ylim(-0.8, 1.8)
    status.set_yticks([])
    status.set_xlabel("time [s]")
    status.set_title("v52 trust gate result per synchronized frame")
    status.legend(fontsize=8, loc="upper left", ncol=2)
    counts = report["counts"]
    metrics = report["cross_bearing_error_deg"]
    status.text(
        0.01, 0.58,
        "\n".join((
            f"Aligned frames: {counts['aligned_frames']}",
            f"Trusted raw frames: {counts['trusted_frames']} "
            f"({100.0 * counts['trusted_ratio']:.1f}%)",
            f"Longest trusted run: {report['longest_trusted_run_s']:.2f} s",
            f"Trusted cross-Tag P95: {metrics['trusted_p95']:.2f} deg"
            if metrics["trusted_p95"] is not None else "Trusted cross-Tag P95: n/a",
            "Metric smoothing: OFF",
            "Interpolation: OFF",
        )),
        transform=status.transAxes, fontsize=12, va="top",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#121f2c", "alpha": 0.9},
    )

    figure.suptitle(
        f"{pair.name}: v51b continuous display vs v52 accuracy-first raw data",
        fontsize=18, fontweight="bold",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "pair": pair.name,
        "trusted_frames": report["counts"]["trusted_frames"],
        "aligned_frames": report["counts"]["aligned_frames"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
