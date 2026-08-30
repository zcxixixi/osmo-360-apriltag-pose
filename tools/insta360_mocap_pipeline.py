#!/usr/bin/env python3
"""One-command Insta360 X5 AprilTag/OptiTrack evaluation pipeline."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2


from tools._root import ROOT
DEFAULT_TAG_MAP = ROOT / "mocap-evaluation/config/insta360_x5_tag_map.json"
DEFAULT_GRIPPER_MESHES = ROOT / "assets/gripper"


@dataclass(frozen=True)
class PipelinePaths:
    run_dir: Path
    vision_dir: Path
    evaluation_dir: Path
    pose_csv: Path
    visual_summary: Path
    evaluation_json: Path
    comparison_video: Path
    manifest: Path
    log: Path


def pipeline_paths(output_root: Path, run_name: str) -> PipelinePaths:
    run_dir = output_root / run_name
    vision_dir = run_dir / "visual"
    evaluation_dir = run_dir / "evaluation"
    return PipelinePaths(
        run_dir=run_dir,
        vision_dir=vision_dir,
        evaluation_dir=evaluation_dir,
        pose_csv=vision_dir / "pose.csv",
        visual_summary=vision_dir / "summary.json",
        evaluation_json=evaluation_dir / "mocap_evaluation.json",
        comparison_video=evaluation_dir / "optitrack_vs_visual_gripper_kalman_rts.mp4",
        manifest=run_dir / "pipeline_manifest.json",
        log=run_dir / "pipeline.log",
    )


def file_identity(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def cuda_available() -> bool:
    try:
        import cupy  # type: ignore

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def resolve_backend(requested: str) -> str:
    available = cuda_available()
    if requested == "cuda" and not available:
        raise SystemExit("CUDA requested but CuPy/device is unavailable; use --backend cpu")
    return "cuda" if requested == "auto" and available else ("cpu" if requested == "auto" else requested)


def validate_video(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise SystemExit(f"cannot open video: {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if width != 2 * height or fps <= 0 or frames <= 0:
        raise SystemExit(f"expected a valid 2:1 panorama; got {width}x{height}, {fps} fps, {frames} frames")
    return {"width": width, "height": height, "fps": fps, "frames": frames}


def copy_input(source: Path, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / source.name
    if destination.exists() and destination.stat().st_size == source.stat().st_size:
        return destination
    shutil.copy2(source, destination)
    return destination


def vision_command(args: argparse.Namespace, video: Path, paths: PipelinePaths, backend: str) -> list[str]:
    return [
        sys.executable, "-m", "tools.osmo_360_offline", str(video),
        "--tag-map", str(args.tag_map), "--sample-fps", str(args.sample_fps),
        "--min-tags", "2", "--max-rmse-px", str(args.max_rmse_px),
        "--view-size", str(args.view_size), "--pnp-points", "corners",
        "--pnp-solver", "ippe", "--full-scan", "--temporal-flow",
        "--redetect-interval", str(args.redetect_interval),
        "--global-refresh-interval", str(args.global_refresh_interval),
        "--global-search-size", str(args.global_search_size),
        "--recovery-scan-interval", str(args.recovery_scan_interval),
        "--camera-model", "insta360-x5",
        "--decoder", args.decoder, "--scan-workers", str(args.scan_workers),
        "--horizontal-step-deg", "30", "--horizontal-fov-deg", "125",
        "--projection-backend", backend, "--max-speed", str(args.max_speed),
        "--official-stitched", "--output-dir", str(paths.run_dir),
        "--session-name", paths.vision_dir.name,
    ]


def evaluation_command(args: argparse.Namespace, motive: Path, paths: PipelinePaths) -> list[str]:
    return [
        sys.executable, "-m", "tools.evaluate_insta360_mocap",
        str(paths.pose_csv), str(motive), "--output-dir", str(paths.evaluation_dir),
        "--initial-time-offset", str(args.initial_time_offset),
        "--search-radius", str(args.time_search_radius),
        "--calibration-fraction", str(args.calibration_fraction),
        "--min-tags", "2", "--min-test-samples", str(args.min_test_samples),
    ]


def render_command(args: argparse.Namespace, video: Path, paths: PipelinePaths) -> list[str]:
    command = [
        sys.executable, "-m", "osmo360.visualization.render_mocap_comparison",
        str(video), str(paths.pose_csv), str(paths.evaluation_dir),
        "--output-fps", str(args.output_fps), "--output", str(paths.comparison_video),
    ]
    if args.gripper_mesh_dir:
        command.extend(("--gripper-mesh-dir", str(args.gripper_mesh_dir)))
    return command


def run_command(stage: str, command: list[str], log_path: Path, dry_run: bool) -> None:
    printable = shlex.join(command)
    print(f"\n[{stage}] {printable}", flush=True)
    if dry_run:
        return
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{datetime.now().isoformat()}] {stage}: {printable}\n")
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=environment,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise SystemExit(f"{stage} failed with exit code {return_code}; see {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-command unstabilized Insta360 X5 visual 6DoF vs OptiTrack evaluation"
    )
    parser.add_argument("video", type=Path, help="unstabilized 2:1 Insta360 Studio MP4")
    parser.add_argument("motive_csv", type=Path, help="Motive rigid-body CSV")
    parser.add_argument("--output-root", type=Path, default=Path("mocap-runs"))
    parser.add_argument("--run-name", default="insta360-x5-optitrack")
    parser.add_argument("--tag-map", type=Path, default=DEFAULT_TAG_MAP)
    parser.add_argument("--gripper-mesh-dir", type=Path, default=DEFAULT_GRIPPER_MESHES)
    parser.add_argument("--confirm-flowstate-off", action="store_true",
                        help="required before the costly visual stage")
    parser.add_argument("--copy-inputs", action="store_true")
    parser.add_argument("--backend", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--sample-fps", type=float, default=50.0)
    parser.add_argument("--view-size", type=int, default=1440)
    parser.add_argument("--global-search-size", type=int, default=720)
    parser.add_argument("--recovery-scan-interval", type=int, default=15)
    parser.add_argument("--redetect-interval", type=int, default=3)
    parser.add_argument("--global-refresh-interval", type=int, default=150)
    parser.add_argument("--decoder", choices=("auto", "cpu", "nvdec"), default="auto")
    parser.add_argument("--scan-workers", type=int, default=4)
    parser.add_argument("--max-rmse-px", type=float, default=8.0)
    parser.add_argument("--max-speed", type=float, default=10.0)
    parser.add_argument("--initial-time-offset", type=float, default=-3.852)
    parser.add_argument("--time-search-radius", type=float, default=1.0)
    parser.add_argument("--calibration-fraction", type=float, default=0.30)
    parser.add_argument("--min-test-samples", type=int, default=200)
    parser.add_argument("--output-fps", type=float, default=25.0)
    parser.add_argument("--from-stage", choices=("vision", "evaluate", "render"), default="vision")
    parser.add_argument("--force", action="store_true", help="rerun selected stages even if outputs exist")
    parser.add_argument("--dry-run", action="store_true", help="validate and print commands only")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.video = args.video.resolve()
    args.motive_csv = args.motive_csv.resolve()
    args.tag_map = args.tag_map.resolve()
    args.gripper_mesh_dir = args.gripper_mesh_dir.resolve() if args.gripper_mesh_dir else None
    args.output_root = args.output_root.resolve()
    for required in (args.video, args.motive_csv, args.tag_map):
        if not required.is_file():
            raise SystemExit(f"missing input: {required}")
    if args.gripper_mesh_dir and not all(
        (args.gripper_mesh_dir / name).is_file()
        for name in ("base_link.STL", "Link1.STL", "Link2.STL", "Link3.STL")
    ):
        raise SystemExit(f"incomplete gripper mesh directory: {args.gripper_mesh_dir}")
    stage_index = {"vision": 0, "evaluate": 1, "render": 2}[args.from_stage]
    if stage_index == 0 and not args.confirm_flowstate_off:
        raise SystemExit("refusing costly run: confirm Studio FlowState/direction lock are off with --confirm-flowstate-off")
    if stage_index <= 2 and shutil.which("gst-launch-1.0") is None:
        raise SystemExit("gst-launch-1.0 is required for H.264 comparison video encoding")
    video_metadata = validate_video(args.video)
    backend = resolve_backend(args.backend)
    paths = pipeline_paths(args.output_root, args.run_name)
    if not args.dry_run:
        paths.run_dir.mkdir(parents=True, exist_ok=True)
        paths.evaluation_dir.mkdir(parents=True, exist_ok=True)
    video, motive = args.video, args.motive_csv
    if args.copy_inputs and not args.dry_run:
        video = copy_input(video, paths.run_dir / "input")
        motive = copy_input(motive, paths.run_dir / "input")

    configuration = {
        "created_at": datetime.now().isoformat(), "backend": backend,
        "video": file_identity(video), "motive_csv": file_identity(motive),
        "tag_map": file_identity(args.tag_map), "video_metadata": video_metadata,
        "parameters": {
            key: getattr(args, key) for key in (
                "sample_fps", "view_size", "global_search_size",
                "recovery_scan_interval", "redetect_interval",
                "global_refresh_interval", "decoder", "scan_workers",
                "max_rmse_px", "max_speed",
                "initial_time_offset", "time_search_radius", "calibration_fraction",
                "min_test_samples", "output_fps",
            )
        },
        "outputs": {key: str(value) for key, value in asdict(paths).items()},
    }
    if paths.manifest.is_file() and not args.force:
        previous = json.loads(paths.manifest.read_text(encoding="utf-8"))
        changed = [
            key for key in ("backend", "video", "motive_csv", "tag_map", "parameters")
            if previous.get(key) != configuration.get(key)
        ]
        if changed:
            raise SystemExit(
                f"run-name already contains incompatible results ({', '.join(changed)} changed); "
                "choose a new --run-name or use --force"
            )
    commands = (
        ("vision", vision_command(args, video, paths, backend), paths.pose_csv),
        ("evaluate", evaluation_command(args, motive, paths), paths.evaluation_json),
        ("render", render_command(args, video, paths), paths.comparison_video),
    )
    for index, (stage, command, expected) in enumerate(commands):
        if index < stage_index:
            continue
        prerequisites = {
            "evaluate": (paths.pose_csv,),
            "render": (paths.pose_csv, paths.evaluation_json),
        }.get(stage, ())
        missing = [path for path in prerequisites if not path.is_file()]
        produced_by = {paths.pose_csv: 0, paths.evaluation_json: 1}
        if args.dry_run:
            missing = [path for path in missing if produced_by.get(path, -1) < stage_index]
        if missing:
            raise SystemExit(f"{stage} prerequisites missing: {missing}")
        if expected.exists() and not args.force:
            print(f"[{stage}] reuse {expected}")
            continue
        if stage == "render" and expected.exists() and args.force and not args.dry_run:
            expected.unlink()
        run_command(stage, command, paths.log, args.dry_run)

    if args.dry_run:
        print("\ndry-run complete")
        return 0
    if paths.evaluation_json.is_file():
        report = json.loads(paths.evaluation_json.read_text(encoding="utf-8"))
        configuration["result_status"] = report["result_status"]
        configuration["publishable_accuracy"] = report["publishable_accuracy"]
        configuration["formal_metrics"] = {
            "position_ate_m": report["position_ate_m"],
            "orientation_error_deg": report["orientation_error_deg"],
            "rpe_20ms": report["rpe_20ms"], "rpe_1s": report["rpe_1s"],
        }
    paths.manifest.write_text(json.dumps(configuration, indent=2, ensure_ascii=False) + "\n",
                              encoding="utf-8")
    print(f"\ncomplete: {paths.run_dir}")
    if paths.comparison_video.is_file():
        print(f"video: {paths.comparison_video}")
    return 0 if configuration.get("publishable_accuracy", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
