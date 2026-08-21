#!/usr/bin/env python3
"""Auto-detect a 360 camera source and build a timestamped 6DoF dataset."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FFMPEG_BIN = ROOT / "work/tools/ffmpeg-master-latest-linux64-gpl/bin"
PANOFORGE_ROOT = ROOT.parent / "panoforge-test"
GRIPPER_MESHES = ROOT / "assets/gripper"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raw 360 camera video to 6DoF trajectory dataset")
    parser.add_argument("input", type=Path)
    parser.add_argument("--camera", choices=("auto", "dji", "insta360", "panorama"), default="auto")
    parser.add_argument("--output-root", type=Path, default=Path("camera-datasets"))
    parser.add_argument("--run-name")
    parser.add_argument("--tag-map", type=Path)
    parser.add_argument("--tag-size", type=float, default=0.088)
    parser.add_argument("--spacing", type=float, default=0.30)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--first-id", type=int, default=0)
    parser.add_argument("--sample-fps", type=float, default=60.0)
    parser.add_argument("--view-size", type=int, default=1440)
    parser.add_argument("--global-search-size", type=int, default=720)
    parser.add_argument("--max-rmse-px", type=float, default=1.2)
    parser.add_argument("--max-processed-frames", type=int)
    parser.add_argument("--stitch-width", type=int, choices=(3840, 6144, 7680), default=3840)
    parser.add_argument("--stitch-encoder", choices=("auto", "cpu", "nvenc"), default="auto")
    parser.add_argument("--projection-backend", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--panoforge-root", type=Path, default=PANOFORGE_ROOT)
    parser.add_argument("--extract-frames", action="store_true")
    parser.add_argument("--skip-preview", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def ffprobe(path: Path) -> dict:
    executable = FFMPEG_BIN / "ffprobe"
    if not executable.is_file():
        raise SystemExit(f"missing bundled ffprobe: {executable}")
    process = subprocess.run(
        [str(executable), "-v", "error", "-show_format", "-show_streams",
         "-print_format", "json", str(path)],
        capture_output=True, text=True, timeout=60,
    )
    if process.returncode:
        raise SystemExit(f"ffprobe failed: {process.stderr.strip()[:500]}")
    return json.loads(process.stdout)


def detect_source(path: Path, override: str = "auto") -> tuple[str, dict]:
    probe = ffprobe(path)
    if override != "auto":
        return override, probe
    suffix = path.suffix.lower()
    if suffix == ".osv":
        return "dji", probe
    if suffix in {".insv", ".lrv"}:
        return "insta360", probe
    text = json.dumps(probe, ensure_ascii=False).lower() + " " + path.name.lower()
    if "insta360" in text or "insta" in text:
        return "insta360", probe
    if "dji" in text or "osmo" in text:
        return "dji", probe
    videos = [stream for stream in probe.get("streams", []) if stream.get("codec_type") == "video"]
    if len(videos) >= 2 and all(
        int(stream.get("width", 0)) == int(stream.get("height", -1))
        for stream in videos[:2]
    ):
        return "dji", probe
    if videos:
        width, height = int(videos[0].get("width", 0)), int(videos[0].get("height", 0))
        if height > 0 and width == height * 2:
            return "panorama", probe
    return "unknown", probe


def cuda_available() -> bool:
    try:
        import cupy  # type: ignore
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def backend(requested: str) -> str:
    available = cuda_available()
    if requested == "cuda" and not available:
        raise SystemExit("CUDA requested but no CuPy CUDA device is available")
    if requested == "auto":
        return "cuda" if available else "cpu"
    return requested


def run(stage: str, command: list[str], log: Path, dry_run: bool) -> None:
    print(f"\n[{stage}] {shlex.join(command)}", flush=True)
    if dry_run:
        return
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{datetime.now(timezone.utc).isoformat()}] {stage}: {shlex.join(command)}\n")
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        code = process.wait()
    if code:
        raise SystemExit(f"{stage} failed with exit code {code}; see {log}")


def file_identity(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def main() -> int:
    args = parse_args()
    source = args.input.resolve()
    if not source.is_file():
        raise SystemExit(f"input does not exist: {source}")
    if args.tag_map and not args.tag_map.resolve().is_file():
        raise SystemExit(f"tag map does not exist: {args.tag_map.resolve()}")
    camera, probe = detect_source(source, args.camera)
    run_name = args.run_name or source.stem
    run_dir = (args.output_root.resolve() / run_name)
    manifest_path = run_dir / "pipeline_manifest.json"
    requested_source = file_identity(source)
    tag_map_identity = file_identity(args.tag_map.resolve()) if args.tag_map else None
    processing_parameters = {
        "camera_override": args.camera,
        "tag_map": tag_map_identity,
        "tag_size": args.tag_size,
        "spacing": args.spacing,
        "rows": args.rows,
        "cols": args.cols,
        "first_id": args.first_id,
        "sample_fps": args.sample_fps,
        "view_size": args.view_size,
        "global_search_size": args.global_search_size,
        "max_rmse_px": args.max_rmse_px,
        "max_processed_frames": args.max_processed_frames,
        "stitch_width": args.stitch_width,
        "stitch_encoder": args.stitch_encoder,
        "projection_backend": args.projection_backend,
    }
    if manifest_path.is_file() and not args.force:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("source") != requested_source:
            raise SystemExit(
                f"run name {run_name!r} belongs to a different input; choose a new --run-name or use --force"
            )
        if previous.get("parameters") != processing_parameters:
            raise SystemExit(
                f"run name {run_name!r} was created with different processing parameters; "
                "choose a new --run-name or use --force"
            )
    if not args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "camera-to-dataset-pipeline/1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": requested_source,
        "parameters": processing_parameters,
        "detected_camera": camera,
        "container_probe": probe,
        "status": "starting",
    }
    print(f"CAMERA_DETECTED {camera} {source.name}")
    if camera == "unknown":
        raise SystemExit("camera could not be identified; use --camera dji or --camera insta360")
    if camera == "insta360" and source.suffix.lower() in {".insv", ".lrv"}:
        manifest.update(
            status="waiting_for_insta360_sdk",
            message="Insta360 raw source recognized; official SDK integration is pending. No fake remux was attempted.",
        )
        if not args.dry_run:
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        print("INSTA360_WAITING_SDK: raw file recognized; dataset processing intentionally not started")
        return 2

    panorama = run_dir / "intermediate/panorama_factory_calibrated.mp4"
    sensor_metadata = run_dir / "sensor-metadata"
    log = run_dir / "pipeline.log"
    visual_dir = run_dir / "visual"
    pose_csv = visual_dir / "pose.csv"
    summary_json = visual_dir / "summary.json"
    preview = run_dir / "preview/trajectory_gripper_6dof.mp4"
    dataset = run_dir / "dataset"
    projection_backend = backend(args.projection_backend)

    video_streams = [
        stream for stream in probe.get("streams", []) if stream.get("codec_type") == "video"
    ]
    already_panorama = bool(
        video_streams
        and int(video_streams[0].get("height", 0)) > 0
        and int(video_streams[0].get("width", 0)) == 2 * int(video_streams[0].get("height", 0))
        and source.suffix.lower() not in {".osv", ".insv"}
    )
    if camera == "dji" and not already_panorama:
        stitch = [
            sys.executable, str(ROOT / "dji_osv_stitch.py"), str(source), str(panorama),
            "--metadata-dir", str(sensor_metadata), "--panoforge-root", str(args.panoforge_root.resolve()),
            "--ffmpeg-bin", str(FFMPEG_BIN), "--width", str(args.stitch_width),
            "--codec", "h264", "--quality", "18", "--encoder", args.stitch_encoder,
        ]
        if args.force:
            stitch.append("--force")
        if args.force or not panorama.is_file():
            run("dji_factory_stitch", stitch, log, args.dry_run)
        else:
            print(f"[dji_factory_stitch] reuse {panorama}")
    else:
        panorama = source

    track = [
        sys.executable, str(ROOT / "osmo_360_offline.py"), str(panorama),
        "--sample-fps", str(args.sample_fps), "--view-size", str(args.view_size),
        "--global-search-size", str(args.global_search_size),
        "--max-rmse-px", str(args.max_rmse_px), "--max-speed", "10",
        "--full-scan", "--temporal-flow", "--redetect-interval", "5",
        "--recovery-scan-interval", "3", "--global-refresh-interval", "60",
        "--projection-backend", projection_backend, "--official-stitched",
        "--output-dir", str(run_dir), "--session-name", "visual",
    ]
    if args.tag_map:
        track.extend(("--tag-map", str(args.tag_map.resolve()), "--min-tags", "2",
                      "--pnp-points", "corners", "--pnp-solver", "ippe"))
    else:
        track.extend((
            "--tag-size", str(args.tag_size), "--spacing", str(args.spacing),
            "--rows", str(args.rows), "--cols", str(args.cols), "--first-id", str(args.first_id),
            "--min-tags", "4", "--pnp-points", "centers", "--pnp-solver", "ippe",
        ))
    imu_csv = sensor_metadata / "imu_perframe.csv"
    if camera == "dji":
        track.extend(("--imu-csv", str(imu_csv)))
    if args.max_processed_frames:
        track.extend(("--max-processed-frames", str(args.max_processed_frames)))
    if args.force or not pose_csv.is_file():
        run("visual_6dof", track, log, args.dry_run)
    else:
        print(f"[visual_6dof] reuse {pose_csv}")

    preview_enabled = not args.skip_preview
    if not args.dry_run and summary_json.is_file():
        visual_summary = json.loads(summary_json.read_text(encoding="utf-8"))
        if int(visual_summary.get("valid_pose_frames", 0)) == 0:
            preview_enabled = False
            print("[preview] skipped because the clip contains no valid 6DoF pose")
    if preview_enabled:
        preview.parent.mkdir(parents=True, exist_ok=True) if not args.dry_run else None
        render = [
            sys.executable, str(ROOT / "render_trajectory_overlay_video.py"),
            str(panorama), str(pose_csv), str(preview), "--ffmpeg", str(FFMPEG_BIN / "ffmpeg"),
            "--fps", "30", "--filter", "kalman", "--median-window", "5",
            "--reference-frame", "start", "--layout", "analysis",
            "--claw-mesh-dir", str(GRIPPER_MESHES),
        ]
        if args.force or not preview.is_file():
            run("preview", render, log, args.dry_run)
        else:
            print(f"[preview] reuse {preview}")

    export = [
        sys.executable, str(ROOT / "export_trajectory_dataset.py"),
        str(panorama), str(pose_csv), str(summary_json), str(dataset),
        "--source-raw", str(source), "--camera-family", "dji_osmo_360" if camera == "dji" else camera,
        "--sensor-metadata-dir", str(sensor_metadata),
        "--detections-jsonl", str(visual_dir / "detections.jsonl"),
    ]
    if preview_enabled:
        export.extend(("--preview", str(preview)))
    if args.extract_frames:
        export.append("--extract-frames")
    run("dataset", export, log, args.dry_run)

    manifest.update(
        status="complete", projection_backend=projection_backend,
        outputs={
            "dataset": str(dataset), "metadata": str(dataset / "metadata.json"),
            "trajectory": str(dataset / "annotations/trajectory_6dof.csv"),
            "preview": str(dataset / "previews/trajectory_overlay.mp4") if preview_enabled else None,
        },
    )
    if not args.dry_run:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    if args.dry_run:
        print(f"DRY_RUN_COMPLETE planned_dataset={dataset}")
    else:
        print(f"DATASET_READY {dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
