from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate, correlation_lags
from scipy.spatial.transform import Rotation

from .dataset import FFPROBE, PIPELINE_REVISION
from .manifest import ManifestError, ROOT

PYTHON = ROOT / ".venv/bin/python"
FFMPEG = FFPROBE.with_name("ffmpeg")
PANEL_A = ROOT / "config/a3_aprilgrid_A_200_205_120mm.json"
PANEL_B = ROOT / "config/a3_aprilgrid_B_210_215_120mm.json"
TAG_TO_TCP = np.asarray([0.10935, 0.0, -0.0095], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    return parser.parse_args()


def run(command: list[str], log: Path, *, gate: bool = False) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(
        command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log.write_text(process.stdout, encoding="utf-8")
    if process.returncode and not (gate and process.returncode == 2):
        raise RuntimeError(
            f"command failed ({process.returncode}): {' '.join(command)}; log={log}"
        )
    return process.returncode


def status_update(path: Path, stage: str, state: str, **details: Any) -> None:
    payload = json.loads(path.read_text()) if path.is_file() else {
        "schema_version": "dual-x5-pair-worker-status/1.0",
        "pipeline_revision": PIPELINE_REVISION,
        "stages": {},
    }
    payload["stages"][stage] = {"state": state, **details}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def extract_audio(source: Path, output: Path, log: Path) -> None:
    if output.is_file():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    run([
        str(FFMPEG), "-y", "-i", str(source), "-vn", "-ac", "1", "-ar", "8000",
        "-c:a", "pcm_s16le", str(output),
    ], log)


def estimate_audio_offset(left_path: Path, right_path: Path, approximate_s: float) -> dict[str, float]:
    left_rate, left = wavfile.read(left_path)
    right_rate, right = wavfile.read(right_path)
    if left_rate != right_rate:
        raise ValueError("left/right audio rates differ")
    left = left.astype(np.float64); right = right.astype(np.float64)
    left -= left.mean(); right -= right.mean()
    corr = correlate(right, left, mode="full", method="fft")
    lags = correlation_lags(len(right), len(left), mode="full")
    expected = int(round(approximate_s * left_rate)); radius = int(5 * left_rate)
    keep = np.abs(lags - expected) <= radius
    local_index = int(np.argmax(corr[keep])); index = int(np.flatnonzero(keep)[local_index])
    lag = int(lags[index])
    denominator = math.sqrt(float(np.dot(left, left) * np.dot(right, right)))
    return {
        "mapping": "right_time_s = left_time_s + offset_s",
        "offset_s": lag / left_rate,
        "correlation": float(corr[index] / denominator),
        "uncertainty_s": 1.0 / left_rate,
    }


def remux(source: Path, stream: int, output: Path, log: Path) -> None:
    if output.is_file():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    run([str(FFMPEG), "-y", "-i", str(source), "-map", f"0:v:{stream}",
         "-c", "copy", str(output)], log)


def cache_lens(video: Path, record: dict[str, Any], stream: int, intercept_s: float,
               output: Path, log: Path) -> None:
    if output.is_file() and output.with_suffix(".json").is_file():
        return
    stride = max(1, round(float(record["fps"]) / 30.0))
    run([
        str(PYTHON), "-m", "tools.cache_fisheye_apriltag_observations", str(video),
        "--x5-offset", record["x5_offset"], "--camera-serial", record["serial"],
        "--stream", str(stream), "--source-width", str(record["lens_size"][0]),
        "--source-height", str(record["lens_size"][1]), "--clock-intercept-s", str(intercept_s),
        "--frame-stride", str(stride), "--rectified-detection", "--rectified-view-size", "720",
        "--output", str(output),
    ], log)


def stitch(source: Path, output: Path, log: Path) -> None:
    if output.is_file():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    run([str(PYTHON), "-m", "tools.insta360_media_stitch", str(source), str(output),
         "--width", "3840", "--stitch-type", "optflow"], log)


def camera_dataset(video: Path, tag_map: Path, output_root: Path, run_name: str,
                   sample_fps: float, log: Path) -> Path:
    pose = output_root / run_name / "visual/pose.csv"
    if pose.is_file():
        return pose
    run([
        str(ROOT / "bin/camera-to-dataset"), str(video), "--camera", "insta360",
        "--tag-map", str(tag_map), "--output-root", str(output_root),
        "--run-name", run_name, "--sample-fps", str(sample_fps),
        "--projection-backend", "cuda", "--skip-preview", "--force",
    ], log)
    if not pose.is_file():
        raise RuntimeError(f"camera pose was not produced: {pose}")
    return pose


def shift_pose_time(source: Path, output: Path, delta_s: float) -> None:
    rows = list(csv.DictReader(source.open(newline="")))
    if not rows:
        raise ValueError(f"empty pose CSV: {source}")
    fields = list(rows[0]); key = "timestamp" if "timestamp" in fields else "timestamp_s"
    for row in rows:
        row[key] = str(float(row[key]) + delta_s)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def export_tcp(camera_csv: Path, own_tag: dict[str, Any], output: Path) -> None:
    rows = list(csv.DictReader(camera_csv.open(newline="")))
    if not rows:
        raise ValueError(f"empty camera trajectory: {camera_csv}")
    added = ["tcp_x_m", "tcp_y_m", "tcp_z_m", "tcp_qx", "tcp_qy", "tcp_qz", "tcp_qw"]
    fields = list(rows[0]) + [key for key in added if key not in rows[0]]
    camera_tag_r = Rotation.from_quat(own_tag["transform"]["quaternion_xyzw"])
    camera_tag_t = np.asarray(own_tag["transform"]["translation_m"], dtype=float)
    camera_tcp_t = camera_tag_t + camera_tag_r.apply(TAG_TO_TCP)
    for row in rows:
        world_camera_t = np.asarray([float(row[key]) for key in ("camera_x_m", "camera_y_m", "camera_z_m")])
        world_camera_r = Rotation.from_quat([float(row[key]) for key in ("qx", "qy", "qz", "qw")])
        world_tcp_t = world_camera_t + world_camera_r.apply(camera_tcp_t)
        world_tcp_r = world_camera_r * camera_tag_r
        for key, value in zip(added[:3], world_tcp_t): row[key] = str(value)
        for key, value in zip(added[3:], world_tcp_r.as_quat()): row[key] = str(value)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def process_pair(root: Path, pair: dict[str, Any], scratch: Path) -> int:
    scratch.mkdir(parents=True, exist_ok=True)
    logs = scratch / "logs"; status = scratch / "status.json"
    left_source = root / pair["left"]["path"]; right_source = root / pair["right"]["path"]
    status_update(status, "identity", "PASS", left=pair["left"], right=pair["right"])

    audio = scratch / "audio"
    extract_audio(left_source, audio / "left.wav", logs / "audio-left.log")
    extract_audio(right_source, audio / "right.wav", logs / "audio-right.log")
    sync = estimate_audio_offset(
        audio / "left.wav", audio / "right.wav", -float(pair["recording_start_delta_s"])
    )
    (audio / "sync.json").write_text(json.dumps(sync, indent=2) + "\n")
    status_update(status, "sync", "PASS", **sync)

    raw = scratch / "raw"
    for side, source, record in (
        ("left", left_source, pair["left"]), ("right", right_source, pair["right"])
    ):
        intercept = 0.0 if side == "left" else float(sync["offset_s"])
        side_root = raw / side
        for stream in (0, 1):
            lens = side_root / f"lens-{stream}.mp4"
            remux(source, stream, lens, logs / f"remux-{side}-{stream}.log")
            cache_lens(lens, record, stream, intercept, side_root / f"lens-{stream}-corners.npz",
                       logs / f"cache-{side}-{stream}.log")
        run([
            str(PYTHON), "-m", "tools.merge_fisheye_observation_caches",
            str(side_root / "lens-0-corners.npz"), str(side_root / "lens-1-corners.npz"),
            "--output", str(side_root / "dual-lens-corners.npz"),
        ], logs / f"merge-{side}.log")
    status_update(status, "dual_lens_cache", "PASS")

    stitched = scratch / "stitched"
    stitch(left_source, stitched / "left-3840.mp4", logs / "stitch-left.log")
    stitch(right_source, stitched / "right-3840.mp4", logs / "stitch-right.log")
    bootstrap = scratch / "bootstrap"; panel_poses: dict[tuple[str, str], Path] = {}
    for side in ("left", "right"):
        video = stitched / f"{side}-3840.mp4"
        panel_poses[(side, "A")] = camera_dataset(
            video, PANEL_A, bootstrap / side, "panel-A", 2.0, logs / f"panel-{side}-A.log"
        )
        panel_poses[(side, "B")] = camera_dataset(
            video, PANEL_B, bootstrap / side, "panel-B", 2.0, logs / f"panel-{side}-B.log"
        )
    calibration = scratch / "calibration"
    run([
        str(PYTHON), "-m", "tools.calibrate_capture_a3_pair",
        "--left-a-pose", str(panel_poses[("left", "A")]),
        "--left-b-pose", str(panel_poses[("left", "B")]),
        "--right-a-pose", str(panel_poses[("right", "A")]),
        "--right-b-pose", str(panel_poses[("right", "B")]),
        "--layout-a", str(PANEL_A), "--layout-b", str(PANEL_B),
        "--pair-id", pair["pair_id"], "--output-dir", str(calibration),
    ], logs / "a3-calibration.log", gate=True)
    world_map = calibration / "session_world_map.json"
    if not world_map.is_file():
        raise RuntimeError("A3 calibration did not produce a session world map")
    status_update(status, "a3_session_map", "PASS_WITH_GATE_REPORT")

    world = scratch / "world"; camera_pose: dict[str, Path] = {}
    for side in ("left", "right"):
        camera_pose[side] = camera_dataset(
            stitched / f"{side}-3840.mp4", world_map, world / side,
            "session-world", 30.0, logs / f"world-{side}.log",
        )
    right_common = world / "right-common-time.csv"
    shift_pose_time(camera_pose["right"], right_common, -float(sync["offset_s"]))
    end_s = min(float(pair["left"]["duration_s"]),
                float(pair["right"]["duration_s"]) - float(sync["offset_s"]))
    joint = scratch / "joint"
    run([
        str(PYTHON), "-m", "tools.joint_dual_camera_pose_graph_cached",
        "--left-cache", str(raw / "left/dual-lens-corners.npz"),
        "--right-cache", str(raw / "right/dual-lens-corners.npz"),
        "--left-initial-pose", str(camera_pose["left"]),
        "--right-initial-pose", str(right_common),
        "--left-panel-map", str(PANEL_A), "--right-panel-map", str(PANEL_B),
        "--initial-world-map", str(world_map),
        "--left-tag-id", str(pair["left"]["base_tag_id"]),
        "--right-tag-id", str(pair["right"]["base_tag_id"]),
        "--start-common-s", "0", "--end-common-s", str(end_s),
        "--sample-stride", "6", "--alternations", "4", "--workers", "8",
        "--anchored-two-pass", "--output-dir", str(joint),
    ], logs / "joint.log", gate=True)
    report = json.loads((joint / "report.json").read_text())
    trajectories = scratch / "trajectories"; trajectories.mkdir(exist_ok=True)
    for side in ("left", "right"):
        camera_output = trajectories / f"{side}_camera_world.csv"
        shutil.copy2(joint / f"{side}_pose.csv", camera_output)
        export_tcp(camera_output, report["own_basetag"][side],
                   trajectories / f"{side}_tcp_world.csv")
    passed = report["status"] == "HOLDOUT_PASS"
    status_update(status, "joint_closure", "PASS" if passed else "FAIL")

    final = root / "final" / PIPELINE_REVISION / "pairs" / pair["pair_id"]
    final.mkdir(parents=True, exist_ok=True)
    for name, source in (("calibration", calibration), ("trajectories", trajectories),
                         ("gates", joint)):
        destination = final / name
        if destination.exists(): shutil.rmtree(destination)
        shutil.copytree(source, destination)
    shutil.copy2(status, final / "status.json")
    shutil.copy2(audio / "sync.json", final / "sync.json")
    return 0 if passed else 2


def main() -> int:
    args = parse_args(); root = args.dataset_root.resolve(strict=True)
    lock_path = root / "final" / PIPELINE_REVISION / "manifest.lock.json"
    if not lock_path.is_file():
        raise ManifestError(f"internal manifest lock is missing: {lock_path}")
    lock = json.loads(lock_path.read_text())
    pair = next((item for item in lock["pairs"] if item["pair_id"] == args.pair_id), None)
    if pair is None:
        raise ManifestError(f"pair not found in internal manifest: {args.pair_id}")
    return process_pair(root, pair, args.scratch_root.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
