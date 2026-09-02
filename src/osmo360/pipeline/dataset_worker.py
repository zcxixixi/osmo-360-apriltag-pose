from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate, correlation_lags
from scipy.spatial.transform import Rotation

from .dataset import FFPROBE, PIPELINE_REVISION
from .instaumi_format import (
    common_window,
    export_aligned_video,
    sha256,
    write_dataset_h5,
)
from .insta360_telemetry import extract_x5_imu
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
    parser.add_argument("--dataset-id", required=True)
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
        str(FFMPEG), "-y", "-i", str(source), "-vn", "-ac", "1", "-ar", "2000",
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


def trajectory_sample_stride(left_fps: float, right_fps: float) -> int:
    return max(1, round(min(left_fps, right_fps) / 30.0))




def cache_signature_matches(
    video: Path, record: dict[str, Any], stream: int, intercept_s: float,
    output: Path, *, verify_video_hash: bool = True,
) -> bool:
    sidecar = output.with_suffix(".json")
    if not output.is_file() or not sidecar.is_file():
        return False
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    clock = metadata.get("clock_mapping", {})
    expected = (
        metadata.get("schema_version") == "fisheye-apriltag-observation-cache/1.0"
        and metadata.get("producer_revision") == "raw-fisheye-cache-v2"
        and metadata.get("video") == str(video.resolve())
        and metadata.get("camera_serial") == record["serial"]
        and metadata.get("stream") == stream
        and metadata.get("source_size") == [1920, 1920]
        and metadata.get("x5_offset") == record["x5_offset"]
        and metadata.get("rectified_detection") is True
        and metadata.get("rectified_view_size") == 720
        and metadata.get("frame_stride") == max(1, round(float(record["fps"]) / 30.0))
        and clock.get("slope") == 1.0
        and math.isclose(float(clock.get("intercept_s", float("nan"))), intercept_s)
    )
    return bool(
        expected
        and (
            not verify_video_hash
            or metadata.get("video_sha256") == sha256(video)
        )
    )


def cache_lens(video: Path, record: dict[str, Any], stream: int, intercept_s: float,
               output: Path, log: Path) -> None:
    if cache_signature_matches(video, record, stream, intercept_s, output):
        return
    output.unlink(missing_ok=True)
    output.with_suffix(".json").unlink(missing_ok=True)
    stride = max(1, round(float(record["fps"]) / 30.0))
    run([
        str(PYTHON), "-m", "tools.cache_fisheye_apriltag_observations", str(video),
        "--x5-offset", record["x5_offset"], "--camera-serial", record["serial"],
        "--stream", str(stream), "--source-width", "1920",
        "--source-height", "1920", "--clock-intercept-s", str(intercept_s),
        "--frame-stride", str(stride), "--rectified-detection", "--rectified-view-size", "720",
        "--output", str(output),
    ], log)
    if not cache_signature_matches(
        video, record, stream, intercept_s, output, verify_video_hash=False,
    ):
        raise RuntimeError(f"generated cache failed signature validation: {output}")


def cached_camera_pose(
    cache: Path, tag_map: Path, output_root: Path, run_name: str,
    sample_stride: int, log: Path, *, initial_pose: Path | None = None,
    start_common_s: float | None = None, end_common_s: float | None = None,
) -> Path:
    pose = output_root / run_name / "pose.csv"
    if pose.is_file():
        return pose
    command = [
        str(PYTHON), "-m", "tools.raw_fisheye_world_pose_cached",
        "--observation-cache", str(cache), "--tag-map", str(tag_map),
        "--sample-stride", str(sample_stride), "--output-dir",
        str(output_root / run_name),
    ]
    if initial_pose is not None:
        command.extend(["--initial-pose", str(initial_pose)])
    if start_common_s is not None:
        command.extend(["--start-common-s", str(start_common_s)])
    if end_common_s is not None:
        command.extend(["--end-common-s", str(end_common_s)])
    run(command, log)
    if not pose.is_file():
        raise RuntimeError(f"cached camera pose was not produced: {pose}")
    return pose




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
    output_root = root / pair["dataset_id"]
    processed = output_root / "processed"
    video_dir = output_root / "video"
    aligned_dir = scratch / "aligned-1920"
    processed.mkdir(parents=True, exist_ok=True)
    (processed / "trajectories").mkdir(exist_ok=True)
    (processed / "gripper").mkdir(exist_ok=True)
    logs = scratch / "logs"; status = processed / "status.json"
    left_source = root / pair["left"]["path"]; right_source = root / pair["right"]["path"]
    status_update(status, "identity", "PASS", left=pair["left"], right=pair["right"])

    audio = scratch / "audio"
    extract_audio(left_source, audio / "left.wav", logs / "audio-left.log")
    extract_audio(right_source, audio / "right.wav", logs / "audio-right.log")
    sync = estimate_audio_offset(
        audio / "left.wav", audio / "right.wav", -float(pair["recording_start_delta_s"])
    )
    (audio / "sync.json").write_text(json.dumps(sync, indent=2) + "\n")
    (processed / "sync.json").write_text(json.dumps(sync, indent=2) + "\n")
    status_update(status, "sync", "PASS", **sync)

    left_start, right_start, common_duration = common_window(
        float(pair["left"]["duration_s"]), float(pair["right"]["duration_s"]),
        float(sync["offset_s"]),
    )
    aligned_lenses = [
        (left_source, aligned_dir / "Left_forward.mp4", 1, left_start, "left-forward"),
        (left_source, aligned_dir / "Left_back.mp4", 0, left_start, "left-back"),
        (right_source, aligned_dir / "Right_forward.mp4", 1, right_start, "right-forward"),
        (right_source, aligned_dir / "Right_back.mp4", 0, right_start, "right-back"),
    ]
    for source, output, stream, start, name in aligned_lenses:
        export_aligned_video(
            FFMPEG, source, output, start_s=start, duration_s=common_duration,
            stream=stream, output_size=1920, log=logs / f"aligned-{name}.log",
        )
    status_update(
        status, "aligned_processing_video", "PASS", duration_s=common_duration,
        left_start_s=left_start, right_start_s=right_start,
    )

    # Diagnostic mode retains all four aligned 1920² videos in scratch.
    if os.environ.get("INSTAUMI_PIPELINE_ALIGNMENT_ONLY", "0") == "1":
        return 0

    raw = scratch / "raw"
    for side, record in (("left", pair["left"]), ("right", pair["right"])):
        side_root = raw / side
        title = side.title()
        for stream, lens_name in ((0, "back"), (1, "forward")):
            lens = aligned_dir / f"{title}_{lens_name}.mp4"
            cache_lens(
                lens, record, stream, 0.0,
                side_root / f"lens-{stream}-corners.npz",
                logs / f"cache-{side}-{stream}.log",
            )
        run([
            str(PYTHON), "-m", "tools.merge_fisheye_observation_caches",
            str(side_root / "lens-0-corners.npz"), str(side_root / "lens-1-corners.npz"),
            "--output", str(side_root / "dual-lens-corners.npz"),
        ], logs / f"merge-{side}.log")
    status_update(status, "dual_lens_cache", "PASS")

    bootstrap = scratch / "bootstrap"; panel_poses: dict[tuple[str, str], Path] = {}
    for side, record in (("left", pair["left"]), ("right", pair["right"])):
        cache = raw / side / "dual-lens-corners.npz"
        sample_stride = max(1, round(float(record["fps"]) / 2.0))
        panel_poses[(side, "A")] = cached_camera_pose(
            cache, PANEL_A, bootstrap / side, "panel-A", sample_stride,
            logs / f"panel-{side}-A.log",
        )
        panel_poses[(side, "B")] = cached_camera_pose(
            cache, PANEL_B, bootstrap / side, "panel-B", sample_stride,
            logs / f"panel-{side}-B.log",
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
    for side, record in (("left", pair["left"]), ("right", pair["right"])):
        camera_pose[side] = cached_camera_pose(
            raw / side / "dual-lens-corners.npz", world_map, world / side,
            "session-world", max(1, round(float(record["fps"]) / 30.0)),
            logs / f"world-{side}.log",
            initial_pose=panel_poses[(side, "A")],
            start_common_s=0.0, end_common_s=common_duration,
        )
    joint_sample_stride = trajectory_sample_stride(
        float(pair["left"]["fps"]), float(pair["right"]["fps"]),
    )
    joint = scratch / "joint"
    run([
        str(PYTHON), "-m", "tools.joint_dual_camera_pose_graph_cached",
        "--left-cache", str(raw / "left/dual-lens-corners.npz"),
        "--right-cache", str(raw / "right/dual-lens-corners.npz"),
        "--left-initial-pose", str(camera_pose["left"]),
        "--right-initial-pose", str(camera_pose["right"]),
        "--left-panel-map", str(PANEL_A), "--right-panel-map", str(PANEL_B),
        "--initial-world-map", str(world_map),
        "--left-tag-id", str(pair["left"]["base_tag_id"]),
        "--right-tag-id", str(pair["right"]["base_tag_id"]),
        "--start-common-s", "0", "--end-common-s", str(common_duration),
        "--sample-stride", str(joint_sample_stride),
        "--alternations", "4", "--workers", "8",
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

    for name, source in (("calibration", calibration), ("trajectories", trajectories),
                         ("gates", joint)):
        destination = processed / name
        if destination.exists(): shutil.rmtree(destination)
        shutil.copytree(source, destination)
    if not passed:
        return 2

    for side in ("Left", "Right"):
        export_aligned_video(
            FFMPEG, aligned_dir / f"{side}_back.mp4", video_dir / f"{side}.mp4",
            start_s=0.0, duration_s=common_duration, stream=0, output_size=1024,
            log=logs / f"aligned-{side.lower()}-1024.log",
        )
    imu = {
        "left": extract_x5_imu(
            left_source,
            scratch / "telemetry/left",
            source_start_s=left_start,
            duration_s=common_duration,
            expected_serial=pair["left"]["serial"],
        ),
        "right": extract_x5_imu(
            right_source,
            scratch / "telemetry/right",
            source_start_s=right_start,
            duration_s=common_duration,
            expected_serial=pair["right"]["serial"],
        ),
    }
    status_update(
        status, "imu", "PASS",
        sample_count={side: int(len(samples.timestamp_ns)) for side, samples in imu.items()},
        source={"left": "left_x5_insv", "right": "right_x5_insv"},
    )
    metadata = write_dataset_h5(
        output_root / "dataset.h5", dataset_id=pair["dataset_id"],
        left_video=video_dir / "Left.mp4", right_video=video_dir / "Right.mp4",
        left_source=left_source, right_source=right_source,
        left_start_s=left_start, right_start_s=right_start, sync=sync,
        ffprobe=FFPROBE, source_records={"left": pair["left"], "right": pair["right"]},
        imu=imu,
    )
    (processed / "review.json").write_text(json.dumps({
        "schema_version": "instaumi-alignment-review/1.0",
        "dataset_id": pair["dataset_id"], "pair_id": pair["pair_id"],
        "duration_s": common_duration, "sync": sync,
        "left_source_start_s": left_start, "right_source_start_s": right_start,
        "video": metadata["video"],
    }, indent=2) + "\n", encoding="utf-8")
    status_update(status, "final_video", "PASS", duration_s=common_duration,
                  left="video/Left.mp4", right="video/Right.mp4")
    shutil.rmtree(aligned_dir)
    return 0


def main() -> int:
    args = parse_args(); root = args.dataset_root.resolve(strict=True)
    lock_path = root / args.dataset_id / "processed" / "manifest.lock.json"
    if not lock_path.is_file():
        raise ManifestError(f"internal manifest lock is missing: {lock_path}")
    lock = json.loads(lock_path.read_text())
    pair = next((item for item in lock["pairs"] if item["pair_id"] == args.pair_id), None)
    if pair is None:
        raise ManifestError(f"pair not found in internal manifest: {args.pair_id}")
    if pair.get("dataset_id") != args.dataset_id:
        raise ManifestError("dataset ID does not match the internal manifest")
    return process_pair(root, pair, args.scratch_root.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
