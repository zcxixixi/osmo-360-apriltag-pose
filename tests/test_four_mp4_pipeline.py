from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from osmo360.pipeline import dataset, four_mp4
from osmo360.pipeline.four_mp4_worker import build_chunk_tasks
from osmo360.pipeline.manifest import ManifestError
from tools.merge_fisheye_observation_chunks import merge_chunks
from tools.cache_fisheye_apriltag_observations import should_run_rectified


LEFT_SERIAL = "IAHEA2606M5WSK"
RIGHT_SERIAL = "IAHEA2606KKUKF"
OFFSET = "m2_100_100_100_0_0_90_100_300_100_0_0_90_400_200_1"


def _make_dataset(
    tmp_path: Path, *, descriptor: bool = True, pair_id: str = "pair-01-test"
) -> Path:
    paths = {}
    for side in ("left", "right"):
        for stream in (0, 1):
            path = tmp_path / "raw" / side / f"lens-{stream}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{side}-{stream}".encode())
            paths[(side, stream)] = path
    if descriptor:
        value = {
            "schema_version": four_mp4.INPUT_SCHEMA,
            "pair_id": pair_id,
            "cameras": {
                "left": {
                    "serial": LEFT_SERIAL,
                    "base_tag_id": 2,
                    "x5_offset": OFFSET,
                    "lenses": ["raw/left/lens-0.mp4", "raw/left/lens-1.mp4"],
                },
                "right": {
                    "serial": RIGHT_SERIAL,
                    "base_tag_id": 3,
                    "x5_offset": OFFSET,
                    "lenses": ["raw/right/lens-0.mp4", "raw/right/lens-1.mp4"],
                },
            },
            "sync": {"offset_s": 0.0125},
            "processing": {
                "cache_workers": 1,
                "threads_per_worker": 2,
                "trajectory_observation_fps": 30,
                "cache_chunk_duration_s": 4,
            },
        }
        (tmp_path / "raw/four-mp4.json").write_text(json.dumps(value))
    return tmp_path


def _probe(_path: Path) -> dict[str, object]:
    return {
        "width": 2880,
        "height": 2880,
        "fps": 30.0,
        "duration_s": 10.0,
        "frame_count": 300,
        "has_audio": False,
    }


def _make_instaumi_dataset(
    tmp_path: Path, *, dataset_id: str = "instaumi_test_000001"
) -> Path:
    video = tmp_path / "video"
    video.mkdir()
    for name in (
        "Left_back.mp4",
        "Left_forward.mp4",
        "Right_back.mp4",
        "Right_forward.mp4",
    ):
        (video / name).write_bytes(name.encode())
    metadata = {
        "schema_version": "1.0.0",
        "dataset_id": dataset_id,
        "created_at_utc": "2026-09-01T08:41:03Z",
        "time": {"reference": "dataset_start"},
        "devices": {
            "left": {"serial_number": LEFT_SERIAL},
            "right": {"serial_number": RIGHT_SERIAL},
        },
        "video": {
            "left": {"frame_count": 300},
            "right": {"frame_count": 300},
        },
    }
    calibration = {
        "time_calibration": {
            "right_left_offset_ns": 2_617_875_000,
            "uncertainty_ns": 500_000,
            "method": "audio_cross_correlation",
        },
        "cameras": {
            side: {"intrinsics": {name: None for name in ("fx", "fy", "cx", "cy")}}
            for side in ("left", "right")
        },
        "extrinsics": {
            "T_right_left": np.eye(4, dtype=int).tolist(),
        },
    }
    string = h5py.string_dtype(encoding="utf-8")
    with h5py.File(tmp_path / "dataset.h5", "w") as handle:
        handle.attrs["schema_name"] = "instaumi"
        handle.create_dataset("/metadata/dataset.json", data=json.dumps(metadata), dtype=string)
        handle.create_dataset(
            "/calib/calibration_full.json", data=json.dumps(calibration), dtype=string
        )
        timestamp = np.arange(300, dtype=np.int64) * 33_333_333
        for side, source_start in (("left", 10_000_000_000), ("right", 7_382_125_000)):
            base = f"/sensor/camera/{side}"
            handle.create_dataset(f"{base}/timestamp_ns", data=timestamp)
            handle.create_dataset(f"{base}/source_timestamp_ns", data=timestamp + source_start)
            handle.create_dataset(f"{base}/frame_index", data=np.arange(300, dtype=np.int64))
            handle.create_dataset(f"{base}/valid", data=np.ones(300, dtype=np.bool_))
    return tmp_path


def test_four_mp4_discovery_and_dataset_dry_run(monkeypatch, tmp_path: Path):
    root = _make_dataset(tmp_path)
    monkeypatch.setattr(four_mp4, "_probe_mp4", _probe)
    monkeypatch.setattr(four_mp4, "_embedded_identity", lambda _path: (None, None))

    lock = four_mp4.discover_four_mp4_dataset(root)
    result = dataset.process_dataset(root, dry_run=True)

    assert lock["pipeline_revision"] == "dual-x5-four-mp4-cpu-v3"
    assert lock["pairs"][0]["left"]["lenses"][0]["stream"] == 0
    assert lock["pairs"][0]["right"]["base_tag_id"] == 3
    assert lock["pairs"][0]["sync"]["offset_s"] == 0.0125
    assert lock["resource_budget"]["maximum_active_cpu_threads"] == 2
    assert result["status"] == "DRY_RUN"
    assert "osmo360.pipeline.four_mp4_worker" in result["command"]


def test_instaumi_h5_is_native_four_mp4_input(monkeypatch, tmp_path: Path):
    root = _make_instaumi_dataset(tmp_path)
    monkeypatch.setattr(four_mp4, "_probe_mp4", _probe)
    monkeypatch.setattr(four_mp4, "_embedded_identity", lambda _path: (None, None))

    lock = four_mp4.discover_four_mp4_dataset(root)
    pair = lock["pairs"][0]

    assert four_mp4.is_four_mp4_dataset(root)
    assert lock["input_format"] == "instaumi-four-fisheye-mp4-hdf5/1.0"
    assert pair["pair_id"] == "instaumi_test_000001"
    assert pair["left"]["lenses"][0]["path"] == "video/Left_back.mp4"
    assert pair["left"]["lenses"][1]["path"] == "video/Left_forward.mp4"
    assert pair["left"]["timeline_camera"] == "left"
    assert pair["sync"]["offset_s"] == 0
    assert pair["sync"]["source_right_left_offset_s"] == pytest.approx(2.617875)
    assert lock["resource_budget"]["profile"] == "fast-cpu"
    assert lock["resource_budget"]["trajectory_observation_fps"] == 30
    assert lock["resource_budget"]["decode_fps"] == 30
    assert lock["resource_budget"]["maximum_active_cpu_threads"] == 16
    assert lock["instaumi"]["calibration_intrinsics_complete"] is False
    assert lock["instaumi"]["extrinsics_status"] == "placeholder_identity"

    tasks = build_chunk_tasks(
        root,
        pair,
        tmp_path / "cache",
        {lens["path"]: "a" * 64 for side in ("left", "right") for lens in pair[side]["lenses"]},
        pair["sync"],
        lock["resource_budget"],
    )
    assert len(tasks) == 4
    assert all("--timeline-h5" in task.command for task in tasks)
    assert all("--native-grayscale-decode" in task.command for task in tasks)
    assert all("--optical-flow-window-size" in task.command for task in tasks)
    assert all(task.expected["frame_stride"] == 1 for task in tasks)
    assert all(task.expected["decode_stride"] == 1 for task in tasks)
    assert all(task.expected["timestamp_source"].startswith("instaumi_h5:") for task in tasks)


def test_four_mp4_requires_factory_offset_when_mp4_has_no_embedded_metadata(
    monkeypatch, tmp_path: Path
):
    root = _make_dataset(tmp_path, descriptor=False)
    monkeypatch.setattr(four_mp4, "_probe_mp4", _probe)
    monkeypatch.setattr(four_mp4, "_embedded_identity", lambda _path: (None, None))

    with pytest.raises(ManifestError, match="x5_offset"):
        four_mp4.discover_four_mp4_dataset(root)


@pytest.mark.parametrize("pair_id", ["../outside", "pair/escape", " bad", ""])
def test_four_mp4_rejects_unsafe_pair_id(monkeypatch, tmp_path: Path, pair_id: str):
    root = _make_dataset(tmp_path, pair_id=pair_id)
    monkeypatch.setattr(four_mp4, "_probe_mp4", _probe)
    monkeypatch.setattr(four_mp4, "_embedded_identity", lambda _path: (None, None))

    with pytest.raises(ManifestError, match="pair_id"):
        four_mp4.discover_four_mp4_dataset(root)


def test_instaumi_rejects_unsafe_dataset_id(monkeypatch, tmp_path: Path):
    root = _make_instaumi_dataset(tmp_path, dataset_id="../../outside")
    monkeypatch.setattr(four_mp4, "_probe_mp4", _probe)
    monkeypatch.setattr(four_mp4, "_embedded_identity", lambda _path: (None, None))

    with pytest.raises(ManifestError, match="metadata.dataset_id"):
        four_mp4.discover_four_mp4_dataset(root)


def test_chunk_plan_is_aligned_resumable_and_cpu_bounded(monkeypatch, tmp_path: Path):
    root = _make_dataset(tmp_path)
    monkeypatch.setattr(four_mp4, "_probe_mp4", _probe)
    monkeypatch.setattr(four_mp4, "_embedded_identity", lambda _path: (None, None))
    lock = four_mp4.discover_four_mp4_dataset(root)
    pair = lock["pairs"][0]
    hashes = {
        lens["path"]: str(index) * 64
        for index, lens in enumerate(
            pair["left"]["lenses"] + pair["right"]["lenses"], 1
        )
    }

    tasks = build_chunk_tasks(
        root,
        pair,
        tmp_path / "cache",
        hashes,
        {"offset_s": 0.0125},
        lock["resource_budget"],
    )

    assert len(tasks) == 12
    lane = [task for task in tasks if task.side == "left" and task.stream == 0]
    assert lane[0].start == 0 and lane[0].end == 119
    assert lane[1].start == 120 and lane[1].end == 239
    assert lane[2].start == 240 and lane[2].end == 299
    assert "--stop-after-end-frame" in lane[0].command
    assert "--seek-to-start" not in lane[0].command
    assert "--seek-to-start" in lane[1].command
    assert lane[0].expected["frame_stride"] == 1
    assert len({(task.side, task.stream) for task in tasks[:4]}) == 4
    assert lane[0].expected["rectified_detection_policy"] == "adaptive"
    assert lane[0].expected["temporal_tracking"] is True
    assert "--temporal-tracking" in lane[0].command


def test_adaptive_rectification_fails_closed_when_a_critical_tag_is_missing():
    assert not should_run_rectified(
        "adaptive", {2, 3, 200, 201}, minimum_direct_tags=4, required_ids={2, 3}
    )
    assert should_run_rectified(
        "adaptive", {2, 200, 201, 202}, minimum_direct_tags=4, required_ids={2, 3}
    )
    assert should_run_rectified(
        "adaptive", {2, 3, 200}, minimum_direct_tags=4, required_ids={2, 3}
    )
    assert should_run_rectified(
        "always", {2, 3, 200, 201}, minimum_direct_tags=4, required_ids={2, 3}
    )


def _write_chunk(path: Path, start: int, end: int) -> None:
    frames = np.arange(start, end + 1, dtype=np.int32)
    detection_frames = np.asarray([start], dtype=np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        timeline_frame_index=frames,
        timeline_local_time_s=frames.astype(np.float64) / 30,
        timeline_common_time_s=frames.astype(np.float64) / 30,
        frame_index=detection_frames,
        local_time_s=detection_frames.astype(np.float64) / 30,
        common_time_s=detection_frames.astype(np.float64) / 30,
        tag_id=np.asarray([200], dtype=np.int32),
        corners_px=np.zeros((1, 4, 2), dtype=np.float32),
        rays_camera=np.zeros((1, 4, 3), dtype=np.float32),
        area_px2=np.ones(1, dtype=np.float32),
        center_px=np.zeros((1, 2), dtype=np.float32),
        detection_source=np.asarray(["direct_raw"], dtype="U24"),
    )
    metadata = {
        "schema_version": "fisheye-apriltag-observation-cache/1.0",
        "video": "/data/lens-0.mp4",
        "video_sha256": "a" * 64,
        "calibration": "embedded_x5_offset",
        "calibration_sha256": "b" * 64,
        "x5_offset": OFFSET,
        "camera_serial": LEFT_SERIAL,
        "stream": 0,
        "source_size": [2880, 2880],
        "fps": 30.0,
        "frame_count": 4,
        "clock_mapping": {
            "formula": "local_time = intercept_s + slope * common_time",
            "intercept_s": 0.0,
            "slope": 1.0,
        },
        "radial_model": "x5-offset-equidistant",
        "rectified_detection": True,
        "rectified_detection_policy": "adaptive",
        "rectified_min_direct_tags": 4,
        "rectified_required_ids": [2, 3],
        "rectified_view_size": 720,
        "rectification_radial_model": "stitch",
        "frame_stride": 2,
        "corner_order": "opencv_aruco_apriltag_canonical",
        "ray_frame": "x5_dual_fisheye_rig_stream0",
        "decoded_frame_range": [start, end],
    }
    path.with_suffix(".json").write_text(json.dumps(metadata))


def test_chunk_merge_reconstructs_one_monotonic_cache(tmp_path: Path):
    first = tmp_path / "chunk-0.npz"
    second = tmp_path / "chunk-1.npz"
    output = tmp_path / "lens.npz"
    _write_chunk(first, 0, 1)
    _write_chunk(second, 2, 3)

    report = merge_chunks([second, first], output)

    with np.load(output) as cache:
        assert cache["timeline_frame_index"].tolist() == [0, 1, 2, 3]
        assert cache["frame_index"].tolist() == [0, 2]
    assert report["chunk_count"] == 2
    assert report["decoded_frame_count"] == 4


def test_temporal_chunk_merge_aggregates_flow_audit(tmp_path: Path):
    first = tmp_path / "temporal-0.npz"
    second = tmp_path / "temporal-1.npz"
    output = tmp_path / "temporal-lens.npz"
    _write_chunk(first, 0, 1)
    _write_chunk(second, 2, 3)
    signature = {
        "temporal_tracking": True,
        "output_stride_frames": 2,
    }
    for path, accepted in ((first, 11), (second, 13)):
        metadata_path = path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text())
        metadata.update({
            "schema_version": "fisheye-apriltag-observation-cache/1.3-temporal",
            "temporal_tracking": True,
            "processing_signature": signature,
            "tracking": {
                "method": "pyramidal LK forward/backward on raw fisheye pixels",
                "integrated_one_pass": True,
                "pose_interpolation_used": False,
                "output_stride_frames": 2,
                "flow_attempted_tag_count": accepted + 1,
                "flow_accepted_tag_count": accepted,
            },
        })
        metadata_path.write_text(json.dumps(metadata))

    report = merge_chunks([first, second], output)

    assert report["schema_version"].endswith("1.3-temporal")
    assert report["tracking"]["flow_attempted_tag_count"] == 26
    assert report["tracking"]["flow_accepted_tag_count"] == 24
    assert report["processing_signature"] == signature
