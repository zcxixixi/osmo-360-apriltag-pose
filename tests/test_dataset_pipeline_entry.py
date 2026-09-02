from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from osmo360.pipeline import dataset
from osmo360.pipeline.dataset_worker import (
    cache_signature_matches,
    estimate_audio_offset,
    trajectory_sample_stride,
)
from osmo360.pipeline.insta360_telemetry import ImuSamples
from osmo360.pipeline.instaumi_format import (
    common_window,
    packet_timeline,
    write_dataset_h5,
)
from osmo360.pipeline.manifest import ManifestError


LEFT_SERIAL = "IAHEA2606M5WSK"
RIGHT_SERIAL = "IAHEA2606KKUKF"
OFFSET = "m2_100_100_100_0_0_90_100_300_100_0_0_90_400_200_1"


def fake_insv(path: Path, serial: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"header {serial} {OFFSET} footer".encode())


def probe(duration: float = 120.0) -> dict[str, float | int]:
    return {
        "width": 2880,
        "height": 2880,
        "fps": 29.97,
        "duration_s": duration,
    }


def test_dataset_path_discovers_registered_left_right_pair(monkeypatch, tmp_path: Path):
    fake_insv(tmp_path / "raw/left/VID_20260901_120000_00_001.insv", LEFT_SERIAL)
    fake_insv(tmp_path / "raw/right/VID_20260901_120002_00_001.insv", RIGHT_SERIAL)
    monkeypatch.setattr(dataset, "_probe", lambda _path: probe())

    lock = dataset.discover_dataset(tmp_path)

    assert lock["pair_count"] == 1
    assert lock["pairs"][0]["pair_id"] == "pair-01-120002"
    assert lock["pairs"][0]["dataset_id"] == "instaumi_000001"
    assert lock["pairs"][0]["left"]["serial"] == LEFT_SERIAL
    assert lock["pairs"][0]["right"]["serial"] == RIGHT_SERIAL
    assert lock["pairs"][0]["left"]["path"].startswith("raw/left/")
    assert lock["ignored_short_recordings"] == []


def test_dataset_rejects_serial_in_wrong_side_directory(monkeypatch, tmp_path: Path):
    fake_insv(tmp_path / "raw/left/VID_20260901_120000_00_001.insv", RIGHT_SERIAL)
    (tmp_path / "raw/right").mkdir(parents=True)
    monkeypatch.setattr(dataset, "_probe", lambda _path: probe())

    with pytest.raises(ManifestError, match="under raw/left"):
        dataset.discover_dataset(tmp_path)


def test_short_recordings_are_ignored_before_pairing(monkeypatch, tmp_path: Path):
    fake_insv(tmp_path / "raw/left/VID_20260901_115900_00_000.insv", LEFT_SERIAL)
    fake_insv(tmp_path / "raw/left/VID_20260901_120000_00_001.insv", LEFT_SERIAL)
    fake_insv(tmp_path / "raw/right/VID_20260901_120002_00_001.insv", RIGHT_SERIAL)
    monkeypatch.setattr(
        dataset,
        "_probe",
        lambda path: probe(5.0 if "115900" in path.name else 120.0),
    )

    lock = dataset.discover_dataset(tmp_path)

    assert lock["pair_count"] == 1
    assert len(lock["ignored_short_recordings"]) == 1


def test_audio_sync_reports_right_time_from_left_offset(tmp_path: Path):
    rate = 1000
    left = np.zeros(2000, dtype=np.int16)
    right = np.zeros(2000, dtype=np.int16)
    left[500:510] = 30000
    right[520:530] = 30000
    left_path = tmp_path / "left.wav"; right_path = tmp_path / "right.wav"
    wavfile.write(left_path, rate, left); wavfile.write(right_path, rate, right)

    result = estimate_audio_offset(left_path, right_path, 0.0)

    assert result["mapping"] == "right_time_s = left_time_s + offset_s"
    assert result["offset_s"] == pytest.approx(0.020, abs=0.001)


def test_common_window_trims_the_leading_side() -> None:
    assert common_window(10.0, 12.0, 0.25) == pytest.approx((0.0, 0.25, 10.0))
    assert common_window(10.0, 12.0, -0.25) == pytest.approx((0.25, 0.0, 9.75))


def test_packet_timeline_sorts_hevc_decode_order_by_pts(monkeypatch) -> None:
    output = "\n".join((
        "0,0.000000,K__",
        "2048,0.133333,___",
        "1024,0.066667,___",
        "512,0.033333,___",
        "1536,0.100000,___",
    ))
    monkeypatch.setattr(
        "osmo360.pipeline.instaumi_format.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, output, ""),
    )

    timeline = packet_timeline(Path("ffprobe"), Path("aligned.mp4"), 1.25)

    assert timeline["frame_index"].tolist() == [0, 1, 2, 3, 4]
    assert timeline["pts"].tolist() == [0, 512, 1024, 1536, 2048]
    assert timeline["timestamp_ns"].tolist() == [
        0, 33_333_000, 66_667_000, 100_000_000, 133_333_000,
    ]
    assert timeline["source_timestamp_ns"].tolist() == [
        1_250_000_000, 1_283_333_000, 1_316_667_000,
        1_350_000_000, 1_383_333_000,
    ]
    assert timeline["keyframe"].tolist() == [1, 0, 0, 0, 0]


def test_packet_timeline_rejects_duplicate_pts(monkeypatch) -> None:
    output = "0,0.000000,K__\n0,0.000000,___"
    monkeypatch.setattr(
        "osmo360.pipeline.instaumi_format.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, output, ""),
    )

    with pytest.raises(ManifestError, match="duplicate or invalid PTS"):
        packet_timeline(Path("ffprobe"), Path("aligned.mp4"), 0.0)


def test_trajectory_sampling_targets_thirty_hz_without_upsampling() -> None:
    assert trajectory_sample_stride(60.0, 59.94) == 2
    assert trajectory_sample_stride(30.0, 29.97) == 1
    assert trajectory_sample_stride(15.0, 15.0) == 1


def test_fisheye_cache_reuse_requires_complete_matching_signature(tmp_path: Path) -> None:
    video = tmp_path / "Left_back.mp4"
    video.write_bytes(b"aligned-hevc")
    cache = tmp_path / "lens-0-corners.npz"
    cache.write_bytes(b"cache")
    record = {
        "serial": LEFT_SERIAL,
        "fps": 30.0,
        "x5_offset": OFFSET,
    }
    metadata = {
        "schema_version": "fisheye-apriltag-observation-cache/1.0",
        "producer_revision": "raw-fisheye-cache-v2",
        "video": str(video.resolve()),
        "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "camera_serial": LEFT_SERIAL,
        "stream": 0,
        "source_size": [1920, 1920],
        "x5_offset": OFFSET,
        "rectified_detection": True,
        "rectified_view_size": 720,
        "frame_stride": 1,
        "clock_mapping": {"intercept_s": 0.0, "slope": 1.0},
    }
    cache.with_suffix(".json").write_text(json.dumps(metadata))

    assert cache_signature_matches(video, record, 0, 0.0, cache)
    video.write_bytes(b"changed-after-cache")
    assert not cache_signature_matches(video, record, 0, 0.0, cache)
    assert cache_signature_matches(
        video, record, 0, 0.0, cache, verify_video_hash=False,
    )
    video.write_bytes(b"aligned-hevc")

    metadata["rectified_view_size"] = 960
    cache.with_suffix(".json").write_text(json.dumps(metadata))
    assert not cache_signature_matches(video, record, 0, 0.0, cache)


def test_instaumi_h5_references_aligned_mp4(monkeypatch, tmp_path: Path) -> None:
    import h5py
    import osmo360.pipeline.instaumi_format as fmt

    video = tmp_path / "video"; video.mkdir()
    left = video / "Left.mp4"; right = video / "Right.mp4"
    left.write_bytes(b"left"); right.write_bytes(b"right")
    monkeypatch.setattr(fmt, "probe_mp4", lambda _ffprobe, path: {
        "width": 1024, "height": 1024, "frame_rate_num": 30,
        "frame_rate_den": 1, "time_base_num": 1, "time_base_den": 90000,
        "frame_count": 2, "duration_ns": 66_666_667, "codec": "h264",
        "pixel_format": "yuv420p", "bitrate_bps": 1000,
        "sha256": path.name,
    })
    monkeypatch.setattr(fmt, "packet_timeline", lambda *_args: {
        "frame_index": np.asarray([0, 1], dtype=np.uint64),
        "timestamp_ns": np.asarray([0, 33_333_333], dtype=np.int64),
        "source_timestamp_ns": np.asarray([0, 33_333_333], dtype=np.int64),
        "pts": np.asarray([0, 3000], dtype=np.int64),
        "keyframe": np.asarray([1, 0], dtype=np.uint8),
        "valid": np.asarray([1, 1], dtype=np.uint8),
    })
    source = {
        "serial": LEFT_SERIAL,
        "lens_size": [1920, 1920],
        "x5_offset": (
            "n2_2663.778_2695.450_2691.260_-0.331_0.192_89.482_"
            "2653.083_8069.790_2689.460_0.386_0.193_90.228_10752_5376_11378"
        ),
    }
    imu_left = ImuSamples(
        timestamp_ns=np.asarray([0, 1_000_000], dtype=np.int64),
        source_timestamp_ns=np.asarray([10_000_000, 11_000_000], dtype=np.int64),
        angular_velocity=np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        linear_acceleration=np.asarray([[0.0, 0.0, 9.81], [0.1, 0.0, 9.80]]),
        valid=np.ones(2, dtype=np.uint8),
        provenance={"firmware_version": "v1.7.8", "camera_serial": LEFT_SERIAL},
    )
    imu_right = ImuSamples(
        timestamp_ns=np.asarray([0, 1_000_000, 2_000_000], dtype=np.int64),
        source_timestamp_ns=np.asarray([20_000_000, 21_000_000, 22_000_000], dtype=np.int64),
        angular_velocity=np.asarray([[0.7, 0.8, 0.9]] * 3),
        linear_acceleration=np.asarray([[0.0, 9.81, 0.0]] * 3),
        valid=np.ones(3, dtype=np.uint8),
        provenance={"firmware_version": "v1.7.8", "camera_serial": RIGHT_SERIAL},
    )
    output = tmp_path / "dataset.h5"
    write_dataset_h5(
        output, dataset_id="instaumi_000001", left_video=left, right_video=right,
        left_source=tmp_path / "left.insv", right_source=tmp_path / "right.insv",
        left_start_s=0, right_start_s=.02,
        sync={"offset_s": .02, "uncertainty_s": .0005}, ffprobe=Path("ffprobe"),
        source_records={"left": source, "right": {**source, "serial": RIGHT_SERIAL}},
        imu={"left": imu_left, "right": imu_right},
    )
    with h5py.File(output) as handle:
        assert handle.attrs["schema_name"] == "instaumi"
        assert handle["sensor/camera/left/video_path"][()].decode() == "video/Left.mp4"
        assert handle["sensor/camera/right/frame_index"].shape == (2,)
        assert handle["sensor/imu/left/angular_velocity"].shape == (2, 3)
        assert handle["sensor/imu/right/angular_velocity"].shape == (3, 3)
        assert handle["sensor/imu/left/source_timestamp_ns"][:].tolist() == [10_000_000, 11_000_000]
        assert handle["sensor/imu/right/source_timestamp_ns"][:].tolist() == [20_000_000, 21_000_000, 22_000_000]
        assert handle["sensor/imu/left/linear_acceleration"][0, 2] == pytest.approx(9.81)
        assert handle["sensor/imu/right/linear_acceleration"][0, 1] == pytest.approx(9.81)
        calibration = json.loads(handle["calib/calibration_full.json"][()].decode())
        intrinsics = calibration["cameras"]["left"]["intrinsics"]
        assert intrinsics["fx"] == pytest.approx(328.3, abs=0.2)
        assert intrinsics["fy"] == intrinsics["fx"]
        assert intrinsics["cx"] == pytest.approx(507.4, abs=0.1)
        assert intrinsics["cy"] == pytest.approx(513.4, abs=0.1)
        assert calibration["cameras"]["left"]["distortion"]["coefficients"] == [0.0] * 4
        assert calibration["extrinsics"]["T_rig_camera_left"] != np.eye(4).tolist()
        metadata = json.loads(handle["metadata/dataset.json"][()].decode())
        dataset_start = metadata["time"]["dataset_start_source_timestamp_ns"]
        for kind in ("camera", "imu"):
            for side in ("left", "right"):
                group = handle[f"sensor/{kind}/{side}"]
                offset = calibration["time_calibration"][f"{side}_{kind}_offset_ns"]
                assert np.array_equal(
                    group["timestamp_ns"][:],
                    group["source_timestamp_ns"][:] + offset - dataset_start,
                )


def test_run_pipeline_shell_requires_only_dataset_path(tmp_path: Path):
    script = Path(__file__).parents[1] / "run_pipeline.sh"
    assert os.access(script, os.X_OK)
    process = subprocess.run([str(script), str(tmp_path)], capture_output=True, text=True)
    assert process.returncode == 2
    assert "raw/left/ and raw/right/" in process.stderr
