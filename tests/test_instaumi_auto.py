from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pytest

import h5py
from osmo360.pipeline import instaumi_auto as auto


def write_raw(root: Path, side: str, name: str, payload: bytes = b"insv") -> Path:
    path = root / "raw" / side / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def write_sha(root: Path, entries: list[tuple[str, Path]]) -> None:
    lines = [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative}" for relative, path in entries]
    (root / "raw/sha256.txt").write_text("\n".join(lines) + "\n")


def test_discovery_requires_sha_entries_for_both_sources(tmp_path: Path) -> None:
    collector = tmp_path / "0901_instaumi_sort_blocks_qsb"
    left = write_raw(collector, "left", "VID_20260901_120000_00_001.insv")
    write_raw(collector, "right", "VID_20260901_120002_00_001.insv")
    write_sha(collector, [(f"left/{left.name}", left)])

    assert auto.discover_pairs(tmp_path) == []


def test_discovery_ignores_non_timestamped_insv_aliases(tmp_path: Path) -> None:
    collector = tmp_path / "0831_instaumi_sort_blocks_lyw"
    left = write_raw(collector, "left", "left.insv")
    right = write_raw(collector, "right", "right.insv")
    write_sha(collector, [(f"left/{left.name}", left), (f"right/{right.name}", right)])

    assert auto.discover_pairs(tmp_path) == []


def test_approved_mapping_pairs_large_start_delta(tmp_path: Path) -> None:
    collector = tmp_path / "0901_instaumi_sort_blocks_sc"
    left = write_raw(collector, "left", "VID_20260901_171010_00_013.insv")
    right = write_raw(collector, "right", "VID_20260901_171049_00_016.insv")
    write_sha(collector, [(f"left/{left.name}", left), (f"right/{right.name}", right)])
    review = tmp_path / "_review/0901_sort_blocks/alignment-review-v1"
    review.mkdir(parents=True)
    (review / "collector_video_mapping.json").write_text(json.dumps({
        "items": [{
            "collector": "sc",
            "usable": True,
            "left_source": left.name,
            "right_source": right.name,
        }],
    }))

    pairs = auto.discover_pairs(tmp_path)

    assert len(pairs) == 1
    assert pairs[0].left.path == left
    assert pairs[0].right.path == right


def test_unreviewed_mapping_is_not_fallback_paired(tmp_path: Path) -> None:
    collector = tmp_path / "0831_instaumi_sort_blocks_lyw"
    left = write_raw(collector, "left", "VID_20260831_163609_00_008.insv")
    right = write_raw(collector, "right", "VID_20260831_163605_00_007.insv")
    write_sha(collector, [(f"left/{left.name}", left), (f"right/{right.name}", right)])
    review = tmp_path / "_review/0901_sort_blocks/alignment-review-v1"
    review.mkdir(parents=True)
    (review / "collector_video_mapping.json").write_text(json.dumps({
        "items": [{
            "collector": "lyw",
            "usable": None,
            "left_source": left.name,
            "right_source": right.name,
        }],
    }))

    assert auto.discover_pairs(tmp_path) == []


def test_source_sha_is_verified_once_until_file_changes(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "raw/left/VID_20260901_120000_00_001.insv"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"source")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    source = auto.Source("left", path, f"left/{path.name}", digest, datetime(2026, 9, 1, 12))
    state = {}
    calls = 0
    real_sha256 = auto._sha256

    def counted(candidate: Path) -> str:
        nonlocal calls
        calls += 1
        return real_sha256(candidate)

    monkeypatch.setattr(auto, "_sha256", counted)
    auto._verify_source(source, state)
    auto._verify_source(source, state)

    assert calls == 1
    path.write_bytes(b"changed")
    with pytest.raises(auto.PipelineFailure, match="SHA-256 mismatch"):
        auto._verify_source(source, state)


def test_encoder_forces_common_frame_rate_and_count(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.insv"
    output = tmp_path / "Left_back.mp4"
    source.write_bytes(b"source")
    captured = []

    def fake_run(command: list[str], _log: Path) -> None:
        captured.append(command)
        Path(command[-1]).write_bytes(b"video")

    monkeypatch.setattr(auto, "_run", fake_run)
    auto._encode_lens(
        source,
        output,
        stream=0,
        start_s=1.25,
        frame_count=3039,
        size=1920,
        log=tmp_path / "encode.log",
    )

    assert output.read_bytes() == b"video"
    command = captured[0]
    assert f"fps={auto.TARGET_FPS},scale=1920:1920:flags=lanczos" in command
    assert command[command.index("-frames:v") + 1] == "3039"
    assert command[command.index("-progress") + 1] == "pipe:1"


def test_encoder_supports_a6000_nvenc(monkeypatch) -> None:
    monkeypatch.setattr(auto, "VIDEO_ENCODER", "hevc_nvenc")

    arguments = auto._video_encoder_args()

    assert arguments[arguments.index("-c:v") + 1] == "hevc_nvenc"
    assert arguments[arguments.index("-preset") + 1] == "p2"


def test_two_shards_are_disjoint_and_cover_every_pair(tmp_path: Path) -> None:
    collector = tmp_path / "0901_instaumi_sort_blocks_qsb"
    entries = []
    for index in range(6):
        left = write_raw(
            collector,
            "left",
            f"VID_20260901_12{index:02d}00_00_{index:03d}.insv",
        )
        right = write_raw(
            collector,
            "right",
            f"VID_20260901_12{index:02d}01_00_{index:03d}.insv",
        )
        entries.extend(((f"left/{left.name}", left), (f"right/{right.name}", right)))
    write_sha(collector, entries)
    script = tmp_path / "process.sh"
    script.write_text("#!/bin/sh\n")
    all_pairs = {pair.key for pair in auto.discover_pairs(tmp_path)}

    shard_results = [
        auto.scan_once(
            tmp_path,
            script,
            max_pairs=10,
            shard_count=2,
            shard_index=index,
            dry_run=True,
        )
        for index in (0, 1)
    ]
    shard_pairs = [
        {item["pair"] for item in result["processed"]}
        for result in shard_results
    ]

    assert shard_pairs[0].isdisjoint(shard_pairs[1])
    assert shard_pairs[0] | shard_pairs[1] == all_pairs
    assert (tmp_path / "_automation/state-0-of-2.json").is_file()
    assert (tmp_path / "_automation/state-1-of-2.json").is_file()


def test_full_export_requires_matching_gripper_profile(
    tmp_path: Path,
    monkeypatch,
) -> None:
    episode = tmp_path / "instaumi_20260901_120000"
    video = episode / "video"
    processed = episode / "processed"
    video.mkdir(parents=True)
    processed.mkdir()
    for name in (
        "Left.mp4",
        "Right.mp4",
        "Left_back.mp4",
        "Left_forward.mp4",
        "Right_back.mp4",
        "Right_forward.mp4",
    ):
        (video / name).write_bytes(b"video")
    (processed / "time_alignment.csv").write_text("header\\n")
    string = h5py.string_dtype(encoding="utf-8")
    with h5py.File(episode / "dataset.h5", "w") as handle:
        handle.create_dataset(
            "metadata/dataset.json",
            data=json.dumps({
                "devices": {
                    "left": {"serial_number": "LEFT"},
                    "right": {"serial_number": "RIGHT"},
                },
            }),
            dtype=string,
        )
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({
        "sides": {
            "left": {"camera_serial": "LEFT"},
            "right": {"camera_serial": "RIGHT"},
        },
    }))
    monkeypatch.setattr(auto, "GRIPPER_PROFILE", profile)

    assert auto._full_export_available(episode)
    profile.write_text(json.dumps({
        "sides": {
            "left": {"camera_serial": "OTHER"},
            "right": {"camera_serial": "RIGHT"},
        },
    }))
    assert not auto._full_export_available(episode)


def test_trajectory_only_completion_is_explicit(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "trajectory.csv").write_text("trajectory")
    auto._demo_path(tmp_path).write_bytes(b"demo")
    auto._atomic_json(processed / "automation_status.json", {
        "status": "COMPLETE",
        "mode": "trajectory_only",
    })

    assert auto._process_complete(tmp_path)


def test_dry_run_skips_complete_episode(tmp_path: Path) -> None:
    collector = tmp_path / "0901_instaumi_sort_blocks_qsb"
    left = write_raw(collector, "left", "VID_20260901_120000_00_001.insv")
    right = write_raw(collector, "right", "VID_20260901_120002_00_001.insv")
    write_sha(collector, [(f"left/{left.name}", left), (f"right/{right.name}", right)])
    episode = collector / "instaumi_20260901_120000"
    processed = episode / "processed"
    processed.mkdir(parents=True)
    for name in ("trajectory.csv", "gripper.csv", "processed.csv", "metadata.csv"):
        (processed / name).write_text(name)
    auto._demo_path(episode).write_bytes(b"demo")
    script = tmp_path / "process.sh"
    script.write_text("#!/bin/sh\n")

    result = auto.scan_once(tmp_path, script, dry_run=True)

    assert result["processed"] == []
    state = json.loads((tmp_path / "_automation/state.json").read_text())
    assert next(iter(state["pairs"].values()))["status"] == "COMPLETE"


def test_dry_run_reports_format_then_process(tmp_path: Path) -> None:
    collector = tmp_path / "0901_instaumi_sort_blocks_qsb"
    left = write_raw(collector, "left", "VID_20260901_120000_00_001.insv")
    right = write_raw(collector, "right", "VID_20260901_120002_00_001.insv")
    write_sha(collector, [(f"left/{left.name}", left), (f"right/{right.name}", right)])
    script = tmp_path / "process.sh"
    script.write_text("#!/bin/sh\n")

    result = auto.scan_once(tmp_path, script, dry_run=True)

    assert result["processed"][0]["action"] == "format_then_process"
