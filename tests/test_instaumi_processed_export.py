from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest

from osmo360.datasets import instaumi_processed_export as export
from osmo360.pipeline.manifest import ManifestError


def _touch_inputs(root: Path) -> None:
    (root / "video").mkdir(parents=True)
    (root / "dataset.h5").write_bytes(b"h5")
    for side in ("Left", "Right"):
        for lens in ("back", "forward"):
            (root / "video" / f"{side}_{lens}.mp4").write_bytes(b"mp4")


def test_signal_profile_is_hash_and_serial_bound() -> None:
    payload, profiles = export.load_profile()

    assert payload["revision_id"] == "instaumi-pair01-gripper-signal-20260902-r1"
    assert profiles["left"].camera_serial == "IAHEA2606M5WSK"
    assert profiles["right"].camera_serial == "IAHEA2606KKUKF"
    assert profiles["left"].base_tag_id == 2
    assert profiles["right"].base_tag_id == 3
    assert profiles["left"].closed_reference_deg > 0
    assert profiles["right"].width_m[-1] > profiles["right"].width_m[0]


def test_nearest_indices_returns_error_without_extrapolation_failure() -> None:
    index, error = export._nearest_indices(
        np.asarray([0.0, 0.1, 0.2]), np.asarray([0.049, 0.151, 0.4])
    )

    assert index.tolist() == [0, 2, 2]
    assert error.tolist() == pytest.approx([0.049, 0.049, 0.2])


def test_h5_preview_missing_falls_back_to_required_back_videos(tmp_path: Path) -> None:
    _touch_inputs(tmp_path)
    metadata = {
        "video": {
            "left": {"path": "video/Left.mp4", "sha256": "0" * 64},
            "right": {"path": "video/Right.mp4", "sha256": "0" * 64},
        }
    }
    with h5py.File(tmp_path / "dataset.h5", "w") as handle:
        handle.require_group("metadata").create_dataset(
            "dataset.json", data=json.dumps(metadata)
        )
        camera = handle.require_group("sensor/camera")
        for side in ("left", "right"):
            group = camera.require_group(side)
            group.create_dataset("timestamp_ns", data=np.asarray([0, 100_000_000]))
            group.create_dataset("video_path", data=f"video/{side.title()}.mp4")

    inputs = export.load_side_inputs(
        tmp_path, {"prefer_h5_preview": True, "input_lens": "back"}
    )

    assert inputs["left"].video == (tmp_path / "video/Left_back.mp4").resolve()
    assert inputs["right"].video == (tmp_path / "video/Right_back.mp4").resolve()
    assert inputs["left"].video_kind == "four_mp4_back_fallback"
    assert inputs["right"].timestamp_s.tolist() == [0.0, 0.1]


def test_export_writes_synchronized_csv_revision_without_removing_existing_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _touch_inputs(tmp_path)
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "time_alignment.csv").write_text("existing\n", encoding="utf-8")
    source_trajectory = tmp_path / "source-trajectory.csv"
    trajectory_fields = [
        "frame",
        "timestamp_s",
        "world_frame",
        "joint_has_pose",
        "left_camera_x_m",
        "right_camera_x_m",
    ]
    trajectory_rows = [
        {
            "frame": str(index),
            "timestamp_s": f"{index * 0.1:.1f}",
            "world_frame": "session_grid_A",
            "joint_has_pose": "1",
            "left_camera_x_m": f"{index + 1}.0",
            "right_camera_x_m": f"{index + 4}.0",
        }
        for index in range(3)
    ]
    with source_trajectory.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=trajectory_fields)
        writer.writeheader()
        writer.writerows(trajectory_rows)

    monkeypatch.setattr(
        export,
        "load_instaumi_config",
        lambda _root: {
            "pair_id": "instaumi_example_000001",
            "cameras": {
                "left": {"serial": "IAHEA2606M5WSK"},
                "right": {"serial": "IAHEA2606KKUKF"},
            },
        },
    )
    monkeypatch.setattr(
        export,
        "_read_trajectory",
        lambda _root, _pair: (
            source_trajectory,
            trajectory_fields,
            trajectory_rows,
            {"status": "SELF_CALIBRATED_PASS"},
        ),
    )
    side_inputs = {
        side: export.SideInput(
            side=side,
            video=tmp_path / "video" / f"{side.title()}_back.mp4",
            video_kind="test_preview",
            timestamp_s=np.asarray([0.0, 0.1, 0.2]),
        )
        for side in ("left", "right")
    }
    monkeypatch.setattr(export, "load_side_inputs", lambda _root, _profile: side_inputs)

    def fake_analyze(source: export.SideInput, _profile, **_kwargs) -> export.SideSignal:
        if source.side == "left":
            opening = np.asarray([1.0, 2.0, 3.0])
            width = np.asarray([0.001, 0.002, 0.003])
            state = np.asarray(["MEASURED", "MEASURED", "RECOVERED_SHORT_GAP"], dtype=object)
        else:
            opening = np.asarray([4.0, np.nan, 6.0])
            width = np.asarray([0.004, np.nan, 0.006])
            state = np.asarray(["MEASURED", "UNAVAILABLE", "MEASURED"], dtype=object)
        return export.SideSignal(
            opening_deg=opening,
            width_m=width,
            state=state,
            timestamp_s=source.timestamp_s,
            source_frame=np.arange(3),
            measured_ratio=2 / 3,
            available_ratio=float(np.isfinite(opening).mean()),
        )

    monkeypatch.setattr(export, "_analyze_side", fake_analyze)

    result = export.export_processed_dataset(tmp_path)

    output = processed / "instaumi-csv-v1"
    assert result["status"] == "COMPLETE"
    assert (processed / "time_alignment.csv").read_text(encoding="utf-8") == "existing\n"
    assert sorted(path.name for path in output.iterdir()) == [
        "gripper.csv",
        "metadata.csv",
        "processed.csv",
        "trajectory.csv",
    ]
    with (output / "gripper.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert rows[0]["left_opening_angle_deg"] == "1.000000"
    assert rows[1]["right_opening_angle_deg"] == ""
    assert rows[1]["right_opening_available"] == "0"
    assert rows[2]["left_opening_measured"] == "0"
    with (output / "processed.csv").open(newline="", encoding="utf-8") as handle:
        combined = list(csv.DictReader(handle))
    assert combined[2]["left_camera_x_m"] == "3.0"
    assert combined[2]["right_opening_width_m"] == "0.006000000"
    assert not list(processed.glob(".instaumi-csv-v1-*"))


def test_export_rejects_processed_symlink(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _touch_inputs(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "processed").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ManifestError, match="symlink"):
        export.export_processed_dataset(tmp_path)


def test_shell_entry_rejects_incomplete_dataset(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "bin/process_instaumi_dataset.sh"
    process = subprocess.run([str(script), str(tmp_path)], capture_output=True, text=True)

    assert process.returncode == 2
    assert "Missing required InstaUMI input" in process.stderr
