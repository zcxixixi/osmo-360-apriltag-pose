from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest

from osmo360.datasets import instaumi_processed_export as export
from osmo360.datasets.instaumi_progress import ProgressSnapshot, _render, progress_snapshot
from osmo360.pipeline.manifest import ManifestError


def _touch_inputs(root: Path) -> None:
    (root / "video").mkdir(parents=True)
    (root / "dataset.h5").write_bytes(b"h5")
    for side in ("Left", "Right"):
        for lens in ("back", "forward"):
            (root / "video" / f"{side}_{lens}.mp4").write_bytes(b"mp4")


def test_signal_profile_is_hash_and_serial_bound() -> None:
    payload, profiles = export.load_profile()

    assert payload["revision_id"] == "instaumi-pair01-gripper-signal-20260902-r3"
    assert payload["prefer_h5_preview"] is False
    assert payload["prefer_fused_trajectory_marker_cache"] is True
    assert payload["processing_width_px"] == 1920
    assert profiles["left"].camera_serial == "IAHEA2606M5WSK"
    assert profiles["right"].camera_serial == "IAHEA2606KKUKF"
    assert profiles["left"].base_tag_id == 2
    assert profiles["right"].base_tag_id == 3
    assert profiles["right"].included_angle_range == (0.0, 80.0)
    assert profiles["right"].dot_selection == "adaptive-black-pad"
    assert profiles["left"].closed_reference_deg > 0
    assert profiles["right"].width_m[-1] > profiles["right"].width_m[0]


def test_nearest_indices_returns_error_without_extrapolation_failure() -> None:
    index, error = export._nearest_indices(
        np.asarray([0.0, 0.1, 0.2]), np.asarray([0.049, 0.151, 0.4])
    )

    assert index.tolist() == [0, 2, 2]
    assert error.tolist() == pytest.approx([0.049, 0.049, 0.2])


def test_cached_yuv_markers_avoid_a_second_video_decode(tmp_path: Path) -> None:
    _, profiles = export.load_profile()
    cache = tmp_path / "right-lens-0.npz"
    left = np.asarray([(800, 1200), (750, 1350), (700, 1500)], dtype=np.float32)
    right = np.asarray([(1120, 1200), (1170, 1350), (1220, 1500)], dtype=np.float32)
    angle = export.included_jaw_angle(left, right)
    np.savez_compressed(
        cache,
        gripper_frame_index=np.arange(30, dtype=np.int32),
        gripper_left_points_px=np.repeat(left[None], 30, axis=0),
        gripper_right_points_px=np.repeat(right[None], 30, axis=0),
        gripper_included_angle_deg=np.full(30, angle, dtype=np.float32),
    )
    source = export.SideInput(
        side="right",
        video=tmp_path / "intentionally-missing.mp4",
        video_kind="fused_trajectory_yuv420_roi_cache",
        timestamp_s=np.arange(30, dtype=np.float64) / 30,
        marker_cache=cache,
    )

    signal = export._analyze_side(
        source,
        profiles["right"],
        processing_width=1920,
        maximum_gap_s=0.25,
    )

    assert signal.source_frame.tolist() == list(range(30))
    assert signal.measured_ratio == 1.0
    assert np.isfinite(signal.opening_deg).all()


def test_h5_preview_missing_falls_back_to_required_back_videos(tmp_path: Path) -> None:
    _touch_inputs(tmp_path)
    metadata = {
        "dataset_id": "instaumi_test_000001",
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


def test_profile_can_force_full_resolution_back_videos(tmp_path: Path) -> None:
    _touch_inputs(tmp_path)
    for side in ("Left", "Right"):
        (tmp_path / "video" / f"{side}.mp4").write_bytes(b"preview")
    metadata = {
        "dataset_id": "instaumi_test_000001",
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
        tmp_path, {"prefer_h5_preview": False, "input_lens": "back"}
    )

    assert inputs["left"].video == (tmp_path / "video/Left_back.mp4").resolve()
    assert inputs["right"].video == (tmp_path / "video/Right_back.mp4").resolve()
    assert inputs["left"].video_kind == "four_mp4_back_fallback"


def test_export_writes_synchronized_csv_revision_without_removing_existing_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _touch_inputs(tmp_path)
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "time_alignment.csv").write_text("existing\n", encoding="utf-8")
    legacy = processed / "instaumi-csv-v1"
    legacy.mkdir()
    for name in export.CSV_NAMES:
        (legacy / name).write_text("old\n", encoding="utf-8")
    source_trajectory = tmp_path / "source-trajectory.csv"
    trajectory_fields = [
        "frame",
        "timestamp_s",
        "world_frame",
        "map_id",
        "joint_has_pose",
        "left_camera_x_m",
        "left_camera_y_m",
        "left_camera_z_m",
        "left_qx",
        "left_qy",
        "left_qz",
        "left_qw",
        "right_camera_x_m",
        "right_camera_y_m",
        "right_camera_z_m",
        "right_qx",
        "right_qy",
        "right_qz",
        "right_qw",
    ]
    trajectory_rows = [
        {
            "frame": str(index),
            "timestamp_s": f"{index * 0.1:.1f}",
            "world_frame": "session_grid_A",
            "map_id": "test-map",
            "joint_has_pose": "1",
            "left_camera_x_m": f"{index + 1}.0",
            "left_camera_y_m": "0.0",
            "left_camera_z_m": "-1.0",
            "left_qx": "0.0",
            "left_qy": "0.0",
            "left_qz": "0.0",
            "left_qw": "1.0",
            "right_camera_x_m": f"{index + 4}.0",
            "right_camera_y_m": "0.0",
            "right_camera_z_m": "-1.0",
            "right_qx": "0.0",
            "right_qy": "0.0",
            "right_qz": "0.0",
            "right_qw": "1.0",
        }
        for index in range(3)
    ]
    with source_trajectory.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=trajectory_fields)
        writer.writeheader()
        writer.writerows(trajectory_rows)
    (tmp_path / "session_world_map.json").write_text(
        json.dumps(
            {
                "schema_version": "world-apriltag-map/1.0",
                "map_id": "test-map",
                "world_frame": "session_grid_A",
                "physical_up_vector": [0, -1, 0],
                "tags": [
                    {
                        "id": 200,
                        "panel": "grid_A",
                        "corners_m": [
                            [-0.1, -0.1, 0],
                            [0.1, -0.1, 0],
                            [0.1, 0.1, 0],
                            [-0.1, 0.1, 0],
                        ],
                    },
                    {
                        "id": 210,
                        "panel": "grid_B",
                        "corners_m": [
                            [0.9, -0.1, 0],
                            [1.1, -0.1, 0],
                            [1.1, 0.1, 0],
                            [0.9, 0.1, 0],
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

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

    output = processed
    assert result["status"] == "COMPLETE"
    assert (processed / "time_alignment.csv").read_text(encoding="utf-8") == "existing\n"
    assert all((output / name).is_file() for name in export.CSV_NAMES)
    assert not legacy.exists()
    with (output / "gripper.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert rows[0]["left_opening_angle_deg"] == "1.000000"
    assert rows[1]["right_opening_angle_deg"] == ""
    assert rows[1]["right_opening_available"] == "0"
    assert rows[2]["left_opening_measured"] == "0"
    with (output / "processed.csv").open(newline="", encoding="utf-8") as handle:
        combined = list(csv.DictReader(handle))
    assert combined[2]["world_frame"] == "world_flu_aprilgrid_midpoint"
    assert combined[2]["left_camera_x_m"] == "-1.000000000"
    assert combined[2]["left_camera_y_m"] == "-2.500000000"
    assert combined[2]["right_opening_width_m"] == "0.006000000"
    with (output / "metadata.csv").open(newline="", encoding="utf-8") as handle:
        metadata = next(csv.DictReader(handle))
    assert metadata["source_world_frame"] == "session_grid_A"
    assert metadata["world_frame"] == "world_flu_aprilgrid_midpoint"
    assert metadata["world_frame_convention"] == "FLU"
    assert metadata["world_x_positive_definition"] == "AprilGrid_back"
    assert not list(processed.glob(".instaumi-csv-v2-direct-*"))


def test_export_rejects_processed_symlink(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _touch_inputs(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "processed").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ManifestError, match="symlink"):
        export.export_processed_dataset(tmp_path)


def _make_pipeline_final(root: Path) -> Path:
    revision = root / "final" / export.PIPELINE_REVISION
    (revision / "pairs").mkdir(parents=True)
    (revision / "manifest.lock.json").write_text(
        json.dumps({"pipeline_revision": export.PIPELINE_REVISION}),
        encoding="utf-8",
    )
    (revision / "status.json").write_text("{}\n", encoding="utf-8")
    return revision


def test_remove_pipeline_final_removes_only_generated_revision(tmp_path: Path) -> None:
    revision = _make_pipeline_final(tmp_path)

    assert export.remove_pipeline_final(tmp_path) is True
    assert not revision.exists()
    assert not (tmp_path / "final").exists()


def test_remove_pipeline_final_preserves_sibling_revision(tmp_path: Path) -> None:
    _make_pipeline_final(tmp_path)
    sibling = tmp_path / "final" / "keep-me"
    sibling.mkdir()
    (sibling / "user.txt").write_text("keep\n", encoding="utf-8")

    assert export.remove_pipeline_final(tmp_path) is True
    assert sibling.is_dir()
    assert (sibling / "user.txt").read_text(encoding="utf-8") == "keep\n"


def test_remove_pipeline_final_rejects_unexpected_content(tmp_path: Path) -> None:
    revision = _make_pipeline_final(tmp_path)
    (revision / "user.txt").write_text("keep\n", encoding="utf-8")

    with pytest.raises(ManifestError, match="unexpected top-level"):
        export.remove_pipeline_final(tmp_path)

    assert revision.is_dir()
    assert (revision / "user.txt").is_file()


def test_shell_entry_rejects_incomplete_dataset(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "bin/process_instaumi_dataset.sh"
    assert "--remove-pipeline-final" in script.read_text(encoding="utf-8")
    process = subprocess.run([str(script), str(tmp_path)], capture_output=True, text=True)

    assert process.returncode == 2
    assert "Missing required InstaUMI input" in process.stderr


def test_progress_snapshot_reports_aggregate_video_frames() -> None:
    snapshot = progress_snapshot(
        {
            "stages": {
                "identity": {"state": "PASS"},
                "sync": {"state": "PASS"},
                "observation_chunks": {
                    "state": "RUNNING",
                    "completed": 3,
                    "total": 12,
                    "completed_frames": 7200,
                    "total_frames": 31068,
                },
            }
        }
    )

    assert snapshot.key == "observation_chunks"
    assert snapshot.completed_frames == 7200
    line = _render(snapshot, elapsed_s=20, stage_elapsed_s=10)
    assert "3/12" in line
    assert "7200/31068" in line
    assert "23.2%" in line
    assert "ETA 00:33" in line


def test_progress_snapshot_exposes_merge_and_tracking_stages() -> None:
    base = {
        "identity": {"state": "PASS"},
        "sync": {"state": "PASS"},
        "observation_chunks": {"state": "REUSED"},
    }
    merging = progress_snapshot({"stages": base})
    tracking = progress_snapshot(
        {
            "stages": {
                **base,
                "dual_lens_observations": {"state": "PASS"},
                "trajectory_tracking": {"state": "RUNNING"},
            }
        }
    )

    assert merging == ProgressSnapshot("dual_lens_observations", "合并双镜头观测", "WAITING")
    assert tracking.key == "trajectory_tracking"
