from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from osmo360.localization import cached_a3_bootstrap
from osmo360.localization.cached_a3_bootstrap import (
    Pose,
    _temporal_gate,
    load_cache_frames,
    pose_to_hand_camera_flu,
    track_cache,
    write_joint_pose_csv,
)
from osmo360.localization.coordinate_frames import (
    X5_STREAM0_OPENCV_FROM_HAND_CAMERA_FLU,
)
from osmo360.localization.instaumi_imu import ImuSeries
from osmo360.localization.raw_fisheye_world_pose import (
    make_kannala_brandt_ray_converter,
    make_x5_offset_ray_converter,
)


def _row(frame: int, time_s: float, x: float | None) -> dict[str, str | int]:
    valid = x is not None
    return {
        "frame": frame,
        "timestamp": f"{time_s:.6f}",
        "camera_x_m": "" if x is None else str(x),
        "camera_y_m": "" if x is None else "0",
        "camera_z_m": "" if x is None else "-0.5",
        "qx": "" if x is None else "0",
        "qy": "" if x is None else "0",
        "qz": "" if x is None else "0",
        "qw": "" if x is None else "1",
        "quality_status": "valid" if valid else "angular_rmse_rejected",
        "angular_rmse_deg": "0.2" if valid else "2.5",
        "detected_tag_count": 4,
        "inlier_tag_count": 4 if valid else 0,
        "measurement_source": "cached_raw_fisheye_bearing_direct",
    }


def test_cache_loader_decompresses_each_npz_member_once(monkeypatch, tmp_path: Path):
    arrays = {
        "frame_index": np.asarray([0, 0, 2], dtype=np.int32),
        "tag_id": np.asarray([5, 5, 6], dtype=np.int32),
        "rays_camera": np.arange(36, dtype=np.float32).reshape(3, 4, 3),
        "area_px2": np.asarray([10.0, 12.0, 8.0], dtype=np.float32),
        "detection_source": np.asarray(["direct", "flow", "direct"]),
        "lens_stream": np.asarray([0, 1, 0], dtype=np.int8),
        "timeline_frame_index": np.asarray([0, 1, 2], dtype=np.int32),
        "timeline_common_time_s": np.asarray([0.0, 0.1, 0.2]),
    }

    class CountingArchive:
        def __init__(self):
            self.reads: Counter[str] = Counter()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __getitem__(self, key: str):
            self.reads[key] += 1
            return arrays[key]

    archive = CountingArchive()
    monkeypatch.setattr(cached_a3_bootstrap.np, "load", lambda _path: archive)

    frames, times = load_cache_frames(tmp_path / "observations.npz")

    assert archive.reads == Counter({key: 1 for key in arrays})
    assert set(frames) == {0, 2}
    assert frames[0][5].area_px2 == 12.0
    assert frames[0][5].source == "flow"
    assert frames[0][5].lens_stream == 1
    assert frames[2][6].lens_stream == 0
    assert np.array_equal(frames[2][6].rays, arrays["rays_camera"][2])
    assert times == {0: 0.0, 1: 0.1, 2: 0.2}


def test_joint_csv_interpolates_both_tracks_in_one_map(tmp_path: Path):
    left = [_row(0, 0.0, 0.0), _row(4, 0.1, None), _row(8, 0.2, 2.0)]
    right = [_row(0, 0.0, 5.0), _row(4, 0.1, 6.0), _row(8, 0.2, 7.0)]
    output = tmp_path / "joint.csv"
    summary = write_joint_pose_csv(output, left, right, map_id="shared-map")
    rows = list(csv.DictReader(output.open(newline="", encoding="utf-8")))

    assert summary["joint_valid_ratio"] == 1.0
    assert summary["joint_measured_ratio"] == 2 / 3
    assert summary["maximum_interpolation_gap_s"]["left"] == 0.2
    assert rows[1]["world_frame"] == "session_grid_A"
    assert rows[1]["map_id"] == "shared-map"
    assert rows[1]["left_pose_state"] == "INTERPOLATED"
    assert float(rows[1]["left_camera_x_m"]) == 1.0
    assert rows[1]["right_pose_state"] == "MEASURED"


def test_joint_csv_preserves_aligned_subframe_phase_and_uses_right_timeline(
    tmp_path: Path,
) -> None:
    left = [_row(0, -0.0005, 0.0), _row(1, 0.0995, 1.0)]
    right = [_row(0, 0.0, 5.0), _row(1, 0.1, 6.0)]
    output = tmp_path / "joint.csv"

    summary = write_joint_pose_csv(output, left, right, map_id="shared-map")
    rows = list(csv.DictReader(output.open(newline="", encoding="utf-8")))

    assert [float(row["timestamp_s"]) for row in rows] == [0.0, 0.1]
    assert summary["canonical_joint_timestamp_source"] == (
        "right_camera_aligned_h5_timeline"
    )
    assert summary["maximum_paired_timestamp_delta_s"] == pytest.approx(0.0005)


def test_joint_csv_retains_untrusted_pose_for_interpolation_longer_than_quarter_second(
    tmp_path: Path,
):
    left = [_row(0, 0.0, 0.0), _row(6, 0.2, None), _row(12, 0.4, 2.0)]
    right = [_row(0, 0.0, 5.0), _row(6, 0.2, 6.0), _row(12, 0.4, 7.0)]
    output = tmp_path / "joint.csv"

    summary = write_joint_pose_csv(output, left, right, map_id="shared-map")
    rows = list(csv.DictReader(output.open(newline="", encoding="utf-8")))

    assert summary["joint_valid_frames"] == 2
    assert summary["joint_valid_ratio"] == 2 / 3
    assert summary["joint_pose_frames"] == 3
    assert summary["joint_pose_ratio"] == 1.0
    assert summary["maximum_allowed_interpolation_gap_s"] == 0.25
    assert summary["maximum_interpolation_gap_s"]["left"] == 0.0
    assert summary["maximum_rejected_interpolation_gap_s"]["left"] == 0.4
    assert summary["untrusted_long_gap_frames"] == 1
    assert summary["untrusted_long_gap_side_frames"] == {"left": 1, "right": 0}
    assert rows[1]["joint_valid"] == "false"
    assert rows[1]["joint_has_pose"] == "true"
    assert rows[1]["left_quality_status"] == "interpolation_untrusted"
    assert rows[1]["left_pose_state"] == "INTERPOLATED_UNTRUSTED"
    assert float(rows[1]["left_camera_x_m"]) == pytest.approx(1.0)
    assert rows[1]["right_pose_state"] == "MEASURED"


def test_joint_csv_holds_nearest_pose_outside_measurement_span(tmp_path: Path):
    left = [_row(0, 0.0, None), _row(1, 0.1, 1.0), _row(2, 0.2, None)]
    right = [_row(0, 0.0, 5.0), _row(1, 0.1, 6.0), _row(2, 0.2, 7.0)]
    output = tmp_path / "joint.csv"

    summary = write_joint_pose_csv(output, left, right, map_id="shared-map")
    rows = list(csv.DictReader(output.open(newline="", encoding="utf-8")))

    assert summary["joint_pose_frames"] == 3
    assert summary["joint_pose_ratio"] == 1.0
    assert summary["held_untrusted_side_frames"] == {"left": 2, "right": 0}
    assert [row["left_pose_state"] for row in rows] == [
        "HELD_UNTRUSTED", "MEASURED", "HELD_UNTRUSTED"
    ]
    assert [float(row["left_camera_x_m"]) for row in rows] == [1.0, 1.0, 1.0]


def test_joint_csv_fails_instead_of_fabricating_when_a_side_has_no_pose(
    tmp_path: Path,
):
    missing = [_row(0, 0.0, None), _row(1, 0.1, None)]
    right = [_row(0, 0.0, 5.0), _row(1, 0.1, 6.0)]

    with pytest.raises(ValueError, match="left trajectory has no accepted pose"):
        write_joint_pose_csv(tmp_path / "joint.csv", missing, right, map_id="shared-map")


@pytest.mark.parametrize("gap", [0.0, -0.1, 0.251, float("inf")])
def test_joint_csv_rejects_unsafe_interpolation_limit(tmp_path: Path, gap: float):
    rows = [_row(0, 0.0, 0.0), _row(1, 0.1, 1.0)]
    with pytest.raises(ValueError, match="0.25"):
        write_joint_pose_csv(
            tmp_path / "joint.csv",
            rows,
            rows,
            map_id="shared-map",
            maximum_interpolation_gap_s=gap,
        )


def test_explicit_h5_kb_model_matches_equivalent_x5_rear_record():
    offset = (
        "n2_2663.778_2695.450_2691.260_-0.331_0.192_89.482_"
        "2653.083_8069.790_2689.460_0.386_0.193_90.228_10752_5376_11378"
    )
    x5, _ = make_x5_offset_ray_converter(
        offset, stream=0, source_width=1920, source_height=1920
    )
    intrinsics = {
        "fx": 328.2338018976478,
        "fy": 328.2338018976478,
        "cx": 507.38628571428563,
        "cy": 513.4190476190475,
        "coefficients": [0.0, 0.0, 0.0, 0.0],
    }
    rig_from_camera = np.eye(4)
    rig_from_camera[:3, :3] = Rotation.from_euler(
        "xy", [-0.331, 0.192], degrees=True
    ).as_matrix()
    explicit, _ = make_kannala_brandt_ray_converter(
        intrinsics,
        rig_from_camera,
        calibration_width=1024,
        calibration_height=1024,
        source_width=1920,
        source_height=1920,
    )
    pixels = np.asarray([
        [960, 960], [100, 960], [1800, 960], [960, 100], [960, 1800]
    ], dtype=float)
    assert np.allclose(explicit(pixels), x5(pixels), atol=2e-8)


def test_absolute_jump_is_rejected_even_for_direct_reacquisition():
    previous = Pose(np.zeros(3), Rotation.identity())
    jumped = Pose(
        np.asarray([0.50, 0.0, 0.0]),
        Rotation.from_euler("z", 60.0, degrees=True),
    )
    weak = _temporal_gate(
        jumped,
        (7.0, previous),
        7.14,
        inlier_tag_count=2,
        sources={"lk_forward_backward"},
    )
    direct = _temporal_gate(
        jumped,
        (7.0, previous),
        7.14,
        inlier_tag_count=5,
        sources={"global_scout_roi_gray"},
    )

    assert weak["rejected"] is True
    assert weak["reason"] == "pose_exceeds_absolute_temporal_limits"
    assert direct["rejected"] is True
    assert direct["reason"] == "pose_exceeds_absolute_temporal_limits"


def test_moderately_fast_same_lens_direct_measurement_remains_accepted():
    previous = Pose(np.zeros(3), Rotation.identity())
    candidate = Pose(
        np.asarray([0.28, 0.0, 0.0]),
        Rotation.from_euler("z", 20.0, degrees=True),
    )

    direct = _temporal_gate(
        candidate,
        (7.0, previous),
        7.14,
        inlier_tag_count=5,
        sources={"global_scout_roi_gray"},
        dominant_lens_stream=0,
        previous_dominant_lens_stream=0,
    )

    assert direct["rejected"] is False
    assert direct["reason"] == "accepted"


def test_recovery_latch_requires_strong_consistent_geometry():
    previous = Pose(np.zeros(3), Rotation.identity())
    candidate = Pose(
        np.asarray([0.05, 0.0, 0.0]),
        Rotation.from_euler("z", 5.0, degrees=True),
    )

    weak = _temporal_gate(
        candidate,
        (7.0, previous),
        7.10,
        inlier_tag_count=3,
        sources={"global_scout_roi_gray"},
        recovery_required=True,
    )
    strong = _temporal_gate(
        candidate,
        (7.0, previous),
        7.10,
        inlier_tag_count=5,
        sources={"global_scout_roi_gray"},
        recovery_required=True,
    )

    assert weak["rejected"] is True
    assert weak["reason"] == "temporal_recovery_requires_strong_consistent_geometry"
    assert strong["rejected"] is False


def test_weak_visual_rotation_that_disagrees_with_gyro_is_rejected():
    times = np.arange(0.0, 0.101, 0.01)
    stationary_imu = ImuSeries(
        side="left",
        timestamp_s=times,
        angular_velocity_hand_rad_s=np.zeros((len(times), 3)),
        calibration_sha256="test",
        dataset_path="test.h5",
    )
    previous = Pose(np.zeros(3), Rotation.identity())
    candidate = Pose(
        np.asarray([0.01, 0.0, 0.0]),
        Rotation.from_euler("z", 30.0, degrees=True),
    )

    result = _temporal_gate(
        candidate,
        (0.0, previous),
        0.1,
        inlier_tag_count=3,
        sources={"global_scout_roi_gray"},
        imu_stream=stationary_imu,
    )

    assert result["imu_prediction_available"] is True
    assert result["imu_visual_rotation_residual_deg"] == pytest.approx(30.0)
    assert result["rejected"] is True
    assert result["reason"] == "weak_visual_rotation_disagrees_with_imu"


def test_fast_lens_handoff_is_rejected_once_without_blocking_same_lens_motion():
    previous = Pose(np.zeros(3), Rotation.identity())
    jumped = Pose(
        np.asarray([0.10, 0.0, 0.0]),
        Rotation.from_euler("z", 20.0, degrees=True),
    )
    handoff = _temporal_gate(
        jumped,
        (6.0, previous),
        6.04,
        inlier_tag_count=6,
        sources={"global_scout_roi_gray"},
        dominant_lens_stream=1,
        previous_dominant_lens_stream=0,
    )
    same_lens = _temporal_gate(
        jumped,
        (6.0, previous),
        6.04,
        inlier_tag_count=6,
        sources={"global_scout_roi_gray"},
        dominant_lens_stream=1,
        previous_dominant_lens_stream=1,
    )
    slow_handoff = _temporal_gate(
        jumped,
        (6.0, previous),
        6.50,
        inlier_tag_count=6,
        sources={"global_scout_roi_gray"},
        dominant_lens_stream=1,
        previous_dominant_lens_stream=0,
    )

    assert handoff["rejected"] is True
    assert handoff["lens_handoff_measurement"] is True
    assert handoff["reason"] == "lens_handoff_pose_exceeds_temporal_limits"
    assert same_lens["rejected"] is False
    assert slow_handoff["rejected"] is False


def test_real_four_mp4_cache_rejects_only_known_fast_lens_handoffs(
    tmp_path: Path,
):
    """Optional production-cache regression, enabled explicitly on the server."""
    root_value = os.environ.get("OSMO_REAL_HANDOFF_CACHE_ROOT")
    if not root_value:
        pytest.skip("set OSMO_REAL_HANDOFF_CACHE_ROOT for production-cache regression")
    root = Path(root_value)
    map_payload = json.loads(
        (root / "tracking/session_world_map.json").read_text(encoding="utf-8")
    )
    panel_a = {
        int(tag["id"]): np.asarray(tag["corners_m"], dtype=np.float64)
        for tag in map_payload["tags"]
        if tag["panel"] == "grid_A"
    }
    transform_payload = map_payload["panel_transform"]
    panel_a_to_b = Pose(
        np.asarray(transform_payload["translation_m"], dtype=np.float64),
        Rotation.from_quat(transform_payload["quaternion_xyzw"]),
    )
    panel_b = {
        int(tag["id"]): panel_a_to_b.rotation.inv().apply(
            np.asarray(tag["corners_m"], dtype=np.float64)
            - panel_a_to_b.position
        )
        for tag in map_payload["tags"]
        if tag["panel"] == "grid_B"
    }

    left_rows, left_summary = track_cache(
        root / "observations/left/dual-lens-corners.npz",
        panel_a,
        panel_b,
        panel_a_to_b,
    )
    right_rows, right_summary = track_cache(
        root / "observations/right/dual-lens-corners.npz",
        panel_a,
        panel_b,
        panel_a_to_b,
    )
    rejected_left = [
        int(row["frame"])
        for row in left_rows
        if row["quality_status"] == "temporal_outlier_rejected"
        and row["temporal_gate_reason"]
        == "lens_handoff_pose_exceeds_temporal_limits"
    ]
    rejected_right = [
        int(row["frame"])
        for row in right_rows
        if row["quality_status"] == "temporal_outlier_rejected"
        and row["temporal_gate_reason"]
        == "lens_handoff_pose_exceeds_temporal_limits"
    ]

    assert rejected_left == [168, 360, 420]
    assert rejected_right == []
    assert left_summary["lens_handoff_temporal_outlier_rejected_frames"] == 3
    assert right_summary["lens_handoff_temporal_outlier_rejected_frames"] == 0
    joint_summary = write_joint_pose_csv(
        tmp_path / "joint.csv",
        left_rows,
        right_rows,
        map_id=map_payload["map_id"],
    )
    assert joint_summary["common_timeline_frames"] == 300
    assert joint_summary["joint_pose_frames"] == 300
    assert joint_summary["joint_valid_frames"] == 266
    assert joint_summary["joint_measured_frames"] == 263
    assert joint_summary["untrusted_long_gap_frames"] == 34


def test_hand_camera_flu_positive_x_is_back_stream_optical_axis():
    bridge = X5_STREAM0_OPENCV_FROM_HAND_CAMERA_FLU
    assert np.allclose(bridge @ [1, 0, 0], [0, 0, 1])
    assert np.allclose(bridge @ [0, 1, 0], [-1, 0, 0])
    assert np.allclose(bridge @ [0, 0, 1], [0, -1, 0])
    assert np.isclose(np.linalg.det(bridge), 1.0)

    internal = Pose(np.asarray([1.0, 2.0, 3.0]), Rotation.identity())
    hand = pose_to_hand_camera_flu(internal)
    assert np.allclose(hand.position, internal.position)
    assert np.allclose(hand.rotation.as_matrix(), bridge)
