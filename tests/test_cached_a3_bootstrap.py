from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from osmo360.localization import cached_a3_bootstrap
from osmo360.localization.cached_a3_bootstrap import (
    Pose,
    _temporal_gate,
    load_cache_frames,
    pose_to_hand_camera_flu,
    write_joint_pose_csv,
)
from osmo360.localization.coordinate_frames import (
    X5_STREAM0_OPENCV_FROM_HAND_CAMERA_FLU,
)
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


def test_sparse_flow_planar_jump_is_rejected_but_direct_reacquisition_is_not():
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
    assert weak["reason"] == "sparse_flow_planar_pose_exceeds_temporal_limits"
    assert direct["rejected"] is False


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
