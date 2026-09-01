from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from osmo360.localization.cached_a3_bootstrap import write_joint_pose_csv
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
