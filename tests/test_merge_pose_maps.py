import csv
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from merge_pose_maps import merge


FIELDS = [
    "frame", "timestamp", "camera_x_m", "camera_y_m", "camera_z_m",
    "roll_deg", "pitch_deg", "yaw_deg", "raw_camera_x_m", "raw_camera_y_m",
    "raw_camera_z_m", "detected_tag_count", "inlier_count", "reprojection_rmse_px",
    "detected_ids", "selected_view", "measurement_source", "quality_status",
]


def write_pose(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)


def row(frame: int, p: np.ndarray, r: Rotation, valid: bool = True) -> dict[str, str]:
    values = dict.fromkeys(FIELDS, "")
    euler = r.as_euler("xyz", degrees=True)
    values.update(
        frame=str(frame), timestamp=f"{frame / 60:.6f}",
        camera_x_m=str(p[0]), camera_y_m=str(p[1]), camera_z_m=str(p[2]),
        raw_camera_x_m=str(p[0]), raw_camera_y_m=str(p[1]), raw_camera_z_m=str(p[2]),
        roll_deg=str(euler[0]), pitch_deg=str(euler[1]), yaw_deg=str(euler[2]),
        measurement_source="direct", quality_status="valid" if valid else "insufficient_tags",
    )
    if not valid:
        for key in (*("camera_x_m", "camera_y_m", "camera_z_m"),):
            values[key] = ""
    return values


def test_merge_uses_secondary_only_for_primary_gaps(tmp_path: Path):
    transform_r = Rotation.from_euler("y", -90, degrees=True)
    transform_t = np.array([-0.7, 0.03, -0.35])
    primary_rows, secondary_rows = [], []
    for frame in range(20):
        p_secondary = np.array([0.2 + frame * 0.002, 0.8, -0.9])
        r_secondary = Rotation.from_euler("xyz", [2, -10 + frame * 0.1, 4], degrees=True)
        p_primary = transform_r.apply(p_secondary) + transform_t
        r_primary = transform_r * r_secondary
        primary_rows.append(row(frame, p_primary, r_primary, valid=frame != 12))
        secondary_rows.append(row(frame, p_secondary, r_secondary))
    primary = tmp_path / "primary.csv"; secondary = tmp_path / "secondary.csv"
    output = tmp_path / "merged.csv"
    write_pose(primary, primary_rows); write_pose(secondary, secondary_rows)
    audit = merge(primary, secondary, output, trusted_secondary=True)

    assert audit["primary_measurements"] == 19
    assert audit["secondary_recovery_measurements"] == 1
    assert audit["merged_valid_ratio"] == 1.0
    assert audit["position_residual_m"]["p95"] < 1e-8
    rows = list(csv.DictReader(output.open()))
    assert rows[12]["measurement_source"].startswith("same_map_relaxed:")
    np.testing.assert_allclose(
        [float(rows[12][key]) for key in ("camera_x_m", "camera_y_m", "camera_z_m")],
        transform_r.apply(np.array([0.224, 0.8, -0.9])) + transform_t,
        atol=1e-8,
    )
