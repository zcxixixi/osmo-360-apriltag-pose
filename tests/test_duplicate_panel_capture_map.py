from pathlib import Path

import cv2
import numpy as np

from osmo_360_offline import (
    View,
    load_capture_duplicate_tag_map,
    load_tag_map,
    perspective_intrinsics,
    solve_view,
)


ROOT = Path(__file__).resolve().parents[1]
CAPTURE_MAP = ROOT / "config/a9abc654_duplicate_panel_capture_map.json"


def _project_detection(raw_id, instance, world_to_camera, tvec, intrinsics):
    pixels, _ = cv2.projectPoints(
        instance.corners_m,
        cv2.Rodrigues(world_to_camera)[0],
        tvec,
        intrinsics,
        None,
    )
    pixels = pixels.reshape(4, 2).astype(np.float32)
    return {
        "id": raw_id,
        "corners_px": pixels,
        "center_px": pixels.mean(axis=0),
        "area_px2": abs(float(cv2.contourArea(pixels))),
        "instance_options": tuple(),
    }


def test_capture_map_preserves_ten_physical_instances_with_duplicate_raw_ids():
    capture_map = load_capture_duplicate_tag_map(CAPTURE_MAP)
    assert capture_map.expected_ids == [128, 129, 130, 131, 132, 133]
    assert capture_map.duplicate_raw_ids == [128, 129, 130, 131]
    assert len(capture_map.expected_virtual_ids) == 10
    assert len(capture_map.metadata["tag_map_sha256"]) == 64
    assert {
        option.virtual_id for option in capture_map.instance_options(128)
    } == {
        "left_vertical_legacy:128",
        "right_grid_legacy:128",
    }


def test_same_decoded_id_twice_is_resolved_only_by_explicit_capture_map():
    capture_map = load_capture_duplicate_tag_map(CAPTURE_MAP)
    camera_position = np.asarray([0.65, -1.0, 0.82])
    camera_to_world = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]
    )
    world_to_camera = camera_to_world.T
    tvec = -world_to_camera @ camera_position
    size = 1200
    intrinsics = perspective_intrinsics(size, 90.0)

    detections = []
    for panel in capture_map.panels.values():
        for raw_id, instance in panel.items():
            detection = _project_detection(
                raw_id, instance, world_to_camera, tvec, intrinsics,
            )
            detection["instance_options"] = tuple(
                capture_map.instance_options(raw_id)
            )
            detections.append(detection)
    diagnostics = {}
    pose = solve_view(
        detections,
        View("front", 0.0, 0.0, 90.0),
        size,
        min_tags=2,
        max_rmse_px=0.1,
        pnp_points="corners",
        pnp_solver="ippe",
        diagnostics=diagnostics,
    )
    assert pose is not None
    assert pose.capture_panel in {
        "left_vertical_legacy", "right_grid_legacy",
    }
    assert all(value.startswith(f"{pose.capture_panel}:") for value in pose.virtual_instance_ids)
    np.testing.assert_allclose(pose.xyz, camera_position, atol=1e-5)
    np.testing.assert_allclose(
        pose.rotation_camera_to_board, camera_to_world, atol=1e-5,
    )
    assert diagnostics["duplicate_panel_resolution"] is True
    assert diagnostics["duplicate_decoded_ids"] == [128, 129, 130, 131]


def test_production_unique_map_rejects_repeated_decode_instead_of_largest_dedupe():
    tag_map = load_tag_map(ROOT / "config/a4_wall_6tag_ids_128_133.json")
    corners = tag_map.corners(128)
    first = {
        "id": 128,
        "corners_px": np.asarray([[10, 10], [30, 10], [30, 30], [10, 30]], np.float32),
        "center_px": np.asarray([20, 20], np.float32),
        "object_center": corners.mean(axis=0),
        "object_corners": corners,
        "area_px2": 400.0,
    }
    second = dict(first)
    second["corners_px"] = first["corners_px"] + 100.0
    second["center_px"] = first["center_px"] + 100.0
    second["area_px2"] = 900.0
    diagnostics = {}
    pose = solve_view(
        [first, second],
        View("front", 0.0, 0.0, 90.0),
        600,
        min_tags=1,
        max_rmse_px=3.0,
        pnp_points="corners",
        diagnostics=diagnostics,
    )
    assert pose is None
    assert diagnostics["visual_pose_failure_reason"] == "duplicate_decoded_id_conflict"
    assert diagnostics["duplicate_resolution"] == "rejected_production_unique_map"

