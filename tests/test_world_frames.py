import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tools.osmo_360_offline import (
    View,
    load_tag_map,
    perspective_intrinsics,
    solve_view,
    view_to_panorama_rotation,
)
from world_frames import RigidTransform, compile_world_tag_map


ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = ROOT / "config/room_corner_10tag_world_provisional.json"


def test_rigid_transform_is_directional_and_round_trips():
    transform = RigidTransform.from_dict({
        "parent_frame": "world",
        "child_frame": "camera",
        "translation_m": [0.3, -0.2, 0.8],
        "quaternion_xyzw": Rotation.from_euler("xyz", [20, -10, 35], degrees=True).as_quat().tolist(),
    })
    points = np.asarray([[0.0, 0.0, 0.0], [0.2, -0.1, 0.4]])
    np.testing.assert_allclose(
        transform.inverse().apply_points(transform.apply_points(points)), points, atol=1e-12
    )
    identity = transform.compose(transform.inverse())
    assert identity.parent_frame == identity.child_frame == "world"
    np.testing.assert_allclose(identity.translation_m, 0.0, atol=1e-12)
    np.testing.assert_allclose(identity.rotation.as_matrix(), np.eye(3), atol=1e-12)


def test_transform_rejects_reflection_and_bad_frame_chain():
    with pytest.raises(ValueError, match="unit length"):
        RigidTransform.from_dict({
            "parent_frame": "world", "child_frame": "camera",
            "translation_m": [0, 0, 0], "quaternion_xyzw": [0, 0, 0, 2],
        })
    a = RigidTransform.from_dict({
        "parent_frame": "world", "child_frame": "camera",
        "translation_m": [0, 0, 0], "quaternion_xyzw": [0, 0, 0, 1],
    })
    with pytest.raises(ValueError, match="cannot compose"):
        a.compose(a)


def test_room_world_map_has_unique_ids_and_measured_geometry():
    payload = compile_world_tag_map(MAP_PATH)
    assert payload["calibration_status"] == "PROVISIONAL_TAPE_MEASURED"
    assert len(payload["tag_map_sha256"]) == 64
    assert sorted(tag["id"] for tag in payload["tags"]) == list(range(128, 138))
    tag_map = load_tag_map(MAP_PATH)
    np.testing.assert_allclose(tag_map.center(134), [0.0, 0.255, 0.9955], atol=1e-7)
    np.testing.assert_allclose(tag_map.center(130), [0.445, 0.0, 1.0155], atol=1e-7)
    assert tag_map.metadata["world_frame"] == "room_world"


def test_world_map_can_exclude_a_mobile_base_tag_id(tmp_path):
    source = compile_world_tag_map(MAP_PATH)
    source.pop("tag_map_sha256", None)
    source["excluded_tag_ids"] = [128]
    filtered_path = tmp_path / "filtered.json"
    filtered_path.write_text(json.dumps(source), encoding="utf-8")

    payload = compile_world_tag_map(filtered_path)

    assert 128 not in {int(tag["id"]) for tag in payload["tags"]}
    assert sorted(int(tag["id"]) for tag in payload["tags"]) == list(range(129, 138))


def test_non_xy_wall_pnp_recovers_world_camera_pose_with_ippe(monkeypatch):
    tag_map = load_tag_map(MAP_PATH)
    camera_position = np.asarray([0.65, -1.0, 0.82])
    camera_to_world = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    world_to_camera = camera_to_world.T
    tvec = -world_to_camera @ camera_position
    size = 1200
    intrinsics = perspective_intrinsics(size, 90.0)
    original_solve_generic = cv2.solvePnPGeneric
    ippe_inputs = []

    def record_solve_generic(object_points, *args, **kwargs):
        ippe_inputs.append((np.asarray(object_points).copy(), kwargs.get("flags")))
        return original_solve_generic(object_points, *args, **kwargs)

    monkeypatch.setattr(cv2, "solvePnPGeneric", record_solve_generic)
    detections = []
    for tag_id in (128, 129, 131, 132):
        corners = tag_map.corners(tag_id).astype(np.float32)
        pixels, _ = cv2.projectPoints(
            corners, cv2.Rodrigues(world_to_camera)[0], tvec, intrinsics, None,
        )
        pixels = pixels.reshape(4, 2).astype(np.float32)
        detections.append({
            "id": tag_id,
            "corners_px": pixels,
            "center_px": pixels.mean(axis=0),
            "object_center": corners.mean(axis=0),
            "object_corners": corners,
            "area_px2": abs(float(cv2.contourArea(pixels))),
        })
    pose = solve_view(
        detections, View("front", 0, 0, 90), size, min_tags=2,
        max_rmse_px=0.1, pnp_points="corners", pnp_solver="ippe",
    )
    assert pose is not None
    assert len(ippe_inputs) == 1
    assert ippe_inputs[0][1] == cv2.SOLVEPNP_IPPE
    np.testing.assert_allclose(ippe_inputs[0][0][:, 2], 0.0, atol=1e-12)
    np.testing.assert_allclose(pose.xyz, camera_position, atol=1e-5)
    np.testing.assert_allclose(pose.rotation_camera_to_board, camera_to_world, atol=1e-5)


def test_arbitrary_plane_ippe_is_consistent_across_perspective_views():
    tag_map = load_tag_map(MAP_PATH)
    camera_position = np.asarray([0.65, -1.0, 0.82])
    camera_to_world = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]
    )
    world_to_panorama = camera_to_world.T
    panorama_tvec = -world_to_panorama @ camera_position
    recovered = []

    for view in (
        View("front", 0.0, 0.0, 90.0),
        View("offset", 18.0, -7.0, 90.0, 4.0),
    ):
        view_to_panorama = view_to_panorama_rotation(
            view.yaw, view.pitch, view.roll,
        )
        world_to_view = view_to_panorama.T @ world_to_panorama
        view_tvec = view_to_panorama.T @ panorama_tvec
        intrinsics = perspective_intrinsics(1200, view.fov)
        detections = []
        for tag_id in (128, 129, 131, 132):
            corners = tag_map.corners(tag_id).astype(np.float32)
            pixels, _ = cv2.projectPoints(
                corners, cv2.Rodrigues(world_to_view)[0], view_tvec,
                intrinsics, None,
            )
            pixels = pixels.reshape(4, 2).astype(np.float32)
            detections.append({
                "id": tag_id,
                "corners_px": pixels,
                "center_px": pixels.mean(axis=0),
                "object_center": corners.mean(axis=0),
                "object_corners": corners,
                "area_px2": abs(float(cv2.contourArea(pixels))),
            })
        pose = solve_view(
            detections, view, 1200, min_tags=2, max_rmse_px=0.1,
            pnp_points="corners", pnp_solver="ippe",
        )
        assert pose is not None
        recovered.append(pose)

    for pose in recovered:
        np.testing.assert_allclose(pose.xyz, camera_position, atol=1e-5)
        np.testing.assert_allclose(
            pose.rotation_camera_to_board, camera_to_world, atol=1e-5,
        )
    np.testing.assert_allclose(recovered[0].xyz, recovered[1].xyz, atol=1e-6)
