import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tools.insta360_offline import (
    _BODY_TO_PANORAMA,
    ImuPanoramaBridgeCalibrator,
    Pose,
    View,
    choose_guarded_visual_reanchor,
    choose_pose,
    default_panorama_to_body_rotation,
    estimate_imu_panorama_bridge,
    find_duplicate_tag_ray_conflicts,
    perspective_intrinsics,
    pose_updates_authoritative_anchor,
    pose_view_to_panorama,
    predict_camera_to_parent_rotation,
    predict_camera_to_parent_rotation_hypotheses,
    propagate_view_with_imu,
    quaternion_to_rotation,
    rotation_residual_deg,
    solve_fixed_attitude_translation,
    solve_view,
    view_to_panorama_rotation,
)
from tools.apriltag_geometry import Grid


def test_kalibr_grid_is_column_major():
    grid = Grid(rows=6, cols=6, tag_size=0.088, spacing_ratio=0.30)
    assert grid.center(1)[0] == pytest.approx(grid.center(0)[0])
    assert grid.center(1)[1] < grid.center(0)[1]
    assert grid.center(6)[0] > grid.center(0)[0]
    assert grid.center(6)[1] == pytest.approx(grid.center(0)[1])
    assert grid.center(35) is not None
    assert grid.center(36) is None


@pytest.mark.parametrize("yaw,pitch", [(0, 0), (45, 0), (-90, 0), (0, 45), (135, -30)])
def test_view_rotation_is_proper(yaw, pitch):
    r = view_to_panorama_rotation(yaw, pitch)
    np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1.0)


def test_forward_ray_matches_view_heading():
    np.testing.assert_allclose(
        view_to_panorama_rotation(90, 0) @ [0, 0, 1], [1, 0, 0], atol=1e-12
    )


def test_imu_view_propagation_preserves_world_facing_basis():
    previous = np.array([1.0, 0.0, 0.0, 0.0])
    angle = np.radians(37.0)
    current = np.array([np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0])
    view = View("tracked", 42.0, -18.0, 85.0, 23.0)
    propagated = propagate_view_with_imu(view, previous, current)
    previous_world_basis = (
        quaternion_to_rotation(previous)
        @ _BODY_TO_PANORAMA.T
        @ view_to_panorama_rotation(view.yaw, view.pitch, view.roll)
    )
    current_world_basis = (
        quaternion_to_rotation(current)
        @ _BODY_TO_PANORAMA.T
        @ view_to_panorama_rotation(
            propagated.yaw, propagated.pitch, propagated.roll
        )
    )
    np.testing.assert_allclose(current_world_basis, previous_world_basis, atol=1e-9)
    np.testing.assert_allclose(
        view_to_panorama_rotation(0, 90) @ [0, 0, 1], [0, -1, 0], atol=1e-12
    )


def test_imu_attitude_propagation_uses_camera_to_parent_direction():
    previous_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    angle = np.radians(37.0)
    current_quaternion = np.array(
        [np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0]
    )
    previous_camera_to_parent = cv2.Rodrigues(
        np.array([0.17, -0.08, 0.21])
    )[0]
    predicted = predict_camera_to_parent_rotation(
        previous_camera_to_parent, previous_quaternion, current_quaternion,
    )
    panorama_current_from_previous = (
        _BODY_TO_PANORAMA
        @ quaternion_to_rotation(current_quaternion).T
        @ quaternion_to_rotation(previous_quaternion)
        @ _BODY_TO_PANORAMA.T
    )
    np.testing.assert_allclose(
        predicted @ panorama_current_from_previous,
        previous_camera_to_parent,
        atol=1e-12,
    )
    np.testing.assert_allclose(predicted.T @ predicted, np.eye(3), atol=1e-12)
    assert np.linalg.det(predicted) == pytest.approx(1.0)


def test_imu_attitude_hypotheses_include_inverse_relative_recovery():
    previous_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    angle = np.radians(37.0)
    current_quaternion = np.array(
        [np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0]
    )
    previous_camera_to_parent = cv2.Rodrigues(
        np.array([0.17, -0.08, 0.21])
    )[0]
    hypotheses = predict_camera_to_parent_rotation_hypotheses(
        previous_camera_to_parent, previous_quaternion, current_quaternion,
    )
    assert [name for name, _rotation in hypotheses] == [
        "nominal", "inverse_relative_recovery",
    ]
    np.testing.assert_allclose(
        hypotheses[0][1],
        predict_camera_to_parent_rotation(
            previous_camera_to_parent,
            previous_quaternion,
            current_quaternion,
        ),
        atol=1e-12,
    )
    relative = (
        _BODY_TO_PANORAMA
        @ quaternion_to_rotation(current_quaternion).T
        @ quaternion_to_rotation(previous_quaternion)
        @ _BODY_TO_PANORAMA.T
    )
    np.testing.assert_allclose(
        hypotheses[1][1], previous_camera_to_parent @ relative, atol=1e-12,
    )


def test_default_imu_panorama_bridge_is_a_proper_equivalent_rotation():
    bridge = default_panorama_to_body_rotation()
    np.testing.assert_allclose(bridge.T @ bridge, np.eye(3), atol=1e-12)
    assert np.linalg.det(bridge) == pytest.approx(1.0)
    np.testing.assert_allclose(bridge, _BODY_TO_PANORAMA.T, atol=1e-12)
    delta = Rotation.from_euler(
        "xyz", [17.0, -23.0, 8.0], degrees=True,
    ).as_matrix()
    np.testing.assert_allclose(
        bridge.T @ delta @ bridge,
        _BODY_TO_PANORAMA @ delta @ _BODY_TO_PANORAMA.T,
        atol=1e-12,
    )


def test_imu_panorama_bridge_recovers_fixed_rotation_with_visual_outlier():
    alignment = Rotation.from_euler(
        "zyx", [31.0, -18.0, 9.0], degrees=True,
    ).as_matrix()
    bridge = Rotation.from_euler(
        "xyz", [-78.0, 24.0, 133.0], degrees=True,
    ).as_matrix()
    imu = [
        Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
        for angles in (
            (0.0, 0.0, 0.0),
            (7.0, 3.0, -2.0),
            (15.0, -5.0, 6.0),
            (24.0, 8.0, 13.0),
            (31.0, -12.0, 21.0),
            (42.0, 17.0, 28.0),
            (55.0, -20.0, 39.0),
        )
    ]
    visual = [alignment @ rotation @ bridge for rotation in imu]
    visual[-1] = visual[-1] @ Rotation.from_euler(
        "y", 105.0, degrees=True,
    ).as_matrix()
    estimate = estimate_imu_panorama_bridge(visual, imu)
    assert estimate is not None
    assert estimate.observation_count == 7
    assert estimate.inlier_count == 6
    assert estimate.excitation_axis_ratio > 0.03
    assert rotation_residual_deg(
        estimate.panorama_to_body, bridge,
    ) < 0.1
    assert rotation_residual_deg(
        estimate.imu_world_to_parent, alignment,
    ) < 0.1
    assert estimate.residual_p95_deg < 0.1


def test_imu_panorama_bridge_rejects_unobservable_single_axis_motion():
    alignment = Rotation.from_euler("x", 20.0, degrees=True).as_matrix()
    bridge = Rotation.from_euler("y", -40.0, degrees=True).as_matrix()
    imu = [
        Rotation.from_euler("z", angle, degrees=True).as_matrix()
        for angle in (0.0, 5.0, 10.0, 20.0, 35.0)
    ]
    visual = [alignment @ rotation @ bridge for rotation in imu]
    assert estimate_imu_panorama_bridge(visual, imu) is None


def test_incremental_imu_panorama_bridge_predicts_absolute_attitude():
    alignment = Rotation.from_euler(
        "xyz", [19.0, -11.0, 37.0], degrees=True,
    ).as_matrix()
    bridge = Rotation.from_euler(
        "xyz", [-72.0, 31.0, 118.0], degrees=True,
    ).as_matrix()
    imu_rotations = [
        Rotation.from_euler("xyz", angles, degrees=True)
        for angles in (
            (0.0, 0.0, 0.0),
            (8.0, 3.0, -1.0),
            (16.0, -4.0, 7.0),
            (27.0, 9.0, 16.0),
        )
    ]
    calibrator = ImuPanoramaBridgeCalibrator(min_observations=4)
    for frame, imu_rotation in enumerate(imu_rotations):
        xyzw = imu_rotation.as_quat()
        quaternion_wxyz = xyzw[[3, 0, 1, 2]]
        calibrator.add_observation(
            frame,
            alignment @ imu_rotation.as_matrix() @ bridge,
            quaternion_wxyz,
        )
    assert calibrator.status == "calibrated"
    current_imu = Rotation.from_euler(
        "xyz", [39.0, -13.0, 29.0], degrees=True,
    )
    current_xyzw = current_imu.as_quat()
    predicted = calibrator.predict(current_xyzw[[3, 0, 1, 2]])
    assert predicted is not None
    assert rotation_residual_deg(
        predicted, alignment @ current_imu.as_matrix() @ bridge,
    ) < 0.1
    audit = calibrator.audit()
    assert audit["status"] == "calibrated"
    assert audit["inlier_count"] == 4
    assert len(audit["panorama_to_body_quaternion_xyzw"]) == 4


def test_duplicate_tag_ray_conflict_distinguishes_overlap_from_two_panels():
    def record(yaw: float, tag_id: int = 128) -> dict:
        return {
            "view": {
                "name": f"h{yaw}", "yaw": yaw, "pitch": 0.0,
                "fov": 90.0, "roll": 0.0,
            },
            "size": 400,
            "detections": [
                {
                    "id": tag_id,
                    "corners_px": [
                        [180.0, 180.0], [220.0, 180.0],
                        [220.0, 220.0], [180.0, 220.0],
                    ],
                }
            ],
        }

    assert find_duplicate_tag_ray_conflicts([record(0.0), record(10.0)]) == []
    assert find_duplicate_tag_ray_conflicts([record(0.0), record(90.0)]) == [128]
    assert find_duplicate_tag_ray_conflicts(
        [record(0.0, 128), record(90.0, 129)]
    ) == []


def test_choose_pose_prefers_attitude_consistency_over_lower_rmse():
    expected = np.eye(3)
    continuous = cv2.Rodrigues(np.radians([1.0, 0.0, 0.0]))[0]
    flipped = cv2.Rodrigues(np.radians([0.0, 62.0, 0.0]))[0]
    good = Pose(
        np.zeros(3), continuous, (0.0, 0.0, 0.0), 8, 1.2, "centered", [128, 129],
        rotation_residual_deg(continuous, expected), "imu_relative",
    )
    lower_rmse_but_flipped = Pose(
        np.zeros(3), flipped, (0.0, 0.0, 0.0), 12, 0.3, "edge", [128, 129, 130],
        rotation_residual_deg(flipped, expected), "imu_relative",
    )
    assert choose_pose([lower_rmse_but_flipped, good]) is good


def test_constrained_translation_is_used_only_when_all_visual_views_fail_gate():
    consistent = Pose(
        np.zeros(3), np.eye(3), (0.0, 0.0, 0.0), 8, 1.1, "good", [128, 129],
        4.0, "imu_relative", 4.0,
    )
    flipped = Pose(
        np.zeros(3), np.eye(3), (0.0, 0.0, 0.0), 8, 0.3, "bad", [128, 129],
        61.0, "imu_relative", 61.0,
    )
    constrained = Pose(
        np.zeros(3), np.eye(3), (0.0, 0.0, 0.0), 8, 0.5, "fixed", [128, 129],
        0.0, "imu_constrained_visual_translation", 61.0,
    )
    assert choose_pose([flipped, constrained, consistent], 30.0) is consistent
    assert choose_pose([flipped, constrained], 30.0) is constrained


def test_single_tag_hybrid_never_replaces_authoritative_multitag_anchor():
    rotation = np.eye(3)
    authoritative = Pose(
        np.array([1.0, 2.0, 3.0]), rotation, (0.0, 0.0, 0.0),
        8, 1.0, "multi", [128, 129], 2.0, "imu_relative",
    )
    single_tag = Pose(
        np.array([9.0, 9.0, 9.0]), rotation, (0.0, 0.0, 0.0),
        4, 1.0, "single", [128], 0.0,
        "imu_constrained_single_tag_translation",
    )
    multitag_fixed = Pose(
        np.array([4.0, 5.0, 6.0]), rotation, (0.0, 0.0, 0.0),
        8, 4.0, "multi_fixed", [128, 129], 0.0,
        "imu_constrained_visual_translation",
    )
    anchor = authoritative
    if pose_updates_authoritative_anchor(single_tag):
        anchor = single_tag
    assert anchor is authoritative
    assert pose_updates_authoritative_anchor(authoritative)
    assert pose_updates_authoritative_anchor(multitag_fixed)


def test_solve_view_uses_attitude_prior_to_select_ippe_branch():
    grid = Grid(2, 2, 0.2, 0.25, 0, "row-major")
    view = View("front", 0.0, 0.0, 90.0)
    size = 1000
    intrinsics = perspective_intrinsics(size, view.fov)
    camera_position = np.array([0.18, -0.15, -1.1])
    camera_to_board = cv2.Rodrigues(np.array([0.18, -0.22, 0.08]))[0]
    board_to_camera = camera_to_board.T
    tvec = -board_to_camera @ camera_position
    detections = []
    for tag_id in range(4):
        corners = grid.corners(tag_id).astype(np.float32)
        pixels, _ = cv2.projectPoints(
            corners, cv2.Rodrigues(board_to_camera)[0], tvec, intrinsics, None,
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

    # Recover IPPE's deliberately worse mirror seed in the same local plane
    # convention as solve_view, then use it as the temporal attitude prior.
    object_points = np.concatenate(
        [detection["object_corners"] for detection in detections]
    ).astype(np.float64)
    image_points = np.concatenate(
        [detection["corners_px"] for detection in detections]
    ).astype(np.float64)
    plane_origin = object_points.mean(axis=0)
    _u, _singular, plane_axes = np.linalg.svd(
        object_points - plane_origin, full_matrices=False,
    )
    plane_basis = np.column_stack(
        (plane_axes[0], plane_axes[1], np.cross(plane_axes[0], plane_axes[1]))
    )
    local_points = (object_points - plane_origin) @ plane_basis
    local_points[:, 2] = 0.0
    ok, local_rvecs, local_tvecs, _errors = cv2.solvePnPGeneric(
        local_points, image_points, intrinsics, None, flags=cv2.SOLVEPNP_IPPE,
    )
    assert ok and len(local_rvecs) == 2
    mirror_to_view = cv2.Rodrigues(local_rvecs[1])[0] @ plane_basis.T
    mirror_tvec = (
        np.asarray(local_tvecs[1]).reshape(3)
        - mirror_to_view @ plane_origin
    )
    mirror_expected = pose_view_to_panorama(
        cv2.Rodrigues(mirror_to_view)[0], mirror_tvec, view,
    )[1]

    pose = solve_view(
        detections, view, size, min_tags=2, max_rmse_px=10.0,
        pnp_points="corners", pnp_solver="ippe",
        expected_rotation_camera_to_board=mirror_expected,
        attitude_source="imu_relative",
    )
    assert pose is not None
    assert pose.attitude_source == "imu_relative"
    assert pose.attitude_residual_deg == pytest.approx(0.0, abs=1e-3)
    # The mirror has a much worse fit; selecting it proves the attitude prior,
    # not RMSE ordering, chose the IPPE branch.
    assert pose.rmse > 1.0


def test_fixed_attitude_translation_recovers_camera_center():
    grid = Grid(2, 2, 0.2, 0.25, 0, "row-major")
    size = 1000
    camera_position = np.array([0.18, -0.15, -1.1])
    camera_to_board = cv2.Rodrigues(np.array([0.18, -0.22, 0.08]))[0]
    board_to_panorama = camera_to_board.T
    panorama_tvec = -board_to_panorama @ camera_position
    object_points = np.concatenate(
        [grid.corners(tag_id) for tag_id in range(4)]
    ).astype(np.float64)
    for view in (
        View("front", 0.0, 0.0, 90.0),
        View("offset", 12.0, -5.0, 90.0, 3.0),
    ):
        intrinsics = perspective_intrinsics(size, view.fov)
        view_to_panorama = view_to_panorama_rotation(
            view.yaw, view.pitch, view.roll,
        )
        board_to_view = view_to_panorama.T @ board_to_panorama
        view_tvec = view_to_panorama.T @ panorama_tvec
        image_points, _ = cv2.projectPoints(
            object_points,
            cv2.Rodrigues(board_to_view)[0],
            view_tvec,
            intrinsics,
            None,
        )
        solved = solve_fixed_attitude_translation(
            object_points,
            image_points.reshape(-1, 2),
            view,
            size,
            camera_to_board,
            max_rmse_px=0.1,
        )
        assert solved is not None
        recovered_position, rmse = solved
        np.testing.assert_allclose(recovered_position, camera_position, atol=1e-8)
        assert rmse < 1e-6


def test_wrong_visual_rotation_uses_imu_attitude_and_visual_translation(monkeypatch):
    grid = Grid(2, 2, 0.2, 0.25, 0, "row-major")
    view = View("front", 0.0, 0.0, 90.0)
    size = 1000
    intrinsics = perspective_intrinsics(size, view.fov)
    camera_position = np.array([0.18, -0.15, -1.1])
    camera_to_board = cv2.Rodrigues(np.array([0.18, -0.22, 0.08]))[0]
    board_to_camera = camera_to_board.T
    true_tvec = -board_to_camera @ camera_position
    detections = []
    for tag_id in range(4):
        corners = grid.corners(tag_id).astype(np.float32)
        pixels, _ = cv2.projectPoints(
            corners,
            cv2.Rodrigues(board_to_camera)[0],
            true_tvec,
            intrinsics,
            None,
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

    wrong_camera_to_board = (
        cv2.Rodrigues(np.radians([0.0, 60.0, 0.0]))[0] @ camera_to_board
    )
    wrong_board_to_camera = wrong_camera_to_board.T
    wrong_rvec = cv2.Rodrigues(wrong_board_to_camera)[0]

    def fake_ransac(object_points, _image_points, _k, _distortion, **_kwargs):
        inliers = np.arange(len(object_points), dtype=np.int32).reshape(-1, 1)
        return True, wrong_rvec.copy(), true_tvec.reshape(3, 1).copy(), inliers

    def preserve_wrong_seed(
        _object_points, _image_points, _k, _distortion, rvec, tvec,
    ):
        return rvec.copy(), tvec.copy()

    monkeypatch.setattr(cv2, "solvePnPRansac", fake_ransac)
    monkeypatch.setattr(cv2, "solvePnPRefineLM", preserve_wrong_seed)
    pose = solve_view(
        detections,
        view,
        size,
        min_tags=2,
        max_rmse_px=0.1,
        pnp_points="corners",
        pnp_solver="iterative",
        expected_rotation_camera_to_board=camera_to_board,
        attitude_source="imu_relative",
        max_attitude_residual_deg=30.0,
        allow_imu_translation_fallback=True,
    )
    assert pose is not None
    assert pose.attitude_source == "imu_constrained_visual_translation"
    assert pose.attitude_residual_deg == pytest.approx(0.0, abs=1e-9)
    assert pose.visual_attitude_residual_deg == pytest.approx(60.0, abs=1e-6)
    np.testing.assert_allclose(pose.rotation_camera_to_board, camera_to_board, atol=1e-12)
    np.testing.assert_allclose(pose.xyz, camera_position, atol=1e-7)
    assert pose.rmse < 1e-4


def test_fixed_attitude_translation_recovers_when_visual_pnp_has_no_candidate(
    monkeypatch,
):
    grid = Grid(1, 2, 0.2, 0.25, 0, "row-major")
    view = View("front", 0.0, 0.0, 90.0)
    size = 1000
    intrinsics = perspective_intrinsics(size, view.fov)
    camera_position = np.array([0.18, -0.15, -1.1])
    camera_to_board = cv2.Rodrigues(np.array([0.18, -0.22, 0.08]))[0]
    board_to_camera = camera_to_board.T
    tvec = -board_to_camera @ camera_position
    detections = []
    for tag_id in range(2):
        corners = grid.corners(tag_id).astype(np.float32)
        pixels, _ = cv2.projectPoints(
            corners,
            cv2.Rodrigues(board_to_camera)[0],
            tvec,
            intrinsics,
            None,
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

    monkeypatch.setattr(
        cv2,
        "solvePnPRansac",
        lambda *_args, **_kwargs: (False, None, None, None),
    )
    diagnostics = {}
    pose = solve_view(
        detections,
        view,
        size,
        min_tags=2,
        max_rmse_px=0.1,
        pnp_points="corners",
        pnp_solver="iterative",
        expected_rotation_camera_to_board=camera_to_board,
        attitude_source="imu_relative",
        max_attitude_residual_deg=30.0,
        allow_imu_translation_fallback=True,
        max_imu_translation_rmse_px=0.1,
        diagnostics=diagnostics,
    )
    assert pose is not None
    assert pose.attitude_source == "imu_constrained_visual_translation"
    assert pose.visual_attitude_residual_deg is None
    np.testing.assert_allclose(pose.rotation_camera_to_board, camera_to_board, atol=1e-12)
    np.testing.assert_allclose(pose.xyz, camera_position, atol=1e-7)
    assert diagnostics["visual_pose_failure_reason"] == "iterative_ransac_failed"
    assert diagnostics["imu_translation_fallback_without_visual_candidate"] is True


def test_inverse_relative_recovery_requires_direct_corner_reprojection():
    grid = Grid(2, 2, 0.2, 0.25, 0, "row-major")
    view = View("front", 0.0, 0.0, 90.0)
    size = 1000
    intrinsics = perspective_intrinsics(size, view.fov)
    camera_position = np.array([0.18, -0.15, -1.1])
    recovered_rotation = cv2.Rodrigues(np.array([0.18, -0.22, 0.08]))[0]
    nominal_rotation = (
        cv2.Rodrigues(np.radians([0.0, 60.0, 0.0]))[0]
        @ recovered_rotation
    )
    world_to_camera = recovered_rotation.T
    tvec = -world_to_camera @ camera_position
    detections = []
    for tag_id in range(4):
        corners = grid.corners(tag_id).astype(np.float32)
        pixels, _ = cv2.projectPoints(
            corners,
            cv2.Rodrigues(world_to_camera)[0],
            tvec,
            intrinsics,
            None,
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

    diagnostics = {}
    pose = solve_view(
        detections,
        view,
        size,
        min_tags=2,
        max_rmse_px=0.1,
        pnp_points="corners",
        pnp_solver="ippe",
        expected_rotation_camera_to_board=nominal_rotation,
        attitude_source="imu_relative",
        max_attitude_residual_deg=30.0,
        allow_imu_translation_fallback=True,
        imu_translation_fallback_rotations=(
            ("inverse_relative_recovery", recovered_rotation),
        ),
        max_imu_translation_rmse_px=0.1,
        diagnostics=diagnostics,
    )
    assert pose is not None
    assert pose.attitude_source == "imu_constrained_visual_translation"
    assert pose.attitude_hypothesis == "inverse_relative_recovery"
    np.testing.assert_allclose(pose.rotation_camera_to_board, recovered_rotation, atol=1e-12)
    np.testing.assert_allclose(pose.xyz, camera_position, atol=1e-7)
    assert pose.visual_attitude_residual_deg == pytest.approx(0.0, abs=1e-3)
    assert diagnostics["attitude_gate_triggered"] is True
    attempts = diagnostics["imu_translation_fallback_attempts"]
    assert attempts[0]["status"] == "rmse_rejected"
    assert attempts[1]["status"] == "accepted"


def test_single_direct_tag_can_recover_translation_only_for_short_imu_gap():
    grid = Grid(1, 1, 0.2, 0.25, 0, "row-major")
    view = View("front", 0.0, 0.0, 90.0)
    size = 1000
    intrinsics = perspective_intrinsics(size, view.fov)
    camera_position = np.array([0.18, -0.15, -1.1])
    camera_to_board = cv2.Rodrigues(np.array([0.18, -0.22, 0.08]))[0]
    board_to_camera = camera_to_board.T
    tvec = -board_to_camera @ camera_position
    corners = grid.corners(0).astype(np.float32)
    pixels, _ = cv2.projectPoints(
        corners,
        cv2.Rodrigues(board_to_camera)[0],
        tvec,
        intrinsics,
        None,
    )
    pixels = pixels.reshape(4, 2).astype(np.float32)
    detection = {
        "id": 0,
        "corners_px": pixels,
        "center_px": pixels.mean(axis=0),
        "object_center": corners.mean(axis=0),
        "object_corners": corners,
        "area_px2": abs(float(cv2.contourArea(pixels))),
    }

    common = dict(
        detections=[detection],
        view=view,
        size=size,
        min_tags=2,
        max_rmse_px=0.1,
        pnp_points="corners",
        pnp_solver="ippe",
        expected_rotation_camera_to_board=camera_to_board,
        attitude_source="imu_relative",
        max_attitude_residual_deg=30.0,
    )
    assert solve_view(**common) is None
    diagnostics = {}
    pose = solve_view(
        **common,
        allow_single_tag_imu_translation_fallback=True,
        max_single_tag_imu_translation_rmse_px=0.1,
        diagnostics=diagnostics,
    )
    assert pose is not None
    assert pose.ids == [0]
    assert pose.inliers == 4
    assert pose.attitude_source == "imu_constrained_single_tag_translation"
    assert pose.visual_attitude_residual_deg is None
    np.testing.assert_allclose(pose.rotation_camera_to_board, camera_to_board, atol=1e-12)
    np.testing.assert_allclose(pose.xyz, camera_position, atol=1e-7)
    assert diagnostics["single_tag_imu_translation_attempted"] is True
    assert diagnostics["single_tag_imu_translation_attempts"][0]["status"] == "accepted"


def test_same_camera_center_from_different_perspective_views():
    camera_xyz_board = np.array([0.22, -0.14, 1.35])
    pano_to_board, _ = cv2.Rodrigues(np.array([0.08, -0.20, 0.04]))
    board_to_pano = pano_to_board.T
    t_pano = -board_to_pano @ camera_xyz_board.reshape(3, 1)
    recovered = []
    for view in (View("front", 0, 0), View("right", 70, -10), View("left", -55, 20)):
        v2p = view_to_panorama_rotation(view.yaw, view.pitch)
        board_to_view = v2p.T @ board_to_pano
        t_view = v2p.T @ t_pano
        rvec, _ = cv2.Rodrigues(board_to_view)
        xyz, rotation = pose_view_to_panorama(rvec, t_view, view)
        recovered.append(xyz)
        np.testing.assert_allclose(rotation, pano_to_board, atol=1e-9)
    for xyz in recovered:
        np.testing.assert_allclose(xyz, camera_xyz_board, atol=1e-9)


def _reanchor_pose(
    xyz: list[float], angle_deg: float, rmse: float, view: str,
) -> Pose:
    rotation = cv2.Rodrigues(np.radians([angle_deg, 0.0, 0.0]))[0]
    return Pose(
        np.asarray(xyz, dtype=float),
        rotation,
        (angle_deg, 0.0, 0.0),
        12,
        rmse,
        view,
        [128, 129, 130],
        110.0,
        "imu_relative",
        110.0,
    )


def test_guarded_visual_reanchor_requires_two_agreeing_direct_views():
    first = _reanchor_pose([0.20, 0.00, 0.00], 30.0, 1.4, "wide")
    second = _reanchor_pose([0.21, 0.01, 0.00], 31.0, 1.1, "tracked")
    sources = {id(first): "direct", id(second): "direct"}
    views = {
        id(first): View("wide", 0.0, 0.0, 110.0),
        id(second): View("tracked", 30.0, 20.0, 70.0),
    }
    decision = choose_guarded_visual_reanchor(
        [first, second],
        sources,
        views,
        np.zeros(3),
        np.eye(3),
        0.5,
        min_tags=2,
        imu_gate_deg=30.0,
        max_rmse_px=2.0,
        max_speed_m_s=2.0,
        max_gap_s=1.5,
        max_position_spread_m=0.08,
        max_attitude_spread_deg=8.0,
        max_angular_speed_deg_s=240.0,
    )
    assert decision is not None
    assert decision.pose.view == "tracked"
    assert decision.pose.attitude_source == "visual_multitag_reanchor"
    assert decision.pose.attitude_hypothesis == "guarded_multiview"
    assert decision.supporting_views == ("wide", "tracked")
    assert decision.position_spread_m == pytest.approx(np.sqrt(2) * 0.01)
    assert decision.attitude_spread_deg == pytest.approx(1.0, abs=1e-9)
    assert decision.speed_m_s < 0.5
    assert decision.angular_speed_deg_s == pytest.approx(62.0, abs=1e-8)


def test_guarded_visual_reanchor_allows_short_three_tag_temporal_branch():
    candidate = _reanchor_pose([0.08, 0.00, 0.00], 20.0, 1.2, "tracked")
    decision = choose_guarded_visual_reanchor(
        [candidate],
        {id(candidate): "direct"},
        {id(candidate): View("tracked", 30.0, 20.0, 70.0)},
        np.zeros(3),
        np.eye(3),
        0.2,
        min_tags=2,
        imu_gate_deg=30.0,
        max_rmse_px=2.0,
        max_speed_m_s=2.0,
        max_gap_s=1.5,
        max_position_spread_m=0.08,
        max_attitude_spread_deg=8.0,
        max_angular_speed_deg_s=240.0,
    )
    assert decision is not None
    assert decision.supporting_views == ("tracked",)
    assert decision.pose.attitude_hypothesis == "guarded_temporal"
    assert decision.angular_speed_deg_s == pytest.approx(100.0, abs=1e-8)


@pytest.mark.parametrize(
    "mutation",
    ("one_view", "optical_flow", "position_flip", "attitude_flip", "too_fast"),
)
def test_guarded_visual_reanchor_rejects_ambiguous_or_discontinuous_pose(mutation):
    first = _reanchor_pose([0.20, 0.00, 0.00], 30.0, 1.4, "wide")
    second = _reanchor_pose([0.21, 0.01, 0.00], 31.0, 1.1, "tracked")
    candidates = [first, second]
    sources = {id(first): "direct", id(second): "direct"}
    views = {
        id(first): View("wide", 0.0, 0.0, 110.0),
        id(second): View("tracked", 30.0, 20.0, 70.0),
    }
    max_speed = 2.0
    if mutation == "one_view":
        candidates = [first]
        first.ids = [128, 129]
    elif mutation == "optical_flow":
        sources[id(first)] = "optical_flow"
        sources[id(second)] = "optical_flow"
    elif mutation == "position_flip":
        second.xyz = np.array([0.50, 0.30, 0.0])
    elif mutation == "attitude_flip":
        second.rotation_camera_to_board = cv2.Rodrigues(
            np.radians([100.0, 0.0, 0.0])
        )[0]
    elif mutation == "too_fast":
        max_speed = 0.1
    decision = choose_guarded_visual_reanchor(
        candidates,
        sources,
        views,
        np.zeros(3),
        np.eye(3),
        0.5,
        min_tags=2,
        imu_gate_deg=30.0,
        max_rmse_px=2.0,
        max_speed_m_s=max_speed,
        max_gap_s=1.5,
        max_position_spread_m=0.08,
        max_attitude_spread_deg=8.0,
        max_angular_speed_deg_s=240.0,
    )
    assert decision is None
