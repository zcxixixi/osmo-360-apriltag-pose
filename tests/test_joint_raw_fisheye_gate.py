import json
from pathlib import Path

import pytest

import numpy as np

from tools.joint_dual_camera_pose_graph_cached import (
    FrameData,
    anchored_cross_direction,
    basetag_pose_matches_expected,
    evaluate_anchored_cross_holdout,
    direct_map,
    load_initial_wall_transform,
    Transform,
    nearest_detection_frame,
    raw_fisheye_cache_audit,
    select_expected_basetag_detection,
    solve_camera_wall_only,
    solve_frames_temporally,
)
from scipy.spatial.transform import Rotation
from tools.joint_camera_correction_cached import nearest_detection_frame as nearest_correction_detection_frame
from tools.calibrate_basetag_reciprocal_cached import (
    load_both_wall_support_times,
    timestamp_supported,
)
from osmo360.localization.raw_fisheye_world_pose import (
    make_x5_offset_ray_converter,
    make_x5_rectified_maps,
)


def write_cache_sidecar(tmp_path: Path, size=(3840, 3840), stream=1):
    video = tmp_path / "stream1.mp4"
    calibration = tmp_path / "calibration.json"
    cache = tmp_path / "observations.npz"
    video.write_bytes(b"raw")
    calibration.write_text("{}")
    cache.write_bytes(b"cache")
    cache.with_suffix(".json").write_text(json.dumps({
        "schema_version": "fisheye-apriltag-observation-cache/1.0",
        "video": str(video),
        "calibration": str(calibration),
        "camera_serial": "SERIAL",
        "stream": stream,
        "source_size": list(size),
        "radial_model": "factory-polynomial",
    }))
    return cache


def test_dual_x5_raw_cache_requires_both_traceable_lens_tracks(tmp_path):
    videos = [tmp_path / "lens0.mp4", tmp_path / "lens1.mp4"]
    for video in videos:
        video.write_bytes(b"raw")
    cache = tmp_path / "dual.npz"
    cache.write_bytes(b"cache")
    cache.with_suffix(".json").write_text(json.dumps({
        "schema_version": "fisheye-apriltag-observation-cache/1.2-dual-lens",
        "camera_serial": "X5-SERIAL",
        "streams": [0, 1],
        "source_videos": list(map(str, videos)),
        "source_size": [2880, 2880],
        "calibration": "embedded_x5_offset",
        "calibration_sha256": ["sha"],
        "x5_offset": "m2_offset",
    }))

    audit = raw_fisheye_cache_audit(cache)

    assert audit["stream"] == [0, 1]
    assert audit["stitching_used"] is False


def test_a3_panel_transform_and_120_mm_map_are_accepted(tmp_path):
    transform_path = tmp_path / "session-map.json"
    transform_path.write_text(json.dumps({
        "panel_transform": {
            "translation_m": [0.56, 0.0, 0.0],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
    }))
    panel_path = tmp_path / "panel.json"
    panel_path.write_text(json.dumps({
        "tag_outer_size_m": 0.12,
        "tags": [{"id": 200, "corners_m": [[0, 0, 0], [0.12, 0, 0], [0.12, 0.12, 0], [0, 0.12, 0]]}],
    }))

    transform = load_initial_wall_transform(transform_path)
    points = direct_map(panel_path)

    assert transform.p == pytest.approx([0.56, 0.0, 0.0])
    assert points[200].shape == (4, 3)


def test_x5_embedded_offset_maps_both_lens_centres_into_one_rig_frame():
    offset = "m2_100_100_100_0_0_90_100_300_100_0_0_90_400_200_1"
    front, front_metadata = make_x5_offset_ray_converter(
        offset, stream=0, source_width=200, source_height=200
    )
    back, back_metadata = make_x5_offset_ray_converter(
        offset, stream=1, source_width=200, source_height=200
    )

    assert front([[100, 100]])[0] == pytest.approx([0, 0, 1])
    assert back([[100, 100]])[0] == pytest.approx([0, 0, -1])
    assert front_metadata["ray_frame"] == back_metadata["ray_frame"]


def test_x5_rectified_maps_stay_on_the_raw_lens_image():
    offset = "m2_100_100_100_0_0_90_100_300_100_0_0_90_400_200_1"
    maps = make_x5_rectified_maps(
        offset, stream=0, source_width=200, source_height=200, view_size=20
    )

    assert len(maps) == 11
    xmap, ymap = maps[0]
    assert xmap.shape == (20, 20)
    assert ymap.shape == (20, 20)
    assert xmap[9:11, 9:11].mean() == pytest.approx(100.0, abs=5.0)
    assert ymap[9:11, 9:11].mean() == pytest.approx(100.0, abs=5.0)


def test_raw_square_fisheye_is_accepted(tmp_path):
    audit = raw_fisheye_cache_audit(write_cache_sidecar(tmp_path))
    assert audit["measurement_input"] == "raw_fisheye"
    assert audit["stitching_used"] is False
    assert audit["synthetic_frames_used"] is False


@pytest.mark.parametrize("size,stream", [((3840, 1920), 1), ((3840, 3840), 0)])
def test_panorama_or_wrong_stream_is_rejected(tmp_path, size, stream):
    with pytest.raises(ValueError):
        raw_fisheye_cache_audit(write_cache_sidecar(tmp_path, size=size, stream=stream))


def test_pairing_snaps_to_an_actual_synchronized_detection_frame():
    frames = np.asarray([0, 3, 6, 9])
    times = np.asarray([-0.038, 0.062, 0.162, 0.262])
    assert nearest_detection_frame(frames, times, 0.100) == 3
    assert nearest_detection_frame(frames, times, 0.220) == 9
    assert nearest_detection_frame(frames, times, 0.400) is None
    assert nearest_correction_detection_frame(frames, times, 0.100) is None
    assert nearest_correction_detection_frame(frames, times, 0.068) == 3


def test_two_perpendicular_walls_reject_planar_mirror_seed():
    true_camera = Transform(
        np.asarray([0.35, -0.08, 0.42]),
        Rotation.from_euler("xyz", [8, -15, 25], degrees=True),
    )
    wall = Transform(
        np.asarray([0.50, 0.0, 0.50]),
        Rotation.from_euler("y", 90, degrees=True),
    )
    left_points = np.asarray([
        [-0.20, -0.20, 0], [0.20, -0.20, 0],
        [0.20, 0.20, 0], [-0.20, 0.20, 0],
    ], dtype=float)
    right_local = left_points.copy()
    right_world = wall.r.apply(right_local) + wall.p

    def rays(points):
        value = true_camera.r.inv().apply(points - true_camera.p)
        return value / np.linalg.norm(value, axis=1, keepdims=True)

    mirrored_seed = Transform(
        np.asarray([-0.35, -0.08, -0.42]),
        Rotation.from_euler("xyz", [170, 15, 25], degrees=True),
    )
    solved = solve_camera_wall_only(
        mirrored_seed,
        [(left_points, rays(left_points))],
        [(right_local, rays(right_world))],
        wall,
    )
    assert np.linalg.norm(solved.p - true_camera.p) < 1e-6
    assert (solved.r.inv() * true_camera.r).magnitude() < 1e-6


def test_single_wall_uses_independent_prior_only_to_choose_ippe_branch():
    true_camera = Transform(
        np.asarray([0.15, -0.05, 0.60]),
        Rotation.from_euler("xyz", [5, 12, -20], degrees=True),
    )
    points = np.asarray([
        [-0.30, -0.20, 0], [0.30, -0.20, 0],
        [0.30, 0.20, 0], [-0.30, 0.20, 0],
        [-0.10, -0.05, 0], [0.10, -0.05, 0],
        [0.10, 0.05, 0], [-0.10, 0.05, 0],
    ], dtype=float)
    rays = true_camera.r.inv().apply(points - true_camera.p)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    solved = solve_camera_wall_only(
        true_camera,
        [(points, rays)],
        [],
        Transform(np.zeros(3), Rotation.identity()),
    )
    assert np.linalg.norm(solved.p - true_camera.p) < 1e-6
    assert (solved.r.inv() * true_camera.r).magnitude() < 1e-6


def test_cross_basetag_rejects_a_same_id_screen_distractor():
    class Cache(dict):
        pass

    cache = Cache(
        tag_id=np.asarray([2, 2]),
        rays_camera=np.asarray([
            [[0.0, 0.0, 1.0]] * 4,
            [[1.0, 0.0, 0.0]] * 4,
        ]),
    )
    expected = Transform(np.asarray([0.0, 0.0, 0.25]), Rotation.identity())
    selected, audit = select_expected_basetag_detection(cache, [0, 1], 2, expected)
    assert selected == 0
    assert audit["candidate_count"] == 2
    assert audit["accepted"] is True
    assert audit["second_center_error_deg"] == pytest.approx(90.0)


def test_cross_basetag_fails_closed_when_only_distractor_is_visible():
    cache = {
        "tag_id": np.asarray([2]),
        "rays_camera": np.asarray([[[1.0, 0.0, 0.0]] * 4]),
    }
    expected = Transform(np.asarray([0.0, 0.0, 0.25]), Rotation.identity())
    selected, audit = select_expected_basetag_detection(
        cache, [0], 2, expected, max_center_error_deg=50.0)
    assert selected is None
    assert audit["accepted"] is False


def test_cross_basetag_pose_rejects_a_screen_plane_near_the_expected_ray():
    expected = Transform(
        np.asarray([0.02, 0.01, 0.25]), Rotation.from_euler("xyz", [2, -4, 8], degrees=True))
    screen = Transform(
        np.asarray([0.03, 0.02, 0.40]), Rotation.from_euler("xyz", [2, 86, 8], degrees=True))
    accepted, audit = basetag_pose_matches_expected(screen, expected)
    assert accepted is False
    assert audit["attitude_error_deg"] == pytest.approx(90.0)


def test_cross_basetag_pose_accepts_mount_motion_with_coarse_position_prior():
    expected = Transform(
        np.asarray([0.02, 0.01, 0.25]), Rotation.from_euler("xyz", [2, -4, 8], degrees=True))
    measured = Transform(
        np.asarray([0.15, -0.03, 0.48]), Rotation.from_euler("xyz", [7, -9, 19], degrees=True))
    accepted, audit = basetag_pose_matches_expected(measured, expected)
    assert accepted is True
    assert audit["position_difference_m"] > 0.20


def test_stronger_two_wall_camera_defines_factor_and_opposite_holdout():
    identity = Transform(np.zeros(3), Rotation.identity())
    left = Transform(np.asarray([0.10, 0.00, 0.30]), Rotation.identity())
    right = Transform(np.asarray([-0.10, 0.00, 0.30]), Rotation.identity())
    own_left = Transform(np.asarray([0.00, 0.00, 0.05]), Rotation.identity())
    own_right = Transform(np.asarray([0.00, 0.00, 0.05]), Rotation.identity())
    dummy = (np.zeros((4, 3)), np.zeros((4, 3)))
    cross_lr = left.inverse().compose(right.compose(own_right))
    cross_rl = right.inverse().compose(left.compose(own_left))
    frame = FrameData(
        0.0, 0, left, right,
        [dummy], [],
        [dummy], [dummy],
        np.zeros((4, 3)), np.zeros((4, 3)), cross_lr, cross_rl,
    )
    assert anchored_cross_direction(frame) == "rl"
    audit = evaluate_anchored_cross_holdout(
        [frame], {0.0: (left, right)}, own_left, own_right)
    assert audit["used_factor_frames"] == {"lr": 0, "rl": 1}
    assert audit["holdout_frames"] == {"lr": 1, "rl": 0}
    assert audit["position_error_mm"]["max"] == pytest.approx(0.0)
    assert audit["rotation_error_deg"]["max"] == pytest.approx(0.0)


def test_temporal_holdout_uses_previous_pose_but_not_heldout_cross(monkeypatch):
    identity = Transform(np.zeros(3), Rotation.identity())
    initial_second = Transform(np.asarray([9.0, 0.0, 0.0]), Rotation.identity())
    dummy = (np.zeros((4, 3)), np.zeros((4, 3)))
    frames = [
        FrameData(0.00, 0, identity, identity, [dummy], [], [dummy], [],
                  None, None, None, None),
        FrameData(0.05, 1, initial_second, initial_second, [dummy], [], [dummy], [],
                  np.zeros((4, 3)), np.zeros((4, 3)), identity, identity),
    ]
    calls = []

    def fake_solve(frame, seed, _wall, _own_left, _own_right, _tag_points,
                   include_cross):
        calls.append((frame.time_s, seed[0].p.copy(), include_cross))
        solved = Transform(np.asarray([frame.time_s + 1.0, 0.0, 0.0]), Rotation.identity())
        return solved, solved

    monkeypatch.setattr(
        "tools.joint_dual_camera_pose_graph_cached.solve_frame", fake_solve)
    solve_frames_temporally(
        frames,
        {0.0: (identity, identity), 0.05: (initial_second, initial_second)},
        identity, identity, identity, np.zeros((4, 3)), {0.0},
    )
    assert calls[0][2] is True
    assert calls[1][2] is False
    assert calls[1][1].tolist() == pytest.approx([1.0, 0.0, 0.0])


def test_reciprocal_calibration_keeps_only_direct_two_wall_pose_rows(tmp_path):
    pose = tmp_path / "pose.csv"
    pose.write_text(
        "timestamp,quality_status,detected_ids\n"
        "0.000,valid,134 135\n"
        "0.033,valid,128 129\n"
        "0.067,invalid,128 134\n"
        "0.100,valid,128 134\n",
        encoding="utf-8",
    )
    times = load_both_wall_support_times(
        pose, {134, 135, 136, 137}, {128, 129, 130, 131, 132, 133})
    assert times.tolist() == pytest.approx([0.100])
    assert timestamp_supported(times, 0.115)
    assert not timestamp_supported(times, 0.121)
