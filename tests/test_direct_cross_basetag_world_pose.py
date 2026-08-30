import numpy as np
from scipy.spatial.transform import Rotation

from calibrate_basetag_reciprocal import Transform
from tools.direct_cross_basetag_world_pose_cached import _continuous_ippe, base_from_observer
from vla_dataset_export import load_pose_csv


def test_direct_cross_chain_recovers_base_without_own_camera_extrinsic():
    world_camera = Transform(
        np.asarray([0.4, -0.2, 0.8]),
        Rotation.from_euler("xyz", [10, -20, 30], degrees=True),
    )
    camera_tag = Transform(
        np.asarray([0.1, 0.02, 0.3]),
        Rotation.from_euler("z", 70, degrees=True),
    )
    base_to_tag = Transform(
        np.asarray([0.02625, 0.0, 0.0196]), Rotation.identity()
    )
    expected = world_camera.compose(camera_tag).compose(base_to_tag.inverse())
    actual = base_from_observer(world_camera, camera_tag, base_to_tag)
    np.testing.assert_allclose(actual.p, expected.p, atol=1e-12)
    np.testing.assert_allclose(actual.r.as_matrix(), expected.r.as_matrix(), atol=1e-12)


def test_base_tag_size_and_offset_do_not_require_tcp_or_camera_mount():
    identity = Transform(np.zeros(3), Rotation.identity())
    base_to_tag = Transform(np.asarray([0.02625, 0.0, 0.0196]), Rotation.identity())
    base = base_from_observer(identity, identity, base_to_tag)
    np.testing.assert_allclose(base.p, [-0.02625, 0.0, -0.0196])


def test_exported_direct_cross_pose_is_a_trusted_world_measurement(tmp_path):
    path = tmp_path / "pose.csv"
    path.write_text(
        "timestamp,x_m,y_m,z_m,qx,qy,qz,qw,parent_frame,child_frame,"
        "measurement_source,quality_status\n"
        "0,0,0,0,0,0,0,1,tag_map,left_base_link,"
        "direct_opposite_basetag_raw_fisheye,valid\n"
        "0.1,0.01,0,0,0,0,0,1,tag_map,left_base_link,"
        "direct_opposite_basetag_raw_fisheye,valid\n",
        encoding="utf-8",
    )
    series = load_pose_csv(path)
    assert series.direct.tolist() == [True, True]
    assert series.tracked.tolist() == [True, True]
    assert series.parent_frame == "tag_map"
    assert series.child_frame == "left_base_link"


def test_ippe_branch_selection_uses_temporal_continuity_not_mount_extrinsic():
    def solution(x, yaw, rmse):
        rotation = Rotation.from_euler("z", yaw, degrees=True)
        return {
            "translation_tag_origin_in_panorama_m": np.asarray([x, 0.0, 0.3]),
            "rotation_tag_to_panorama": rotation.as_matrix(),
            "angular_rmse_deg": rmse,
        }

    previous = Transform(
        np.asarray([0.1, 0.0, 0.3]), Rotation.from_euler("z", 5, degrees=True)
    )
    # The mirror branch has fractionally better reprojection but is physically
    # discontinuous.  It must not be selected merely to reduce RMSE.
    selected = _continuous_ippe(
        [solution(-0.1, 175, 0.1), solution(0.101, 6, 0.2)], previous
    )
    np.testing.assert_allclose(selected.p, [0.101, 0.0, 0.3])
