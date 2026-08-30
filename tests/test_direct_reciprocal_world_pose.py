import numpy as np
from scipy.spatial.transform import Rotation
from types import SimpleNamespace

from calibrate_basetag_reciprocal import Transform
from tools.direct_reciprocal_world_pose_cached import choose_episode_anchor, direct_camera_pair


class Frame:
    initial_left = Transform(np.zeros(3), Rotation.identity())
    initial_right = Transform(np.zeros(3), Rotation.identity())
    left_leftwall = left_rightwall = right_leftwall = right_rightwall = []


def test_right_anchor_chain_recovers_left_camera(monkeypatch):
    world_right = Transform(np.array([1., 2., 3.]), Rotation.from_euler("z", 20, degrees=True))
    world_left = Transform(np.array([.8, 2.1, 3.]), Rotation.from_euler("z", 190, degrees=True))
    own_left = Transform(np.array([.03, 0., .02]), Rotation.identity())
    frame = Frame()
    frame.cross_rl_pose = world_right.inverse().compose(world_left.compose(own_left))
    frame.cross_lr_pose = None
    monkeypatch.setattr("tools.direct_reciprocal_world_pose_cached.solve_camera_wall_only",
                        lambda *args: world_right)
    left, right = direct_camera_pair(frame, "right", Transform(np.zeros(3), Rotation.identity()),
                                     own_left, own_left)
    assert np.allclose(left.p, world_left.p)
    assert np.allclose(left.r.as_matrix(), world_left.r.as_matrix())
    assert np.allclose(right.p, world_right.p)


def test_direct_chain_requires_same_frame_cross_measurement():
    frame = Frame(); frame.cross_rl_pose = None; frame.cross_lr_pose = None
    own = Transform(np.zeros(3), Rotation.identity())
    assert direct_camera_pair(frame, "right", own, own, own) is None


def test_anchor_is_chosen_once_from_episode_support(monkeypatch):
    frames = [SimpleNamespace(cross_lr_pose=None, cross_rl_pose=object()) for _ in range(3)]
    monkeypatch.setattr("tools.direct_reciprocal_world_pose_cached.wall_support_score",
                        lambda frame, side: (1, 4) if side else (0, 2))
    anchor, scores = choose_episode_anchor(frames)
    assert anchor == "right"
    assert scores["right"]["usable_cross_frames"] == 3
