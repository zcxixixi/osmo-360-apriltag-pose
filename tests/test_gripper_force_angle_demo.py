import cv2
import numpy as np

from render_gripper_force_angle_demo import (
    JawFrame,
    bounded_interpolate,
    local_to_point,
    normalize_contact_intensity,
    observe_frame,
    point_to_local,
)


def test_jaw_local_coordinates_round_trip_marker_scale():
    frame = JawFrame(
        origin=np.array([100.0, 80.0]),
        axis=np.array([0.6, -0.8]),
        normal=np.array([0.8, 0.6]),
        scale_px=50.0,
    )
    coordinate = np.array([0.75, -0.20])

    point = local_to_point(coordinate, frame)

    np.testing.assert_allclose(point_to_local(point, frame), coordinate, atol=1e-12)


def test_bounded_interpolation_recovers_only_short_gaps():
    values = np.array([0.0, np.nan, 2.0, np.nan, np.nan, np.nan, 8.0])

    result, recovered = bounded_interpolate(values, maximum_gap=1)

    np.testing.assert_allclose(result[:3], [0.0, 1.0, 2.0])
    assert recovered.tolist() == [False, True, False, False, False, False, False]
    assert np.isnan(result[3:6]).all()


def test_contact_intensity_removes_unloaded_opening_angle_coupling():
    opening = np.repeat(np.arange(11.0), 20)
    raw = 1.0 + 0.5 * opening + np.tile(np.linspace(0.0, 0.09, 20), 11)
    contact = np.zeros(len(raw), dtype=bool)
    contact[18::20] = True
    contact[19::20] = True
    raw[contact] += 5.0
    valid = np.arange(len(raw))

    intensity, _, _ = normalize_contact_intensity(raw, opening, valid)

    assert np.percentile(intensity[~contact], 90) < 5.0
    assert np.median(intensity[contact]) > 90.0
    assert abs(np.corrcoef(opening[~contact], intensity[~contact])[0, 1]) < 0.2


def test_x5_profile_detects_jaw_axes_and_pad_dots():
    image = np.zeros((1920, 1920, 3), dtype=np.uint8)
    yellow = (0, 255, 255)
    cv2.fillConvexPoly(
        image,
        np.array([[840, 1620], [910, 1200], [950, 1210], [890, 1640]], np.int32),
        yellow,
    )
    cv2.fillConvexPoly(
        image,
        np.array([[1080, 1620], [1010, 1200], [970, 1210], [1030, 1640]], np.int32),
        yellow,
    )
    cv2.ellipse(image, (925, 1280), (8, 6), 0, 0, 360, (20, 20, 20), -1)
    cv2.ellipse(image, (995, 1280), (8, 6), 0, 0, 360, (20, 20, 20), -1)

    observation = observe_frame(image, "insta360-x5-front")

    assert np.isfinite(observation.included_angle_deg)
    assert 5.0 <= observation.included_angle_deg <= 50.0
    assert observation.dot_left is not None
    assert observation.dot_right is not None
