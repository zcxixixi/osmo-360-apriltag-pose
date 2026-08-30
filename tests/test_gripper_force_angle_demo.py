import cv2
import numpy as np
import pytest

from render_gripper_force_angle_demo import (
    JawFrame,
    DotObservation,
    ForceModel,
    FrameObservation,
    apply_one_sided_opening_fallback,
    apply_fixed_relative_force,
    bounded_interpolate,
    local_to_point,
    draw_detection_overlay,
    contact_event_audit,
    labeled_contact_gap_audit,
    normalize_contact_intensity,
    opening_angles,
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


def test_labeled_contact_audit_preserves_gap_and_uses_unloaded_opening_baseline():
    opening = np.tile(np.linspace(0.0, 10.0, 20), 5)
    times = np.arange(len(opening), dtype=float) / 10.0
    contact = (times >= 4.0) & (times <= 6.0)
    gaps = 100.0 + 5.0 * opening - 8.0 * contact
    yellow = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    observations = [
        FrameObservation(
            yellow,
            yellow,
            DotObservation(np.array([0.0, 0.0]), 50.0, 0.8),
            DotObservation(np.array([gap, 0.0]), 50.0, 0.8),
            40.0,
        )
        for gap in gaps
    ]

    audit, labels, measured_gaps, residuals, supported = labeled_contact_gap_audit(
        observations, opening, 10.0, [[4.0, 6.0]]
    )

    assert audit is not None
    assert labels[39] == "UNLOADED"
    assert labels[40] == "CONTACT"
    assert labels[60] == "CONTACT"
    assert labels[61] == "UNLOADED"
    np.testing.assert_allclose(measured_gaps, gaps)
    assert np.median(residuals[contact & supported]) == pytest.approx(-8.0)
    assert audit["geometry_check"]["unloaded_model"]["slope_px_per_deg"] == pytest.approx(5.0)

    no_label_audit, no_labels, no_label_gaps, _, _ = labeled_contact_gap_audit(
        observations, opening, 10.0, []
    )
    assert no_label_audit is None
    assert set(no_labels) == {"UNLABELED"}
    np.testing.assert_allclose(no_label_gaps, gaps)

    events = contact_event_audit(opening, measured_gaps, 10.0, [4.0])
    assert events["events"][0]["nearest_frame_measured"] is True
    assert events["events"][0]["nearest_black_dot_gap_px"] == pytest.approx(gaps[40])


def test_frozen_closed_reference_does_not_renormalize_each_capture():
    observations = [
        FrameObservation(None, None, None, None, included)
        for included in np.linspace(40.0, 48.0, 40)
    ]

    opening, closed_reference = opening_angles(
        observations, closed_reference=46.75
    )

    assert closed_reference == 46.75
    assert opening[0] == pytest.approx(6.75)
    assert opening[-1] == 0.0



def test_one_sided_right_axis_is_explicit_low_confidence_measurement():
    right_axis = np.array([[1.0, -1.0], [0.5, -0.5], [0.0, 0.0]])
    observations = [
        FrameObservation(None, right_axis, None, None, np.nan),
        FrameObservation(right_axis, right_axis, None, None, 45.0),
    ]
    bilateral = np.array([np.nan, 1.75])
    hardware_angle = {
        "single_side_fallback": {
            "available_side": "right",
            "heading_center_deg": -45.0,
            "coefficients_high_to_low": [0.0, 1.0, 2.0],
            "validated_output_range_deg": [0.0, 15.0],
            "measurement_state": "MEASURED_ONE_SIDED_RIGHT_LOW_CONFIDENCE",
            "model": "quadratic_relative_axis_heading",
            "blocked_holdout": {"p95_deg": 2.0},
        }
    }

    opening, states, audit = apply_one_sided_opening_fallback(
        observations, bilateral, hardware_angle
    )

    assert opening.tolist() == pytest.approx([2.0, 1.75])
    assert states.tolist() == [
        "MEASURED_ONE_SIDED_RIGHT_LOW_CONFIDENCE",
        "MEASURED",
    ]
    assert audit["one_sided_right_frames"] == 1


def test_fixed_relative_force_rejects_free_motion_noise():
    left_dot = DotObservation(np.array([0.0, 0.0]), 50.0, 0.8)
    observations = [
        FrameObservation(
            np.ones((3, 2)),
            np.ones((3, 2)),
            left_dot,
            DotObservation(np.array([gap, 0.0]), 50.0, 0.8),
            45.0,
        )
        for gap in (103.0, 110.0, 125.0)
    ]
    revision = {
        "revision_id": "force-test",
        "quantity": "hardware_revision_relative_fingertip_force",
        "model": {
            "expected_gap_polynomial_high_to_low": [100.0],
            "opening_support_deg": [0.0, 10.0],
            "noise_floor_px": 5.0,
            "full_scale_signal_px": 10.0,
        },
    }

    force, audit = apply_fixed_relative_force(
        observations, np.array([1.0, 1.0, 1.0]), revision
    )

    assert force.tolist() == pytest.approx([0.0, 50.0, 100.0])
    assert audit["fixed_scale_across_captures"] is True
    assert audit["zero_force_frame_ratio"] == pytest.approx(1 / 3)

def test_x5_profile_detects_jaw_axes_and_pad_dots():
    image = np.full((1920, 1920, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (650, 1240), (870, 1680), (20, 20, 20), -1)
    cv2.rectangle(image, (1050, 1240), (1270, 1680), (20, 20, 20), -1)
    yellow = (0, 255, 255)
    for point in [(800, 1360), (770, 1450), (720, 1560)]:
        cv2.circle(image, point, 12, yellow, -1)
    for point in [(1120, 1360), (1150, 1450), (1200, 1560)]:
        cv2.circle(image, point, 12, yellow, -1)
    cv2.circle(image, (925, 1280), 32, yellow, -1)
    cv2.circle(image, (995, 1280), 32, yellow, -1)
    cv2.ellipse(image, (925, 1280), (8, 6), 0, 0, 360, (20, 20, 20), -1)
    cv2.ellipse(image, (995, 1280), (8, 6), 0, 0, 360, (20, 20, 20), -1)

    observation = observe_frame(
        image, "insta360-x5-front", "physical-marker-triad"
    )
    force_image = image.copy()

    assert np.isfinite(observation.included_angle_deg)
    assert 35.0 <= observation.included_angle_deg <= 80.0
    assert observation.dot_left is not None

    model = ForceModel(
        left_local=np.zeros(2),
        right_local=np.zeros(2),
        left_shape=np.ones(2),
        right_shape=np.ones(2),
        baseline=0.0,
        noise_mad=0.0,
        noise_floor=0.0,
        full_scale=1.0,
    )
    draw_detection_overlay(image, observation, model, opening_angle_deg=12.3)
    draw_detection_overlay(
        force_image,
        observation,
        model,
        opening_angle_deg=12.3,
        fingertip_force_percent=80.0,
    )

    angle_arc_region = image[1380:1570, 820:1100]
    amber_like = (
        (angle_arc_region[..., 0] < 120)
        & (angle_arc_region[..., 1] > 120)
        & (angle_arc_region[..., 2] > 180)
    )
    assert np.count_nonzero(amber_like) > 20
    assert image[80:230, 620:1380].mean() < 200.0
    force_arrow_region = force_image[1240:1320, 890:1030]
    force_amber_like = (
        (force_arrow_region[..., 0] < 120)
        & (force_arrow_region[..., 1] > 120)
        & (force_arrow_region[..., 2] > 180)
    )
    assert np.count_nonzero(force_amber_like) > 20
    assert observation.dot_right is not None


def test_x5_adaptive_dot_selection_handles_vertical_image_shift():
    image = np.full((1920, 1920, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (620, 940), (880, 1450), (20, 20, 20), -1)
    cv2.rectangle(image, (1040, 940), (1300, 1450), (20, 20, 20), -1)
    yellow = (0, 255, 255)
    for point in [(800, 1050), (760, 1140), (700, 1280)]:
        cv2.circle(image, point, 12, yellow, -1)
    for point in [(1120, 1050), (1160, 1140), (1220, 1280)]:
        cv2.circle(image, point, 12, yellow, -1)

    fixed = observe_frame(
        image, "insta360-x5-front", "physical-marker-triad"
    )
    adaptive = observe_frame(
        image,
        "insta360-x5-front",
        "physical-marker-triad",
        (25.0, 80.0),
        "adaptive-black-pad",
    )

    assert not np.isfinite(fixed.included_angle_deg)
    assert np.isfinite(adaptive.included_angle_deg)
    assert adaptive.yellow_left is not None
    assert adaptive.yellow_right is not None


def test_x5_accepted_pca_mode_tracks_yellow_jaw_contours():
    image = np.full((1920, 1920, 3), 255, dtype=np.uint8)
    yellow = (0, 255, 255)
    for center, angle in (((760, 1450), 15.0), ((1160, 1450), -15.0)):
        box = cv2.boxPoints((center, (80, 400), angle))
        cv2.fillConvexPoly(image, np.round(box).astype(np.int32), yellow)

    observation = observe_frame(image, "insta360-x5-front", "pca-axis")

    assert observation.yellow_left is not None
    assert observation.yellow_right is not None
    assert observation.included_angle_deg == pytest.approx(30.0, abs=1.0)
