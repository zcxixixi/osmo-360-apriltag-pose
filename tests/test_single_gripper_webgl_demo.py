from pathlib import Path

from osmo360.visualization.render_single_gripper_webgl_demo import (
    force_measurement_available,
    single_capture_id,
)


def test_single_capture_id_comes_from_source_insv_name():
    source = Path("/captures/VID_20260901_155528_00_003.insv")

    assert single_capture_id(source, 30.0) == "VID_20260901_155528_00_003-single-30fps"


def test_single_capture_id_preserves_non_integer_frame_rate():
    source = Path("capture.insv")

    assert single_capture_id(source, 59.94) == "capture-single-59.94fps"


def test_one_sided_low_confidence_force_remains_available():
    assert force_measurement_available(
        True, 42.0, "MEASURED_ONE_SIDED_LEFT_LOW_CONFIDENCE"
    )
    assert not force_measurement_available(
        True, float("nan"), "MEASURED_ONE_SIDED_LEFT_LOW_CONFIDENCE"
    )
    assert not force_measurement_available(
        False, 42.0, "MEASURED_ONE_SIDED_LEFT_LOW_CONFIDENCE"
    )
