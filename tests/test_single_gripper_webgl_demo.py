from pathlib import Path

from render_single_gripper_webgl_demo import single_capture_id


def test_single_capture_id_comes_from_source_osv_name():
    source = Path("/captures/CAM_20260828101530_0064_D.OSV")

    assert single_capture_id(source, 100.0) == "CAM_20260828101530_0064_D-single-100fps"


def test_single_capture_id_preserves_non_integer_frame_rate():
    source = Path("capture.OSV")

    assert single_capture_id(source, 59.94) == "capture-single-59.94fps"
