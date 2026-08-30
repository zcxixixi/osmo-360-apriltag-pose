from pathlib import Path

from osmo_360_offline import (
    infer_camera_model,
    resolve_decoder,
    resolve_projection,
)


def test_auto_camera_profiles_from_export_names_and_resolution():
    assert infer_camera_model(
        Path("CAM_20260820144829_0021_D_3000_360_scaled.mp4"), 3000, 1500, "auto"
    ) == "dji-osmo-360"
    assert infer_camera_model(
        Path("VID_INSTA360_X5_NO_FLOWSTATE.mp4"), 7680, 3840, "auto"
    ) == "insta360-x5"
    assert infer_camera_model(
        Path("unknown.mp4"), 7680, 3840, "auto"
    ) == "insta360-x5"
    assert infer_camera_model(
        Path("unknown.mp4"), 5000, 2500, "auto"
    ) == "generic"


def test_measured_auto_processing_strategy():
    assert resolve_decoder("auto", "dji-osmo-360") == "cpu"
    assert resolve_decoder("auto", "insta360-x5") == "cpu"
    assert resolve_decoder("nvdec", "insta360-x5") == "nvdec"
    assert resolve_projection("auto", "dji-osmo-360", 3000) == "cpu"
    assert resolve_projection("auto", "dji-osmo-360", 6000) == "cuda"
    assert resolve_projection("auto", "insta360-x5", 7680) == "cuda"
    assert resolve_projection("cpu", "insta360-x5", 7680) == "cpu"
