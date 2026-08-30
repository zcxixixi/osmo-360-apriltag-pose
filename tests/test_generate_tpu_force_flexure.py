import json
import sys

import numpy as np

import generate_tpu_force_flexure as flexure


def test_single_flexure_is_one_watertight_printable_component():
    mesh = flexure.make_flexure()

    assert mesh.is_watertight
    assert len(mesh.split()) == 1
    np.testing.assert_allclose(mesh.extents, [28.25, 12.25, 9.25])
    assert mesh.volume > 1000.0


def test_generator_writes_single_pair_preview_and_report(tmp_path, monkeypatch):
    output = tmp_path / "flexure"
    monkeypatch.setattr(sys, "argv", ["generate_tpu_force_flexure.py", "--output-dir", str(output)])

    assert flexure.main() == 0

    report = json.loads((output / "build_report.json").read_text())
    assert (output / "TPU_force_flexure_single_r1.STL").is_file()
    assert (output / "TPU_force_flexure_pair_r1.STL").is_file()
    assert (output / "TPU_force_flexure_preview_r1.png").is_file()
    assert report["single"]["watertight_components"] is True
    assert report["single"]["component_count"] == 1
    assert report["pair"]["component_count"] == 2
    np.testing.assert_allclose(report["pair"]["extents_mm"], [62.25, 12.25, 9.25])
    assert report["pair"]["bounds_mm"][0][1] == report["single"]["bounds_mm"][0][1]
    assert report["pair"]["bounds_mm"][1][1] == report["single"]["bounds_mm"][1][1]
    assert report["mechanical"]["target_visual_travel_mm"] == [3.0, 5.0]
    assert report["mechanical"]["marker_boss_face"].startswith("camera-facing")


def test_frozen_hardware_revision_matches_print_assets():
    revision_path = (
        flexure.ROOT
        / "config/rig_revisions/gripper_force_flexure_tpu_20260830_r1.json"
    )
    revision = json.loads(revision_path.read_text())

    assert revision["status"] == "PROTOTYPE_PRINT_REQUIRED_NOT_CALIBRATED"
    generator = revision["generator"]
    assert flexure.sha256(flexure.ROOT / generator["path"]) == generator["sha256"]
    for output in revision["outputs"].values():
        assert flexure.sha256(flexure.ROOT / output["path"]) == output["sha256"]
