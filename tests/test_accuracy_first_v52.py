import json
from pathlib import Path

import pytest

from tools.build_accuracy_first_v52 import gate_reason, reject_adjacent_jumps
from osmo360.rig_revision import load_rig_revision, sha256
from osmo360.localization.world_frames import compile_world_tag_map


REPO = Path(__file__).resolve().parents[1]
REVISION = REPO / "config/rig_revisions/dual_gripper_v52_dev_20260826_r1.json"
NEW_GRIPPER_REVISION = (
    REPO / "config/rig_revisions/dual_gripper_v52_new_gripper_20260826_r2.json"
)


def pose_row(ids: str, rmse: float = 0.5) -> dict[str, str]:
    return {
        "quality_status": "valid",
        "camera_x_m": "0.1",
        "camera_y_m": "0.2",
        "camera_z_m": "0.3",
        "qx": "0",
        "qy": "0",
        "qz": "0",
        "qw": "1",
        "detected_ids": ids,
        "angular_rmse_deg": str(rmse),
    }


def test_checked_revision_loads_without_legacy_fallback():
    bundle = load_rig_revision(REVISION)
    assert bundle["revision"]["revision_id"] == "dual-gripper-v52-dev-20260826-r1"
    assert bundle["hardware"]["robots"]["left"]["camera_serial"] == "95SXN9H0423SGG"
    assert bundle["policy"]["allow_metric_smoothing"] is False


def test_new_gripper_revision_pins_cad_and_preserves_tag_transform():
    bundle = load_rig_revision(NEW_GRIPPER_REVISION)
    assert bundle["cad_revision"]["revision_id"] == "gripper-cad-v52-new-r1"
    assert bundle["geometry"]["base_to_tag"]["translation_m"] == [0.02625, 0.0, 0.0196]
    assert bundle["geometry"]["base_to_tcp"]["status"] == (
        "LEGACY_FIXED_REFERENCE_PENDING_NEW_CONTACT_TCP_ACCEPTANCE"
    )
    assert bundle["revision"]["training_ready"] is False


def test_revision_hash_mismatch_fails_closed(tmp_path):
    payload = json.loads(REVISION.read_text(encoding="utf-8"))
    payload["hardware"]["sha256"] = "0" * 64
    path = tmp_path / "bad-revision.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hardware hash mismatch"):
        load_rig_revision(path)


def test_diagnostic_world_requires_explicit_opt_in(tmp_path):
    revision = json.loads(REVISION.read_text(encoding="utf-8"))
    source_map = Path(revision["world_tag_map"]["path"])
    world = json.loads(source_map.read_text(encoding="utf-8"))
    world["calibration_status"] = "DIAGNOSTIC_CAPTURE_ONLY"
    diagnostic_map = tmp_path / "diagnostic-map.json"
    diagnostic_map.write_text(json.dumps(world))
    compiled = compile_world_tag_map(diagnostic_map)
    revision["training_ready"] = False
    revision["world_tag_map"] = {
        "path": str(diagnostic_map),
        "sha256": sha256(diagnostic_map),
        "compiled_sha256": compiled["tag_map_sha256"],
        "map_id": world["map_id"],
    }
    path = tmp_path / "diagnostic-revision.json"
    path.write_text(json.dumps(revision))

    with pytest.raises(ValueError, match="world Tag map is not VERIFIED"):
        load_rig_revision(path)
    assert load_rig_revision(path, allow_diagnostic_world=True)["revision"]["training_ready"] is False


def test_gate_requires_two_wall_panels_and_strict_residual():
    groups = [set(range(128, 134)), set(range(134, 138))]
    good = pose_row("128 129 134 135", 1.0)
    assert gate_reason(good, good, panel_groups=groups, maximum_rmse_deg=2.0) is None
    single = pose_row("128 129 130", 0.2)
    assert gate_reason(single, good, panel_groups=groups, maximum_rmse_deg=2.0) == (
        "left_single_panel_depth_untrusted"
    )
    weak = pose_row("128 129 134 135", 2.1)
    assert gate_reason(weak, good, panel_groups=groups, maximum_rmse_deg=2.0) == (
        "left_angular_rmse_rejected"
    )


def test_adjacent_position_jump_rejects_both_measurements():
    rows = []
    for index, position in enumerate((0.0, 0.08, 0.081)):
        row = {
            "timestamp": index / 30.0,
            "quality_status": "direct_trusted",
            "trusted": True,
        }
        for role in ("left", "right"):
            row.update({
                f"{role}_base_x_m": position,
                f"{role}_base_y_m": 0.0,
                f"{role}_base_z_m": 0.0,
            })
        rows.append(row)
    assert reject_adjacent_jumps(rows, 0.05) == 2
    assert rows[0]["quality_status"] == "position_jump_rejected"
    assert rows[1]["quality_status"] == "position_jump_rejected"
    assert rows[2]["quality_status"] == "direct_trusted"
