import json
from pathlib import Path

import pytest

from osmo360.pipeline.cli import list_commands
from osmo360.pipeline.devices import (
    assign_device,
    load_inventory,
    parse_camera_sdk_output,
    register_devices,
)
from osmo360.pipeline.device_ui import PAGE
from osmo360.pipeline.manifest import ManifestError, load_manifest, sha256
from osmo360.pipeline.process import process_capture
from osmo360.pipeline.review import build_review_bundle


def _identity(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256(path)}


def _manifest(tmp_path: Path) -> Path:
    raw = tmp_path / "capture.insv"
    identity = tmp_path / "identity.json"
    cad = tmp_path / "cad.zip"
    rig = tmp_path / "rig.json"
    angle = tmp_path / "angle.json"
    marker = tmp_path / "marker.json"
    scene = tmp_path / "scene.html"
    raw.write_bytes(b"raw")
    identity.write_text('{"serial":"TESTSERIAL01"}\n')
    cad.write_bytes(b"cad")
    rig.write_text('{"revision_id":"rig-test"}\n')
    angle.write_text('{"revision_id":"angle-test"}\n')
    marker.write_text('{"revision_id":"marker-test"}\n')
    scene.write_text("<html>fetch('timeline.json');front-video.mp4</html>\n")
    force_dir = tmp_path / "force"
    timeline_dir = tmp_path / "timeline"
    bundle_dir = tmp_path / "bundle"
    data = {
        "schema_version": "capture-manifest/1.0",
        "capture_id": "x5-test-capture",
        "status": "DIAGNOSTIC",
        "camera": {
            "serial": "TESTSERIAL01",
            "model": "Insta360 X5",
            "role": "physical_right",
            "base_tag_id": 3,
        },
        "inputs": {
            "raw_video": _identity(raw),
            "camera_identity": _identity(identity),
            "new_cad_source": _identity(cad),
        },
        "revisions": {
            "rig": _identity(rig),
            "jaw_angle": _identity(angle),
            "marker_layout": _identity(marker),
            "renderer": _identity(scene),
        },
        "pipeline": {
            "profile": "single_x5_camera_local",
            "maximum_recovery_gap_s": 0.25,
            "contact_events_s": [1.0],
        },
        "outputs": {
            "force_angle_dir": str(force_dir),
            "timeline_dir": str(timeline_dir),
            "review_bundle_dir": str(bundle_dir),
        },
        "review": {"name": "test"},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    return path


def _write_outputs(manifest_path: Path) -> None:
    manifest = load_manifest(manifest_path)
    force_dir = manifest.output_path("force_angle_dir")
    timeline_dir = manifest.output_path("timeline_dir")
    force_dir.mkdir()
    timeline_dir.mkdir()
    audit = {
        "status": "DIAGNOSTIC",
        "source": {
            "osv_sha256": manifest.data["inputs"]["raw_video"]["sha256"],
            "base_tag_id": 3,
            "camera_serial": manifest.data["camera"]["serial"],
        },
        "rig_revision": {"sha256": manifest.data["revisions"]["rig"]["sha256"]},
        "angle": {
            "revision": {"sha256": manifest.data["revisions"]["jaw_angle"]["sha256"]}
        },
    }
    (force_dir / "audit.json").write_text(json.dumps(audit))
    (force_dir / "jaw_angle_marker_overlay.mp4").write_bytes(b"video")
    angle_id = json.loads(
        manifest.identity_path("revisions", "jaw_angle").read_text()
    )["revision_id"]
    timeline = {
        "camera_serial": manifest.data["camera"]["serial"],
        "localization": {"base_tag_id": 3},
        "angle_models": {"left": {"revision": {"id": angle_id}}},
    }
    (timeline_dir / "single_gripper_webgl_timeline.json").write_text(
        json.dumps(timeline)
    )


def test_manifest_verifies_identities_and_builds_dry_run(tmp_path):
    path = _manifest(tmp_path)
    manifest = load_manifest(path)

    result = process_capture(manifest, dry_run=True)

    assert result["capture_id"] == "x5-test-capture"
    assert "--camera-local-basetag" in result["commands"][1]
    assert "--x5-angle-revision" in result["commands"][0]


def test_manifest_rejects_hash_drift(tmp_path):
    path = _manifest(tmp_path)
    data = json.loads(path.read_text())
    data["inputs"]["raw_video"]["sha256"] = "0" * 64
    path.write_text(json.dumps(data))
    with pytest.raises(ManifestError, match="hash mismatch"):
        load_manifest(path)


def test_manifest_optional_relative_force_revision_reaches_processor(tmp_path):
    path = _manifest(tmp_path)
    force_revision = tmp_path / "relative-force.json"
    force_revision.write_text('{"revision_id":"force-test"}\n')
    data = json.loads(path.read_text())
    data["revisions"]["relative_force"] = _identity(force_revision)
    path.write_text(json.dumps(data))

    result = process_capture(load_manifest(path), dry_run=True)
    command = result["commands"][0]

    assert "--relative-force-revision" in command
    assert command[command.index("--relative-force-revision") + 1] == str(
        force_revision
    )



def test_existing_outputs_become_immutable_review_bundle(tmp_path):
    path = _manifest(tmp_path)
    _write_outputs(path)
    manifest = load_manifest(path)

    process = process_capture(manifest)
    bundle = build_review_bundle(manifest)
    reused = build_review_bundle(manifest)

    directory = Path(bundle["directory"])
    assert process["force_reused"] is True
    assert process["timeline_reused"] is True
    assert bundle["reused"] is False
    assert reused["reused"] is True
    assert (directory / "timeline.json").is_file()
    assert (directory / "front-video.mp4").is_file()
    assert list(directory.glob("scene.*.html"))
    assert json.loads((directory / "checksums.json").read_text()) == bundle["checksums"]


def test_command_registry_hides_legacy_by_default():
    assert "legacy" not in list_commands(False)
    assert "legacy" in list_commands(True)


def test_camera_sdk_fleet_registration_preserves_assignments(tmp_path):
    output = "\n".join(
        [
            "serial:IAHEA2606KMURQ ;camera type:Insta360 X5 ;fw version:v1.7.8",
            "serial:IAHEA2606KTEST ;camera type:Insta360 X5 ;fw version:v1.7.7",
        ]
    )
    devices = parse_camera_sdk_output(output)
    inventory_path = tmp_path / "inventory.json"

    register_devices(devices, inventory_path)
    assign_device(
        "IAHEA2606KMURQ",
        role="physical_right",
        base_tag_id=3,
        path=inventory_path,
    )
    updated = register_devices(
        [
            {
                "serial": "IAHEA2606KMURQ",
                "model": "Insta360 X5",
                "firmware": "v1.7.9",
            }
        ],
        inventory_path,
    )

    assert sorted(updated["devices"]) == ["IAHEA2606KMURQ", "IAHEA2606KTEST"]
    assert updated["devices"]["IAHEA2606KMURQ"]["firmware"] == "v1.7.9"
    assert updated["devices"]["IAHEA2606KMURQ"]["assignment"] == {
        "role": "physical_right",
        "base_tag_id": 3,
    }
    assert load_inventory(inventory_path) == updated


def test_visual_device_manager_exposes_required_buttons():
    assert "扫描已连接 X5" in PAGE
    assert "登记全部" in PAGE
    assert "保存分配" in PAGE
    assert "同步到服务器" in PAGE
    assert "/api/scan" in PAGE
    assert "/api/register" in PAGE
    assert "/api/assign" in PAGE

    assert "/api/sync" in PAGE