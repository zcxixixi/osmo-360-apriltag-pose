import json
from pathlib import Path

import pytest

from osmo360.pipeline.devices import load_device_pairs, resolve_device_pair
from osmo360.pipeline.manifest import ManifestError


ROOT = Path(__file__).resolve().parents[1]


def test_right1_camera_and_sd_card_share_one_asset_identity():
    inventory = json.loads(
        (ROOT / "config/devices/x5_inventory.json").read_text(encoding="utf-8")
    )
    media = json.loads(
        (ROOT / "config/devices/media_inventory.json").read_text(encoding="utf-8")
    )

    camera = inventory["devices"]["IAHEA2606KM43A"]
    asset = media["assets"]["right-1"]
    assert camera["asset_label"] == "right-1"
    assert camera["assignment"] == {
        "role": "physical_left",
        "base_tag_id": 2,
        "label": "right-1",
    }
    assert asset["camera_serial"] == camera["serial"]
    assert asset["sd_card"]["asset_label"] == "right-1"
    assert asset["sd_card"]["reader_usb_serial_observed"] == "000000002957"
    assert asset["sd_card"]["filesystem_uuid_status"] == "PENDING_NEXT_MOUNT"


def test_left1_camera_and_sd_are_bound_by_capture_evidence():
    inventory = json.loads(
        (ROOT / "config/devices/x5_inventory.json").read_text(encoding="utf-8")
    )
    media = json.loads(
        (ROOT / "config/devices/media_inventory.json").read_text(encoding="utf-8")
    )

    camera = inventory["devices"]["IAHEA2606M5WSK"]
    asset = media["assets"]["left-1"]
    assert camera["asset_label"] == "left-1"
    assert camera["assignment"] == {
        "role": "physical_left",
        "base_tag_id": 2,
        "label": "left-1",
    }
    assert asset["camera_serial"] == camera["serial"]
    assert asset["verified_geometry_binding"]["base_tag_id"] == 2
    assert asset["sd_card"]["filesystem_uuid"] == "6632-3830"
    assert asset["sd_card"]["evidence_capture_sha256"] == (
        "c42d4caf09d06af0820bc4af4a088f555c9775a7693ed1248801da5ae442376d"
    )


def test_dual_gripper_pair_is_serial_bound_and_resolvable():
    pairs = load_device_pairs()
    pair = pairs["pairs"]["dual-x5-gripper-pair-01"]
    assert pair["left"]["serial"] == "IAHEA2606M5WSK"
    assert pair["left"]["role"] == "physical_left"
    assert pair["left"]["base_tag_id"] == 2
    assert pair["right"]["serial"] == "IAHEA2606KKUKF"
    assert pair["right"]["role"] == "physical_right"
    assert pair["right"]["base_tag_id"] == 3
    assert pair["future_capture_profile"] == {
        "camera_model": "insta360-x5",
        "mode": "4K30",
        "nominal_width": 3840,
        "nominal_height": 1920,
        "nominal_fps": 30.0,
        "required_lens_tracks": 2,
        "stitching": "official full-panorama",
    }
    pair_id, resolved = resolve_device_pair(
        {"IAHEA2606KKUKF", "IAHEA2606M5WSK"}
    )
    assert pair_id == "dual-x5-gripper-pair-01"
    assert resolved == pair


def test_user_verified_kmdgp_kmurq_pair_is_serial_bound_and_resolvable():
    inventory = json.loads(
        (ROOT / "config/devices/x5_inventory.json").read_text(encoding="utf-8")
    )["devices"]
    assert inventory["IAHEA2606KMDGP"]["assignment"] == {
        "role": "physical_left",
        "base_tag_id": 2,
        "label": "left-gripper-basetag2",
    }
    assert inventory["IAHEA2606KMURQ"]["assignment"] == {
        "role": "physical_right",
        "base_tag_id": 3,
        "label": "right-gripper-basetag3",
    }

    pair_id, pair = resolve_device_pair(
        {"IAHEA2606KMURQ", "IAHEA2606KMDGP"}
    )
    assert pair_id == "dual-x5-gripper-pair-02"
    assert pair["left"]["serial"] == "IAHEA2606KMDGP"
    assert pair["right"]["serial"] == "IAHEA2606KMURQ"
    assert pair["verification"]["method"] == (
        "user identification of serial-bound INSV stream-0 frames"
    )


def test_unknown_serial_combination_is_rejected():
    with pytest.raises(ManifestError, match="exactly one device pair"):
        resolve_device_pair({"IAHEA2606M5WSK", "UNKNOWN"})
