import json
from pathlib import Path


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


def test_left1_camera_is_named_without_guessing_geometry_or_sd_identity():
    inventory = json.loads(
        (ROOT / "config/devices/x5_inventory.json").read_text(encoding="utf-8")
    )
    media = json.loads(
        (ROOT / "config/devices/media_inventory.json").read_text(encoding="utf-8")
    )

    camera = inventory["devices"]["IAHEA2606M5WSK"]
    asset = media["assets"]["left-1"]
    assert camera["asset_label"] == "left-1"
    assert camera["assignment"] is None
    assert asset["camera_serial"] == camera["serial"]
    assert asset["verified_geometry_binding"]["status"] == "PENDING_CAPTURE_EVIDENCE"
    assert asset["sd_card"]["status"] == "PENDING_MEDIA_EVIDENCE"
