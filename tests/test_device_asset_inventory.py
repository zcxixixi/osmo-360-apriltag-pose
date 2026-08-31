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
    asset = media["assets"]["右1"]
    assert camera["asset_label"] == "右1"
    assert camera["assignment"] == {
        "role": "physical_left",
        "base_tag_id": 2,
        "label": "右1",
    }
    assert asset["camera_serial"] == camera["serial"]
    assert asset["sd_card"]["asset_label"] == "右1"
    assert asset["sd_card"]["reader_usb_serial_observed"] == "000000002957"
    assert asset["sd_card"]["filesystem_uuid_status"] == "PENDING_NEXT_MOUNT"
