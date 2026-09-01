from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
import subprocess
from pathlib import Path
from typing import Any

from .manifest import ManifestError, ROOT


CAMERA_SDK_ROOT = ROOT / "work/insta360-sdk/camera-2.1.1"
CAMERA_SDK_BINARY = CAMERA_SDK_ROOT / "bin/CameraSDKTest"
CAMERA_SDK_LIBRARY = CAMERA_SDK_ROOT / "lib"
DEFAULT_INVENTORY = ROOT / "config/devices/x5_inventory.json"
DEFAULT_PAIRS = ROOT / "config/devices/x5_pairs.json"
SDK_REVISION_ID = "insta360-linux-camera-2.1.1-media-3.1.1"
DEFAULT_SERVER = os.environ.get("OSMO_VISUALIZATION_URL", "http://192.168.111.62:7865")
DEVICE_PATTERN = re.compile(
    r"serial:(?P<serial>[A-Z0-9]+)\s*;camera type:(?P<model>[^;]+?)\s*;fw version:(?P<firmware>[^;\r\n]+)"
)


def parse_camera_sdk_output(output: str) -> list[dict[str, str]]:
    devices: dict[str, dict[str, str]] = {}
    for match in DEVICE_PATTERN.finditer(output):
        serial = match.group("serial").strip()
        devices[serial] = {
            "serial": serial,
            "model": match.group("model").strip(),
            "firmware": match.group("firmware").strip(),
        }
    return [devices[serial] for serial in sorted(devices)]


def scan_devices(*, timeout: float = 30.0) -> list[dict[str, str]]:
    if not CAMERA_SDK_BINARY.is_file() or not CAMERA_SDK_LIBRARY.is_dir():
        raise ManifestError("CameraSDK 2.1.1 is not deployed")
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = str(CAMERA_SDK_LIBRARY)
    process = subprocess.run(
        [str(CAMERA_SDK_BINARY)],
        input="0\n",
        capture_output=True,
        text=True,
        timeout=timeout,
        env=environment,
        cwd=ROOT,
    )
    output = process.stdout + process.stderr
    devices = parse_camera_sdk_output(output)
    if not devices:
        detail = "no device found" if "no device found" in output else output.strip()[-500:]
        raise ManifestError(f"CameraSDK discovered no devices: {detail}")
    return devices


def load_inventory(path: Path = DEFAULT_INVENTORY) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": "x5-device-inventory/1.0",
            "sdk_revision_id": SDK_REVISION_ID,
            "devices": {},
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != "x5-device-inventory/1.0":
        raise ManifestError("invalid X5 device inventory schema")
    if not isinstance(data.get("devices"), dict):
        raise ManifestError("X5 device inventory devices must be an object")
    return data


def load_device_pairs(path: Path = DEFAULT_PAIRS) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != "x5-device-pairs/1.0":
        raise ManifestError("invalid X5 device-pairs schema")
    pairs = data.get("pairs")
    if not isinstance(pairs, dict):
        raise ManifestError("X5 device pairs must be an object")
    for pair_id, pair in pairs.items():
        left, right = pair.get("left", {}), pair.get("right", {})
        if left.get("role") != "physical_left" or left.get("base_tag_id") != 2:
            raise ManifestError(f"{pair_id} has an invalid left assignment")
        if right.get("role") != "physical_right" or right.get("base_tag_id") != 3:
            raise ManifestError(f"{pair_id} has an invalid right assignment")
        if left.get("serial") == right.get("serial"):
            raise ManifestError(f"{pair_id} reuses one serial on both sides")
    return data


def resolve_device_pair(
    serials: set[str], path: Path = DEFAULT_PAIRS,
) -> tuple[str, dict[str, Any]]:
    matches = []
    for pair_id, pair in load_device_pairs(path)["pairs"].items():
        expected = {pair["left"]["serial"], pair["right"]["serial"]}
        if serials == expected:
            matches.append((pair_id, pair))
    if len(matches) != 1:
        raise ManifestError(
            f"expected exactly one device pair for serials {sorted(serials)}, found {len(matches)}"
        )
    return matches[0]


def write_inventory(inventory: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def register_devices(
    devices: list[dict[str, str]],
    path: Path = DEFAULT_INVENTORY,
) -> dict[str, Any]:
    inventory = load_inventory(path)
    for device in devices:
        serial = device["serial"]
        existing = inventory["devices"].get(serial, {})
        inventory["devices"][serial] = {
            **existing,
            "serial": serial,
            "model": device["model"],
            "firmware": device["firmware"],
            "identity_status": "SDK_VERIFIED",
            "sdk_revision_id": SDK_REVISION_ID,
            "assignment": existing.get("assignment"),
        }
    inventory["devices"] = {
        serial: inventory["devices"][serial]
        for serial in sorted(inventory["devices"])
    }
    write_inventory(inventory, path)
    return inventory


def assign_device(
    serial: str,
    *,
    role: str,
    base_tag_id: int,
    label: str | None = None,
    path: Path = DEFAULT_INVENTORY,
) -> dict[str, Any]:
    inventory = load_inventory(path)
    if serial not in inventory["devices"]:
        raise ManifestError(
            f"serial {serial!r} is not registered; run 'umi devices register' first"
        )
    if role not in {"physical_left", "physical_right"}:
        raise ManifestError("role must be physical_left or physical_right")
    if base_tag_id not in {2, 3}:
        raise ManifestError("base_tag_id must be 2 or 3")
    assignment = {"role": role, "base_tag_id": base_tag_id}
    if label:
        assignment["label"] = label
    inventory["devices"][serial]["assignment"] = assignment
    write_inventory(inventory, path)
    return inventory


def sync_inventory(
    path: Path = DEFAULT_INVENTORY,
    server: str = DEFAULT_SERVER,
) -> dict[str, Any]:
    inventory = load_inventory(path)
    body = json.dumps(inventory, ensure_ascii=False).encode()
    request = urllib.request.Request(
        server.rstrip("/") + "/api/devices",
        data=body,
        method="PUT",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        try:
            message = json.load(error).get("error", error.reason)
        except json.JSONDecodeError:
            message = error.reason
        raise ManifestError(f"device inventory upload failed: {message}") from error
    except urllib.error.URLError as error:
        raise ManifestError(f"device inventory upload failed: {error.reason}") from error
