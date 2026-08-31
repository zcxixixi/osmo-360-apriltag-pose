import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_node_kit_packages_only_the_supplied_public_key(tmp_path: Path) -> None:
    module = load_module(
        "package_node_kit", ROOT / "tools/compute_node/package_node_kit.py"
    )
    key = tmp_path / "node.pub"
    public_key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITest test-node"
    key.write_text(public_key + "\n", encoding="utf-8")
    output = tmp_path / "kit"
    module.package_node_kit(key, output)
    assert (output / "WORKSTATION_PUBLIC_KEY.pub").read_text().strip() == public_key
    assert (output / "RESULTS").is_dir()
    assert (output / "RUN_ON_UBUNTU.sh").is_file()
    assert "Terminal=true" in (output / "RUN_ON_UBUNTU.desktop").read_text()
    assert not any(path.name.startswith("id_") for path in output.rglob("*"))


def test_node_kit_rejects_non_public_key(tmp_path: Path) -> None:
    module = load_module(
        "package_node_kit_invalid", ROOT / "tools/compute_node/package_node_kit.py"
    )
    key = tmp_path / "bad.key"
    key.write_text("not-a-public-key\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        module.package_node_kit(key, tmp_path / "kit")


def test_vendor_sdk_verifier_detects_tampering(tmp_path: Path) -> None:
    module = load_module("verify_vendor_sdk", ROOT / "tools/verify_vendor_sdk.py")
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    binary = sdk / "lib.so"
    binary.write_bytes(b"verified-sdk")
    expected = hashlib.sha256(binary.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "revision_id": "test-sdk",
        "required_files": {"lib.so": expected},
    }), encoding="utf-8")
    assert module.verify_sdk(sdk, manifest)["valid"]
    binary.write_bytes(b"tampered")
    assert not module.verify_sdk(sdk, manifest)["valid"]


def test_compute_node_shell_scripts_parse() -> None:
    scripts = [
        ROOT / "tools/compute_node/collect_node_info.sh",
        ROOT / "tools/compute_node/enable_ssh_node.sh",
        ROOT / "tools/compute_node/run_on_ubuntu.sh",
    ]
    subprocess.run(["bash", "-n", *map(str, scripts)], check=True)
