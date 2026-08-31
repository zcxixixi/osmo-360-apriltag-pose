#!/usr/bin/env python3
import argparse
from pathlib import Path
import shutil

SUPPORTED_KEY_PREFIXES = (
    "ssh-ed25519 ", "ssh-rsa ", "ecdsa-sha2-nistp256 ",
    "ecdsa-sha2-nistp384 ", "ecdsa-sha2-nistp521 ",
)
FILES = (
    "README.md", "collect_node_info.sh", "enable_ssh_node.sh",
    "gpu_smoke_test.py", "run_on_ubuntu.sh",
)
DESKTOP = """[Desktop Entry]
Type=Application
Version=1.0
Name=OSMO GPU compute-node onboarding
Comment=Inventory this node, enable SSH, and save reports to removable storage
Exec=bash -c "cd -- \\\"\\$(dirname -- \\\"\\$1\\\")\\\" && bash ./RUN_ON_UBUNTU.sh" _ %k
Icon=utilities-terminal
Terminal=true
Categories=System;
StartupNotify=true
"""


def package_node_kit(public_key_file: Path, output: Path) -> None:
    public_key = public_key_file.read_text(encoding="utf-8").splitlines()[0].strip()
    if not public_key.startswith(SUPPORTED_KEY_PREFIXES):
        raise ValueError("unsupported or malformed OpenSSH public key")
    source = Path(__file__).resolve().parent
    output.mkdir(parents=True, exist_ok=True)
    (output / "RESULTS").mkdir(exist_ok=True)
    for name in FILES:
        target_name = "README.txt" if name == "README.md" else name
        shutil.copy2(source / name, output / target_name)
    (output / "WORKSTATION_PUBLIC_KEY.pub").write_text(public_key + "\n", encoding="utf-8")
    launcher = output / "RUN_ON_UBUNTU.desktop"
    launcher.write_text(DESKTOP, encoding="utf-8")
    for name in (
        "collect_node_info.sh", "enable_ssh_node.sh", "gpu_smoke_test.py",
        "run_on_ubuntu.sh", "RUN_ON_UBUNTU.desktop",
    ):
        (output / name).chmod(0o755)
    shutil.copy2(output / "run_on_ubuntu.sh", output / "RUN_ON_UBUNTU.sh")
    (output / "RUN_ON_UBUNTU.sh").chmod(0o755)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-key-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package_node_kit(args.public_key_file, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
