from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .manifest import ManifestError, ROOT, load_manifest, sha256
from .devices import (
    DEFAULT_INVENTORY,
    assign_device,
    load_inventory,
    register_devices,
    scan_devices,
)
from .process import process_capture
from .review import review_capture


REGISTRY = ROOT / "config" / "pipeline_registry.json"


def inspect_capture(path: str | Path) -> dict[str, Any]:
    manifest = load_manifest(path)
    outputs = {
        name: {
            "path": str(manifest.output_path(name)),
            "exists": manifest.output_path(name).exists(),
        }
        for name in manifest.data["outputs"]
    }
    return {
        "capture_id": manifest.capture_id,
        "status": manifest.data["status"],
        "camera": manifest.data["camera"],
        "pipeline": manifest.data["pipeline"],
        "manifest_sha256": sha256(manifest.path),
        "identities_verified": True,
        "outputs": outputs,
    }


def list_commands(include_legacy: bool) -> dict[str, Any]:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    result = {"active": registry["active"]}
    if include_legacy:
        result["legacy"] = registry["legacy"]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="umi",
        description="Manifest-driven 360-camera gripper pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect", help="verify and summarize a capture manifest")
    inspect_parser.add_argument("manifest", type=Path)
    process_parser = subparsers.add_parser("process", help="run or verify the capture processing pipeline")
    process_parser.add_argument("manifest", type=Path)
    devices_parser = subparsers.add_parser(
        "devices", help="scan and register an X5 fleet through CameraSDK"
    )
    device_subparsers = devices_parser.add_subparsers(
        dest="device_command", required=True
    )
    device_subparsers.add_parser("scan", help="list all currently connected X5 devices")
    register_parser = device_subparsers.add_parser(
        "register", help="merge connected X5 devices into the persistent inventory"
    )
    register_parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    assign_parser = device_subparsers.add_parser(
        "assign", help="bind a registered serial to a physical role and BaseTag"
    )
    assign_parser.add_argument("serial")
    assign_parser.add_argument(
        "--role", required=True, choices=("physical_left", "physical_right")
    )
    assign_parser.add_argument("--base-tag-id", required=True, type=int, choices=(2, 3))
    assign_parser.add_argument("--label")
    assign_parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    list_parser = device_subparsers.add_parser(
        "list", help="show the persistent X5 inventory"
    )
    list_parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    process_parser.add_argument("--dry-run", action="store_true")
    review_parser = subparsers.add_parser("review", help="build an immutable review bundle")
    review_parser.add_argument("manifest", type=Path)
    review_parser.add_argument("--publish", action="store_true")
    review_parser.add_argument("--dry-run", action="store_true")
    command_parser = subparsers.add_parser("commands", help="list supported command paths")
    command_parser.add_argument("--legacy", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "inspect":
            result = inspect_capture(args.manifest)
        elif args.command == "process":
            result = process_capture(
                load_manifest(args.manifest), dry_run=args.dry_run
            )
        elif args.command == "review":
            result = review_capture(
                load_manifest(args.manifest),
                publish=args.publish,
                dry_run=args.dry_run,
            )
        elif args.command == "devices":
            if args.device_command == "scan":
                result = {"devices": scan_devices()}
            elif args.device_command == "register":
                result = register_devices(scan_devices(), args.inventory.resolve())
            elif args.device_command == "assign":
                result = assign_device(
                    args.serial,
                    role=args.role,
                    base_tag_id=args.base_tag_id,
                    label=args.label,
                    path=args.inventory.resolve(),
                )
            else:
                result = load_inventory(args.inventory.resolve())
        else:
            result = list_commands(args.legacy)
    except (ManifestError, OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
