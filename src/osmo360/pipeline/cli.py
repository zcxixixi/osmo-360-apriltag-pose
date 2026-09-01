from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .manifest import ManifestError, ROOT, load_manifest, sha256
from .devices import (
    DEFAULT_SERVER,
    DEFAULT_INVENTORY,
    assign_device,
    load_inventory,
    register_devices,
    scan_devices,
    sync_inventory,
)
from .device_ui import serve_device_ui
from .process import process_capture
from .review import review_capture
from .progress_ui import serve_progress


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


def verify_baselines() -> dict[str, Any]:
    modules = (
        "osmo360.verification.verify_dual_gripper_v50_baseline",
        "osmo360.verification.verify_x5_one_sided_force_baseline",
    )
    python = (
        ROOT / ".venv/bin/python"
        if (ROOT / ".venv/bin/python").is_file()
        else Path(sys.executable)
    )
    results = []
    for module in modules:
        process = subprocess.run(
            [str(python), "-m", module],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        results.append(json.loads(process.stdout))
    return {"status": "PASS", "baselines": results}


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
    sync_parser = device_subparsers.add_parser(
        "sync", help="upload the persistent X5 inventory to the LAN server"
    )
    sync_parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    sync_parser.add_argument("--server", default=DEFAULT_SERVER)
    assign_parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    ui_parser = device_subparsers.add_parser(
        "ui", help="open the local visual X5 fleet manager"
    )
    ui_parser.add_argument("--host", default="127.0.0.1")
    ui_parser.add_argument("--port", type=int, default=7866)
    ui_parser.add_argument("--no-browser", action="store_true")
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
    subparsers.add_parser("verify", help="run every accepted successor baseline")
    progress_parser = subparsers.add_parser(
        "progress", help="serve a live pipeline status dashboard"
    )
    progress_parser.add_argument("status", type=Path)
    progress_parser.add_argument("--host", default="127.0.0.1")
    progress_parser.add_argument("--port", type=int, default=7868)
    progress_parser.add_argument("--no-browser", action="store_true")
    review_ui_parser = subparsers.add_parser(
        "review-ui", help="serve the simple human data-quality review platform"
    )
    review_ui_parser.add_argument("dataset_root", type=Path)
    review_ui_parser.add_argument("--host", default="127.0.0.1")
    review_ui_parser.add_argument("--port", type=int, default=7869)
    review_ui_parser.add_argument("--no-browser", action="store_true")
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
        elif args.command == "verify":
            result = verify_baselines()
        elif args.command == "progress":
            serve_progress(
                args.status,
                host=args.host,
                port=args.port,
                open_browser=not args.no_browser,
            )
            return 0
        elif args.command == "review-ui":
            from .review_ui import serve_review_ui
            serve_review_ui(
                args.dataset_root,
                host=args.host,
                port=args.port,
                open_browser=not args.no_browser,
            )
            return 0
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
            elif args.device_command == "sync":
                result = sync_inventory(
                    args.inventory.resolve(),
                    args.server,
                )
            elif args.device_command == "ui":
                serve_device_ui(
                    host=args.host,
                    port=args.port,
                    open_browser=not args.no_browser,
                )
                return 0
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
