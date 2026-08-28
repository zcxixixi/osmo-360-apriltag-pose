#!/usr/bin/env python3
"""Validate one v52+ rig revision and print its resolved immutable identity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rig_revision import load_rig_revision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rig_revision", type=Path)
    args = parser.parse_args()
    bundle = load_rig_revision(args.rig_revision)
    hardware = bundle["hardware"]
    print(json.dumps({
        "status": "PASS",
        "revision_id": bundle["revision"]["revision_id"],
        "revision_sha256": bundle["revision_sha256"],
        "hardware_sha256": bundle["revision"]["hardware"]["sha256"],
        "geometry_sha256": bundle["revision"]["gripper_geometry"]["sha256"],
        "world_map_sha256": bundle["world_map"]["tag_map_sha256"],
        "roles": {
            role: {
                "camera_serial": robot["camera_serial"],
                "base_tag_id": robot["base_tag_id"],
                "mount_revision": robot["mount_revision"],
            }
            for role, robot in hardware["robots"].items()
        },
        "training_ready": bundle["revision"]["training_ready"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
