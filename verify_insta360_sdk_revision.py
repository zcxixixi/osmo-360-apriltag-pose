#!/usr/bin/env python3
"""Validate the installed Insta360 SDK revision without system-wide installation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from insta360_sdk_revision import DEFAULT_REVISION, load_insta360_sdk_revision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("revision", type=Path, nargs="?", default=DEFAULT_REVISION)
    parser.add_argument("--verify-source-archive", action="store_true")
    args = parser.parse_args()
    bundle = load_insta360_sdk_revision(
        args.revision, verify_source_archive=args.verify_source_archive
    )
    revision = bundle["revision"]
    print(json.dumps({
        "status": "PASS",
        "revision_id": revision["revision_id"],
        "revision_sha256": bundle["revision_sha256"],
        "platform": revision["platform"],
        "media_sdk_version": revision["media_sdk"]["version"],
        "camera_sdk_version": revision["camera_sdk"]["version"],
        "media_root": str(bundle["media_root"]),
        "camera_root": str(bundle["camera_root"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
