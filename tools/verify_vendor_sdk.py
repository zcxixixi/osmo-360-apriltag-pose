#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sdk(root: Path, manifest_path: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = []
    for relative, expected in manifest["required_files"].items():
        path = root / relative
        actual = sha256(path) if path.is_file() else None
        files.append({
            "path": relative,
            "present": path.is_file(),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "valid": actual == expected,
        })
    return {
        "schema_version": "vendor-sdk-verification/v1",
        "revision_id": manifest["revision_id"],
        "sdk_root": str(root.resolve()),
        "valid": all(item["valid"] for item in files),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdk-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify_sdk(args.sdk_root, args.manifest)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
