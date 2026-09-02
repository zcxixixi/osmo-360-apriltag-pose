#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

COLLECTOR_PATTERN = re.compile(r"^\d{4}_instaumi_[a-z0-9_]+$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_files(collector: Path) -> list[tuple[str, Path]]:
    result = []
    for side in ("left", "right"):
        directory = collector / "raw" / side
        if not directory.is_dir():
            raise RuntimeError(f"missing source directory: {directory}")
        paths = sorted(directory.glob("*.insv"))
        if not paths:
            raise RuntimeError(f"source directory contains no INSV: {directory}")
        result.extend((f"{side}/{path.name}", path) for path in paths)
    return result


def run(command: list[str]) -> None:
    process = subprocess.run(command)
    if process.returncode:
        raise RuntimeError(f"command failed ({process.returncode}): {' '.join(command)}")


def upload_collector(collector: Path, host: str, destination_root: str) -> dict[str, Any]:
    destination = f"{destination_root.rstrip('/')}/{collector.name}/raw"
    run([
        "ssh", "-F", "/dev/null", "-o", "BatchMode=yes",
        host, "mkdir", "-p", destination,
    ])
    run([
        "rsync",
        "-a",
        "--partial",
        "--append-verify",
        "--human-readable",
        "--info=progress2",
        "--exclude=sha256.txt",
        "-e",
        "ssh -F /dev/null -o BatchMode=yes",
        f"{collector / 'raw'}/",
        f"{host}:{destination}/",
    ])

    files = source_files(collector)
    hashes = [(relative, sha256(path)) for relative, path in files]
    with tempfile.TemporaryDirectory(prefix="instaumi-sha-") as temporary:
        checksum = Path(temporary) / "sha256.txt"
        checksum.write_text(
            "".join(f"{digest}  {relative}\n" for relative, digest in hashes),
            encoding="utf-8",
        )
        run([
            "rsync",
            "-a",
            "--delay-updates",
            "-e",
            "ssh -F /dev/null -o BatchMode=yes",
            str(checksum),
            f"{host}:{destination}/sha256.txt",
        ])
    return {
        "collector": collector.name,
        "file_count": len(files),
        "size_bytes": sum(path.stat().st_size for _, path in files),
        "destination": destination,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume-upload InstaUMI collector folders and publish SHA completion markers"
    )
    parser.add_argument("source_root", type=Path)
    parser.add_argument("--host", default="osmo-server")
    parser.add_argument(
        "--destination-root",
        default="/home/ps/current-robotics-data-2/total_annotation/umi_insta360",
    )
    parser.add_argument(
        "--status",
        type=Path,
        default=Path.home() / ".local/state/instaumi-upload/status.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.source_root.resolve(strict=True)
    collectors = [
        path for path in sorted(root.iterdir())
        if path.is_dir() and COLLECTOR_PATTERN.fullmatch(path.name)
    ]
    if not collectors:
        raise SystemExit(f"no InstaUMI collector directories found under {root}")
    total_bytes = sum(
        path.stat().st_size
        for collector in collectors
        for _, path in source_files(collector)
    )
    status: dict[str, Any] = {
        "schema_version": "instaumi-drive-upload/1.0",
        "status": "RUNNING",
        "source_root": str(root),
        "destination_root": args.destination_root,
        "total_collectors": len(collectors),
        "total_bytes": total_bytes,
        "completed": [],
        "updated_at_utc": utc_now(),
    }
    atomic_json(args.status, status)
    try:
        for collector in collectors:
            status["current_collector"] = collector.name
            status["stage"] = "rsync_then_sha256"
            status["updated_at_utc"] = utc_now()
            atomic_json(args.status, status)
            result = upload_collector(collector, args.host, args.destination_root)
            status["completed"].append(result)
            status["completed_bytes"] = sum(item["size_bytes"] for item in status["completed"])
            status["updated_at_utc"] = utc_now()
            atomic_json(args.status, status)
    except Exception as error:
        status["status"] = "FAILED"
        status["error"] = f"{type(error).__name__}: {error}"
        status["updated_at_utc"] = utc_now()
        atomic_json(args.status, status)
        raise
    status["status"] = "COMPLETE"
    status["stage"] = "complete"
    status.pop("current_collector", None)
    status["updated_at_utc"] = utc_now()
    atomic_json(args.status, status)
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
