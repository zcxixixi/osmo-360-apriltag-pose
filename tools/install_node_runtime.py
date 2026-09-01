#!/usr/bin/env python3
"""Install the pinned Node.js runtime into the gitignored project tool tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

from tools._root import ROOT

REVISION = ROOT / "config/runtime_revisions/node_linux_x64_24_20_0.json"
MAX_NODE_ARCHIVE_BYTES = 128 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_revision(path: Path) -> dict:
    revision = json.loads(path.read_text(encoding="utf-8"))
    if revision.get("schema_version") != "node-runtime-revision/1.0":
        raise ValueError(f"unsupported Node runtime revision: {path}")
    if revision.get("platform") != "linux-x86_64":
        raise ValueError(f"unsupported Node runtime platform: {revision.get('platform')}")
    return revision


def _download(url: str, output: Path) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "nodejs.org"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ValueError("Node archive URL must be an HTTPS URL on nodejs.org")
    try:
        _ = parsed.port
    except ValueError as error:
        raise ValueError("Node archive URL contains an invalid port") from error
    request = urllib.request.Request(url, headers={"User-Agent": "osmo360-node-installer/1"})
    # The source is pinned to nodejs.org above and verified by SHA-256 after download.
    with urllib.request.urlopen(request, timeout=30) as response, output.open("wb") as handle:  # nosec B310
        final_url = response.geturl()
        final = urlsplit(final_url)
        if final.scheme != "https" or final.hostname != "nodejs.org":
            raise ValueError("Node archive redirect left https://nodejs.org")
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > MAX_NODE_ARCHIVE_BYTES:
            raise ValueError("Node archive exceeds the download size limit")
        downloaded = 0
        while chunk := response.read(4 * 1024 * 1024):
            downloaded += len(chunk)
            if downloaded > MAX_NODE_ARCHIVE_BYTES:
                raise ValueError("Node archive exceeds the download size limit")
            handle.write(chunk)


def _node_version(binary: Path) -> str:
    result = subprocess.run(
        [str(binary), "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return result.stdout.strip().removeprefix("v")


def install_node_runtime(
    *,
    revision_path: Path = REVISION,
    archive_path: Path | None = None,
    repo_root: Path = ROOT,
) -> dict:
    revision = _load_revision(revision_path)
    root = repo_root.resolve()
    install_root = Path(revision["install_root"])
    runtime_dir = Path(revision["runtime_dir"])
    if install_root.is_absolute() or ".." in install_root.parts:
        raise ValueError("Node install_root must stay inside the repository")
    if runtime_dir.name != str(runtime_dir) or runtime_dir.name in {"", ".", ".."}:
        raise ValueError("Node runtime_dir must be one safe path component")
    destination = (
        root / install_root / runtime_dir
    ).resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError("Node runtime destination escapes the repository") from exc
    binary = destination / "bin/node"
    if destination.exists():
        if _sha256(binary) != revision["binary"]["sha256"]:
            raise RuntimeError(f"existing Node binary hash mismatch: {binary}")
        if _node_version(binary) != revision["node_version"]:
            raise RuntimeError(
                f"existing runtime does not match {revision['revision_id']}: {destination}"
            )
        return {
            "status": "reused",
            "revision_id": revision["revision_id"],
            "node": str(binary),
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".node-install-", dir=destination.parent
    ) as temporary:
        temporary_root = Path(temporary)
        archive = (
            Path(archive_path).expanduser().resolve()
            if archive_path is not None
            else temporary_root / revision["archive"]["name"]
        )
        if archive_path is None:
            _download(revision["archive"]["url"], archive)
        actual_sha256 = _sha256(archive)
        expected_sha256 = revision["archive"]["sha256"]
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Node archive SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
            )

        extraction = temporary_root / "extract"
        extraction.mkdir()
        with tarfile.open(archive, "r:xz") as bundle:
            bundle.extractall(extraction, filter="data")
        extracted = extraction / revision["runtime_dir"]
        if not extracted.is_dir():
            raise ValueError(
                f"Node archive is missing expected root {revision['runtime_dir']}"
            )
        top_level = [path.name for path in extraction.iterdir()]
        if top_level != [revision["runtime_dir"]]:
            raise ValueError(f"unexpected Node archive roots: {sorted(top_level)}")
        extracted_binary = extracted / "bin/node"
        if _sha256(extracted_binary) != revision["binary"]["sha256"]:
            raise ValueError("extracted Node binary SHA-256 does not match the revision")
        if _node_version(extracted_binary) != revision["node_version"]:
            raise ValueError("extracted Node binary version does not match the revision")
        marker = {
            "schema_version": "installed-node-runtime/1.0",
            "revision_id": revision["revision_id"],
            "archive_sha256": expected_sha256,
            "node_binary_sha256": revision["binary"]["sha256"],
            "node_version": revision["node_version"],
        }
        (extracted / ".osmo360-runtime.json").write_text(
            json.dumps(marker, indent=2) + "\n", encoding="utf-8"
        )
        os.rename(extracted, destination)

    return {
        "status": "installed",
        "revision_id": revision["revision_id"],
        "node": str(binary),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="verified offline archive")
    parser.add_argument("--revision", type=Path, default=REVISION)
    args = parser.parse_args()
    result = install_node_runtime(
        revision_path=args.revision.resolve(), archive_path=args.archive
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
