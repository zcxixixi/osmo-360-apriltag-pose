#!/usr/bin/env python3
"""Install the pinned offline FFmpeg runtime into the project tool tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tarfile
import tempfile
from pathlib import Path

from osmo360.ffmpeg_runtime import _version
from tools._root import ROOT


REVISION = ROOT / "config/runtime_revisions/ffmpeg_linux_x64_9_0_1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_revision(path: Path) -> dict:
    revision = json.loads(path.read_text(encoding="utf-8"))
    if revision.get("schema_version") != "ffmpeg-runtime-revision/1.0":
        raise ValueError(f"unsupported FFmpeg runtime revision: {path}")
    if revision.get("platform") != "linux-x86_64":
        raise ValueError(f"unsupported FFmpeg platform: {revision.get('platform')}")
    return revision


def _safe_destination(repo_root: Path, revision: dict) -> Path:
    root = repo_root.resolve()
    install_root = Path(revision["install_root"])
    runtime_dir = Path(revision["runtime_dir"])
    if install_root.is_absolute() or ".." in install_root.parts:
        raise ValueError("FFmpeg install_root must stay inside the repository")
    if runtime_dir.name != str(runtime_dir) or runtime_dir.name in {"", ".", ".."}:
        raise ValueError("FFmpeg runtime_dir must be one safe path component")
    destination = root / install_root / runtime_dir
    resolved = destination.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("FFmpeg runtime destination escapes the repository") from exc
    if destination.is_symlink():
        raise ValueError("FFmpeg runtime destination must not be a symlink")
    return destination


def _validate_tree(root: Path, revision: dict) -> dict[str, str]:
    expected_root = root / revision["runtime_dir"]
    if not expected_root.is_dir() or expected_root.is_symlink():
        raise ValueError(f"archive is missing expected root {revision['runtime_dir']}")
    for path in expected_root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"FFmpeg archive must not contain symlinks: {path}")
    hashes: dict[str, str] = {}
    for name in ("ffmpeg", "ffprobe"):
        item = revision["binaries"][name]
        binary = expected_root / item["path"]
        if not binary.is_file():
            raise ValueError(f"FFmpeg archive is missing {item['path']}")
        actual = _sha256(binary)
        if actual != item["sha256"]:
            raise ValueError(
                f"{name} SHA-256 mismatch: expected {item['sha256']}, got {actual}"
            )
        parsed, version = _version(binary, name)
        if version != revision["ffmpeg_version"]:
            raise ValueError(f"{name} version {version} does not match the revision")
        if parsed != tuple(map(int, revision["ffmpeg_version"].split("."))):
            raise ValueError(f"{name} version tuple does not match the revision")
        binary.chmod(0o755)
        hashes[name] = actual
    return hashes


def install_ffmpeg_runtime(
    *,
    archive_path: Path,
    revision_path: Path = REVISION,
    repo_root: Path = ROOT,
) -> dict:
    revision = _load_revision(revision_path)
    destination = _safe_destination(repo_root, revision)
    if destination.exists():
        if destination.is_symlink() or not destination.is_dir():
            raise RuntimeError(f"existing FFmpeg runtime is not a real directory: {destination}")
        hashes = {}
        for name in ("ffmpeg", "ffprobe"):
            binary = destination / revision["binaries"][name]["path"]
            if binary.is_symlink() or not binary.is_file():
                raise RuntimeError(f"existing {name} binary is not a real file: {binary}")
            actual = _sha256(binary)
            if actual != revision["binaries"][name]["sha256"]:
                raise RuntimeError(f"existing {name} binary hash mismatch: {binary}")
            _version(binary, name)
            hashes[name] = actual
        return {
            "status": "reused",
            "revision_id": revision["revision_id"],
            "ffmpeg": str(destination / revision["binaries"]["ffmpeg"]["path"]),
            "ffprobe": str(destination / revision["binaries"]["ffprobe"]["path"]),
            "binary_sha256": hashes,
        }

    archive = archive_path.expanduser().resolve(strict=True)
    actual_archive_sha256 = _sha256(archive)
    if actual_archive_sha256 != revision["archive"]["sha256"]:
        raise ValueError(
            "FFmpeg archive SHA-256 mismatch: expected "
            f"{revision['archive']['sha256']}, got {actual_archive_sha256}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".ffmpeg-install-", dir=destination.parent
    ) as temporary:
        temporary_root = Path(temporary)
        with tarfile.open(archive, "r:xz") as bundle:
            bundle.extractall(temporary_root, filter="data")
        top_level = [path.name for path in temporary_root.iterdir()]
        if top_level != [revision["runtime_dir"]]:
            raise ValueError(f"unexpected FFmpeg archive roots: {sorted(top_level)}")
        hashes = _validate_tree(temporary_root, revision)
        extracted = temporary_root / revision["runtime_dir"]
        marker = {
            "schema_version": "installed-ffmpeg-runtime/1.0",
            "revision_id": revision["revision_id"],
            "archive_sha256": actual_archive_sha256,
            "binary_sha256": hashes,
            "upstream_source": revision["upstream_source"],
            "network_protocols": revision["build"]["network_protocols"],
        }
        (extracted / ".osmo360-runtime.json").write_text(
            json.dumps(marker, indent=2) + "\n", encoding="utf-8"
        )
        for path in extracted.rglob("*"):
            if path.is_file() and path.name not in {"ffmpeg", "ffprobe"}:
                path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        os.rename(extracted, destination)

    return {
        "status": "installed",
        "revision_id": revision["revision_id"],
        "ffmpeg": str(destination / revision["binaries"]["ffmpeg"]["path"]),
        "ffprobe": str(destination / revision["binaries"]["ffprobe"]["path"]),
        "binary_sha256": hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--revision", type=Path, default=REVISION)
    args = parser.parse_args()
    result = install_ffmpeg_runtime(
        archive_path=args.archive,
        revision_path=args.revision.resolve(),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
