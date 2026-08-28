#!/usr/bin/env python3
"""Fail-closed loader for the locally deployed Insta360 Linux SDK revision."""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_REVISION = ROOT / "config/sdk_revisions/insta360_linux_camera_2_1_1_media_3_1_1.json"
SCHEMA = "insta360-sdk-revision/1.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str, base: Path = ROOT) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _verify_file(root: Path, reference: dict[str, Any], label: str) -> Path:
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError(f"{label} must contain exactly path and sha256")
    path = (root / str(reference["path"])).resolve()
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    actual = sha256(path)
    if actual != reference["sha256"]:
        raise ValueError(f"{label} hash mismatch: expected {reference['sha256']}, got {actual}")
    return path


def load_insta360_sdk_revision(
    path: Path = DEFAULT_REVISION, *, verify_source_archive: bool = False,
) -> dict[str, Any]:
    revision_path = path.resolve()
    revision = json.loads(revision_path.read_text(encoding="utf-8"))
    if revision.get("schema_version") != SCHEMA:
        raise ValueError("unsupported Insta360 SDK revision schema")
    if revision.get("platform") != "linux-x86_64":
        raise ValueError("Insta360 SDK revision is not for linux-x86_64")
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise ValueError("current host is incompatible with the Insta360 SDK revision")
    if verify_source_archive:
        source = revision.get("source_archive", {})
        archive = _resolve(str(source.get("path", "")))
        if not archive.is_file() or sha256(archive) != source.get("sha256"):
            raise ValueError("Insta360 source archive is missing or has a hash mismatch")
    media = revision.get("media_sdk", {})
    camera = revision.get("camera_sdk", {})
    media_root = _resolve(str(media.get("root", "")))
    camera_root = _resolve(str(camera.get("root", "")))
    media_binary = _verify_file(media_root, media.get("binary"), "MediaSDK binary")
    media_library = _verify_file(media_root, media.get("library"), "MediaSDK library")
    media_model = _verify_file(media_root, media.get("model_probe"), "MediaSDK model")
    camera_binary = _verify_file(camera_root, camera.get("binary"), "CameraSDK binary")
    camera_library = _verify_file(camera_root, camera.get("library"), "CameraSDK library")
    models = media_root / "models"
    if not models.is_dir():
        raise ValueError(f"MediaSDK models directory is missing: {models}")
    return {
        "revision": revision,
        "revision_path": revision_path,
        "revision_sha256": sha256(revision_path),
        "media_root": media_root,
        "media_binary": media_binary,
        "media_library": media_library,
        "media_model": media_model,
        "camera_root": camera_root,
        "camera_binary": camera_binary,
        "camera_library": camera_library,
    }
