from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .manifest import CaptureManifest, ManifestError, ROOT, sha256
from .process import process_capture


def _bundle_sources(manifest: CaptureManifest) -> dict[str, Path]:
    force_dir = manifest.output_path("force_angle_dir")
    timeline_dir = manifest.output_path("timeline_dir")
    scene = manifest.identity_path("revisions", "renderer")
    scene_hash = manifest.data["revisions"]["renderer"]["sha256"]
    return {
        "timeline.json": timeline_dir / "single_gripper_webgl_timeline.json",
        "front-video.mp4": force_dir / "jaw_angle_marker_overlay.mp4",
        "audit.json": force_dir / "audit.json",
        "capture-manifest.json": manifest.path,
        f"scene.{scene_hash[:12]}.html": scene,
        "scene.html": scene,
    }


def _verify_existing_bundle(directory: Path) -> dict[str, str]:
    checksums_path = directory / "checksums.json"
    if not checksums_path.is_file():
        raise ManifestError(f"existing review bundle is incomplete: {directory}")
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    if not isinstance(checksums, dict) or not checksums:
        raise ManifestError("review bundle checksums.json is invalid")
    for name, expected in checksums.items():
        path = directory / name
        if not path.is_file() or sha256(path) != expected:
            raise ManifestError(f"review bundle identity mismatch: {name}")
    return checksums


def build_review_bundle(manifest: CaptureManifest) -> dict[str, Any]:
    directory = manifest.output_path("review_bundle_dir")
    if directory.exists():
        checksums = _verify_existing_bundle(directory)
        return {"directory": str(directory), "reused": True, "checksums": checksums}

    sources = _bundle_sources(manifest)
    for name, source in sources.items():
        if not source.is_file():
            raise ManifestError(f"review source is missing: {name}: {source}")
    temporary = directory.with_name(directory.name + ".part")
    if temporary.exists():
        raise ManifestError(f"stale temporary review bundle exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        for name, source in sources.items():
            shutil.copy2(source, temporary / name)
        checksums = {
            name: sha256(temporary / name)
            for name in sorted(sources)
        }
        (temporary / "checksums.json").write_text(
            json.dumps(checksums, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        directory.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(directory)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {"directory": str(directory), "reused": False, "checksums": checksums}


def publish_review_bundle(manifest: CaptureManifest, directory: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "tools.upload_visualization_bundle",
        "--timeline",
        str(directory / "timeline.json"),
        "--video",
        str(directory / "front-video.mp4"),
        "--scene",
        str(directory / "scene.html"),
        "--name",
        str(manifest.data.get("review", {}).get("name", manifest.capture_id)),
    ]
    server = manifest.data.get("review", {}).get("server")
    if server:
        command.extend(("--server", str(server)))
    process = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(process.stdout)


def review_capture(
    manifest: CaptureManifest,
    *,
    publish: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    if dry_run:
        process = process_capture(manifest, dry_run=True)
        return {
            "capture_id": manifest.capture_id,
            "dry_run": True,
            "process": process,
            "bundle_dir": str(manifest.output_path("review_bundle_dir")),
            "publish": publish,
        }
    process = process_capture(manifest)
    bundle = build_review_bundle(manifest)
    result: dict[str, Any] = {
        "capture_id": manifest.capture_id,
        "process": process,
        "bundle": bundle,
    }
    if publish:
        result["published"] = publish_review_bundle(
            manifest, Path(bundle["directory"])
        )
    return result
