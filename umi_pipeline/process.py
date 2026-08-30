from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .manifest import CaptureManifest, ManifestError, ROOT, sha256


def _force_command(manifest: CaptureManifest) -> list[str]:
    data = manifest.data
    pipeline = data["pipeline"]
    command = [
        sys.executable,
        str(ROOT / "render_gripper_force_angle_demo.py"),
        str(manifest.identity_path("inputs", "raw_video")),
        "--source-osv",
        str(manifest.identity_path("inputs", "raw_video")),
        "--rig-revision",
        str(manifest.identity_path("revisions", "rig")),
        "--allow-diagnostic-rig",
        "--x5-angle-revision",
        str(manifest.identity_path("revisions", "jaw_angle")),
        "--base-tag-id",
        str(data["camera"]["base_tag_id"]),
        "--output-dir",
        str(manifest.output_path("force_angle_dir")),
        "--camera-profile",
        "insta360-x5-front",
        "--maximum-recovery-gap-s",
        str(pipeline["maximum_recovery_gap_s"]),
        "--display-relative-fingertip-force",
    ]
    for event in pipeline.get("contact_events_s", []):
        command.extend(("--contact-event-s", str(float(event))))
    return command


def _timeline_command(manifest: CaptureManifest) -> list[str]:
    data = manifest.data
    force_dir = manifest.output_path("force_angle_dir")
    return [
        sys.executable,
        str(ROOT / "render_single_gripper_webgl_demo.py"),
        str(manifest.identity_path("inputs", "raw_video")),
        "--source-osv",
        str(manifest.identity_path("inputs", "raw_video")),
        "--calibration",
        str(manifest.identity_path("inputs", "camera_identity")),
        "--force-angle-csv",
        str(force_dir / "force_angle_observations.csv"),
        "--force-angle-audit",
        str(force_dir / "audit.json"),
        "--rig-revision",
        str(manifest.identity_path("revisions", "rig")),
        "--marker-layout",
        str(manifest.identity_path("revisions", "marker_layout")),
        "--new-cad-source",
        str(manifest.identity_path("inputs", "new_cad_source")),
        "--output-dir",
        str(manifest.output_path("timeline_dir")),
        "--timeline-only",
        "--allow-diagnostic-rig",
        "--camera-hardware-model",
        "insta360-x5",
        "--camera-local-basetag",
    ]


def _verify_force_output(manifest: CaptureManifest) -> dict[str, Any]:
    audit_path = manifest.output_path("force_angle_dir") / "audit.json"
    if not audit_path.is_file():
        raise ManifestError(f"force output is incomplete: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected = manifest.data
    if audit.get("source", {}).get("osv_sha256") != expected["inputs"]["raw_video"]["sha256"]:
        raise ManifestError("force output raw-video identity mismatch")
    if audit.get("source", {}).get("base_tag_id") != expected["camera"]["base_tag_id"]:
        raise ManifestError("force output BaseTag binding mismatch")
    if audit.get("rig_revision", {}).get("sha256") != expected["revisions"]["rig"]["sha256"]:
        raise ManifestError("force output rig revision mismatch")
    if audit.get("angle", {}).get("revision", {}).get("sha256") != expected["revisions"]["jaw_angle"]["sha256"]:
        raise ManifestError("force output jaw-angle revision mismatch")
    return audit


def _verify_timeline_output(manifest: CaptureManifest) -> Path:
    path = manifest.output_path("timeline_dir") / "single_gripper_webgl_timeline.json"
    if not path.is_file():
        raise ManifestError(f"timeline output is incomplete: {path}")
    timeline = json.loads(path.read_text(encoding="utf-8"))
    if timeline.get("localization", {}).get("base_tag_id") != manifest.data["camera"]["base_tag_id"]:
        raise ManifestError("timeline BaseTag binding mismatch")
    expected_id = json.loads(
        manifest.identity_path("revisions", "jaw_angle").read_text(encoding="utf-8")
    )["revision_id"]
    actual_id = timeline.get("angle_models", {}).get("left", {}).get("revision", {}).get("id")
    if actual_id != expected_id:
        raise ManifestError("timeline jaw-angle revision mismatch")
    return path


def process_capture(manifest: CaptureManifest, *, dry_run: bool = False) -> dict[str, Any]:
    force_command = _force_command(manifest)
    timeline_command = _timeline_command(manifest)
    if dry_run:
        return {
            "capture_id": manifest.capture_id,
            "dry_run": True,
            "commands": [force_command, timeline_command],
        }

    force_dir = manifest.output_path("force_angle_dir")
    force_audit = force_dir / "audit.json"
    force_reused = force_audit.is_file()
    if force_dir.exists() and not force_reused:
        raise ManifestError(f"refusing incomplete existing force directory: {force_dir}")
    if not force_reused:
        subprocess.run(force_command, cwd=ROOT, check=True)
    audit = _verify_force_output(manifest)

    timeline_dir = manifest.output_path("timeline_dir")
    timeline_path = timeline_dir / "single_gripper_webgl_timeline.json"
    timeline_reused = timeline_path.is_file()
    if timeline_dir.exists() and not timeline_reused:
        raise ManifestError(f"refusing incomplete existing timeline directory: {timeline_dir}")
    if not timeline_reused:
        subprocess.run(timeline_command, cwd=ROOT, check=True)
    timeline_path = _verify_timeline_output(manifest)
    return {
        "capture_id": manifest.capture_id,
        "force_reused": force_reused,
        "timeline_reused": timeline_reused,
        "force_status": audit["status"],
        "timeline": str(timeline_path),
        "timeline_sha256": sha256(timeline_path),
    }
