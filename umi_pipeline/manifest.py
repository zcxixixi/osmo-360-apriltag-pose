from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CAPTURE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,79}$")


class ManifestError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CaptureManifest:
    path: Path
    data: dict[str, Any]

    @property
    def capture_id(self) -> str:
        return str(self.data["capture_id"])

    def identity_path(self, section: str, name: str) -> Path:
        identity = self.data[section][name]
        value = Path(identity["path"])
        return value if value.is_absolute() else ROOT / value

    def output_path(self, name: str) -> Path:
        value = Path(self.data["outputs"][name])
        if not value.is_absolute():
            raise ManifestError(f"outputs.{name}.path must be absolute")
        return value

    def verify_identity(self, section: str, name: str) -> Path:
        identity = self.data[section][name]
        path = self.identity_path(section, name)
        if not path.is_file():
            raise ManifestError(f"{section}.{name} is missing: {path}")
        expected = str(identity.get("sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ManifestError(f"{section}.{name}.sha256 is invalid")
        actual = sha256(path)
        if actual != expected:
            raise ManifestError(
                f"{section}.{name} hash mismatch: expected {expected}, got {actual}"
            )
        return path

    def verify(self) -> None:
        if self.data.get("schema_version") != "capture-manifest/1.0":
            raise ManifestError("schema_version must be capture-manifest/1.0")
        if not CAPTURE_ID.fullmatch(self.capture_id):
            raise ManifestError(f"invalid capture_id: {self.capture_id!r}")
        camera = self.data.get("camera", {})
        if camera.get("model") != "Insta360 X5":
            raise ManifestError("the first manifest profile supports Insta360 X5 only")
        serial = camera.get("serial")
        if serial is not None and not re.fullmatch(r"[A-Z0-9]{10,20}", str(serial)):
            raise ManifestError("camera.serial must be 10-20 uppercase alphanumeric characters")
        if camera.get("role") not in {"physical_left", "physical_right"}:
            raise ManifestError("camera.role must be physical_left or physical_right")
        if camera.get("base_tag_id") not in {2, 3}:
            raise ManifestError("camera.base_tag_id must be 2 or 3")
        pipeline = self.data.get("pipeline", {})
        if pipeline.get("profile") != "single_x5_camera_local":
            raise ManifestError("unsupported pipeline.profile")
        if float(pipeline.get("maximum_recovery_gap_s", -1)) != 0.25:
            raise ManifestError("maximum_recovery_gap_s must be 0.25")
        required_inputs = {"raw_video", "camera_identity", "new_cad_source"}
        required_revisions = {"rig", "jaw_angle", "marker_layout", "renderer"}
        if set(self.data.get("inputs", {})) != required_inputs:
            raise ManifestError(f"inputs must contain exactly {sorted(required_inputs)}")
        if set(self.data.get("revisions", {})) != required_revisions:
            raise ManifestError(
                f"revisions must contain exactly {sorted(required_revisions)}"
            )
        for section, names in (
            ("inputs", required_inputs),
            ("revisions", required_revisions),
        ):
            for name in names:
                self.verify_identity(section, name)
        identity = json.loads(
            self.identity_path("inputs", "camera_identity").read_text(encoding="utf-8")
        )
        if serial is not None and identity.get("serial") != serial:
            raise ManifestError("camera serial does not match inputs.camera_identity")
        outputs = self.data.get("outputs", {})
        if set(outputs) != {"force_angle_dir", "timeline_dir", "review_bundle_dir"}:
            raise ManifestError("outputs must define force_angle_dir, timeline_dir, review_bundle_dir")
        for name in outputs:
            self.output_path(name)


def load_manifest(path: str | Path, *, verify: bool = True) -> CaptureManifest:
    manifest_path = Path(path).expanduser().resolve(strict=True)
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ManifestError(f"invalid manifest JSON: {error}") from error
    if not isinstance(data, dict):
        raise ManifestError("manifest must be a JSON object")
    manifest = CaptureManifest(manifest_path, data)
    if verify:
        manifest.verify()
    return manifest
