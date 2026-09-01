from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


from osmo360.paths import ROOT
CAPTURE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,79}$")
PATH_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ManifestError(ValueError):
    pass


def validate_path_component(value: Any, *, field: str) -> str:
    """Return a manifest identifier that is safe as exactly one path component."""
    if not isinstance(value, str):
        raise ManifestError(f"{field} must be a string path component")
    component = value
    if component in {".", ".."} or PATH_COMPONENT.fullmatch(component) is None:
        raise ManifestError(
            f"{field} must be 1-128 ASCII letters, digits, '.', '_' or '-' and "
            "must start with a letter or digit"
        )
    return component


def confined_path(root: Path, *parts: str, field: str = "path") -> Path:
    """Resolve a path and fail closed if symlinks or traversal escape ``root``."""
    boundary = root.expanduser().resolve()
    raw_candidate = boundary.joinpath(*parts)
    try:
        relative = raw_candidate.relative_to(boundary)
    except ValueError:
        relative = None
    if relative is not None:
        current = boundary
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise ManifestError(f"{field} must not traverse a symlink: {current}")
    candidate = raw_candidate.resolve()
    try:
        candidate.relative_to(boundary)
    except ValueError as error:
        raise ManifestError(f"{field} must stay inside {boundary}: {candidate}") from error
    if candidate == boundary:
        raise ManifestError(f"{field} must not replace its root directory: {boundary}")
    return candidate


def publish_directory(source: Path, destination: Path, *, allowed_root: Path) -> None:
    """Publish a directory through a staged, recoverable same-filesystem rename."""
    source = source.resolve(strict=True)
    if not source.is_dir():
        raise ManifestError(f"publish source is not a directory: {source}")
    boundary = allowed_root.resolve(strict=True)
    raw_destination = destination.expanduser().absolute()
    try:
        relative = raw_destination.relative_to(boundary)
    except ValueError as error:
        raise ManifestError(
            f"publish destination must stay inside {boundary}: {raw_destination}"
        ) from error
    current = boundary
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ManifestError(f"publish destination must not traverse a symlink: {current}")
    destination = raw_destination.resolve()
    try:
        destination.relative_to(boundary)
    except ValueError as error:
        raise ManifestError(
            f"publish destination must stay inside {boundary}: {destination}"
        ) from error
    if destination == boundary:
        raise ManifestError(f"refusing to replace publish root: {boundary}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Re-resolve after mkdir so a pre-existing parent symlink cannot bypass the check.
    destination = destination.resolve()
    try:
        destination.relative_to(boundary)
    except ValueError as error:
        raise ManifestError(
            f"publish destination escaped {boundary} after directory creation: {destination}"
        ) from error

    nonce = uuid.uuid4().hex
    staged = destination.with_name(f".{destination.name}.publish-{nonce}")
    backup = destination.with_name(f".{destination.name}.backup-{nonce}")
    moved_previous = False
    try:
        shutil.copytree(source, staged)
        if destination.exists():
            if not destination.is_dir() or destination.is_symlink():
                raise ManifestError(
                    f"publish destination must be a real directory: {destination}"
                )
            destination.replace(backup)
            moved_previous = True
        staged.replace(destination)
    except BaseException:
        if moved_previous and backup.exists() and not destination.exists():
            backup.replace(destination)
        raise
    finally:
        if staged.exists():
            shutil.rmtree(staged)
    if moved_previous:
        shutil.rmtree(backup)


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
        revision_names = set(self.data.get("revisions", {}))
        allowed_revisions = required_revisions | {"relative_force"}
        if not required_revisions.issubset(revision_names) or not revision_names.issubset(
            allowed_revisions
        ):
            raise ManifestError(
                f"revisions must contain {sorted(required_revisions)} and may include relative_force"
            )
        for section, names in (
            ("inputs", required_inputs),
            ("revisions", revision_names),
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
