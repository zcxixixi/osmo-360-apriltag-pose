#!/usr/bin/env python3
"""Fail-closed loader for immutable v52+ dual-gripper rig revisions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from osmo360.localization.world_frames import compile_world_tag_map


from osmo360.paths import ROOT
SCHEMA = "dual-gripper-rig-revision/1.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _load_reference(revision: dict[str, Any], key: str) -> tuple[Path, dict[str, Any]]:
    reference = revision.get(key)
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError(f"rig revision {key} must contain exactly path and sha256")
    path = _resolve(str(reference["path"]))
    if not path.is_file():
        raise ValueError(f"rig revision {key} does not exist: {path}")
    actual = sha256(path)
    if actual != reference["sha256"]:
        raise ValueError(
            f"rig revision {key} hash mismatch: expected {reference['sha256']}, got {actual}"
        )
    return path, json.loads(path.read_text(encoding="utf-8"))


def _vector(value: Any, *, length: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {length} finite numbers")
    return array


def _validate_hardware(hardware: dict[str, Any]) -> None:
    if hardware.get("schema_version") != "gripper-hardware-reference/1.0":
        raise ValueError("unsupported hardware schema")
    robots = hardware.get("robots")
    if not isinstance(robots, dict) or set(robots) != {"left", "right"}:
        raise ValueError("hardware must define exactly physical left and right roles")
    serials: list[str] = []
    tag_ids: list[int] = []
    for role in ("left", "right"):
        robot = robots[role]
        serial = robot.get("camera_serial")
        if not isinstance(serial, str) or not serial:
            raise ValueError(f"hardware {role} camera_serial is missing")
        serials.append(serial)
        tag_ids.append(int(robot["base_tag_id"]))
        transform = robot.get("camera_to_eef_reference")
        if not isinstance(transform, dict):
            raise ValueError(f"hardware {role} camera_to_eef_reference is missing")
        _vector(transform.get("translation_m"), length=3, name=f"hardware {role} translation")
        _vector(transform.get("quaternion_xyzw"), length=4, name=f"hardware {role} quaternion")
        if robot.get("camera_to_tcp_verified") is not True:
            raise ValueError(f"hardware {role} camera-to-TCP is not verified")
    if len(set(serials)) != 2:
        raise ValueError("left and right camera serials must be unique")
    if len(set(tag_ids)) != 2:
        raise ValueError("left and right BaseTag IDs must be unique")


def _validate_geometry(geometry: dict[str, Any], hardware: dict[str, Any]) -> None:
    if geometry.get("schema_version") != "gripper-rigid-geometry/1.0":
        raise ValueError("unsupported gripper geometry schema")
    tag_size = float(geometry.get("basetag", {}).get("tag_outer_size_m", 0.0))
    if tag_size <= 0.0:
        raise ValueError("BaseTag size must be positive")
    base_tag = geometry.get("base_to_tag", {})
    base_tcp = geometry.get("base_to_tcp", {})
    if (base_tag.get("parent_frame"), base_tag.get("child_frame")) != (
        "base_link", "basetag"
    ):
        raise ValueError("base_to_tag frame direction is invalid")
    if (base_tcp.get("parent_frame"), base_tcp.get("child_frame")) != (
        "base_link", "gripper_tcp"
    ):
        raise ValueError("base_to_tcp frame direction is invalid")
    _vector(base_tag.get("translation_m"), length=3, name="base_to_tag translation")
    _vector(base_tag.get("quaternion_xyzw"), length=4, name="base_to_tag quaternion")
    geometry_tcp = _vector(
        base_tcp.get("translation_m"), length=3, name="base_to_tcp translation"
    )
    _vector(base_tcp.get("quaternion_xyzw"), length=4, name="base_to_tcp quaternion")
    hardware_tcp = _vector(
        hardware.get("eef_reference", {}).get("base_to_tcp_translation_m"),
        length=3,
        name="hardware base_to_tcp translation",
    )
    if not np.array_equal(geometry_tcp, hardware_tcp):
        raise ValueError("hardware and geometry define conflicting base_to_tcp translations")


def _validate_cad_revision(
    geometry: dict[str, Any],
) -> tuple[Path | None, dict[str, Any] | None]:
    reference = geometry.get("cad_revision")
    if reference is None:
        return None, None
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("cad_revision must contain exactly path and sha256")
    path = _resolve(str(reference["path"]))
    if not path.is_file():
        raise ValueError(f"CAD revision does not exist: {path}")
    actual = sha256(path)
    if actual != reference["sha256"]:
        raise ValueError(
            f"CAD revision hash mismatch: expected {reference['sha256']}, got {actual}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "gripper-cad-revision/1.0":
        raise ValueError("unsupported CAD revision schema")
    if payload.get("mesh_units") != "m":
        raise ValueError("v52 CAD meshes must use metre units")
    mesh_directory = _resolve(str(payload["mesh_directory"]))
    expected_meshes = {"base_link.STL", "Link1.STL", "Link2.STL", "Link3.STL"}
    meshes = payload.get("meshes")
    if not isinstance(meshes, dict) or set(meshes) != expected_meshes:
        raise ValueError("CAD revision must pin exactly four gripper meshes")
    for name, expected_hash in meshes.items():
        mesh = mesh_directory / name
        if not mesh.is_file() or sha256(mesh) != expected_hash:
            raise ValueError(f"CAD mesh hash mismatch: {mesh}")
    for key in ("urdf", "source_pad"):
        item = payload.get(key)
        if not isinstance(item, dict) or "path" not in item or "sha256" not in item:
            raise ValueError(f"CAD revision {key} reference is incomplete")
        asset = _resolve(str(item["path"]))
        if not asset.is_file() or sha256(asset) != item["sha256"]:
            raise ValueError(f"CAD revision {key} hash mismatch: {asset}")
    source_pad = payload["source_pad"]
    if source_pad.get("source_units") != "mm" or source_pad.get("scale_to_mesh_units") != 0.001:
        raise ValueError("millimetre pad STL must declare an explicit 0.001 scale")
    joint3 = _vector(
        payload.get("joint_origins_m", {}).get("joint3_basetag"),
        length=3,
        name="CAD joint3 BaseTag origin",
    )
    base_tag = _vector(
        geometry.get("base_to_tag", {}).get("translation_m"),
        length=3,
        name="geometry base_to_tag translation",
    )
    if not np.array_equal(joint3, base_tag):
        raise ValueError("CAD joint3 and geometry base_to_tag conflict")
    if payload.get("tcp_status") != geometry.get("base_to_tcp", {}).get("status"):
        raise ValueError("CAD and geometry TCP status conflict")
    return path, payload


def _validate_world_map(
    revision: dict[str, Any], path: Path, payload: dict[str, Any],
    *, allow_diagnostic_world: bool = False,
) -> dict[str, Any]:
    reference = revision["world_tag_map"]
    expected_keys = {"path", "sha256", "compiled_sha256", "map_id"}
    if set(reference) != expected_keys:
        raise ValueError("world_tag_map must pin path, file hash, compiled hash, and map_id")
    if sha256(path) != reference["sha256"]:
        raise ValueError("world Tag map file hash mismatch")
    compiled = compile_world_tag_map(path)
    if compiled.get("tag_map_sha256") != reference["compiled_sha256"]:
        raise ValueError("world Tag map compiled hash mismatch")
    if payload.get("map_id") != reference["map_id"]:
        raise ValueError("world Tag map ID mismatch")
    if payload.get("calibration_status") != "VERIFIED":
        diagnostic = "DIAGNOSTIC" in str(payload.get("calibration_status", ""))
        if not (
            allow_diagnostic_world
            and revision.get("training_ready") is False
            and diagnostic
        ):
            raise ValueError("world Tag map is not VERIFIED")
    return compiled


def _validate_policy(
    revision: dict[str, Any], hardware: dict[str, Any], world_map: dict[str, Any]
) -> None:
    policy = revision.get("accuracy_first_policy")
    if not isinstance(policy, dict):
        raise ValueError("accuracy_first_policy is missing")
    required = {
        "maximum_pose_angular_rmse_deg", "require_both_wall_panels",
        "wall_panel_id_groups", "maximum_cross_bearing_error_deg",
        "maximum_trusted_position_step_m", "maximum_interpolation_gap_s",
        "allow_metric_smoothing", "allow_dual_fisheye_position_fill",
        "allow_hidden_contact_constraint",
    }
    if set(policy) != required:
        raise ValueError("accuracy_first_policy fields do not match the v52 schema")
    if policy["require_both_wall_panels"] is not True:
        raise ValueError("v52 accuracy-first policy must require both wall panels")
    for key in (
        "allow_metric_smoothing", "allow_dual_fisheye_position_fill",
        "allow_hidden_contact_constraint",
    ):
        if policy[key] is not False:
            raise ValueError(f"v52 accuracy-first policy forbids {key}")
    groups = [set(map(int, group)) for group in policy["wall_panel_id_groups"]]
    if len(groups) != 2 or not groups[0] or not groups[1] or groups[0] & groups[1]:
        raise ValueError("wall panel Tag groups must be two non-empty disjoint sets")
    expected_world_ids = {int(tag["id"]) for tag in world_map["tags"]}
    if groups[0] | groups[1] != expected_world_ids:
        raise ValueError("wall panel Tag groups do not match the compiled world map")
    base_ids = {int(robot["base_tag_id"]) for robot in hardware["robots"].values()}
    if base_ids & expected_world_ids:
        raise ValueError("BaseTag IDs must not overlap wall Tag IDs")


def load_rig_revision(
    path: Path, *, allow_diagnostic_world: bool = False,
) -> dict[str, Any]:
    revision_path = path.resolve()
    revision = json.loads(revision_path.read_text(encoding="utf-8"))
    if revision.get("schema_version") != SCHEMA:
        raise ValueError(f"unsupported rig revision schema: {revision.get('schema_version')}")
    if revision.get("role_binding_source") != "hardware":
        raise ValueError("role_binding_source must be hardware")
    hardware_path, hardware = _load_reference(revision, "hardware")
    geometry_path, geometry = _load_reference(revision, "gripper_geometry")
    world_reference = revision.get("world_tag_map")
    if not isinstance(world_reference, dict) or "path" not in world_reference:
        raise ValueError("world_tag_map reference is missing")
    world_path = _resolve(str(world_reference["path"]))
    if not world_path.is_file():
        raise ValueError(f"world Tag map does not exist: {world_path}")
    world_payload = json.loads(world_path.read_text(encoding="utf-8"))
    _validate_hardware(hardware)
    _validate_geometry(geometry, hardware)
    cad_revision_path, cad_revision = _validate_cad_revision(geometry)
    compiled_world = _validate_world_map(
        revision, world_path, world_payload,
        allow_diagnostic_world=allow_diagnostic_world,
    )
    _validate_policy(revision, hardware, compiled_world)
    frames = revision.get("expected_frames", {})
    if frames.get("world_frame") != compiled_world.get("world_frame"):
        raise ValueError("rig revision world frame conflicts with world Tag map")
    if not np.array_equal(
        _vector(frames.get("physical_up_vector"), length=3, name="physical up"),
        _vector(world_payload.get("physical_up_vector"), length=3, name="map physical up"),
    ):
        raise ValueError("rig revision physical up conflicts with world Tag map")
    return {
        "revision": revision,
        "revision_path": revision_path,
        "revision_sha256": sha256(revision_path),
        "hardware": hardware,
        "hardware_path": hardware_path,
        "geometry": geometry,
        "geometry_path": geometry_path,
        "cad_revision": cad_revision,
        "cad_revision_path": cad_revision_path,
        "world_map": compiled_world,
        "world_map_path": world_path,
        "policy": revision["accuracy_first_policy"],
    }
