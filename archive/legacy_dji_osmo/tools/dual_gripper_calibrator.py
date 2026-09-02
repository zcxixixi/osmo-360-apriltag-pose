#!/usr/bin/env python3
"""Interactive two-gripper relative-pose and URDF articulation calibrator."""

from __future__ import annotations

import json
import io
import math
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file, send_from_directory
import cv2
import numpy as np


from tools._root import ROOT
ASSET_ROOT = ROOT / "assets/osmo_rig/osmo定位.SLDASM"
MESH_ROOT = ASSET_ROOT / "meshes"
OLD_MESH_ROOT = ROOT.parent / "claw-urdf/extracted/claw-urdf/osmo定位.SLDASM/meshes"
PAD_COMPARE_ROOT = ROOT / "assets/hardware_compare"
OUTPUT_ROOT = ROOT / "sessions/dual-gripper-calibrations"
TEMPLATE_ROOT = ROOT / "dual_gripper_calibrator_web"
LOCAL_DATA_ROOT = Path(os.environ.get("DUAL_GRIPPER_DATA_ROOT", ROOT / ".local-data"))
TIMELINE_PATH = LOCAL_DATA_ROOT / "dual_gripper_timeline.json"
HARDWARE_MODEL_PATH = ROOT / "config/hardware_model.json"
UMI_OUTPUT_ROOT = LOCAL_DATA_ROOT / "vla-episode"
_UMI_ARRAYS = None

app = Flask(__name__, template_folder=str(TEMPLATE_ROOT))


def finite_number(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是数字") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} 必须是有限数字")
    return result


def validate_payload(raw: object) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("配置必须是 JSON 对象")
    pose = raw.get("right_gripper_in_left_frame")
    if not isinstance(pose, dict):
        raise ValueError("缺少 right_gripper_in_left_frame")
    translation = pose.get("translation_m")
    rotation = pose.get("rotation_rpy_deg")
    if not isinstance(translation, list) or len(translation) != 3:
        raise ValueError("translation_m 必须包含 3 个数")
    if not isinstance(rotation, list) or len(rotation) != 3:
        raise ValueError("rotation_rpy_deg 必须包含 3 个数")
    translation = [finite_number(v, f"translation_m[{i}]") for i, v in enumerate(translation)]
    rotation = [finite_number(v, f"rotation_rpy_deg[{i}]") for i, v in enumerate(rotation)]
    if max(abs(value) for value in translation) > 10.0:
        raise ValueError("相对位置超过 10 m，请检查单位")
    openings = raw.get("preview_opening_deg", {})
    if not isinstance(openings, dict):
        openings = {}
    left_opening = finite_number(openings.get("left", 20.0), "left opening")
    right_opening = finite_number(openings.get("right", 20.0), "right opening")
    if not (0 <= left_opening <= 90 and 0 <= right_opening <= 90):
        raise ValueError("开合角必须在 0–90°")
    center_poses = raw.get("grippers_in_center_frame", {})
    validated_center = {}
    for side, fallback_translation, fallback_rotation in (
        ("left", [-0.5 * value for value in translation], [0.0, 0.0, 0.0]),
        ("right", [0.5 * value for value in translation], rotation),
    ):
        candidate = center_poses.get(side, {}) if isinstance(center_poses, dict) else {}
        candidate_translation = candidate.get("translation_m", fallback_translation)
        candidate_rotation = candidate.get("rotation_rpy_deg", fallback_rotation)
        if not isinstance(candidate_translation, list) or len(candidate_translation) != 3:
            raise ValueError(f"{side} center translation 必须包含 3 个数")
        if not isinstance(candidate_rotation, list) or len(candidate_rotation) != 3:
            raise ValueError(f"{side} center rotation 必须包含 3 个数")
        validated_center[side] = {
            "translation_m": [finite_number(v, f"{side} translation[{i}]") for i, v in enumerate(candidate_translation)],
            "rotation_rpy_deg": [finite_number(v, f"{side} rotation[{i}]") for i, v in enumerate(candidate_rotation)],
        }
    return {
        "schema_version": "dual-gripper-relative-extrinsic/v1",
        "calibration_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reference_frame": "midpoint_between_gripper_bases",
        "grippers_in_center_frame": validated_center,
        "right_gripper_in_left_frame": {
            "translation_m": translation,
            "rotation_rpy_deg": rotation,
            "rotation_order": "XYZ",
        },
        "preview_opening_deg": {"left": left_opening, "right": right_opening},
        "model": {
            "urdf": "assets/osmo_rig/osmo定位.SLDASM/urdf/osmo定位.SLDASM.urdf",
            "display_origin": "base_midpoint",
            "units": "m, deg",
        },
        "note": str(raw.get("note", ""))[:500],
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/animation")
def animation():
    return render_template("animation.html")


@app.get("/hardware")
def hardware():
    return send_file(TEMPLATE_ROOT / "hardware.html", mimetype="text/html", conditional=True)


@app.get("/hardware-compare")
def hardware_compare():
    return send_file(TEMPLATE_ROOT / "hardware_compare.html", mimetype="text/html", conditional=True)


@app.get("/pad-compare")
def pad_compare():
    return send_file(TEMPLATE_ROOT / "pad_compare.html", mimetype="text/html", conditional=True)


@app.get("/api/hardware-model")
def hardware_model():
    if not HARDWARE_MODEL_PATH.is_file():
        return jsonify(error="hardware model not found"), 404
    return send_file(HARDWARE_MODEL_PATH, mimetype="application/json", conditional=True)


def umi_arrays():
    global _UMI_ARRAYS
    if _UMI_ARRAYS is None:
        _UMI_ARRAYS = np.load(UMI_OUTPUT_ROOT / "episode_arrays.npz")
    return _UMI_ARRAYS


def umi_dataset_available() -> bool:
    return all((UMI_OUTPUT_ROOT / name).is_file() for name in (
        "episode_arrays.npz",
        "episode_metadata.json",
        "quality_report.json",
    ))


@app.get("/umi")
def umi_explainer():
    return send_file(TEMPLATE_ROOT / "umi.html", mimetype="text/html", conditional=True)


@app.get("/api/umi-summary")
def umi_summary():
    if not umi_dataset_available():
        return jsonify(error="UMI dataset not loaded; set DUAL_GRIPPER_DATA_ROOT"), 404
    arrays = umi_arrays()
    metadata = json.loads((UMI_OUTPUT_ROOT / "episode_metadata.json").read_text(encoding="utf-8"))
    report = json.loads((UMI_OUTPUT_ROOT / "quality_report.json").read_text(encoding="utf-8"))
    return jsonify(
        metadata=metadata,
        quality=report,
        timestamp_s=np.round(arrays["timestamp_s"], 3).tolist(),
        robots=[{
            "position_m": np.round(arrays[f"robot{i}_eef_pos"], 5).tolist(),
            "raw_position_m": np.round(arrays[f"robot{i}_eef_pos_raw"], 5).tolist()
            if f"robot{i}_eef_pos_raw" in arrays else np.round(arrays[f"robot{i}_eef_pos"], 5).tolist(),
            "rotation_axis_angle": np.round(arrays[f"robot{i}_eef_rot_axis_angle"], 5).tolist(),
            "gripper_width_mm": np.round(arrays[f"robot{i}_gripper_width"][:, 0] * 1000, 2).tolist(),
            "tracked": arrays[f"robot{i}_pose_tracked"].astype(int).tolist(),
            "direct_tag": arrays[f"robot{i}_pose_measured"].astype(int).tolist(),
            "recovered": arrays[f"robot{i}_pose_recovered"].astype(int).tolist()
            if f"robot{i}_pose_recovered" in arrays else np.zeros(len(arrays["timestamp_s"]), dtype=int).tolist(),
            "outlier_rejected": arrays[f"robot{i}_pose_outlier_rejected"].astype(int).tolist()
            if f"robot{i}_pose_outlier_rejected" in arrays else np.zeros(len(arrays["timestamp_s"]), dtype=int).tolist(),
        } for i in range(2)],
        action_shape=list(arrays["action"].shape),
        action_valid_ratio=float(np.mean(arrays["action_valid"])) if "action_valid" in arrays else 1.0,
    )


@app.get("/api/umi-frame/<int:camera_id>/<int:frame>")
def umi_frame(camera_id: int, frame: int):
    if camera_id not in (0, 1):
        return jsonify(error="camera must be 0 or 1"), 404
    if not umi_dataset_available():
        return jsonify(error="UMI dataset not loaded; set DUAL_GRIPPER_DATA_ROOT"), 404
    images = umi_arrays()[f"camera{camera_id}_rgb"]
    index = int(np.clip(frame, 0, len(images) - 1))
    bgr = cv2.cvtColor(images[index], cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        return jsonify(error="frame encoding failed"), 500
    response = send_file(io.BytesIO(encoded.tobytes()), mimetype="image/jpeg", max_age=0)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["X-UMI-Frame"] = str(index)
    return response


@app.get("/api/animation-timeline")
def animation_timeline():
    if not TIMELINE_PATH.is_file():
        return jsonify(error="animation timeline not found"), 404
    return send_file(TIMELINE_PATH, mimetype="application/json", conditional=True)


@app.get("/mesh/<path:name>")
def mesh(name: str):
    if name not in {"base_link.STL", "Link1.STL", "Link2.STL", "Link3.STL"}:
        return jsonify(error="unknown mesh"), 404
    return send_from_directory(MESH_ROOT, name)


@app.get("/compare-mesh/<version>/<path:name>")
def compare_mesh(version: str, name: str):
    if name not in {"base_link.STL", "Link1.STL", "Link2.STL", "Link3.STL"}:
        return jsonify(error="unknown mesh"), 404
    roots = {"old": OLD_MESH_ROOT, "current": MESH_ROOT}
    root = roots.get(version)
    if root is None:
        return jsonify(error="unknown version"), 404
    return send_from_directory(root, name)


@app.get("/pad-mesh/<version>")
def pad_mesh(version: str):
    files = {
        "old": "old_pad_right.STL",
        "new": "new_pad_vol2_aligned.STL",
    }
    name = files.get(version)
    if name is None:
        return jsonify(error="unknown version"), 404
    return send_from_directory(PAD_COMPARE_ROOT, name)


@app.post("/api/save")
def save():
    try:
        payload = validate_payload(request.get_json(silent=True))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_ROOT / f"{payload['calibration_id']}.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return jsonify(
        calibration_id=payload["calibration_id"],
        path=str(output.resolve()),
        config=payload,
    )


@app.get("/api/health")
def health():
    return jsonify(ok=True, meshes=all((MESH_ROOT / name).is_file() for name in (
        "base_link.STL", "Link1.STL", "Link2.STL", "Link3.STL",
    )))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7861, threaded=True)
