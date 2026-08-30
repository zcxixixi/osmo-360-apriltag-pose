#!/usr/bin/env python3
"""Verify the frozen dual-gripper v50 visual baseline and its provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_LOCK = ROOT / "config/baselines/dual_gripper_v50_accepted_baseline.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quaternion_step_deg(quaternions: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=float)
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    cosine = np.clip(np.abs(np.sum(quaternions[:-1] * quaternions[1:], axis=1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(cosine))


def check(condition: bool, name: str, value: Any, failures: list[dict[str, Any]]) -> None:
    if not condition:
        failures.append({"name": name, "value": value})


def in_range(value: float, bounds: list[float]) -> bool:
    return float(bounds[0]) <= value <= float(bounds[1])


def verify_hashes(lock: dict[str, Any], groups: list[str],
                  failures: list[dict[str, Any]]) -> int:
    checked = 0
    for group in groups:
        for item in lock[group]:
            path = Path(item["path"])
            if not path.is_absolute():
                path = ROOT / path
            if not path.is_file():
                failures.append({"name": f"{group}.file_exists", "value": str(path)})
                continue
            actual = sha256(path)
            checked += 1
            check(actual == item["sha256"], f"{group}.sha256", {
                "path": str(path), "expected": item["sha256"], "actual": actual,
            }, failures)
    return checked


def verify_action(lock: dict[str, Any], failures: list[dict[str, Any]]) -> dict[str, Any]:
    action = lock["accepted_metrics"]["claw_to_claw_action"]
    gates = lock["regression_gates"]["claw_to_claw_action"]
    timeline_path = next(
        Path(item["path"]) for item in lock["protected_artifacts"]
        if item["path"].endswith("dual_gripper_claw_to_claw_action_v50_fixed_timeline.json")
    )
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    frames = timeline["frames"]
    left = np.asarray([frame["left"]["p"] for frame in frames], dtype=float)
    right = np.asarray([frame["right"]["p"] for frame in frames], dtype=float)
    separation = np.linalg.norm(left - right, axis=1)
    left_step = quaternion_step_deg(np.asarray([frame["left"]["q"] for frame in frames]))
    right_step = quaternion_step_deg(np.asarray([frame["right"]["q"] for frame in frames]))
    visible = {
        side: float(np.mean([frame[side].get("visible", True) for frame in frames]))
        for side in ("left", "right")
    }
    measured = {
        "frames": len(frames),
        "duration_s": float(timeline["duration_s"]),
        "visible_ratio": visible,
        "tcp_separation_min_m": float(separation.min()),
        "tcp_separation_median_m": float(np.median(separation)),
        "tcp_separation_p95_m": float(np.quantile(separation, 0.95)),
        "closest_time_s": float(frames[int(np.argmin(separation))]["t"]),
        "left_orientation_step_p95_deg": float(np.quantile(left_step, 0.95)),
        "right_orientation_step_p95_deg": float(np.quantile(right_step, 0.95)),
    }
    check(len(frames) == action["frames"], "action.frames", measured["frames"], failures)
    check(math.isclose(measured["duration_s"], gates["duration_s"], abs_tol=1e-9),
          "action.duration_s", measured["duration_s"], failures)
    check(min(visible.values()) >= gates["weak_tracked_ratio_min"],
          "action.visible_ratio", visible, failures)
    check(in_range(measured["tcp_separation_min_m"], gates["tcp_separation_min_m_range"]),
          "action.tcp_separation_min_m", measured["tcp_separation_min_m"], failures)
    check(in_range(measured["tcp_separation_median_m"], gates["tcp_separation_median_m_range"]),
          "action.tcp_separation_median_m", measured["tcp_separation_median_m"], failures)
    check(measured["left_orientation_step_p95_deg"] <= gates["left_orientation_step_p95_deg_max"],
          "action.left_orientation_step_p95_deg",
          measured["left_orientation_step_p95_deg"], failures)
    check(measured["right_orientation_step_p95_deg"] <= gates["right_orientation_step_p95_deg_max"],
          "action.right_orientation_step_p95_deg",
          measured["right_orientation_step_p95_deg"], failures)

    report_path = next(
        Path(item["path"]) for item in lock["protected_artifacts"]
        if item["path"].endswith("fused-world-v50-v15-regression-fixed/report.json")
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    invariants = lock["algorithm_invariants"]
    check(report["weak_attitude_source"] == invariants["weak_attitude_source"],
          "fusion.weak_attitude_source", report["weak_attitude_source"], failures)
    for key in ("screen_same_id_used", "contact_constraint_used", "synthetic_frames_used"):
        check(report[key] is invariants[key], f"fusion.{key}", report[key], failures)
    coverage = report["coverage"]
    check(coverage["weak_ratio"] >= gates["weak_tracked_ratio_min"],
          "fusion.weak_ratio", coverage["weak_ratio"], failures)
    check(coverage["untrusted_long_gap_frames"] <= gates["untrusted_long_gap_frames_max"],
          "fusion.untrusted_long_gap_frames", coverage["untrusted_long_gap_frames"], failures)
    check(coverage["maximum_allowed_interpolation_gap_s"]
          == invariants["maximum_interpolation_gap_s"],
          "fusion.maximum_allowed_interpolation_gap_s",
          coverage["maximum_allowed_interpolation_gap_s"], failures)
    return measured


def verify_calibration(lock: dict[str, Any], failures: list[dict[str, Any]]) -> dict[str, Any]:
    audit_path = next(
        Path(item["path"]) for item in lock["protected_artifacts"]
        if item["path"].endswith("v15_regression_audit_v49_vs_v50.json")
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    candidate = audit["candidates"][
        "together_fused_world_v50_v15_regression_fixed_full_timeline.json"
    ]
    gates = lock["regression_gates"]["calibration_vs_v15"]
    for side in ("left", "right"):
        values = candidate[side]
        position = values["position_rigid_aligned_residual_mm"]["p95"]
        orientation = values["orientation_constant_frame_aligned_error_deg"]["p95"]
        correlation = values["orientation_increment_correlation"]
        check(position <= gates["position_rigid_aligned_p95_mm_max"],
              f"calibration.{side}.position_p95_mm", position, failures)
        check(orientation <= gates["orientation_constant_frame_aligned_p95_deg_max"],
              f"calibration.{side}.orientation_p95_deg", orientation, failures)
        check(correlation >= gates["orientation_increment_correlation_min"],
              f"calibration.{side}.orientation_increment_correlation", correlation, failures)
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--skip-large-input-hashes", action="store_true")
    parser.add_argument("--skip-code-hashes", action="store_true")
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    failures: list[dict[str, Any]] = []
    groups = ["protected_artifacts"]
    if not args.skip_code_hashes:
        groups.append("protected_code")
    if not args.skip_large_input_hashes:
        groups.append("protected_inputs")
    checked = verify_hashes(lock, groups, failures)
    action = verify_action(lock, failures)
    calibration = verify_calibration(lock, failures)
    result = {
        "baseline_id": lock["baseline_id"],
        "status": "PASS" if not failures else "FAIL",
        "hashes_checked": checked,
        "action_metrics": action,
        "calibration_metrics": calibration,
        "failures": failures,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
