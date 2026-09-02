#!/usr/bin/env python3
"""Verify the frozen X5 BaseTag2 one-sided relative-force visual baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


from osmo360.paths import ROOT
BASELINE = ROOT / "config/baselines/x5_left_one_sided_force_insta_only_accepted_20260902.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_hashes(entries: list[dict], failures: list[str]) -> int:
    checked = 0
    for entry in entries:
        value = Path(entry["path"])
        path = value if value.is_absolute() else ROOT / value
        if not path.is_file():
            failures.append(f"missing protected file: {path}")
            continue
        actual = sha256(path)
        checked += 1
        if actual != entry["sha256"]:
            failures.append(
                f"hash mismatch for {path}: expected {entry['sha256']}, got {actual}"
            )
    return checked


def main() -> int:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    failures: list[str] = []
    if baseline.get("schema_version") != "x5-one-sided-force-baseline-lock/1.0":
        failures.append("invalid baseline schema")
    checked = check_hashes(baseline["protected_code"], failures)
    checked += check_hashes(baseline["protected_artifacts"], failures)

    audit_path = Path(baseline["protected_artifacts"][0]["path"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    gates = baseline["regression_gates"]
    tag_counts = audit["source"]["base_tag_frame_counts"]
    if int(tag_counts["2"]) < int(gates["base_tag2_frames_min"]):
        failures.append("BaseTag2 frame count regressed")
    if int(tag_counts["3"]) > int(gates["base_tag3_frames_max"]):
        failures.append("unexpected BaseTag3 frames")
    angle = audit["angle"]
    if float(angle["available_frame_ratio"]) < float(
        gates["angle_available_frame_ratio_min"]
    ):
        failures.append("angle availability regressed")
    fallback = angle["one_sided_fallback"]
    if int(fallback["one_sided_left_frames"]) < int(
        gates["one_sided_left_frames_min"]
    ):
        failures.append("left-only angle coverage regressed")
    if int(fallback["one_sided_right_frames"]) < int(
        gates["one_sided_right_frames_min"]
    ):
        failures.append("right-only angle coverage regressed")
    force = audit["force"]
    if float(force["measured_frame_ratio"]) < float(
        gates["force_measured_frame_ratio_min"]
    ):
        failures.append("relative-force coverage regressed")

    timeline_entry = next(
        entry
        for entry in baseline["protected_artifacts"]
        if entry["path"].endswith("single_gripper_webgl_timeline.json")
    )
    timeline = json.loads(Path(timeline_entry["path"]).read_text(encoding="utf-8"))
    target_index = int(baseline["accepted_metrics"]["occlusion_visual_check"]["frame"])
    target = next(
        frame for frame in timeline["frames"] if int(frame["source_index"]) == target_index
    )["left"]
    low, high = map(float, gates["occlusion_frame_force_percent_range"])
    measured_force = float(target["contact_intensity_percent"])
    if not low <= measured_force <= high:
        failures.append(
            f"occlusion-frame force {measured_force} is outside [{low}, {high}]"
        )
    if target["contact_measurement_state"] != gates["occlusion_frame_required_state"]:
        failures.append("occlusion-frame measurement state regressed")

    result = {
        "baseline_id": baseline["baseline_id"],
        "status": "PASS" if not failures else "FAIL",
        "hashes_checked": checked,
        "angle_available_frame_ratio": angle["available_frame_ratio"],
        "force_measured_frame_ratio": force["measured_frame_ratio"],
        "occlusion_frame": {
            "source_index": target_index,
            "force_percent": measured_force,
            "measurement_state": target["contact_measurement_state"],
        },
        "failures": failures,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
