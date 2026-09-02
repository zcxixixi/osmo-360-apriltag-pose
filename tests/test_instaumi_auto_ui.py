from __future__ import annotations

import hashlib
import json
from pathlib import Path

from osmo360.pipeline.instaumi_auto import discover_pairs
from osmo360.pipeline.instaumi_auto_ui import build_status, create_app


def make_pair(root: Path, minute: int) -> None:
    collector = root / "0901_instaumi_sort_blocks_qsb"
    entries = []
    for side, second in (("left", 0), ("right", 1)):
        path = collector / "raw" / side / f"VID_20260901_12{minute:02d}{second:02d}_00_{minute:03d}.insv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{side}-{minute}".encode())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append(f"{digest}  {side}/{path.name}")
    sha_path = collector / "raw/sha256.txt"
    existing = sha_path.read_text() if sha_path.is_file() else ""
    sha_path.write_text(existing + "\n".join(entries) + "\n")


def test_dashboard_aggregates_running_failed_and_complete(tmp_path: Path) -> None:
    for minute in range(3):
        make_pair(tmp_path, minute)
    pairs = discover_pairs(tmp_path)
    automation = tmp_path / "_automation"
    automation.mkdir()
    state = {
        "node": "a6000",
        "pairs": {
            pairs[0].key: {"status": "RUNNING", "stage": "format", "node": "a6000"},
            pairs[1].key: {"status": "FAILED", "stage": "format", "error": "bad input"},
        },
    }
    (automation / "state-1-of-2.json").write_text(json.dumps(state))
    logs = automation / "logs" / pairs[0].collector_root.name / pairs[0].episode_name
    logs.mkdir(parents=True)
    (logs / "format_status.json").write_text(json.dumps({
        "stage": "video",
        "frame_count_per_video": 100,
    }))
    (logs / "video-left-back.progress").write_text("frame=50\nprogress=continue\n")
    complete = pairs[2].collector_root / pairs[2].episode_name / "processed"
    complete.mkdir(parents=True)
    for name in ("trajectory.csv", "gripper.csv", "processed.csv", "metadata.csv"):
        (complete / name).write_text(name)
    (
        complete
        / f"{pairs[2].episode_name}_imu_assisted_demo.mp4"
    ).write_bytes(b"demo")

    result = build_status(tmp_path)

    assert result["counts"] == {
        "total": 3,
        "complete": 1,
        "running": 1,
        "waiting": 0,
        "failed": 1,
    }
    running = next(task for task in result["tasks"] if task["status"] == "RUNNING")
    assert running["node"] == "a6000"
    assert running["stage"] == "video encoding"
    assert running["progress"] > 10


def test_dashboard_http_api(tmp_path: Path) -> None:
    make_pair(tmp_path, 0)
    client = create_app(tmp_path).test_client()

    page = client.get("/")
    response = client.get("/api/status")

    assert page.status_code == 200
    assert "InstaUMI" in page.get_data(as_text=True)
    assert response.status_code == 200
    assert response.get_json()["counts"]["total"] == 1
