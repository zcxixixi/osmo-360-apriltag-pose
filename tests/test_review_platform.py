import csv
import json
from pathlib import Path

import numpy as np
import pytest

from osmo360.pipeline.review_store import REASONS, ReviewStore, summarize_pair
from osmo360.pipeline.review_ui import PAGE, pair_timeline, suggest_segments


def write_pair(root: Path, pair_id: str = "pair-01-test") -> Path:
    directory = root / "preprocess" / pair_id
    directory.mkdir(parents=True)
    (directory / "metrics.json").write_text(json.dumps({
        "input_duration_seconds_per_role": 3,
        "wall_clock_seconds": 1.5,
        "roles": {
            "left": {"counts": {"fallback": 1}},
            "right": {"counts": {"fallback": 2}},
        },
    }), encoding="utf-8")
    rows = [
        {"role": role, "common_time_s": second, "ids": [200, 201, 202]}
        for second in range(3) for role in ("left", "right")
    ]
    (directory / "tag_detections.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    with (directory / "gripper_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "role", "common_time_s", "source_frame", "yellow_pixels", "candidate_count",
        ])
        writer.writeheader()
        for second in range(3):
            for role in ("left", "right"):
                writer.writerow({
                    "role": role, "common_time_s": second, "source_frame": second * 60,
                    "yellow_pixels": 1200, "candidate_count": 4,
                })
    rgb = np.full((3, 8, 8, 3), 127, dtype=np.uint8)
    np.savez_compressed(directory / "rgb_samples.npz", left=rgb, right=rgb)
    bundle = directory / "review_bundle"
    bundle.mkdir()
    (bundle / "timeline.json").write_text('{"schema_version":"test","frames":[{}]}')
    (bundle / "front-video.mp4").write_bytes(b"video")
    (bundle / "scene.html").write_text("<html></html>")
    (bundle / "visualization.json").write_text(
        '{"view_url":"http://example.test/view/pair"}'
    )
    return directory


def test_pair_summary_and_simple_page(tmp_path: Path) -> None:
    directory = write_pair(tmp_path)
    summary = summarize_pair(directory)
    assert summary["auto_status"] == "pass_candidate"
    assert summary["tag_usable_ratio"] == 1.0
    assert summary["gripper_candidate_ratio"] == 1.0
    assert "画面正常，通过并看下一条" in PAGE
    assert REASONS["sync"] == "左右对不上"
    timeline = pair_timeline(directory)
    assert timeline["tags"] == [[0, 3.0], [1, 3.0], [2, 3.0]]
    assert timeline["gripper"] == [[0, 4.0], [1, 4.0], [2, 4.0]]


def test_review_history_reprocess_queue_and_stale_hash(tmp_path: Path) -> None:
    directory = write_pair(tmp_path)
    store = ReviewStore(tmp_path)
    items = store.scan()
    assert len(items) == 1
    pair_id = items[0]["pair_id"]
    store.add_review(pair_id, decision="approved", reasons=[], notes="清楚", reviewer="审核员A")
    event = store.add_review(
        pair_id, decision="reprocess", reasons=["sync"], notes="左右错位", reviewer="审核员B"
    )
    assert event["decision"] == "reprocess"
    queue = json.loads(store.queue_path.read_text())
    assert queue["items"][0]["pair_id"] == pair_id
    assert queue["items"][0]["reasons"] == ["sync"]
    store.add_review(pair_id, decision="approved", reasons=[], notes="复查通过", reviewer="审核员A")
    assert json.loads(store.queue_path.read_text())["items"] == []
    assert len(store.history(pair_id)) == 3
    assert store.snapshot.is_file()
    metrics = json.loads((directory / "metrics.json").read_text())
    metrics["wall_clock_seconds"] = 2.0
    (directory / "metrics.json").write_text(json.dumps(metrics))
    rescanned = store.scan()[0]
    assert rescanned["stale_review"]
    assert rescanned["decision"] is None


def test_reprocess_requires_reason_and_reviewer(tmp_path: Path) -> None:
    write_pair(tmp_path)
    store = ReviewStore(tmp_path)
    pair_id = store.scan()[0]["pair_id"]
    with pytest.raises(ValueError, match="reason"):
        store.add_review(pair_id, decision="reprocess", reasons=[], notes="", reviewer="A")
    with pytest.raises(ValueError, match="reviewer"):
        store.add_review(pair_id, decision="approved", reasons=[], notes="", reviewer="")
    with pytest.raises(ValueError, match="invalid reasons"):
        store.add_review(pair_id, decision="rejected", reasons=["unknown"], notes="", reviewer="A")


def test_approval_requires_3d_video_alignment(tmp_path: Path) -> None:
    directory = write_pair(tmp_path)
    for path in (directory / "review_bundle").iterdir():
        path.unlink()
    store = ReviewStore(tmp_path)
    pair_id = store.scan()[0]["pair_id"]
    with pytest.raises(ValueError, match="3D"):
        store.add_review(
            pair_id, decision="approved", reasons=[], notes="", reviewer="审核员"
        )


def test_segment_annotation_and_approved_export(tmp_path: Path) -> None:
    write_pair(tmp_path)
    store = ReviewStore(tmp_path)
    pair_id = store.scan()[0]["pair_id"]
    segment = store.add_segment(
        pair_id, start_s=0.5, end_s=2.5, label="pick", success=True,
        notes="拿起红色积木", reviewer="审核员A",
    )
    assert store.list_segments(pair_id)[0]["label"] == "pick"
    store.review_segment(
        segment["id"], decision="approved", reasons=[], notes="3D对齐",
        reviewer="审核员B",
    )
    exported = json.loads(store.export_path.read_text())
    assert exported["items"][0]["start_s"] == 0.5
    assert exported["items"][0]["end_s"] == 2.5


def test_motion_based_segment_suggestions(tmp_path: Path) -> None:
    directory = write_pair(tmp_path)
    frames = []
    for index in range(100):
        x = 0.0 if index < 20 else min((index - 20) * 0.01, 0.3)
        frames.append({
            "t": index / 20,
            "left": {"p": [x, 0, 0]},
            "right": {"p": [x, 0.1, 0]},
        })
    (directory / "review_bundle" / "timeline.json").write_text(json.dumps({
        "fps": 20, "frames": frames,
    }))
    suggestions = suggest_segments(directory)
    assert suggestions
    assert suggestions[0]["end_s"] > suggestions[0]["start_s"]
