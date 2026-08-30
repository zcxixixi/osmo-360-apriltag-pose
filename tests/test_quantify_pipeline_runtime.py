import sys

import pytest

from tools.quantify_pipeline_runtime import measure, summarize_manifest


def test_measure_reports_throughput_and_realtime_factor(tmp_path):
    result = measure(
        [sys.executable, "-c", "pass"],
        stage="smoke",
        media_duration_s=2.0,
        frame_count=60,
        cwd=tmp_path,
    )

    assert result["returncode"] == 0
    assert result["wall_s"] > 0
    assert result["realtime_factor"] == pytest.approx(result["wall_s"] / 2.0)
    assert result["throughput_fps"] == pytest.approx(60 / result["wall_s"])


def test_summary_uses_maximum_stage_per_parallel_phase():
    summary = summarize_manifest({
        "label": "dual-camera",
        "media_duration_s": 20.0,
        "stages": [
            {"name": "stitch-left", "phase": "stitch", "wall_s": 10.0},
            {"name": "stitch-right", "phase": "stitch", "wall_s": 12.0},
            {"name": "pose-left", "phase": "analysis", "wall_s": 80.0},
            {"name": "pose-right", "phase": "analysis", "wall_s": 100.0},
            {"name": "publish", "phase": "publish", "wall_s": 2.0},
        ],
    })

    assert summary["sequential"]["wall_s"] == 204.0
    assert summary["phase_parallel"]["wall_s"] == 114.0
    assert summary["bottleneck"]["stage"] == "pose-right"
    assert summary["phase_parallel"]["realtime_factor"] == pytest.approx(5.7)
