import json

from osmo360.pipeline.progress_ui import PAGE, load_status


def test_progress_ui_loads_status_and_contains_dashboard_contract(tmp_path):
    path = tmp_path / "pipeline_status.json"
    path.write_text(
        json.dumps({
            "run_name": "block-sorting",
            "stages": [{"name": "stitch", "state": "running", "progress": 42}],
        }),
        encoding="utf-8",
    )

    status = load_status(path)

    assert status["run_name"] == "block-sorting"
    assert status["stages"][0]["progress"] == 42
    assert "/api/status" in PAGE
    assert "DUAL X5 · UMI DATA PIPELINE" in PAGE
