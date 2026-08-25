from dual_gripper_calibrator import app, validate_payload


def test_validate_and_save(tmp_path, monkeypatch):
    import dual_gripper_calibrator as module

    monkeypatch.setattr(module, "OUTPUT_ROOT", tmp_path)
    client = app.test_client()
    payload = {
        "right_gripper_in_left_frame": {
            "translation_m": [0.35, 0, 0.02],
            "rotation_rpy_deg": [0, 0, 180],
        },
        "preview_opening_deg": {"left": 12, "right": 18},
        "grippers_in_center_frame": {
            "left": {"translation_m": [-0.175, 0, -0.01], "rotation_rpy_deg": [5, 0, 0]},
            "right": {"translation_m": [0.175, 0, 0.01], "rotation_rpy_deg": [-5, 0, 180]},
        },
        "note": "face to face",
    }
    response = client.post("/api/save", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["config"]["reference_frame"] == "midpoint_between_gripper_bases"
    assert data["config"]["right_gripper_in_left_frame"]["translation_m"] == [0.35, 0.0, 0.02]
    assert data["config"]["grippers_in_center_frame"]["left"]["translation_m"] == [-0.175, -0.0, -0.01]
    assert data["config"]["grippers_in_center_frame"]["right"]["translation_m"] == [0.175, 0.0, 0.01]
    assert data["config"]["grippers_in_center_frame"]["left"]["rotation_rpy_deg"] == [5.0, 0.0, 0.0]
    assert (tmp_path / f"{data['calibration_id']}.json").is_file()


def test_rejects_bad_translation():
    try:
        validate_payload({"right_gripper_in_left_frame": {
            "translation_m": [999, 0, 0], "rotation_rpy_deg": [0, 0, 0]
        }})
    except ValueError as exc:
        assert "10 m" in str(exc)
    else:
        raise AssertionError("invalid translation was accepted")


def test_animation_page_is_available_and_missing_timeline_is_explicit():
    client = app.test_client()
    page = client.get("/animation")
    timeline = client.get("/api/animation-timeline")
    assert page.status_code == 200
    assert "双夹爪 6DoF 交互动画" in page.get_data(as_text=True)
    assert timeline.status_code == 404
    assert "not found" in timeline.get_json()["error"]


def test_hardware_model_page_and_api_are_available():
    client = app.test_client()
    page = client.get("/hardware")
    model = client.get("/api/hardware-model")
    assert page.status_code == 200
    assert "Osmo 360 × 夹爪硬件模型" in page.get_data(as_text=True)
    assert model.status_code == 200
    payload = model.get_json()
    assert payload["model"] == "DJI Osmo 360"
    assert payload["frames"]["tcp"]["translation_in_base_m"] == [0.1356, 0.0, 0.0101]
    assert set(payload["mounts"]) == {"left", "right"}


def test_umi_explainer_page_handles_missing_local_dataset():
    client = app.test_client()
    page = client.get("/umi")
    summary = client.get("/api/umi-summary")
    frame = client.get("/api/umi-frame/0/180")
    assert page.status_code == 200
    assert "这18秒视频，已经变成可审计的训练数据" in page.get_data(as_text=True)
    assert summary.status_code == 404
    assert "DUAL_GRIPPER_DATA_ROOT" in summary.get_json()["error"]
    assert frame.status_code == 404
    assert "DUAL_GRIPPER_DATA_ROOT" in frame.get_json()["error"]
