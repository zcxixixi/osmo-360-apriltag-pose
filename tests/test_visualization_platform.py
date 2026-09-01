import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "dual_gripper_3d" / "platform_server.mjs"
MESH_DIR = ROOT / "assets" / "gripper_v52_new_r1" / "meshes"
WRITE_TOKEN = "a" * 64


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request(
    url: str,
    method: str = "GET",
    body: bytes | None = None,
    content_type: str | None = None,
    token: str | None = None,
):
    headers = {"Content-Type": content_type} if content_type else {}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=body, method=method, headers=headers)
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status, response.read(), response.headers


def test_processed_bundle_becomes_viewable_animation(tmp_path):
    port = _free_port()
    environment = os.environ.copy()
    environment.pop("OSMO_PLATFORM_WRITE_TOKEN_FILE", None)
    environment["OSMO_PLATFORM_WRITE_TOKEN"] = WRITE_TOKEN
    process = subprocess.Popen(
        [
            "node",
            str(SERVER),
            "--data-dir",
            str(tmp_path),
            "--mesh-dir",
            str(MESH_DIR),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(50):
            try:
                _request(f"{base}/api/projects")
                break
            except urllib.error.URLError:
                time.sleep(0.02)
        else:
            raise AssertionError("platform server did not become ready")
        _, health, _ = _request(f"{base}/healthz")
        assert json.loads(health) == {
            "status": "ok",
            "service": "osmo-motion-studio",
            "api_version": "v1",
            "write_authentication": "bearer",
        }
        _, capabilities, _ = _request(f"{base}/api/capabilities")
        assert json.loads(capabilities)["input_mode"] == "processed_bundle"
        capability_data = json.loads(capabilities)
        assert capability_data["write_authentication"] == {
            "type": "bearer",
            "required": True,
        }
        assert capability_data["required_files"]["scene"]["name"] == "scene.html"
        assert capability_data["renderer"]["scene"] == "project-versioned"
        _, empty_inventory, _ = _request(f"{base}/api/devices")
        assert json.loads(empty_inventory)["devices"] == {}
        inventory = {
            "schema_version": "x5-device-inventory/1.0",
            "sdk_revision_id": "sdk-test",
            "devices": {
                "IAHEA2606KMURQ": {
                    "serial": "IAHEA2606KMURQ",
                    "model": "Insta360 X5",
                    "firmware": "v1.7.8",
                    "assignment": {
                        "role": "physical_right",
                        "base_tag_id": 3,
                    },
                }
            },
        }
        with pytest.raises(urllib.error.HTTPError) as unauthorized:
            _request(
                f"{base}/api/devices",
                "PUT",
                json.dumps(inventory).encode(),
                "application/json",
            )
        assert unauthorized.value.code == 401
        assert unauthorized.value.headers["WWW-Authenticate"].startswith("Bearer ")
        assert not (tmp_path / "x5_device_inventory.json").exists()
        status, saved_inventory, _ = _request(
            f"{base}/api/devices",
            "PUT",
            json.dumps(inventory).encode(),
            "application/json",
            WRITE_TOKEN,
        )
        assert status == 200
        assert json.loads(saved_inventory)["count"] == 1
        _, loaded_inventory, _ = _request(f"{base}/api/devices")
        assert json.loads(loaded_inventory) == inventory


        with pytest.raises(urllib.error.HTTPError) as bad_token:
            _request(
                f"{base}/api/projects",
                "POST",
                json.dumps({"name": "unauthorized"}).encode(),
                "application/json",
                "wrong-token",
            )
        assert bad_token.value.code == 401

        status, body, _ = _request(
            f"{base}/api/projects",
            "POST",
            json.dumps({"name": "API test"}).encode(),
            "application/json",
            WRITE_TOKEN,
        )
        assert status == 201
        project_id = json.loads(body)["project"]["id"]

        timeline = {
            "schema_version": "single-gripper-webgl/v1",
            "render_mode": "single_gripper_world_diagnostic",
            "fps": 100.0,
            "duration_s": 0.01,
            "frames": [
                {
                    "t": 0.0,
                    "left": {"p": [0, 0, 0], "q": [0, 0, 0, 1], "joints": [0, 0]},
                    "right": {"p": [0, 0, 0], "q": [0, 0, 0, 1], "joints": [0, 0]},
                }
            ],
        }
        _request(
            f"{base}/api/projects/{project_id}/timeline",
            "PUT",
            json.dumps(timeline).encode(),
            "application/json",
            WRITE_TOKEN,
        )
        _request(
            f"{base}/api/projects/{project_id}/video",
            "PUT",
            b"not-empty",
            "video/mp4",
            WRITE_TOKEN,
        )
        versioned_scene = (
            b"<!doctype html><div id='force-panel'>VERSIONED-SCENE-TEST</div>"
            b"<script>fetch('timeline.json');const video='front-video.mp4';</script>"
        )
        _request(
            f"{base}/api/projects/{project_id}/scene",
            "PUT",
            versioned_scene,
            "text/html",
            WRITE_TOKEN,
        )
        status, body, _ = _request(
            f"{base}/api/projects/{project_id}/publish",
            "POST",
            b"",
            token=WRITE_TOKEN,
        )
        published = json.loads(body)
        assert status == 200
        assert published["project"]["status"] == "ready"
        assert published["project"]["summary"]["frames"] == 1
        assert published["project"]["view_url"] == f"{base}/view/{project_id}/?interactive=1"
        _, project_body, _ = _request(f"{base}/api/projects/{project_id}")
        assert json.loads(project_body)["project"]["links"]["publish"].endswith(
            f"/api/projects/{project_id}/publish"
        )

        status, scene, _ = _request(f"{base}/view/{project_id}/")
        assert status == 200
        assert scene == versioned_scene
        assert b"VERSIONED-SCENE-TEST" in scene

        _, listed, _ = _request(f"{base}/api/projects")
        assert json.loads(listed)["projects"][0]["id"] == project_id
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_platform_server_fails_closed_without_write_token(tmp_path):
    environment = os.environ.copy()
    environment.pop("OSMO_PLATFORM_WRITE_TOKEN", None)
    environment.pop("OSMO_PLATFORM_WRITE_TOKEN_FILE", None)
    result = subprocess.run(
        [
            "node",
            str(SERVER),
            "--data-dir",
            str(tmp_path),
            "--mesh-dir",
            str(MESH_DIR),
            "--host",
            "127.0.0.1",
            "--port",
            str(_free_port()),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=environment,
        timeout=5,
    )
    assert result.returncode != 0
    assert "platform write token is required" in result.stderr
