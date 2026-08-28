import json
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "dual_gripper_3d" / "platform_server.mjs"
MESH_DIR = ROOT / "assets" / "gripper_v52_new_r1" / "meshes"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request(url: str, method: str = "GET", body: bytes | None = None, content_type: str | None = None):
    headers = {"Content-Type": content_type} if content_type else {}
    request = urllib.request.Request(url, data=body, method=method, headers=headers)
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status, response.read(), response.headers


def test_processed_bundle_becomes_viewable_animation(tmp_path):
    port = _free_port()
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
        }
        _, capabilities, _ = _request(f"{base}/api/capabilities")
        assert json.loads(capabilities)["input_mode"] == "processed_bundle"


        status, body, _ = _request(
            f"{base}/api/projects",
            "POST",
            json.dumps({"name": "API test"}).encode(),
            "application/json",
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
        )
        _request(
            f"{base}/api/projects/{project_id}/video",
            "PUT",
            b"not-empty",
            "video/mp4",
        )
        status, body, _ = _request(f"{base}/api/projects/{project_id}/publish", "POST", b"")
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
        assert b"single_gripper_scene" not in scene
        assert b"force-panel" in scene

        _, listed, _ = _request(f"{base}/api/projects")
        assert json.loads(listed)["projects"][0]["id"] == project_id
    finally:
        process.terminate()
        process.wait(timeout=5)
