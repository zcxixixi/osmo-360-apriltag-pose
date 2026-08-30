import json
from pathlib import Path

import pytest

from tools.insta360_sdk_revision import load_insta360_sdk_revision, sha256


def make_revision(tmp_path: Path) -> Path:
    media = tmp_path / "media"
    camera = tmp_path / "camera"
    files = {
        "media_binary": media / "usr/bin/MediaSDKTest",
        "media_library": media / "usr/lib/libMediaSDK.so",
        "media_model": media / "models/ai_stitcher_v2.ins",
        "camera_binary": camera / "bin/CameraSDKTest",
        "camera_library": camera / "lib/libCameraSDK.so",
    }
    for index, path in enumerate(files.values()):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"asset-{index}".encode())
    payload = {
        "schema_version": "insta360-sdk-revision/1.0",
        "revision_id": "test-sdk",
        "platform": "linux-x86_64",
        "source_archive": {"path": "/not-required.zip", "sha256": "0" * 64},
        "media_sdk": {
            "version": "3.1.1",
            "root": str(media),
            "binary": {"path": "usr/bin/MediaSDKTest", "sha256": sha256(files["media_binary"])},
            "library": {"path": "usr/lib/libMediaSDK.so", "sha256": sha256(files["media_library"])},
            "model_probe": {"path": "models/ai_stitcher_v2.ins", "sha256": sha256(files["media_model"])},
        },
        "camera_sdk": {
            "version": "2.1.1",
            "root": str(camera),
            "binary": {"path": "bin/CameraSDKTest", "sha256": sha256(files["camera_binary"])},
            "library": {"path": "lib/libCameraSDK.so", "sha256": sha256(files["camera_library"])},
        },
        "usage": {},
    }
    revision = tmp_path / "revision.json"
    revision.write_text(json.dumps(payload), encoding="utf-8")
    return revision


def test_sdk_revision_loads_versioned_local_deployment(tmp_path: Path):
    bundle = load_insta360_sdk_revision(make_revision(tmp_path))
    assert bundle["revision"]["revision_id"] == "test-sdk"
    assert bundle["media_binary"].name == "MediaSDKTest"
    assert bundle["camera_library"].name == "libCameraSDK.so"


def test_sdk_revision_hash_mismatch_fails_closed(tmp_path: Path):
    revision = make_revision(tmp_path)
    payload = json.loads(revision.read_text(encoding="utf-8"))
    payload["media_sdk"]["library"]["sha256"] = "f" * 64
    revision.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="MediaSDK library hash mismatch"):
        load_insta360_sdk_revision(revision)
