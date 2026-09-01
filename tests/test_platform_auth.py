from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from osmo360.pipeline import devices
from osmo360.pipeline.platform_auth import (
    load_platform_write_token,
    platform_authorization_headers,
    validate_http_url,
)
from tools import upload_visualization_bundle as uploader

TOKEN = "b" * 64


def _token_file(tmp_path: Path) -> Path:
    path = tmp_path / "platform-write-token"
    path.write_text(TOKEN + "\n", encoding="utf-8")
    path.chmod(0o600)
    return path


def test_private_token_file_generates_bearer_header(monkeypatch, tmp_path):
    monkeypatch.delenv("OSMO_PLATFORM_WRITE_TOKEN", raising=False)
    monkeypatch.delenv("OSMO_PLATFORM_WRITE_TOKEN_FILE", raising=False)
    token_file = _token_file(tmp_path)

    assert load_platform_write_token(token_file) == TOKEN
    assert platform_authorization_headers(token_file) == {
        "Authorization": f"Bearer {TOKEN}"
    }


def test_group_readable_token_file_is_rejected(monkeypatch, tmp_path):
    monkeypatch.delenv("OSMO_PLATFORM_WRITE_TOKEN", raising=False)
    monkeypatch.delenv("OSMO_PLATFORM_WRITE_TOKEN_FILE", raising=False)
    token_file = _token_file(tmp_path)
    token_file.chmod(0o640)

    with pytest.raises(RuntimeError, match="mode 0600"):
        load_platform_write_token(token_file)


def test_device_inventory_sync_sends_bearer_token(monkeypatch, tmp_path):
    token_file = _token_file(tmp_path)
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "schema_version": "x5-device-inventory/1.0",
                "sdk_revision_id": "sdk-test",
                "devices": {},
            }
        ),
        encoding="utf-8",
    )
    captured = []

    def fake_urlopen(request, timeout):
        captured.append((request, timeout))
        return io.BytesIO(b'{"status":"saved","count":0}')

    monkeypatch.setattr(devices.urllib.request, "urlopen", fake_urlopen)

    result = devices.sync_inventory(
        inventory, "http://platform.invalid:7865", token_file
    )

    assert result["status"] == "saved"
    request, timeout = captured[0]
    assert timeout == 30
    assert request.get_header("Authorization") == f"Bearer {TOKEN}"


def test_upload_file_sends_bearer_token(monkeypatch, tmp_path):
    source = tmp_path / "timeline.json"
    source.write_text("{}", encoding="utf-8")
    captured = {"headers": {}}

    class FakeResponse:
        status = 200
        reason = "OK"

        @staticmethod
        def read():
            return b'{"bytes":2}'

    class FakeConnection:
        def __init__(self, host, port, timeout):
            captured.update(host=host, port=port, timeout=timeout)

        def putrequest(self, method, target):
            captured.update(method=method, target=target)

        def putheader(self, key, value):
            captured["headers"][key] = value

        def endheaders(self):
            return None

        def send(self, chunk):
            captured["body"] = captured.get("body", b"") + chunk

        @staticmethod
        def getresponse():
            return FakeResponse()

        @staticmethod
        def close():
            return None

    monkeypatch.setattr(uploader.http.client, "HTTPConnection", FakeConnection)

    result = uploader.put_file(
        "http://platform.invalid:7865/api/projects/test/timeline",
        source,
        "application/json",
        {"Authorization": f"Bearer {TOKEN}"},
    )

    assert result == {"bytes": 2}
    assert captured["headers"]["Authorization"] == f"Bearer {TOKEN}"
    assert captured["body"] == b"{}"


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://platform.invalid/file",
        "http://user:password@platform.invalid/api",
        "http://platform.invalid/api#fragment",
        "http://platform.invalid/api\\redirect",
        "http://platform.invalid:invalid/api",
    ],
)
def test_http_client_rejects_ambiguous_or_non_http_urls(url):
    with pytest.raises(ValueError):
        validate_http_url(url)


def test_device_inventory_sync_rejects_non_http_server_before_reading_token(tmp_path):
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "schema_version": "x5-device-inventory/1.0",
                "sdk_revision_id": "sdk-test",
                "devices": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(devices.ManifestError, match="invalid visualization server URL"):
        devices.sync_inventory(inventory, "file:///tmp/platform")


def test_visualization_uploader_rejects_non_http_urls(tmp_path):
    source = tmp_path / "timeline.json"
    source.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="scheme must be http or https"):
        uploader.request_json("file:///etc/passwd")
    with pytest.raises(ValueError, match="scheme must be http or https"):
        uploader.put_file("file:///tmp/output", source, "application/json")
