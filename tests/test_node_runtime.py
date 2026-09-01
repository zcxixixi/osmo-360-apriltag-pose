import hashlib
import io
import json
import tarfile
import urllib.request
from pathlib import Path

import pytest

from osmo360.visualization import node_runtime
from osmo360.visualization.node_runtime import (
    PINNED_RUNTIME_DIR,
    NodeRuntimeError,
    resolve_node_binary,
)
from tools.install_node_runtime import _download, install_node_runtime


def _fake_node(path: Path, version: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\nprintf '%s\\n' 'v{version}'\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_pinned_node_is_preferred_over_an_old_system_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pinned = _fake_node(
        tmp_path / "repo/work/tools" / PINNED_RUNTIME_DIR / "bin/node",
        "24.20.0",
    )
    system = _fake_node(tmp_path / "system/node", "18.20.8")
    monkeypatch.setattr(
        node_runtime,
        "PINNED_BINARY_SHA256",
        hashlib.sha256(pinned.read_bytes()).hexdigest(),
    )

    resolved = resolve_node_binary(
        repo_root=tmp_path / "repo",
        environ={"PATH": str(system.parent)},
    )

    assert resolved == pinned.resolve()


def test_explicit_old_node_fails_instead_of_silently_falling_back(tmp_path: Path) -> None:
    old = _fake_node(tmp_path / "node", "20.20.2")

    with pytest.raises(NodeRuntimeError, match=r"require >= 22\.12\.0"):
        resolve_node_binary(
            repo_root=tmp_path,
            environ={"OSMO_NODE_BINARY": str(old), "PATH": ""},
        )


def test_supported_system_node_is_allowed_when_pinned_runtime_is_absent(
    tmp_path: Path,
) -> None:
    system = _fake_node(tmp_path / "bin/node", "22.12.0")

    assert resolve_node_binary(
        repo_root=tmp_path / "repo",
        environ={"PATH": str(system.parent)},
    ) == system.resolve()


def test_installer_rejects_archive_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "malicious.tar.xz"
    with tarfile.open(archive, "w:xz") as bundle:
        directory = tarfile.TarInfo("node-test-linux-x64")
        directory.type = tarfile.DIRTYPE
        bundle.addfile(directory)
        escaped = tarfile.TarInfo("../escaped")
        payload = b"must not escape"
        escaped.size = len(payload)
        bundle.addfile(escaped, io.BytesIO(payload))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    revision = tmp_path / "revision.json"
    revision.write_text(
        json.dumps(
            {
                "schema_version": "node-runtime-revision/1.0",
                "revision_id": "node-test",
                "platform": "linux-x86_64",
                "node_version": "24.20.0",
                "install_root": "work/tools",
                "runtime_dir": "node-test-linux-x64",
                "binary": {"path": "bin/node", "sha256": "0" * 64},
                "archive": {
                    "name": archive.name,
                    "url": "https://invalid.example/archive.tar.xz",
                    "sha256": digest,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(tarfile.FilterError):
        install_node_runtime(
            revision_path=revision,
            archive_path=archive,
            repo_root=tmp_path / "repo",
        )

    assert not (tmp_path / "escaped").exists()


def test_node_installer_rejects_non_official_download_hosts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("network must not be reached")

    monkeypatch.setattr(urllib.request, "urlopen", fail_if_called)

    with pytest.raises(ValueError, match="HTTPS URL on nodejs.org"):
        _download("file:///tmp/node.tar.xz", tmp_path / "node.tar.xz")
    with pytest.raises(ValueError, match="HTTPS URL on nodejs.org"):
        _download("https://example.invalid/node.tar.xz", tmp_path / "node.tar.xz")

    assert called is False
