from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from osmo360 import ffmpeg_runtime
from osmo360.ffmpeg_runtime import (
    FFmpegRuntimeError,
    PINNED_RUNTIME_DIR,
    resolve_ffmpeg_runtime,
)
from tools.install_ffmpeg_runtime import install_ffmpeg_runtime


def _fake_program(path: Path, program: str, version: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"#!/bin/sh\nprintf '%s\\n' '{program} version {version} test'\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _revision(path: Path, archive: Path, root_name: str, binaries: dict) -> Path:
    payload = {
        "schema_version": "ffmpeg-runtime-revision/1.0",
        "revision_id": "ffmpeg-test-9.0.1",
        "platform": "linux-x86_64",
        "ffmpeg_version": "9.0.1",
        "minimum_supported_version": "9.0.1",
        "install_root": "work/tools",
        "runtime_dir": root_name,
        "archive": {"name": archive.name, "sha256": _sha256(archive)},
        "binaries": binaries,
        "upstream_source": {"sha256": "0" * 64},
        "build": {"network_protocols": False},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_pinned_ffmpeg_is_preferred_over_legacy_system_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pinned_bin = tmp_path / "repo/work/tools" / PINNED_RUNTIME_DIR / "bin"
    pinned_ffmpeg = _fake_program(pinned_bin / "ffmpeg", "ffmpeg", "9.0.1")
    pinned_ffprobe = _fake_program(pinned_bin / "ffprobe", "ffprobe", "9.0.1")
    system_bin = tmp_path / "system"
    _fake_program(system_bin / "ffmpeg", "ffmpeg", "4.4.2")
    _fake_program(system_bin / "ffprobe", "ffprobe", "4.4.2")
    monkeypatch.setattr(ffmpeg_runtime, "PINNED_FFMPEG_SHA256", _sha256(pinned_ffmpeg))
    monkeypatch.setattr(ffmpeg_runtime, "PINNED_FFPROBE_SHA256", _sha256(pinned_ffprobe))

    runtime = resolve_ffmpeg_runtime(
        repo_root=tmp_path / "repo", environ={"PATH": str(system_bin)}
    )

    assert runtime.ffmpeg == pinned_ffmpeg.resolve()
    assert runtime.ffprobe == pinned_ffprobe.resolve()
    assert runtime.version == "9.0.1"


def test_explicit_legacy_ffmpeg_fails_closed(tmp_path: Path) -> None:
    bin_dir = tmp_path / "legacy"
    _fake_program(bin_dir / "ffmpeg", "ffmpeg", "4.4.2")
    _fake_program(bin_dir / "ffprobe", "ffprobe", "4.4.2")

    with pytest.raises(FFmpegRuntimeError, match=r"require >= 9\.0\.1"):
        resolve_ffmpeg_runtime(
            repo_root=tmp_path / "repo",
            environ={"OSMO_FFMPEG_BIN": str(bin_dir), "PATH": ""},
        )


def test_offline_ffmpeg_installer_validates_and_reuses_archive(tmp_path: Path) -> None:
    source = tmp_path / "source/ffmpeg-test/bin"
    ffmpeg = _fake_program(source / "ffmpeg", "ffmpeg", "9.0.1")
    ffprobe = _fake_program(source / "ffprobe", "ffprobe", "9.0.1")
    archive = tmp_path / "ffmpeg-test.tar.xz"
    with tarfile.open(archive, "w:xz") as bundle:
        bundle.add(source.parent, arcname="ffmpeg-test")
    revision = _revision(
        tmp_path / "revision.json",
        archive,
        "ffmpeg-test",
        {
            "ffmpeg": {"path": "bin/ffmpeg", "sha256": _sha256(ffmpeg)},
            "ffprobe": {"path": "bin/ffprobe", "sha256": _sha256(ffprobe)},
        },
    )

    installed = install_ffmpeg_runtime(
        archive_path=archive, revision_path=revision, repo_root=tmp_path / "repo"
    )
    reused = install_ffmpeg_runtime(
        archive_path=tmp_path / "not-needed.tar.xz",
        revision_path=revision,
        repo_root=tmp_path / "repo",
    )

    assert installed["status"] == "installed"
    assert reused["status"] == "reused"
    marker = tmp_path / "repo/work/tools/ffmpeg-test/.osmo360-runtime.json"
    assert json.loads(marker.read_text(encoding="utf-8"))["network_protocols"] is False


def test_ffmpeg_installer_rejects_archive_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "malicious.tar.xz"
    with tarfile.open(archive, "w:xz") as bundle:
        escaped = tarfile.TarInfo("../escaped")
        payload = b"must not escape"
        escaped.size = len(payload)
        bundle.addfile(escaped, io.BytesIO(payload))
    revision = _revision(
        tmp_path / "revision.json",
        archive,
        "ffmpeg-test",
        {
            "ffmpeg": {"path": "bin/ffmpeg", "sha256": "0" * 64},
            "ffprobe": {"path": "bin/ffprobe", "sha256": "0" * 64},
        },
    )

    with pytest.raises(tarfile.FilterError):
        install_ffmpeg_runtime(
            archive_path=archive,
            revision_path=revision,
            repo_root=tmp_path / "repo",
        )

    assert not (tmp_path / "escaped").exists()
