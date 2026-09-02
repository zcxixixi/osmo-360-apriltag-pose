from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tools import upload_instaumi_drive as upload


def make_collector(tmp_path: Path) -> Path:
    collector = tmp_path / "0902_instaumi_sort_blocks_qsb"
    for side in ("left", "right"):
        directory = collector / "raw" / side
        directory.mkdir(parents=True)
        (directory / f"VID_20260902_12000{side == 'right'}_00_001.insv").write_bytes(
            side.encode()
        )
    return collector


def test_source_files_requires_both_camera_sides(tmp_path: Path) -> None:
    collector = tmp_path / "0902_instaumi_sort_blocks_qsb"
    (collector / "raw/left").mkdir(parents=True)
    (collector / "raw/left/video.insv").write_bytes(b"left")

    with pytest.raises(RuntimeError, match="missing source directory"):
        upload.source_files(collector)


def test_upload_publishes_sha_only_after_raw_rsync(
    tmp_path: Path,
    monkeypatch,
) -> None:
    collector = make_collector(tmp_path)
    commands = []

    monkeypatch.setattr(upload, "run", lambda command: commands.append(command))

    result = upload.upload_collector(collector, "nas", "/data/instaumi")

    assert result["file_count"] == 2
    assert commands[0][:5] == ["ssh", "-o", "BatchMode=yes", "nas", "mkdir"]
    assert "--exclude=sha256.txt" in commands[1]
    assert commands[1][-1].endswith("/0902_instaumi_sort_blocks_qsb/raw/")
    assert commands[2][-1].endswith("/raw/sha256.txt")
    expected = {
        hashlib.sha256(b"left").hexdigest(),
        hashlib.sha256(b"right").hexdigest(),
    }
    assert len(expected) == 2
