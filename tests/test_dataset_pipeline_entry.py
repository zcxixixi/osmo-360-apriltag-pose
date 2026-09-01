from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from osmo360.pipeline import dataset
from osmo360.pipeline.dataset_worker import estimate_audio_offset
from osmo360.pipeline.manifest import ManifestError


LEFT_SERIAL = "IAHEA2606M5WSK"
RIGHT_SERIAL = "IAHEA2606KKUKF"
OFFSET = "m2_100_100_100_0_0_90_100_300_100_0_0_90_400_200_1"


def fake_insv(path: Path, serial: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"header {serial} {OFFSET} footer".encode())


def probe(duration: float = 120.0) -> dict[str, float | int]:
    return {
        "width": 2880,
        "height": 2880,
        "fps": 29.97,
        "duration_s": duration,
    }


def test_dataset_path_discovers_registered_left_right_pair(monkeypatch, tmp_path: Path):
    fake_insv(tmp_path / "raw/left/VID_20260901_120000_00_001.insv", LEFT_SERIAL)
    fake_insv(tmp_path / "raw/right/VID_20260901_120002_00_001.insv", RIGHT_SERIAL)
    monkeypatch.setattr(dataset, "_probe", lambda _path: probe())

    lock = dataset.discover_dataset(tmp_path)

    assert lock["pair_count"] == 1
    assert lock["pairs"][0]["pair_id"] == "pair-01-120002"
    assert lock["pairs"][0]["left"]["serial"] == LEFT_SERIAL
    assert lock["pairs"][0]["right"]["serial"] == RIGHT_SERIAL
    assert lock["pairs"][0]["left"]["path"].startswith("raw/left/")
    assert lock["ignored_short_recordings"] == []


def test_dataset_rejects_serial_in_wrong_side_directory(monkeypatch, tmp_path: Path):
    fake_insv(tmp_path / "raw/left/VID_20260901_120000_00_001.insv", RIGHT_SERIAL)
    (tmp_path / "raw/right").mkdir(parents=True)
    monkeypatch.setattr(dataset, "_probe", lambda _path: probe())

    with pytest.raises(ManifestError, match="under raw/left"):
        dataset.discover_dataset(tmp_path)


def test_short_recordings_are_ignored_before_pairing(monkeypatch, tmp_path: Path):
    fake_insv(tmp_path / "raw/left/VID_20260901_115900_00_000.insv", LEFT_SERIAL)
    fake_insv(tmp_path / "raw/left/VID_20260901_120000_00_001.insv", LEFT_SERIAL)
    fake_insv(tmp_path / "raw/right/VID_20260901_120002_00_001.insv", RIGHT_SERIAL)
    monkeypatch.setattr(
        dataset,
        "_probe",
        lambda path: probe(5.0 if "115900" in path.name else 120.0),
    )

    lock = dataset.discover_dataset(tmp_path)

    assert lock["pair_count"] == 1
    assert len(lock["ignored_short_recordings"]) == 1


def test_audio_sync_reports_right_time_from_left_offset(tmp_path: Path):
    rate = 1000
    left = np.zeros(2000, dtype=np.int16)
    right = np.zeros(2000, dtype=np.int16)
    left[500:510] = 30000
    right[520:530] = 30000
    left_path = tmp_path / "left.wav"; right_path = tmp_path / "right.wav"
    wavfile.write(left_path, rate, left); wavfile.write(right_path, rate, right)

    result = estimate_audio_offset(left_path, right_path, 0.0)

    assert result["mapping"] == "right_time_s = left_time_s + offset_s"
    assert result["offset_s"] == pytest.approx(0.020, abs=0.001)


def test_run_pipeline_shell_requires_only_dataset_path(tmp_path: Path):
    script = Path(__file__).parents[1] / "run_pipeline.sh"
    assert os.access(script, os.X_OK)
    process = subprocess.run([str(script), str(tmp_path)], capture_output=True, text=True)
    assert process.returncode == 2
    assert "dataset.h5 + video/*.mp4" in process.stderr
    assert "raw/left + raw/right" in process.stderr
