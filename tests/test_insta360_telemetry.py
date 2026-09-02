from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

import osmo360.pipeline.insta360_telemetry as telemetry
from osmo360.pipeline.insta360_telemetry import extract_x5_imu


PARSER = '''#!/usr/bin/env python3
import pathlib
import sys
source = pathlib.Path(sys.argv[1])
source.with_suffix(source.suffix + ".csv").write_text(
    '"serial_number","SERIAL_LEFT"\\n'
    '"camera_type","Insta360 X5"\\n'
    '"fw_version","v1.7.8"\\n'
    '"first_frame_timestamp",1000000\\n'
    '"gyro_timestamp",2.0\\n'
    '"loopIteration","time","gyroADC[0]","gyroADC[1]","gyroADC[2]","accSmooth[0]","accSmooth[1]","accSmooth[2]"\\n'
    '0,0,180,0,-180,0,0,20088.576\\n'
    '1,1000,90,45,0,2048,0,0\\n'
    '2,2000,0,0,0,0,2048,0\\n'
)
print("Detected camera: Insta360 Insta360 X5")
'''


def make_parser(tmp_path: Path) -> Path:
    parser = tmp_path / "gyro2bb"
    parser.write_text(PARSER)
    parser.chmod(0o755)
    return parser


def test_extract_x5_imu_preserves_source_clock_and_converts_si_units(
    monkeypatch, tmp_path: Path,
) -> None:
    source = tmp_path / "capture.insv"
    source.write_bytes(b"insv")
    parser = make_parser(tmp_path)
    monkeypatch.setattr(
        telemetry, "TELEMETRY_PARSER_SHA256",
        hashlib.sha256(parser.read_bytes()).hexdigest(),
    )
    monkeypatch.setenv("OSMO_X5_TELEMETRY_PARSER", str(parser))

    imu = extract_x5_imu(
        source, tmp_path / "scratch", source_start_s=0.0,
        duration_s=0.002, expected_serial="SERIAL_LEFT",
    )

    assert imu.timestamp_ns.tolist() == [0, 1_000_000]
    assert imu.source_timestamp_ns.tolist() == [1_002_000_000, 1_003_000_000]
    assert imu.angular_velocity[0].tolist() == pytest.approx([np.pi, 0, -np.pi])
    assert imu.linear_acceleration[0].tolist() == pytest.approx([0, 0, 9.808875])
    assert imu.valid.tolist() == [1, 1]
    assert imu.provenance["firmware_version"] == "v1.7.8"


def test_extract_x5_imu_rejects_wrong_camera_serial(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "capture.insv"
    source.write_bytes(b"insv")
    parser = make_parser(tmp_path)
    monkeypatch.setattr(
        telemetry, "TELEMETRY_PARSER_SHA256",
        hashlib.sha256(parser.read_bytes()).hexdigest(),
    )
    monkeypatch.setenv("OSMO_X5_TELEMETRY_PARSER", str(parser))

    with pytest.raises(ValueError, match="serial mismatch"):
        extract_x5_imu(
            source, tmp_path / "scratch", source_start_s=0.0,
            duration_s=0.002, expected_serial="OTHER_SERIAL",
        )
