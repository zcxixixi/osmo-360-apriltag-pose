from pathlib import Path

from osmo360.pipeline.insta360_telemetry import _read_blackbox_csv


def test_blackbox_metadata_preserves_nested_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "imu.csv"
    path.write_text(
        '"serial_number","IAHEA2606M5WSK"\n'
        '"gyro_cfg_info",{"acc_range":32,"gyro_range":2000}\n'
        '"loopIteration","time","gyroADC[0]","gyroADC[1]","gyroADC[2]",'
        '"accSmooth[0]","accSmooth[1]","accSmooth[2]"\n'
        '0,1000,1,2,3,4,5,6\n',
        encoding="utf-8",
    )

    metadata, samples = _read_blackbox_csv(path)

    assert metadata["serial_number"] == "IAHEA2606M5WSK"
    assert metadata["gyro_cfg_info"] == {
        "acc_range": 32,
        "gyro_range": 2000,
    }
    assert samples.shape == (1, 7)
