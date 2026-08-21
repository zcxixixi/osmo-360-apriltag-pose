from pathlib import Path

import numpy as np

import camera_to_dataset
from export_trajectory_dataset import relative_pose


def video_probe(width=3000, height=3000, count=2, encoder=""):
    return {
        "format": {"tags": {"encoder": encoder}},
        "streams": [
            {"codec_type": "video", "codec_name": "hevc", "width": width, "height": height}
            for _ in range(count)
        ],
    }


def test_camera_detection_uses_raw_container_extensions(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(camera_to_dataset, "ffprobe", lambda _path: video_probe())
    assert camera_to_dataset.detect_source(tmp_path / "clip.OSV")[0] == "dji"
    assert camera_to_dataset.detect_source(tmp_path / "clip.insv")[0] == "insta360"


def test_camera_detection_accepts_stitched_panorama(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        camera_to_dataset,
        "ffprobe",
        lambda _path: video_probe(3840, 1920, 1),
    )
    assert camera_to_dataset.detect_source(tmp_path / "stitched.mp4")[0] == "panorama"


def test_first_pose_coordinate_frame_is_identity():
    origin = np.array([1.2, -0.4, 0.8, 10.0, -7.0, 35.0])
    position, rotation = relative_pose(origin.copy(), origin)
    assert np.allclose(position, 0.0)
    assert np.allclose(rotation.as_quat(), [0.0, 0.0, 0.0, 1.0], atol=1e-12)


def test_packaged_dji_entrypoints_and_panoforge_exist():
    assert (camera_to_dataset.ROOT / "camera-to-dataset").is_file()
    assert (camera_to_dataset.ROOT / "dji_osv_stitch.py").is_file()
    assert (camera_to_dataset.PANOFORGE_ROOT / "app/core/maps.py").is_file()
