import json
import sys

import cv2

from tools import generate_a3_aprilgrid_pair as generator


def test_default_layout_nearly_fills_a3_with_140mm_pitch():
    layout = generator.layout_for((200, 201, 202, 203, 204, 205), 120.0, 20.0)

    assert [tag["black_outer_top_left_mm"] for tag in layout] == [
        [10.0, 18.5], [150.0, 18.5], [290.0, 18.5],
        [10.0, 158.5], [150.0, 158.5], [290.0, 158.5],
    ]
    assert layout[0]["corners_m"] == [
        [-0.20, -0.13, 0.0], [-0.08, -0.13, 0.0],
        [-0.08, -0.01, 0.0], [-0.20, -0.01, 0.0],
    ]


def test_generator_emits_two_detectable_a3_sheets(tmp_path, monkeypatch):
    output = tmp_path / "pair"
    monkeypatch.setattr(sys, "argv", [
        "generate_a3_aprilgrid_pair.py", "--output-dir", str(output), "--dpi", "72",
    ])

    assert generator.main() == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["revision_id"] == "a3-aprilgrid-pair-200-205_210-215-120mm-20mm-20260828-r1"
    assert manifest["acceptance"]["black_square_mm"] == [119.5, 120.5]
    assert manifest["acceptance"]["center_pitch_mm"] == [139.5, 140.5]
    assert [sheet["ids"] for sheet in manifest["sheets"]] == [
        [200, 201, 202, 203, 204, 205],
        [210, 211, 212, 213, 214, 215],
    ]

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    for sheet, expected in zip(manifest["sheets"], (range(200, 206), range(210, 216))):
        image = cv2.imread(str(output / sheet["png"]), cv2.IMREAD_GRAYSCALE)
        _, ids, _ = detector.detectMarkers(image)
        assert ids is not None
        assert sorted(ids.flatten().tolist()) == list(expected)
        assert (output / sheet["pdf"]).is_file()
        assert (output / sheet["layout"]).is_file()
