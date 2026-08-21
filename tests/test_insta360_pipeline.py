from argparse import Namespace
from pathlib import Path

from insta360_mocap_pipeline import (
    DEFAULT_GRIPPER_MESHES,
    DEFAULT_TAG_MAP,
    evaluation_command,
    pipeline_paths,
    render_command,
    vision_command,
)


def arguments() -> Namespace:
    return Namespace(
        tag_map=DEFAULT_TAG_MAP,
        sample_fps=50.0,
        max_rmse_px=8.0,
        view_size=1440,
        global_search_size=720,
        recovery_scan_interval=15,
        max_speed=10.0,
        initial_time_offset=-3.852,
        time_search_radius=1.0,
        calibration_fraction=0.30,
        min_test_samples=200,
        output_fps=25.0,
        gripper_mesh_dir=DEFAULT_GRIPPER_MESHES,
    )


def test_packaged_tag_map_and_gripper_assets_exist():
    assert DEFAULT_TAG_MAP.is_file()
    assert all(
        (DEFAULT_GRIPPER_MESHES / name).is_file()
        for name in ("base_link.STL", "Link1.STL", "Link2.STL", "Link3.STL")
    )


def test_pipeline_commands_preserve_formal_evaluation_contract(tmp_path: Path):
    args = arguments()
    paths = pipeline_paths(tmp_path, "run")
    video = tmp_path / "video.mp4"
    motive = tmp_path / "motive.csv"
    vision = vision_command(args, video, paths, "cuda")
    evaluation = evaluation_command(args, motive, paths)
    render = render_command(args, video, paths)

    assert vision[vision.index("--sample-fps") + 1] == "50.0"
    assert vision[vision.index("--min-tags") + 1] == "2"
    assert vision[vision.index("--projection-backend") + 1] == "cuda"
    assert "--temporal-flow" not in vision
    assert evaluation[evaluation.index("--calibration-fraction") + 1] == "0.3"
    assert evaluation[evaluation.index("--min-test-samples") + 1] == "200"
    assert render[render.index("--gripper-mesh-dir") + 1] == str(DEFAULT_GRIPPER_MESHES)
    assert render[render.index("--output-fps") + 1] == "25.0"


def test_pipeline_output_layout(tmp_path: Path):
    paths = pipeline_paths(tmp_path, "trial")
    assert paths.pose_csv == tmp_path / "trial/visual/pose.csv"
    assert paths.evaluation_json == tmp_path / "trial/evaluation/mocap_evaluation.json"
    assert paths.comparison_video.name == "optitrack_vs_visual_gripper_kalman_rts.mp4"
