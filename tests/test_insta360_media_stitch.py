from argparse import Namespace
from pathlib import Path

from tools.insta360_media_stitch import camera_model_hint, media_sdk_command


def test_media_sdk_command_keeps_stabilization_disabled(tmp_path: Path):
    sdk = tmp_path / "sdk"
    (sdk / "usr/bin").mkdir(parents=True)
    (sdk / "usr/lib").mkdir()
    (sdk / "models").mkdir()
    (sdk / "usr/bin/MediaSDKTest").touch()
    source = tmp_path / "clip.insv"
    source.touch()
    args = Namespace(
        sdk_root=sdk, input=[source], output=tmp_path / "panorama.mp4", width=3840,
        stitch_type="optflow", disable_cuda=False, soft_decode=False, soft_encode=False,
    )

    command, environment = media_sdk_command(args)

    assert command[command.index("-output_size") + 1] == "3840x1920"
    assert "-enable_flowstate" not in command
    assert "-enable_directionlock" not in command
    assert environment["LD_LIBRARY_PATH"].split(":", 1)[0] == str(sdk / "usr/lib")


def test_camera_model_hint_reads_insv_footer(tmp_path: Path):
    source = tmp_path / "clip.insv"
    source.write_bytes(b"video" + b"\x00" * 32 + b"Insta360 X5\x00metadata")

    assert camera_model_hint(source) == "Insta360 X5"
