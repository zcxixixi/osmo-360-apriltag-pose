from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from osmo360.ffmpeg_runtime import project_ffmpeg_runtime
from osmo360.pipeline.ffmpeg_gray_pipe import FFmpegGrayPipe, probe_video_stream


def _video(tmp_path: Path) -> Path:
    output = tmp_path / "frames.mp4"
    runtime = project_ffmpeg_runtime()
    subprocess.run(
        [
            str(runtime.ffmpeg),
            "-nostdin", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=30:duration=1",
            "-c:v", "mpeg4", "-q:v", "2", "-pix_fmt", "yuv420p", str(output),
        ],
        check=True,
    )
    return output


def _read(pipe: FFmpegGrayPipe) -> np.ndarray:
    with pipe:
        return np.asarray([pipe.read() for _ in range(pipe.expected_frames)])


def test_ffmpeg_gray_pipe_stride_and_seek_are_frame_exact(tmp_path: Path):
    video = _video(tmp_path)
    info = probe_video_stream(video)
    assert info == type(info)(width=64, height=48, fps=30.0, frame_count=30)

    full = _read(FFmpegGrayPipe(
        video,
        width=64,
        height=48,
        fps=30.0,
        start_frame=0,
        end_frame=29,
        frame_stride=1,
        decoder_threads=1,
    ))
    even = _read(FFmpegGrayPipe(
        video,
        width=64,
        height=48,
        fps=30.0,
        start_frame=0,
        end_frame=29,
        frame_stride=2,
        decoder_threads=1,
    ))
    chunk = _read(FFmpegGrayPipe(
        video,
        width=64,
        height=48,
        fps=30.0,
        start_frame=10,
        end_frame=20,
        frame_stride=2,
        decoder_threads=1,
    ))

    assert full.shape == (30, 48, 64)
    assert np.array_equal(even, full[::2])
    assert np.array_equal(chunk, full[10:21:2])
    assert pipe_provenance(video)["decoder_transport"] == "ffmpeg_rawvideo_pipe"


def pipe_provenance(video: Path) -> dict[str, object]:
    pipe = FFmpegGrayPipe(
        video,
        width=64,
        height=48,
        fps=30.0,
        start_frame=0,
        end_frame=0,
        frame_stride=1,
    )
    provenance = pipe.provenance
    with pipe:
        pipe.read()
    return provenance
