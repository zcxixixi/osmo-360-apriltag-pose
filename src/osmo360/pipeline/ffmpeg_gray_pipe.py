"""Verified FFmpeg raw-gray frame transport for the CPU four-MP4 pipeline."""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import BinaryIO

import numpy as np

from osmo360.ffmpeg_runtime import FFmpegRuntime, project_ffmpeg_runtime


class FFmpegGrayPipeError(RuntimeError):
    """Raised when FFmpeg cannot deliver the exact requested frame sequence."""


@dataclass(frozen=True)
class VideoStreamInfo:
    width: int
    height: int
    fps: float
    frame_count: int
    pixel_format: str
    color_range: str | None


def probe_video_stream(
    video: Path,
    *,
    runtime: FFmpegRuntime | None = None,
) -> VideoStreamInfo:
    """Read the first video's fixed geometry and timing with pinned ffprobe."""
    source = video.resolve(strict=True)
    selected_runtime = runtime or project_ffmpeg_runtime()
    result = subprocess.run(
        [
            str(selected_runtime.ffprobe),
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames,pix_fmt,color_range",
            "-of", "json",
            str(source),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode:
        raise FFmpegGrayPipeError(
            f"ffprobe failed for {source}: {result.stderr.strip()}"
        )
    try:
        streams = json.loads(result.stdout)["streams"]
        stream = streams[0]
        fps = float(Fraction(str(stream["avg_frame_rate"])))
        info = VideoStreamInfo(
            width=int(stream["width"]),
            height=int(stream["height"]),
            fps=fps,
            frame_count=int(stream["nb_frames"]),
            pixel_format=str(stream["pix_fmt"]),
            color_range=(
                str(stream["color_range"])
                if stream.get("color_range") is not None
                else None
            ),
        )
    except (IndexError, KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise FFmpegGrayPipeError(
            f"ffprobe returned incomplete fixed-frame metadata for {source}"
        ) from exc
    if info.width <= 0 or info.height <= 0 or info.fps <= 0 or info.frame_count <= 0:
        raise FFmpegGrayPipeError(f"invalid video stream metadata for {source}: {info}")
    return info


class FFmpegGrayPipe:
    """Stream selected luma planes from FFmpeg stdout into NumPy arrays.

    The subprocess is deliberately bounded by an exact source-frame interval,
    stride, output frame count, fixed resolution, and verified project runtime.
    No shell is involved.  stderr is spooled to a temporary file so the child
    can never deadlock on an undrained diagnostic pipe.
    """

    def __init__(
        self,
        video: Path,
        *,
        width: int,
        height: int,
        fps: float,
        start_frame: int,
        end_frame: int,
        frame_stride: int,
        decoder_threads: int = 0,
        runtime: FFmpegRuntime | None = None,
    ) -> None:
        source = video.resolve(strict=True)
        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError("width, height, and fps must be positive")
        if start_frame < 0 or end_frame < start_frame or frame_stride <= 0:
            raise ValueError("invalid FFmpeg frame interval or stride")
        if decoder_threads < 0:
            raise ValueError("decoder_threads must be non-negative")
        self.video = source
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.start_frame = int(start_frame)
        self.end_frame = int(end_frame)
        self.frame_stride = int(frame_stride)
        self.expected_frames = (end_frame - start_frame) // frame_stride + 1
        self.frame_bytes = self.width * self.height
        self.runtime = runtime or project_ffmpeg_runtime()
        self._read_frames = 0
        self._closed = False
        self._stderr: BinaryIO = tempfile.TemporaryFile(mode="w+b")

        command = [
            str(self.runtime.ffmpeg),
            "-nostdin", "-hide_banner", "-loglevel", "error",
        ]
        if decoder_threads:
            command.extend(("-threads", str(decoder_threads)))
        if start_frame:
            command.extend(("-ss", f"{start_frame / fps:.12f}"))
        filters = ["extractplanes=y"]
        if frame_stride > 1:
            filters.insert(0, f"select=not(mod(n\\,{frame_stride}))")
        command.extend((
            "-i", str(source),
            "-map", "0:v:0", "-an", "-sn", "-dn",
            "-vf", ",".join(filters),
            "-fps_mode", "passthrough",
            "-frames:v", str(self.expected_frames),
            "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
        ))
        self.command = tuple(command)
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=self._stderr,
                bufsize=0,
            )
        except OSError:
            self._stderr.close()
            raise
        if self._process.stdout is None:  # pragma: no cover - subprocess contract
            self._abort()
            raise FFmpegGrayPipeError("FFmpeg stdout pipe was not created")

    @property
    def provenance(self) -> dict[str, object]:
        return {
            "decoder_transport": "ffmpeg_rawvideo_pipe",
            "pixel_format": "gray8_luma",
            "seek_method": "accurate_input_timestamp_seek",
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "frame_stride": self.frame_stride,
            "expected_frames": self.expected_frames,
            "ffmpeg": self.runtime.provenance(),
        }

    def _diagnostic(self) -> str:
        self._stderr.flush()
        self._stderr.seek(0)
        return self._stderr.read().decode("utf-8", errors="replace").strip()

    def _read_exact(self, size: int) -> bytes:
        assert self._process.stdout is not None
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = self._process.stdout.read(remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def read(self) -> np.ndarray:
        if self._closed:
            raise FFmpegGrayPipeError("cannot read from a closed FFmpeg pipe")
        if self._read_frames >= self.expected_frames:
            raise StopIteration
        payload = self._read_exact(self.frame_bytes)
        if len(payload) != self.frame_bytes:
            return_code = self._process.wait(timeout=10)
            raise FFmpegGrayPipeError(
                "FFmpeg rawvideo pipe ended early at frame "
                f"{self._read_frames}/{self.expected_frames} with {len(payload)}/"
                f"{self.frame_bytes} bytes (exit={return_code}): {self._diagnostic()}"
            )
        self._read_frames += 1
        return np.frombuffer(payload, dtype=np.uint8).reshape(self.height, self.width)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        assert self._process.stdout is not None
        if self._read_frames != self.expected_frames:
            self._abort()
            raise FFmpegGrayPipeError(
                f"FFmpeg pipe closed after {self._read_frames}/{self.expected_frames} frames"
            )
        extra = self._process.stdout.read(1)
        return_code = self._process.wait(timeout=30)
        self._process.stdout.close()
        diagnostic = self._diagnostic()
        self._stderr.close()
        if extra or return_code:
            raise FFmpegGrayPipeError(
                f"FFmpeg rawvideo completion mismatch (extra={bool(extra)}, "
                f"exit={return_code}): {diagnostic}"
            )

    def _abort(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
        if self._process.stdout is not None:
            self._process.stdout.close()
        self._stderr.close()
        self._closed = True

    def __enter__(self) -> "FFmpegGrayPipe":
        return self

    def __exit__(self, exc_type, _exc, _traceback) -> bool:
        if exc_type is None:
            self.close()
        else:
            self._abort()
        return False


@dataclass(frozen=True)
class YUV420Frame:
    luma: np.ndarray
    chroma_u: np.ndarray
    chroma_v: np.ndarray


class FFmpegYUV420Pipe:
    """Stream exact full-range YUV420 frames for shared gray/color analysis."""

    def __init__(
        self,
        video: Path,
        *,
        width: int,
        height: int,
        fps: float,
        start_frame: int,
        end_frame: int,
        frame_stride: int,
        decoder_threads: int = 0,
        runtime: FFmpegRuntime | None = None,
    ) -> None:
        source = video.resolve(strict=True)
        if width <= 0 or height <= 0 or width % 2 or height % 2 or fps <= 0:
            raise ValueError("YUV420 width/height must be positive and even; fps must be positive")
        if start_frame < 0 or end_frame < start_frame or frame_stride <= 0:
            raise ValueError("invalid FFmpeg frame interval or stride")
        if decoder_threads < 0:
            raise ValueError("decoder_threads must be non-negative")
        self.video = source
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.start_frame = int(start_frame)
        self.end_frame = int(end_frame)
        self.frame_stride = int(frame_stride)
        self.expected_frames = (end_frame - start_frame) // frame_stride + 1
        self.frame_bytes = self.width * self.height * 3 // 2
        self.runtime = runtime or project_ffmpeg_runtime()
        stream_info = probe_video_stream(source, runtime=self.runtime)
        if stream_info.pixel_format not in {"yuv420p", "yuvj420p"}:
            raise ValueError(
                "shared YUV420 transport requires an 8-bit 4:2:0 source, got "
                f"{stream_info.pixel_format}"
            )
        self.output_pixel_format = (
            "yuvj420p"
            if stream_info.pixel_format == "yuvj420p" or stream_info.color_range == "pc"
            else "yuv420p"
        )
        self._read_frames = 0
        self._closed = False
        self._stderr: BinaryIO = tempfile.TemporaryFile(mode="w+b")

        command = [
            str(self.runtime.ffmpeg),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
        ]
        if decoder_threads:
            command.extend(("-threads", str(decoder_threads)))
        if start_frame:
            command.extend(("-ss", f"{start_frame / fps:.12f}"))
        filters = []
        if frame_stride > 1:
            filters.append(f"select=not(mod(n\\,{frame_stride}))")
        command.extend(("-i", str(source), "-map", "0:v:0", "-an", "-sn", "-dn"))
        if filters:
            command.extend(("-vf", ",".join(filters)))
        command.extend(
            (
                "-fps_mode",
                "passthrough",
                "-frames:v",
                str(self.expected_frames),
                "-f",
                "rawvideo",
                "-pix_fmt",
                self.output_pixel_format,
                "pipe:1",
            )
        )
        self.command = tuple(command)
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=self._stderr,
                bufsize=0,
            )
        except OSError:
            self._stderr.close()
            raise
        if self._process.stdout is None:  # pragma: no cover - subprocess contract
            self._abort()
            raise FFmpegGrayPipeError("FFmpeg stdout pipe was not created")

    @property
    def provenance(self) -> dict[str, object]:
        return {
            "decoder_transport": "ffmpeg_rawvideo_pipe",
            "pixel_format": f"{self.output_pixel_format}_shared_luma_chroma",
            "seek_method": "accurate_input_timestamp_seek",
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "frame_stride": self.frame_stride,
            "expected_frames": self.expected_frames,
            "ffmpeg": self.runtime.provenance(),
        }

    def _diagnostic(self) -> str:
        self._stderr.flush()
        self._stderr.seek(0)
        return self._stderr.read().decode("utf-8", errors="replace").strip()

    def _read_exact(self, size: int) -> bytes:
        assert self._process.stdout is not None
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = self._process.stdout.read(remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def read(self) -> YUV420Frame:
        if self._closed:
            raise FFmpegGrayPipeError("cannot read from a closed FFmpeg pipe")
        if self._read_frames >= self.expected_frames:
            raise StopIteration
        payload = self._read_exact(self.frame_bytes)
        if len(payload) != self.frame_bytes:
            return_code = self._process.wait(timeout=10)
            raise FFmpegGrayPipeError(
                "FFmpeg YUV420 pipe ended early at frame "
                f"{self._read_frames}/{self.expected_frames} with {len(payload)}/"
                f"{self.frame_bytes} bytes (exit={return_code}): {self._diagnostic()}"
            )
        self._read_frames += 1
        flat = np.frombuffer(payload, dtype=np.uint8)
        luma_size = self.width * self.height
        chroma_size = luma_size // 4
        return YUV420Frame(
            luma=flat[:luma_size].reshape(self.height, self.width),
            chroma_u=flat[luma_size : luma_size + chroma_size].reshape(
                self.height // 2, self.width // 2
            ),
            chroma_v=flat[luma_size + chroma_size :].reshape(
                self.height // 2, self.width // 2
            ),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        assert self._process.stdout is not None
        if self._read_frames != self.expected_frames:
            self._abort()
            raise FFmpegGrayPipeError(
                f"FFmpeg YUV420 pipe closed after {self._read_frames}/{self.expected_frames} frames"
            )
        extra = self._process.stdout.read(1)
        return_code = self._process.wait(timeout=30)
        self._process.stdout.close()
        diagnostic = self._diagnostic()
        self._stderr.close()
        if extra or return_code:
            raise FFmpegGrayPipeError(
                f"FFmpeg YUV420 completion mismatch (extra={bool(extra)}, "
                f"exit={return_code}): {diagnostic}"
            )

    def _abort(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
        if self._process.stdout is not None:
            self._process.stdout.close()
        self._stderr.close()
        self._closed = True

    def __enter__(self) -> "FFmpegYUV420Pipe":
        return self

    def __exit__(self, exc_type, _exc, _traceback) -> bool:
        if exc_type is None:
            self.close()
        else:
            self._abort()
        return False
