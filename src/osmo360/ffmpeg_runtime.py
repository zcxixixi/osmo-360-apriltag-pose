"""Resolve and verify the pinned project FFmpeg/FFprobe runtime."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


MINIMUM_VERSION = (9, 0, 1)
PINNED_RUNTIME_DIR = "ffmpeg-9.0.1-linux-x86_64"
PINNED_REVISION_ID = "ffmpeg-linux-x64-9.0.1-osmo1"
PINNED_FFMPEG_SHA256 = "91f3138dafa5ecfaee9156f4323c43809e64c05b5e612cb8528453ec09fa1143"
PINNED_FFPROBE_SHA256 = "cc11804f067a81a229b419acc4486aa6f1ee345103edd81360ce69f631bef15c"


class FFmpegRuntimeError(RuntimeError):
    """Raised when no verified and supported FFmpeg runtime is available."""


@dataclass(frozen=True)
class FFmpegRuntime:
    ffmpeg: Path
    ffprobe: Path
    version: str
    revision_id: str | None
    ffmpeg_sha256: str
    ffprobe_sha256: str

    def provenance(self) -> dict[str, str | None]:
        return {
            "version": self.version,
            "revision_id": self.revision_id,
            "ffmpeg_sha256": self.ffmpeg_sha256,
            "ffprobe_sha256": self.ffprobe_sha256,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version(binary: Path, program: str) -> tuple[tuple[int, int, int], str]:
    try:
        result = subprocess.run(
            [str(binary), "-version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise FFmpegRuntimeError(f"cannot execute {program} runtime {binary}: {exc}") from exc
    first_line = result.stdout.splitlines()[0] if result.stdout else ""
    match = re.match(rf"{re.escape(program)} version (\d+)\.(\d+)\.(\d+)(?:\s|$)", first_line)
    if match is None:
        raise FFmpegRuntimeError(
            f"{program} runtime {binary} returned an invalid version: {first_line!r}"
        )
    parts = tuple(int(part) for part in match.groups())
    return parts, ".".join(match.groups())


def _validate_binary(
    path: Path,
    program: str,
    *,
    expected_sha256: str | None = None,
) -> tuple[str, str]:
    if not path.is_file() or not os.access(path, os.X_OK):
        raise FFmpegRuntimeError(f"{program} runtime is missing or not executable: {path}")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o022:
        raise FFmpegRuntimeError(f"{program} runtime is group/other-writable: {path}")
    actual_sha256 = _sha256(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise FFmpegRuntimeError(
            f"pinned {program} SHA-256 mismatch: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )
    parsed, version = _version(path, program)
    if parsed < MINIMUM_VERSION:
        required = ".".join(map(str, MINIMUM_VERSION))
        raise FFmpegRuntimeError(
            f"{program} {version} at {path} is unsupported; require >= {required}. "
            "Run `.venv/bin/python -m tools.install_ffmpeg_runtime --archive PATH`."
        )
    return version, actual_sha256


def _validate_pair(
    bin_dir: Path,
    *,
    expected_ffmpeg_sha256: str | None = None,
    expected_ffprobe_sha256: str | None = None,
    revision_id: str | None = None,
    require_real_files: bool = False,
) -> FFmpegRuntime:
    candidates = (bin_dir / "ffmpeg", bin_dir / "ffprobe")
    if require_real_files:
        if bin_dir.is_symlink() or any(path.is_symlink() for path in candidates):
            raise FFmpegRuntimeError("pinned FFmpeg runtime must not contain symlinks")
        if any(path.stat().st_uid != os.getuid() for path in candidates):
            raise FFmpegRuntimeError("pinned FFmpeg runtime is owned by another user")
    ffmpeg, ffprobe = (path.resolve() for path in candidates)
    ffmpeg_version, ffmpeg_sha256 = _validate_binary(
        ffmpeg, "ffmpeg", expected_sha256=expected_ffmpeg_sha256
    )
    ffprobe_version, ffprobe_sha256 = _validate_binary(
        ffprobe, "ffprobe", expected_sha256=expected_ffprobe_sha256
    )
    if ffmpeg_version != ffprobe_version:
        raise FFmpegRuntimeError(
            f"ffmpeg/ffprobe version mismatch: {ffmpeg_version} != {ffprobe_version}"
        )
    return FFmpegRuntime(
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        version=ffmpeg_version,
        revision_id=revision_id,
        ffmpeg_sha256=ffmpeg_sha256,
        ffprobe_sha256=ffprobe_sha256,
    )


def resolve_ffmpeg_runtime(
    *,
    repo_root: Path | None = None,
    environ: dict[str, str] | None = None,
) -> FFmpegRuntime:
    """Prefer an explicit or pinned runtime and reject legacy FFmpeg builds."""

    environment = os.environ if environ is None else environ
    root = (
        Path(__file__).resolve().parents[2]
        if repo_root is None
        else Path(repo_root).resolve()
    )
    configured = environment.get("OSMO_FFMPEG_BIN", "").strip()
    if configured:
        return _validate_pair(Path(configured).expanduser().resolve())

    pinned = root / "work" / "tools" / PINNED_RUNTIME_DIR / "bin"
    if pinned.exists():
        return _validate_pair(
            pinned,
            expected_ffmpeg_sha256=PINNED_FFMPEG_SHA256,
            expected_ffprobe_sha256=PINNED_FFPROBE_SHA256,
            revision_id=PINNED_REVISION_ID,
            require_real_files=True,
        )

    ffmpeg = shutil.which("ffmpeg", path=environment.get("PATH"))
    ffprobe = shutil.which("ffprobe", path=environment.get("PATH"))
    if ffmpeg is None or ffprobe is None:
        raise FFmpegRuntimeError(
            "no supported FFmpeg runtime found; run "
            "`.venv/bin/python -m tools.install_ffmpeg_runtime --archive PATH`"
        )
    ffmpeg_path = Path(ffmpeg).resolve()
    ffprobe_path = Path(ffprobe).resolve()
    if ffmpeg_path.parent != ffprobe_path.parent:
        raise FFmpegRuntimeError("system ffmpeg and ffprobe must come from the same directory")
    return _validate_pair(ffmpeg_path.parent)


@lru_cache(maxsize=1)
def project_ffmpeg_runtime() -> FFmpegRuntime:
    """Resolve and hash the process-wide project runtime once."""

    return resolve_ffmpeg_runtime()
