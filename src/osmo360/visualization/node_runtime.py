"""Resolve a supported Node.js runtime for project renderers and services."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path


MINIMUM_NODE_VERSION = (22, 12, 0)
PINNED_RUNTIME_DIR = "node-v24.20.0-linux-x64"
PINNED_BINARY_SHA256 = "89af8424dd53e560b1933f87ba650d8bf57c83ca5a04600eefb31f416aabbae7"


class NodeRuntimeError(RuntimeError):
    """Raised when no supported Node.js runtime is available."""


def _version(binary: Path) -> tuple[int, int, int]:
    try:
        result = subprocess.run(
            [str(binary), "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise NodeRuntimeError(f"cannot execute Node.js runtime {binary}: {exc}") from exc
    match = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", result.stdout.strip())
    if match is None:
        raise NodeRuntimeError(
            f"Node.js runtime {binary} returned an invalid version: {result.stdout.strip()!r}"
        )
    return tuple(int(part) for part in match.groups())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate(binary: Path, *, expected_sha256: str | None = None) -> Path:
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise NodeRuntimeError(f"Node.js runtime is missing or not executable: {binary}")
    if expected_sha256 is not None:
        actual_sha256 = _sha256(binary)
        if actual_sha256 != expected_sha256:
            raise NodeRuntimeError(
                f"pinned Node.js binary SHA-256 mismatch: expected {expected_sha256}, "
                f"got {actual_sha256}"
            )
    version = _version(binary)
    if version < MINIMUM_NODE_VERSION:
        required = ".".join(map(str, MINIMUM_NODE_VERSION))
        actual = ".".join(map(str, version))
        raise NodeRuntimeError(
            f"Node.js {actual} at {binary} is unsupported; require >= {required}. "
            "Run `.venv/bin/python -m tools.install_node_runtime`."
        )
    return binary.resolve()


def resolve_node_binary(
    *,
    repo_root: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Prefer an explicit or pinned runtime and reject EOL system versions."""

    environment = os.environ if environ is None else environ
    root = (
        Path(__file__).resolve().parents[3]
        if repo_root is None
        else Path(repo_root).resolve()
    )
    configured = environment.get("OSMO_NODE_BINARY", "").strip()
    if configured:
        return _validate(Path(configured).expanduser().resolve())

    pinned = root / "work" / "tools" / PINNED_RUNTIME_DIR / "bin" / "node"
    if pinned.exists():
        return _validate(pinned, expected_sha256=PINNED_BINARY_SHA256)

    system = shutil.which("node", path=environment.get("PATH"))
    if system is None:
        raise NodeRuntimeError(
            "no Node.js runtime found; run `.venv/bin/python -m tools.install_node_runtime`"
        )
    return _validate(Path(system))
