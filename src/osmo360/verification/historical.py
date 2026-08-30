"""Verify immutable baseline files from their bound historical Git commits."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from osmo360.paths import ROOT


BINDINGS = ROOT / "config/baselines/historical_source_commits.json"


def baseline_commit(relative_baseline_path: str) -> str:
    data = json.loads(BINDINGS.read_text(encoding="utf-8"))
    return str(data["bindings"][relative_baseline_path])


def git_blob_sha256(commit: str, relative_path: str) -> str:
    process = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(process.stdout).hexdigest()


def verify_historical_file(commit: str, path: str, expected_sha256: str) -> None:
    value = Path(path)
    if value.is_absolute():
        actual = hashlib.sha256(value.read_bytes()).hexdigest()
    else:
        actual = git_blob_sha256(commit, path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"historical baseline mismatch for {path}: expected {expected_sha256}, got {actual}"
        )
