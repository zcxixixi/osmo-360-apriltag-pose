import json
import os
import subprocess
from pathlib import Path

from osmo360.visualization.node_runtime import resolve_node_binary


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATHS = ROOT / "dual_gripper_3d/runtime_paths.mjs"


def _resolve(environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    source = (
        "import {resolveChromeExecutable} from "
        + json.dumps(RUNTIME_PATHS.as_uri())
        + "; console.log(resolveChromeExecutable());"
    )
    return subprocess.run(
        [str(resolve_node_binary()), "--input-type=module", "--eval", source],
        capture_output=True,
        text=True,
        env=environment,
        timeout=5,
    )


def test_explicit_chrome_binary_is_used(tmp_path: Path) -> None:
    chrome = tmp_path / "chrome"
    chrome.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    chrome.chmod(0o755)
    environment = os.environ.copy()
    environment["OSMO_CHROME_BINARY"] = str(chrome)

    result = _resolve(environment)

    assert result.returncode == 0
    assert result.stdout.strip() == str(chrome)


def test_invalid_explicit_chrome_fails_instead_of_falling_back(tmp_path: Path) -> None:
    environment = os.environ.copy()
    missing = tmp_path / "missing-chrome"
    environment["OSMO_CHROME_BINARY"] = str(missing)

    result = _resolve(environment)

    assert result.returncode != 0
    assert f"configured Chrome binary is missing or not executable: {missing}" in result.stderr
