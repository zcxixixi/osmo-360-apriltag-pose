#!/usr/bin/env python3
"""Install the locked renderer dependencies with the supported Node runtime."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from osmo360.visualization.node_runtime import resolve_node_binary
from tools._root import ROOT


def npm_command(node: Path) -> list[str]:
    npm_cli = node.parents[1] / "lib/node_modules/npm/bin/npm-cli.js"
    if not npm_cli.is_file():
        raise FileNotFoundError(f"npm CLI is missing from Node runtime: {npm_cli}")
    return [str(node), str(npm_cli)]


def install_node_dependencies(package_root: Path) -> dict:
    node = resolve_node_binary()
    npm = npm_command(node)
    subprocess.run(
        [*npm, "ci", "--ignore-scripts", "--prefix", str(package_root)],
        check=True,
    )
    audit = subprocess.run(
        [*npm, "audit", "--json", "--prefix", str(package_root)],
        check=False,
        capture_output=True,
        text=True,
    )
    report = json.loads(audit.stdout)
    vulnerabilities = report.get("metadata", {}).get("vulnerabilities", {})
    if audit.returncode != 0 or vulnerabilities.get("total", 0) != 0:
        raise RuntimeError(
            "npm audit found vulnerabilities: "
            + json.dumps(vulnerabilities, sort_keys=True)
        )
    return {
        "node": str(node),
        "package_root": str(package_root.resolve()),
        "dependencies": report.get("metadata", {}).get("dependencies", {}),
        "vulnerabilities": vulnerabilities,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-root",
        type=Path,
        default=ROOT / "dual_gripper_3d",
    )
    args = parser.parse_args()
    print(json.dumps(install_node_dependencies(args.package_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
