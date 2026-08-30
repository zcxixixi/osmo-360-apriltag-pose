import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_ROOT_FILES = {
    ".gitignore",
    "AGENTS.md",
    "README.md",
    "pyproject.toml",
    "requirements-train-cu130.txt",
    "umi",
    "uv.lock",
}


def tracked_root_files() -> set[str]:
    output = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        path
        for path in output.splitlines()
        if path and len(Path(path).parts) == 1
    }


def test_repository_root_contains_only_project_entry_files():
    assert tracked_root_files() <= ALLOWED_ROOT_FILES


def test_top_level_guides_remain_concise():
    assert len((ROOT / "README.md").read_text(encoding="utf-8").splitlines()) <= 450
    assert len((ROOT / "AGENTS.md").read_text(encoding="utf-8").splitlines()) <= 220
