"""Render concise terminal progress for the resumable InstaUMI pipeline."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osmo360.pipeline.four_mp4 import PIPELINE_REVISION
from osmo360.pipeline.instaumi import load_instaumi_config
from osmo360.pipeline.manifest import confined_path, validate_path_component


@dataclass(frozen=True)
class ProgressSnapshot:
    key: str
    label: str
    state: str
    completed: int = 0
    total: int = 0
    completed_frames: int = 0
    total_frames: int = 0


def progress_snapshot(payload: dict[str, Any]) -> ProgressSnapshot:
    stages = payload.get("stages", {})
    identity = stages.get("identity", {})
    if identity.get("state") != "PASS":
        return ProgressSnapshot(
            "identity",
            "校验输入与缓存身份",
            str(identity.get("state", "WAITING")),
        )

    sync = stages.get("sync", {})
    if sync.get("state") != "PASS":
        return ProgressSnapshot("sync", "同步左右相机时间轴", str(sync.get("state", "WAITING")))

    observations = stages.get("observation_chunks", {})
    if observations.get("state") not in {"PASS", "REUSED"}:
        return ProgressSnapshot(
            "observation_chunks",
            "四路视频解码与灰度检测",
            str(observations.get("state", "WAITING")),
            completed=int(observations.get("completed", 0)),
            total=int(observations.get("total", 0)),
            completed_frames=int(observations.get("completed_frames", 0)),
            total_frames=int(observations.get("total_frames", 0)),
        )

    dual = stages.get("dual_lens_observations", {})
    if dual.get("state") != "PASS":
        return ProgressSnapshot(
            "dual_lens_observations",
            "合并双镜头观测",
            str(dual.get("state", "WAITING")),
        )

    tracking = stages.get("trajectory_tracking", {})
    if tracking.get("state") != "PASS":
        return ProgressSnapshot(
            "trajectory_tracking",
            "求解双手联合轨迹",
            str(tracking.get("state", "WAITING")),
        )
    return ProgressSnapshot("complete", "轨迹处理完成", "PASS")


def _format_duration(seconds: float) -> str:
    value = max(0, int(round(seconds)))
    minutes, remainder = divmod(value, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{remainder:02d}"
    return f"{minutes:02d}:{remainder:02d}"


def _render(snapshot: ProgressSnapshot, elapsed_s: float, stage_elapsed_s: float) -> str:
    prefix = f"[轨迹 1/2][{_format_duration(elapsed_s)}] {snapshot.label}"
    if snapshot.key != "observation_chunks" or snapshot.total <= 0:
        return prefix
    ratio = (
        snapshot.completed_frames / snapshot.total_frames
        if snapshot.total_frames > 0
        else snapshot.completed / snapshot.total
    )
    ratio = min(1.0, max(0.0, ratio))
    details = f" {snapshot.completed}/{snapshot.total} 块"
    if snapshot.total_frames > 0:
        details += (
            f" | {snapshot.completed_frames}/{snapshot.total_frames} 帧"
            "（四路合计）"
        )
    details += f" | {ratio * 100:5.1f}%"
    if 0 < ratio < 1 and stage_elapsed_s > 0:
        eta_s = stage_elapsed_s * (1.0 - ratio) / ratio
        details += f" | ETA {_format_duration(eta_s)}"
    return prefix + details


def _status_path(root: Path) -> Path:
    config = load_instaumi_config(root)
    dataset_id = validate_path_component(root.name, field="dataset directory name")
    pair_id = validate_path_component(config.get("pair_id"), field="pair_id")
    cache_base = Path(
        os.environ.get("OSMO_PIPELINE_CACHE", str(root / ".osmo-cache"))
    ).expanduser().resolve()
    return confined_path(
        cache_base,
        dataset_id,
        PIPELINE_REVISION,
        pair_id,
        "status.json",
        field="pipeline progress status",
    )


def monitor(root: Path, *, started_at_ns: int, interval_s: float = 0.5) -> int:
    status_path = _status_path(root.resolve(strict=True))
    started = time.monotonic()
    stage_key = ""
    stage_started = started
    last_non_tty_key: tuple[Any, ...] | None = None
    interactive = sys.stderr.isatty()
    stopping = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        while not stopping:
            snapshot = ProgressSnapshot("waiting", "准备轨迹流水线", "WAITING")
            try:
                if status_path.is_file() and status_path.stat().st_mtime_ns >= started_at_ns:
                    payload = json.loads(status_path.read_text(encoding="utf-8"))
                    snapshot = progress_snapshot(payload)
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass
            now = time.monotonic()
            if snapshot.key != stage_key:
                stage_key = snapshot.key
                stage_started = now
            line = _render(snapshot, now - started, now - stage_started)
            if interactive:
                print(f"\r\033[2K{line}", end="", file=sys.stderr, flush=True)
            non_tty_key = (
                snapshot.key,
                snapshot.state,
                snapshot.completed,
                snapshot.total,
                snapshot.completed_frames,
                snapshot.total_frames,
            )
            if not interactive and non_tty_key != last_non_tty_key:
                print(line, file=sys.stderr, flush=True)
                last_non_tty_key = non_tty_key
            if snapshot.key == "complete":
                if interactive:
                    print(file=sys.stderr, flush=True)
                return 0
            time.sleep(interval_s)
    finally:
        if interactive and stage_key != "complete":
            print(file=sys.stderr, flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--started-at-ns", type=int, required=True)
    parser.add_argument("--interval-s", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return monitor(
        args.dataset_root,
        started_at_ns=args.started_at_ns,
        interval_s=args.interval_s,
    )


if __name__ == "__main__":
    raise SystemExit(main())
