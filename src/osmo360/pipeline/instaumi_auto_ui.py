from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template_string

from .instaumi_auto import (
    _full_outputs_complete,
    _process_complete,
    discover_pairs,
)

PROCESS_PERCENT = re.compile(r"\|\s*(\d+(?:\.\d+)?)%")

PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>InstaUMI 自动处理</title>
<style>
:root{color-scheme:dark;--bg:#0d1117;--panel:#161b22;--line:#30363d;--text:#e6edf3;--muted:#8b949e;--blue:#58a6ff;--green:#3fb950;--red:#f85149;--amber:#d29922}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 ui-sans-serif,system-ui;padding:24px}.wrap{max-width:1500px;margin:auto}h1{font-size:24px;margin:0 0 4px}.sub{color:var(--muted);margin-bottom:20px}.cards{display:grid;grid-template-columns:repeat(5,minmax(120px,1fr));gap:12px;margin-bottom:18px}.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px}.card b{font-size:26px;display:block}.card span{color:var(--muted)}table{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line)}th,td{padding:10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{color:var(--muted);position:sticky;top:0;background:var(--panel)}.bar{width:180px;height:10px;background:#30363d;border-radius:5px;overflow:hidden}.fill{height:100%;background:var(--blue)}.COMPLETE{color:var(--green)}.FAILED{color:var(--red)}.RUNNING{color:var(--blue)}.WAITING{color:var(--muted)}.mono{font-family:ui-monospace,monospace;font-size:12px}.error{max-width:420px;color:var(--red);white-space:normal}.small{color:var(--muted);font-size:12px}@media(max-width:900px){.cards{grid-template-columns:repeat(2,1fr)}table{font-size:12px}.hide-small{display:none}}
</style>
</head>
<body><div class="wrap">
<h1>InstaUMI 自动处理</h1><div class="sub" id="updated">正在读取状态…</div>
<div class="cards" id="cards"></div>
<table><thead><tr><th>状态</th><th>数据集</th><th>节点</th><th>阶段</th><th>进度</th><th class="hide-small">源文件</th><th>信息</th></tr></thead><tbody id="rows"></tbody></table>
</div><script>
const esc=s=>String(s??'').replace(/[&<>\"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}[c]));
async function refresh(){
 const r=await fetch('/api/status',{cache:'no-store'});const d=await r.json();
 document.getElementById('updated').textContent=`每 3 秒刷新 · ${d.updated_at} · 分片 ${d.shards.join(', ')||'无'}`;
 const labels=[['total','总计'],['complete','已完成'],['running','处理中'],['waiting','等待'],['failed','失败']];
 document.getElementById('cards').innerHTML=labels.map(([k,l])=>`<div class="card"><b>${d.counts[k]}</b><span>${l}</span></div>`).join('');
 document.getElementById('rows').innerHTML=d.tasks.map(t=>`<tr><td class="${t.status}"><b>${esc(t.status)}</b></td><td><b>${esc(t.episode)}</b><div class="small">${esc(t.collector)}</div></td><td>${esc(t.node||'-')}</td><td>${esc(t.stage)}</td><td><div>${t.progress.toFixed(1)}%</div><div class="bar"><div class="fill" style="width:${t.progress}%"></div></div></td><td class="mono hide-small">L ${esc(t.left)}<br>R ${esc(t.right)}</td><td class="error">${esc(t.error||t.output||'')}</td></tr>`).join('');
}
refresh();setInterval(refresh,3000);
</script></body></html>"""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _latest_pair_states(automation_root: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    merged: dict[str, dict[str, Any]] = {}
    shards = []
    paths = sorted(automation_root.glob("state*.json"))
    if any("-of-" in path.stem for path in paths):
        paths = [path for path in paths if path.name != "state.json"]
    for path in paths:
        state = _load_json(path)
        shard = path.stem.removeprefix("state") or "legacy"
        shards.append(f"{shard}:{state.get('node', '-')}")
        for key, value in state.get("pairs", {}).items():
            if not isinstance(value, dict):
                continue
            current = merged.get(key)
            if current is None or str(value.get("updated_at_utc", "")) >= str(
                current.get("updated_at_utc", "")
            ):
                merged[key] = value
    return merged, shards


def _ffmpeg_progress(log_root: Path, frame_count: int) -> float:
    if frame_count <= 0:
        return 0.0
    total = 0
    for path in log_root.glob("video-*.progress"):
        values: dict[str, str] = {}
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    values[key] = value
        except OSError:
            continue
        total += min(frame_count, int(values.get("frame", 0)))
    return min(100.0, 100.0 * total / (6 * frame_count))


def _process_progress(log: Path) -> float:
    try:
        text = log.read_text(encoding="utf-8")
    except OSError:
        return 0.0
    values = [float(match.group(1)) for match in PROCESS_PERCENT.finditer(text)]
    if "[完成]" in text:
        return 100.0
    return max(values, default=0.0)


def _task_progress(data_root: Path, pair, state: dict[str, Any]) -> tuple[str, float]:
    status = str(state.get("status", "WAITING"))
    stage = str(state.get("stage", "waiting"))
    if status == "COMPLETE":
        return "complete", 100.0
    logs = data_root / "_automation/logs" / pair.collector_root.name / pair.episode_name
    if stage == "format":
        detail = _load_json(logs / "format_status.json")
        format_stage = str(detail.get("stage", "sha"))
        if format_stage == "audio":
            return "audio sync", 5.0
        if format_stage == "video":
            video = _ffmpeg_progress(logs, int(detail.get("frame_count_per_video", 0)))
            return "video encoding", 10.0 + 75.0 * video / 100.0
        if format_stage == "imu":
            return "IMU extraction", 88.0
        if format_stage == "hdf5":
            return "HDF5", 94.0
        if format_stage == "complete":
            return "format complete", 96.0
        return "SHA verification", 2.0
    if stage == "trajectory":
        downstream = _process_progress(logs / "process.log")
        return "trajectory/gripper", 96.0 + 3.0 * downstream / 100.0
    if stage == "render":
        return "trajectory video", 99.0
    if status == "FAILED":
        return "failed", 0.0
    return stage, 0.0


def build_status(data_root: Path) -> dict[str, Any]:
    data_root = data_root.resolve(strict=True)
    automation_root = data_root / "_automation"
    states, shards = _latest_pair_states(automation_root)
    tasks = []
    for pair in discover_pairs(data_root):
        episode_path = pair.collector_root / pair.episode_name
        state = dict(states.get(pair.key, {}))
        if _process_complete(episode_path):
            state["status"] = "COMPLETE"
            state["stage"] = "complete"
        status = str(state.get("status", "WAITING"))
        if status not in {"COMPLETE", "RUNNING", "FAILED"}:
            status = "WAITING"
        stage, progress = _task_progress(data_root, pair, state)
        tasks.append({
            "status": status,
            "collector": pair.collector_root.name,
            "episode": pair.episode_name,
            "node": state.get("node"),
            "stage": stage,
            "progress": progress,
            "left": pair.left.path.name,
            "right": pair.right.path.name,
            "error": state.get("error"),
            "output": str(episode_path / "processed") if status == "COMPLETE" else None,
        })
    order = {"RUNNING": 0, "FAILED": 1, "WAITING": 2, "COMPLETE": 3}
    tasks.sort(key=lambda item: (order[item["status"]], item["episode"], item["collector"]))
    counts = {
        "total": len(tasks),
        "complete": sum(task["status"] == "COMPLETE" for task in tasks),
        "running": sum(task["status"] == "RUNNING" for task in tasks),
        "waiting": sum(task["status"] == "WAITING" for task in tasks),
        "failed": sum(task["status"] == "FAILED" for task in tasks),
    }
    return {
        "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "counts": counts,
        "shards": shards,
        "tasks": tasks,
    }


def create_app(data_root: Path) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template_string(PAGE)

    @app.get("/api/status")
    def api_status():
        return jsonify(build_status(data_root))

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the InstaUMI automation progress dashboard")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/ps/current-robotics-data-2/total_annotation/umi_insta360"),
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7871)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    create_app(args.data_root).run(host=args.host, port=args.port, threaded=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
