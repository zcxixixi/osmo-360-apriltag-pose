#!/usr/bin/env python3
"""Local controller: offline panorama by default, legacy RTMP preserved."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request, send_from_directory

from tools._root import ROOT
SESSIONS = ROOT / "sessions"
INPUT = ROOT / "work/input"
app = Flask(__name__)
lock = threading.Lock()
state: dict[str, object] = {"process": None, "session": None, "log": None, "mode": None}

PAGE = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Osmo 360 AprilGrid</title><style>
:root{color-scheme:dark;font-family:system-ui}body{margin:0;background:#090c10;color:#eef2f6}main{width:min(1000px,92vw);margin:35px auto;padding:30px;background:#151a21;border:1px solid #2c3541;border-radius:22px}h1{margin:0}.sub{color:#9aa6b4;margin:8px 0 24px}.tabs{display:flex;gap:8px;margin-bottom:20px}.tab{background:#252d38;color:#ccd5df}.tab.active{background:#5bbcff;color:#071018}.panel{display:none}.panel.active{display:block}.controls{display:grid;grid-template-columns:2fr repeat(3,1fr);gap:12px;align-items:end}label{display:grid;gap:6px;color:#9aa6b4;font-size:13px}input,select{padding:11px;border:1px solid #3a4553;border-radius:9px;background:#0d1218;color:white}button{padding:13px 20px;border:0;border-radius:10px;background:#35d078;color:#052211;font-weight:700;cursor:pointer}button.stop{background:#ff6464}.status{margin-top:20px;display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.metric{background:#0d1218;padding:14px;border-radius:10px}.metric span{font-size:12px;color:#83909f}.metric b{display:block;font-size:20px;margin-top:4px}#log{white-space:pre-wrap;max-height:180px;overflow:auto;background:#090c10;padding:12px;color:#b9c5d2}#result img{width:100%;margin-top:18px;border-radius:12px}.warning{color:#ffc857}a{color:#65bdff}@media(max-width:700px){.controls,.status{grid-template-columns:1fr 1fr}}</style></head><body><main>
<h1>Osmo 360 · AprilGrid</h1><div class="sub">默认处理本地 2:1 全景文件；结果按 demo/approximate 标注。</div><div class="tabs"><button class="tab active" onclick="tab('offline')">全景文件处理</button><button class="tab" onclick="tab('legacy')">Legacy / 实验 RTMP</button></div>
<section id="offline" class="panel active"><div class="controls"><label>本地视频<select id="video"></select></label><label>Tag 边长 m<input id="tag" type="number" value="0.088" step="0.001"></label><label>spacing<input id="spacing" type="number" value="0.30" step="0.01"></label><label>采样 fps<input id="fps" type="number" value="5" step="1"></label></div><p><label><span><input id="official" type="checkbox"> 输入是 DJI Studio 官方导出的 2:1 全景</span></label></p><p class="warning">OSV/LRF 仅列出供检查；处理入口只接受可靠拼接的 2:1 MP4，LRF 不作正式输入。</p><button id="run" onclick="toggleOffline()">开始处理</button></section>
<section id="legacy" class="panel"><p>旧单镜头低清 RTMP 原型，仅供实验。</p><button id="legacyRun" onclick="toggleLegacy()">开始 Legacy 实时处理</button></section>
<div class="status"><div class="metric"><span>状态</span><b id="running">空闲</b></div><div class="metric"><span>处理帧</span><b id="frames">0</b></div><div class="metric"><span>有效位姿</span><b id="poses">0</b></div><div class="metric"><span>累计 ID</span><b id="tags">0 / 36</b></div></div><p id="message"></p><pre id="log"></pre><div id="result"></div>
<script>let active=false,lastImage='';function tab(id){document.querySelectorAll('.panel').forEach(x=>x.classList.toggle('active',x.id===id));document.querySelectorAll('.tab').forEach((x,i)=>x.classList.toggle('active',(i===0)===(id==='offline')))}async function videos(){let j=await(await fetch('/api/videos')).json();video.innerHTML=j.videos.map(v=>`<option value="${v.path}">${v.label}</option>`).join('')||'<option>未找到视频</option>'}async function post(url,body={}){let r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),j=await r.json();if(!r.ok)throw Error(j.error);message.textContent=j.message||'';await refresh()}async function toggleOffline(){try{active?await post('/api/stop'):await post('/api/start/offline',{path:video.value,tag_size:+tag.value,spacing:+spacing.value,sample_fps:+fps.value,official_stitched:official.checked})}catch(e){message.textContent=e.message}}async function toggleLegacy(){try{active?await post('/api/stop'):await post('/api/start/legacy',{tag_size:+tag.value,spacing:+spacing.value})}catch(e){message.textContent=e.message}}async function refresh(){let j=await(await fetch('/api/status',{cache:'no-store'})).json();active=j.running;running.textContent=active?'处理中':'空闲';frames.textContent=j.processed_frames||j.frame||j.frames||0;poses.textContent=j.valid_pose_frames||j.pose_frames||0;tags.textContent=(j.seen_ids||j.recognized_ids||[]).length+' / 36';run.textContent=active?'安全停止':'开始处理';run.className=active?'stop':'';legacyRun.textContent=active?'安全停止':'开始 Legacy 实时处理';legacyRun.className=active?'stop':'';log.textContent=j.log_tail||'';if(j.image&&j.image!==lastImage){lastImage=j.image;result.innerHTML=`<a href="${j.image}"><img src="${j.image}?t=${Date.now()}"></a><p>${(j.downloads||[]).map(x=>`<a href="${x.url}" download>${x.name}</a>`).join(' · ')}</p>`}}videos();refresh();setInterval(refresh,1000)</script></main></body></html>"""


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def safe_roots() -> list[Path]:
    user = os.environ.get("USER", "")
    return [INPUT, Path("/media") / user, Path("/run/media") / user, Path("/mnt")]


def scan_videos() -> list[dict]:
    found = []
    for root in safe_roots():
        if not root.exists():
            continue
        try:
            for path in root.rglob("*"):
                if path.is_file() and path.suffix.lower() in {".osv", ".lrf", ".mp4"}:
                    found.append(
                        {
                            "path": str(path.resolve()),
                            "label": f"{path.name} · {path.stat().st_size / 1048576:.1f} MB",
                        }
                    )
        except (OSError, PermissionError):
            pass
    return sorted(found, key=lambda x: x["path"])


def validate_selected(raw: str) -> Path:
    selected = Path(raw).resolve(strict=True)
    if selected.suffix.lower() not in {".osv", ".lrf", ".mp4"}:
        raise ValueError("不支持的文件类型")
    if not any(
        root.exists()
        and (selected == root.resolve() or root.resolve() in selected.parents)
        for root in safe_roots()
    ):
        raise ValueError("路径不在允许目录")
    return selected


def launch(command: list[str], mode: str, session: Path) -> None:
    session.mkdir(parents=True, exist_ok=False)
    log = (session / "launcher.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    state.update(process=process, session=session, log=log, mode=mode)


@app.get("/")
def index():
    return render_template_string(PAGE)


@app.get("/api/videos")
def api_videos():
    return jsonify(videos=scan_videos())


@app.post("/api/start/offline")
def start_offline():
    p = request.get_json(silent=True) or {}
    try:
        selected = validate_selected(str(p.get("path", "")))
        tag = float(p.get("tag_size", 0.088))
        spacing = float(p.get("spacing", 0.3))
        fps = float(p.get("sample_fps", 5))
    except (OSError, ValueError) as e:
        return jsonify(error=str(e)), 400
    if selected.suffix.lower() != ".mp4":
        return jsonify(
            error="OSV 必须先分析/可靠拼接；LRF 只供预览。位姿入口仅接受 2:1 MP4。"
        ), 400
    if tag <= 0 or spacing < 0 or fps <= 0:
        return jsonify(error="参数无效"), 400
    with lock:
        proc = state.get("process")
        if isinstance(proc, subprocess.Popen) and proc.poll() is None:
            return jsonify(error="已有任务运行中"), 409
        name = datetime.now().strftime("%Y%m%d-%H%M%S")
        session = SESSIONS / name
        status = session / "status.json"
        cmd = [
            sys.executable,
            "-m",
            "tools.insta360_offline",
            str(selected),
            "--tag-size",
            str(tag),
            "--spacing",
            str(spacing),
            "--sample-fps",
            str(fps),
            "--output-dir",
            str(SESSIONS),
            "--session-name",
            name,
            "--status-file",
            str(status),
        ]
        if bool(p.get("official_stitched")):
            cmd.append("--official-stitched")
        launch(cmd, "offline", session)
    return jsonify(message="全景离线处理已开始")


@app.post("/api/start/legacy")
def start_legacy():
    p = request.get_json(silent=True) or {}
    try:
        tag = float(p.get("tag_size", 0.088))
        spacing = float(p.get("spacing", 0.3))
    except (TypeError, ValueError):
        return jsonify(error="参数无效"), 400
    if tag <= 0 or spacing < 0:
        return jsonify(error="参数无效"), 400
    with lock:
        proc = state.get("process")
        if isinstance(proc, subprocess.Popen) and proc.poll() is None:
            return jsonify(error="已有任务运行中"), 409
        name = "legacy-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        session = SESSIONS / name
        status = session / "status.json"
        cmd = [
            sys.executable,
            "-m",
            "tools.osmo_apriltag_demo",
            "--tag-size",
            str(tag),
            "--spacing",
            str(spacing),
            "--no-display",
            "--csv",
            str(session / "pose.csv"),
            "--jsonl",
            str(session / "detections.jsonl"),
            "--summary",
            str(session / "summary.json"),
            "--live-status",
            str(status),
        ]
        launch(cmd, "legacy", session)
    return jsonify(message="Legacy RTMP 实验已开始")


@app.post("/api/stop")
def stop():
    with lock:
        proc = state.get("process")
        if not isinstance(proc, subprocess.Popen) or proc.poll() is not None:
            return jsonify(error="没有运行中的任务"), 409
        os.killpg(proc.pid, signal.SIGINT)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=5)
    return jsonify(message="已安全停止；已处理数据已保留")


@app.get("/api/status")
def status():
    with lock:
        proc, session = state.get("process"), state.get("session")
        running = isinstance(proc, subprocess.Popen) and proc.poll() is None
        if not running and state.get("log"):
            state["log"].close()
            state["log"] = None
    out: dict = {"running": running}
    if isinstance(session, Path):
        out.update(read_json(session / ("status.json" if running else "summary.json")))
        logs = [session / "processor.log", session / "launcher.log"]
        text = "\n".join(
            p.read_text(encoding="utf-8", errors="replace") for p in logs if p.exists()
        )
        out["log_tail"] = "\n".join(text.splitlines()[-12:])
        image = session / "relative_coordinates.png"
        if image.exists():
            out["image"] = f"/results/{session.name}/{image.name}"
        out["downloads"] = [
            {"name": p.name, "url": f"/results/{session.name}/{p.name}"}
            for p in session.iterdir()
            if p.name
            in {
                "pose.csv",
                "detections.jsonl",
                "summary.json",
                "relative_coordinates.png",
                "processor.log",
            }
        ]
    return jsonify(out)


@app.get("/results/<session>/<filename>")
def results(session: str, filename: str):
    return send_from_directory(SESSIONS / session, filename)


if __name__ == "__main__":
    SESSIONS.mkdir(parents=True, exist_ok=True)
    INPUT.mkdir(parents=True, exist_ok=True)
    app.run(host="127.0.0.1", port=7860, threaded=True)
