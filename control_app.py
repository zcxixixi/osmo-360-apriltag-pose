#!/usr/bin/env python3
"""Local start/stop controller for the Osmo 360 AprilGrid live processor."""

from __future__ import annotations

import csv
import json
import math
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, jsonify, render_template_string, request, send_from_directory


ROOT = Path(__file__).resolve().parent
SESSIONS = ROOT / "sessions"
app = Flask(__name__)
lock = threading.Lock()
state: dict[str, object] = {"process": None, "session": None, "started_at": None, "log": None}


PAGE = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Osmo 360 AprilGrid</title>
  <style>
    :root { color-scheme: dark; font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
    body { margin:0; min-height:100vh; background:#0b0d10; color:#eef2f6; display:grid; place-items:center; }
    main { width:min(920px,92vw); padding:38px; background:#15191f; border:1px solid #29303a; border-radius:24px; box-shadow:0 24px 80px #0008; }
    h1 { margin:0 0 8px; font-size:30px; } .sub { color:#99a4b2; margin-bottom:28px; }
    .video-wrap { margin-bottom:24px; overflow:hidden; border-radius:16px; border:1px solid #303946; background:#050607; aspect-ratio:16/9; }
    .video-wrap iframe { width:100%; height:100%; display:block; border:0; }
    .controls { display:flex; gap:14px; align-items:end; flex-wrap:wrap; }
    label { color:#99a4b2; font-size:13px; display:grid; gap:7px; }
    input { width:130px; padding:12px; border-radius:10px; border:1px solid #36404d; background:#0d1116; color:white; font-size:16px; }
    button { min-width:220px; padding:15px 24px; border:0; border-radius:12px; background:#22c55e; color:#06150b; font-size:17px; font-weight:750; cursor:pointer; }
    button.stop { background:#fb5b5b; color:#1c0505; } button:disabled { opacity:.45; cursor:wait; }
    .status { margin-top:24px; padding:18px; border-radius:14px; background:#0d1116; display:grid; grid-template-columns:repeat(4,1fr); gap:16px; }
    .metric b { display:block; font-size:22px; margin-top:5px; } .metric span { color:#7f8a98; font-size:12px; }
    #message { margin-top:16px; color:#f6c65b; } #result { margin-top:24px; }
    #result img { width:100%; border-radius:14px; border:1px solid #303946; }
    a { color:#64b5ff; } @media(max-width:650px){.status{grid-template-columns:1fr 1fr}main{padding:22px}}
  </style>
</head>
<body><main>
  <h1>Osmo 360 · AprilGrid</h1>
  <div class="sub">实时识别、记录相机相对坐标，停止后生成轨迹图片</div>
  <div class="video-wrap">
    <iframe src="http://127.0.0.1:8889/osmo/live?controls=true&muted=true&autoplay=true&playsInline=true" scrolling="no" allow="autoplay; fullscreen; picture-in-picture" title="Osmo 360 实时画面"></iframe>
  </div>
  <div class="controls">
    <label>Tag 黑色边长（米）<input id="tagSize" type="number" value="0.088" min="0.001" step="0.001"></label>
    <label>Tag 间距比例<input id="spacing" type="number" value="0.30" min="0" step="0.01"></label>
    <button id="toggle" onclick="toggleRun()">开始实时处理</button>
  </div>
  <div class="status">
    <div class="metric"><span>状态</span><b id="running">已停止</b></div>
    <div class="metric"><span>处理帧</span><b id="frames">0</b></div>
    <div class="metric"><span>有效位姿帧</span><b id="poses">0</b></div>
    <div class="metric"><span>累计 Tag</span><b id="tags">0 / 36</b></div>
  </div>
  <div id="message"></div>
  <div id="result"></div>
</main>
<script>
let running=false, busy=false, lastImage='';
async function toggleRun(){
  if(busy)return; busy=true; document.getElementById('toggle').disabled=true;
  const starting=!running, url=running?'/api/stop':'/api/start';
  const body=running?{}:{tag_size:Number(tagSize.value),spacing:Number(spacing.value)};
  try{const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const j=await r.json();if(!r.ok)throw Error(j.error||'操作失败');message.textContent=j.message||'';if(starting)document.getElementById('result').innerHTML='';}
  catch(e){message.textContent=e.message;} finally{busy=false;document.getElementById('toggle').disabled=false;await refresh();}
}
async function refresh(){
  const r=await fetch('/api/status',{cache:'no-store'}),j=await r.json();running=j.running;
  const b=document.getElementById('toggle');b.textContent=running?'停止并生成结果':'开始实时处理';b.className=running?'stop':'';
  document.getElementById('running').textContent=running?'处理中':'已停止';
  document.getElementById('frames').textContent=j.frame||j.frames||0;
  document.getElementById('poses').textContent=j.pose_frames||0;
  document.getElementById('tags').textContent=(j.seen_ids||[]).length+' / 36';
  if(j.image&&j.image!==lastImage){lastImage=j.image;document.getElementById('result').innerHTML='<a href="'+j.image+'" download><img src="'+j.image+'?t='+Date.now()+'"></a><p><a href="'+j.image+'" download>下载坐标轨迹 PNG</a></p>';}
}
setInterval(refresh,1000);refresh();
</script></body></html>
"""


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def generate_plot(session: Path) -> Path:
    csv_path = session / "pose.csv"
    summary = read_json(session / "summary.json")
    rows: list[dict[str, str]] = []
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

    valid: list[tuple[float, float, float, float]] = []
    for row in rows:
        try:
            x, y, z = float(row["camera_x_m"]), float(row["camera_y_m"]), float(row["camera_z_m"])
            error = float(row["reprojection_rmse_px"])
            if all(math.isfinite(v) and abs(v) < 50 for v in (x, y, z)) and error <= 10:
                valid.append((x, y, z, error))
        except (KeyError, TypeError, ValueError):
            continue

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 10), facecolor="#0b0d10")
    seen = summary.get("seen_ids", [])
    fig.suptitle(f"Osmo 360 relative trajectory · {len(valid)} pose frames · Tags {len(seen)}/36", fontsize=18, y=.97)

    if not valid:
        axis = fig.add_subplot(111)
        axis.axis("off")
        axis.text(.5, .55, "No valid AprilGrid pose was recorded", ha="center", va="center", fontsize=24)
        axis.text(.5, .45, f"Detected IDs: {seen}", ha="center", va="center", fontsize=13, color="#9aa5b3")
    else:
        data = np.asarray(valid)
        x, y, z, error = data.T
        color = np.linspace(0, 1, len(data))
        ax3 = fig.add_subplot(221, projection="3d")
        ax3.plot(x, y, z, color="#5bbcff", linewidth=1.5, alpha=.8)
        sc = ax3.scatter(x, y, z, c=color, cmap="viridis", s=10)
        ax3.scatter(x[0], y[0], z[0], c="#4ade80", s=90, label="Start")
        ax3.scatter(x[-1], y[-1], z[-1], c="#fb5b5b", s=90, label="End")
        ax3.scatter([0], [0], [0], c="white", marker="s", s=80, label="AprilGrid")
        ax3.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)", title="3D camera trajectory")
        ax3.legend(loc="upper right")
        for position, a, b, title in [(222, x, y, "Top · XY"), (223, x, z, "Side · XZ"), (224, y, z, "Side · YZ")]:
            ax = fig.add_subplot(position)
            ax.plot(a, b, color="#5bbcff", linewidth=1.4)
            ax.scatter(a, b, c=color, cmap="viridis", s=8)
            ax.scatter(a[0], b[0], c="#4ade80", s=70)
            ax.scatter(a[-1], b[-1], c="#fb5b5b", s=70)
            ax.scatter([0], [0], c="white", marker="s", s=55)
            ax.set_title(title); ax.grid(alpha=.18); ax.set_aspect("equal", adjustable="datalim")
        fig.colorbar(sc, ax=fig.axes, shrink=.55, pad=.03, label="Session progress")
        fig.text(.02, .015, f"Detected IDs: {seen}   ·   Median reprojection error: {np.median(error):.2f}px", color="#aab4c0")
        if not summary.get("calibrated_intrinsics", False):
            fig.text(.98, .015, "APPROXIMATE INTRINSICS — DEMO-GRADE COORDINATES", ha="right", color="#f6c65b", weight="bold")

    output = session / "relative_coordinates.png"
    fig.savefig(output, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output


@app.get("/")
def index():
    return render_template_string(PAGE)


@app.post("/api/start")
def start():
    payload = request.get_json(silent=True) or {}
    tag_size = float(payload.get("tag_size", 0.088))
    spacing = float(payload.get("spacing", 0.30))
    if tag_size <= 0 or spacing < 0:
        return jsonify(error="尺寸参数无效"), 400
    with lock:
        process = state.get("process")
        if isinstance(process, subprocess.Popen) and process.poll() is None:
            return jsonify(error="实时处理已经在运行"), 409
        session = SESSIONS / datetime.now().strftime("%Y%m%d-%H%M%S")
        session.mkdir(parents=True, exist_ok=False)
        log_handle = (session / "processor.log").open("w", encoding="utf-8")
        command = [
            sys.executable, str(ROOT / "osmo_apriltag_demo.py"),
            "--tag-size", str(tag_size), "--spacing", str(spacing), "--no-display",
            "--csv", str(session / "pose.csv"), "--jsonl", str(session / "detections.jsonl"),
            "--summary", str(session / "summary.json"), "--live-status", str(session / "status.json"),
        ]
        process = subprocess.Popen(command, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT)
        state.update(process=process, session=session, started_at=time.time(), log=log_handle)
    return jsonify(message="实时处理已开始")


@app.post("/api/stop")
def stop():
    with lock:
        process = state.get("process")
        session = state.get("session")
        if not isinstance(process, subprocess.Popen) or process.poll() is not None or not isinstance(session, Path):
            return jsonify(error="没有正在运行的处理任务"), 409
        process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait(timeout=5)
    with lock:
        log_handle = state.get("log")
        if log_handle:
            log_handle.close()
        state["process"] = None
        state["log"] = None
    generate_plot(session)
    return jsonify(message="处理已停止，坐标图片已生成")


@app.get("/api/status")
def status():
    with lock:
        process = state.get("process")
        session = state.get("session")
        running = isinstance(process, subprocess.Popen) and process.poll() is None
    result: dict[str, object] = {"running": running}
    if isinstance(session, Path):
        result.update(read_json(session / ("status.json" if running else "summary.json")))
        image = session / "relative_coordinates.png"
        if image.exists():
            result["image"] = f"/results/{session.name}/{image.name}"
    return jsonify(result)


@app.get("/results/<session>/<filename>")
def results(session: str, filename: str):
    return send_from_directory(SESSIONS / session, filename, as_attachment=False)


if __name__ == "__main__":
    SESSIONS.mkdir(parents=True, exist_ok=True)
    app.run(host="127.0.0.1", port=7860, debug=False, threaded=True)
