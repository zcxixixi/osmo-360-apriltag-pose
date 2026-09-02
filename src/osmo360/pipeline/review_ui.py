from __future__ import annotations

import csv
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse
import webbrowser

import cv2
import numpy as np

from .review_store import DECISIONS, REASONS, SEGMENT_LABELS, ReviewStore

PAGE = r'''<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>UMI 数据审核</title><style>
[hidden]{display:none!important}:root{font-family:Inter,"Noto Sans SC",system-ui,sans-serif;color:#17202a;background:#f3f6f8}*{box-sizing:border-box}body{margin:0;overflow:hidden}button,input{font:inherit}.top{height:64px;background:#122431;color:#fff;display:flex;align-items:center;justify-content:space-between;padding:0 20px}.top h1{font-size:20px;margin:0}.layout{display:grid;grid-template-columns:330px 1fr;height:calc(100vh - 64px);min-height:0}aside{background:#fff;border-right:1px solid #dce4e8;padding:12px;overflow-y:auto;min-height:0}.filters{display:flex;gap:6px;flex-wrap:wrap;position:sticky;top:-12px;background:#fff;padding:12px 0 8px;z-index:2}.filters button{border:1px solid #ccd7dd;background:#fff;border-radius:18px;padding:6px 10px;cursor:pointer}.filters button.active{background:#173d52;color:#fff}.item{border:2px solid #e2e9ed;border-radius:12px;padding:11px;margin:7px 0;cursor:pointer}.item.active{border-color:#168bd2;background:#eef8fe}.item h3{margin:0;font-size:15px}.row{display:flex;align-items:center;justify-content:space-between;gap:10px}.badge{font-size:12px;padding:4px 8px;border-radius:16px;background:#e9eef1}.approved{background:#dff5e4;color:#17662b}.rejected{background:#fee1e1;color:#8d2020}.pending{color:#53636d}main{padding:14px 18px;overflow-y:auto;min-height:0}.empty{display:grid;place-items:center;height:100%;color:#6c7d87}.title{font-size:18px;font-weight:800}.viewer{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:10px 0}.view{background:#0b1115;border-radius:12px;overflow:hidden;position:relative;height:min(38vh,430px)}.view label{position:absolute;z-index:1;left:9px;top:8px;background:#0009;color:#fff;padding:4px 8px;border-radius:10px}.view video,.view img{width:100%;height:100%;object-fit:contain;display:block}.panel{background:#fff;border-radius:12px;padding:12px;margin-top:10px}.timebar{display:grid;grid-template-columns:90px auto 1fr;align-items:center;gap:10px}.timebar input{width:100%}.play{border:0;border-radius:9px;padding:9px 14px;background:#173d52;color:#fff;font-weight:750;cursor:pointer}.review{display:flex;gap:9px;align-items:center;flex-wrap:wrap}.review input{width:150px;border:1px solid #cbd7dd;border-radius:8px;padding:9px}.review button,.trim button,.align button,.marker button{border:0;border-radius:9px;padding:10px 14px;font-weight:750;cursor:pointer}.export{display:inline-block;text-decoration:none;border-radius:9px;padding:9px 13px;background:#eef8fe;color:#075f91;font-weight:750}.yes{background:#24a148;color:#fff}.no{background:#da3b3b;color:#fff}.next{background:#e8edf0}.trim{display:flex;align-items:center;gap:9px;flex-wrap:wrap}.start{background:#168bd2;color:#fff}.end{background:#e18b00;color:#fff}.marker{display:flex;align-items:center;gap:8px;padding:8px 0;border-top:1px solid #e2e9ed}.marker .seek{background:#eef8fe;color:#075f91}.marker .delete{margin-left:auto;background:#f5f5f5;color:#8d2020;padding:6px 9px}.message{min-height:20px;margin-top:7px;color:#a33220;font-weight:650}.align{display:flex;gap:7px;align-items:center;flex-wrap:wrap;margin-top:8px}.align input{width:100px;padding:7px}.align button{border:1px solid #cbd7dd;background:#fff;padding:7px 9px}details{margin-top:10px;color:#53636d}@media(max-width:900px){body{overflow:auto}.layout{display:block;height:auto}aside{max-height:260px}.viewer{grid-template-columns:1fr}.view{height:300px}main{overflow:visible}}
</style></head><body><header class="top"><h1>UMI 数据审核</h1><div id="summary"></div></header><div class="layout"><aside><div class="filters" id="filters"></div><div id="list"></div></aside><main><div id="empty" class="empty">请选择一条数据</div><div id="detail" hidden><div class="row"><div class="title" id="title"></div><div><a class="export" href="/api/keyframes-export" download="keyframes.json">导出 JSON</a> <button class="next" onclick="nextItem()">下一条</button></div></div><div class="viewer"><div class="view"><label>左</label><img id="left"><video id="leftVideo" controls muted playsinline hidden></video></div><div class="view"><label>右</label><img id="right"><video id="rightVideo" controls muted playsinline hidden></video></div></div><div class="panel timebar"><b id="time">0.000 秒</b><button class="play" onclick="togglePlayback()">播放 / 暂停</button><input id="slider" type="range" min="0" max="0" step="0.033333" value="0"></div><section class="panel"><div class="review"><b>数据能用吗？</b><input id="reviewer" placeholder="审核人"><button class="yes" id="yesButton" onclick="saveUsability(true)">能用</button><button class="no" onclick="saveUsability(false)">不能用</button><span id="reviewState"></span></div><div class="message" id="message"></div></section><section class="panel" id="trimPanel" hidden><div class="trim"><b>裁剪点</b><button class="start" onclick="addKeyframe('useful_start')">开始</button><button class="end" onclick="addKeyframe('useful_end')">结束</button><span>当前 <b id="cutTime">0.000 秒</b></span></div><div class="message" id="keyframeMessage"></div><div id="keyframes"></div></section><details class="panel" id="alignmentDetails"><summary>左右对齐微调</summary><div class="align"><button onclick="adjustAlignment(1/videoFps)">右提前1帧</button><button onclick="adjustAlignment(-1/videoFps)">右延后1帧</button><input id="alignmentOffset" type="number" min="-30" max="30" step="0.001" onchange="setAlignment(Number(this.value))"><span id="alignmentValue"></span><button onclick="saveAlignment()">保存</button></div><div class="message" id="alignmentMessage"></div></details></div></main></div><script>
let items=[],current=null,currentData=null,filter='all',manualOffset=0,videoFps=30;const labels={approved:'可用',rejected:'不可用',reprocess:'需处理',reprocessed:'已处理',pending:'待审核'};const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
async function api(url,opt){const r=await fetch(url,opt);const j=await r.json();if(!r.ok)throw Error(j.error||'操作失败');return j}function status(i){return i.decision&&!i.stale_review?i.decision:'pending'}
function renderFilters(){const fs=[['all','全部'],['pending','待审核'],['approved','可用'],['rejected','不可用']];filters.innerHTML=fs.map(x=>`<button class="${filter===x[0]?'active':''}" onclick="filter='${x[0]}';renderList()">${x[1]}</button>`).join('')}
function renderList(){renderFilters();const shown=items.filter(i=>filter==='all'||status(i)===filter);list.innerHTML=shown.map(i=>`<div class="item ${current===i.pair_id?'active':''}" onclick="openItem('${i.pair_id}')"><div class="row"><h3>${esc(i.pair_id)}</h3><span class="badge ${status(i)}">${labels[status(i)]||labels.pending}</span></div><small>${Math.round(i.metrics.duration_s||i.metrics.rgb_samples||0)} 秒</small></div>`).join('');summary.textContent=`${items.filter(i=>status(i)!=='pending').length} / ${items.length}`}
async function load(){items=await api('/api/items');renderList();if(items.length)await openItem(items.find(i=>status(i)==='pending')?.pair_id||items[0].pair_id)}
function commonTime(){return leftVideo.hidden?Number(slider.value):Number(leftVideo.currentTime||slider.value)}function updateTime(value){const v=Math.max(0,Number(value)||0);time.textContent=`${v.toFixed(3)} 秒`;cutTime.textContent=`${v.toFixed(3)} 秒`;slider.value=v}
function rightTarget(){return Math.max(0,Math.min(rightVideo.duration||Infinity,commonTime()+manualOffset))}function syncRight(force=false){const target=rightTarget();if(force||Math.abs(rightVideo.currentTime-target)>.06)rightVideo.currentTime=target}function togglePlayback(){if(leftVideo.hidden)return;if(leftVideo.paused){syncRight(true);Promise.allSettled([leftVideo.play(),rightVideo.play()])}else{leftVideo.pause();rightVideo.pause()}}
async function openItem(id){current=id;currentData=await api(`/api/items/${id}`);const saved=await api(`/api/items/${id}/alignment`);manualOffset=Number(saved.right_time_offset_s||0);videoFps=Number(currentData.metrics.video_fps||30);alignmentOffset.value=manualOffset.toFixed(3);renderAlignment();empty.hidden=true;detail.hidden=false;title.textContent=id;const aligned=!!currentData.metrics.aligned_video_ready;left.hidden=right.hidden=aligned;leftVideo.hidden=rightVideo.hidden=!aligned;alignmentDetails.hidden=!aligned;if(aligned){leftVideo.src=`/api/items/${id}/video?role=left`;rightVideo.src=`/api/items/${id}/video?role=right`;leftVideo.onplay=()=>{syncRight(true);rightVideo.play().catch(()=>{})};leftVideo.onpause=()=>rightVideo.pause();leftVideo.ontimeupdate=()=>{syncRight();updateTime(leftVideo.currentTime)}}slider.max=Math.max(0,Number(currentData.metrics.duration_s||currentData.metrics.rgb_samples||0));slider.step=(1/videoFps).toFixed(6);slider.value=0;reviewer.value=localStorage.reviewer||currentData.reviewer||'';updateTime(0);const s=status(currentData);reviewState.textContent=labels[s]||labels.pending;trimPanel.hidden=s!=='approved';yesButton.disabled=!currentData.metrics.review_ready;await loadKeyframes();renderList()}
slider.oninput=()=>{const v=Number(slider.value);if(!leftVideo.hidden){leftVideo.currentTime=v;syncRight(true)}else{left.src=`/api/items/${current}/frame?role=left&index=${Math.floor(v)}`;right.src=`/api/items/${current}/frame?role=right&index=${Math.floor(v)}`}updateTime(v)};
async function saveUsability(usable){message.textContent='';const name=reviewer.value.trim();if(!name){message.textContent='请填写审核人';return}localStorage.reviewer=name;try{await api(`/api/items/${current}/review`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({decision:usable?'approved':'rejected',reasons:usable?[]:['other'],notes:'',reviewer:name})});items=await api('/api/items');if(usable)await openItem(current);else nextItem()}catch(e){message.textContent=e.message}}
async function addKeyframe(label){keyframeMessage.textContent='';const name=reviewer.value.trim();if(!name){keyframeMessage.textContent='请填写审核人';return}try{await api(`/api/items/${current}/keyframes`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({time_sec:commonTime(),label,reviewer:name})});await loadKeyframes()}catch(e){keyframeMessage.textContent=e.message}}
async function loadKeyframes(){if(!current)return;const rows=await api(`/api/items/${current}/keyframes`);keyframes.innerHTML=rows.map(x=>`<div class="marker"><button class="seek" onclick="seekTo(${x.time_sec})">${x.label==='useful_start'?'开始':'结束'} ${x.time_sec.toFixed(3)} 秒 · 帧 ${x.frame}</button><button class="delete" onclick="deleteKeyframe('${x.id}')">删除</button></div>`).join('')}
function seekTo(v){slider.value=v;slider.oninput()}async function deleteKeyframe(id){try{await api(`/api/keyframes/${id}`,{method:'DELETE'});await loadKeyframes()}catch(e){keyframeMessage.textContent=e.message}}
function renderAlignment(){alignmentValue.textContent=`${manualOffset>=0?'+':''}${manualOffset.toFixed(3)} 秒`;alignmentOffset.value=manualOffset.toFixed(3)}function setAlignment(v){manualOffset=Math.max(-30,Math.min(30,Number(v)||0));renderAlignment();syncRight(true)}function adjustAlignment(v){setAlignment(manualOffset+v)}async function saveAlignment(){alignmentMessage.textContent='';const name=reviewer.value.trim();if(!name){alignmentMessage.textContent='请填写审核人';return}try{await api(`/api/items/${current}/alignment`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({right_time_offset_s:manualOffset,reviewer:name,notes:''})});alignmentMessage.textContent='已保存'}catch(e){alignmentMessage.textContent=e.message}}
function nextItem(){const i=items.findIndex(x=>x.pair_id===current);const next=items.slice(i+1).find(x=>status(x)==='pending')||items.find(x=>status(x)==='pending')||items[(i+1)%items.length];if(next)openItem(next.pair_id)}
load().catch(e=>{empty.textContent=e.message});
</script></body></html>'''


@lru_cache(maxsize=2)
def load_rgb_samples(path: str, mtime_ns: int) -> dict[str, np.ndarray]:
    del mtime_ns
    with np.load(path) as arrays:
        return {"left": arrays["left"].copy(), "right": arrays["right"].copy()}


def pair_timeline(directory: Path) -> dict[str, list[list[float]]]:
    tag_values: dict[int, list[int]] = {}
    tag_path = directory / "tag_detections.jsonl"
    if tag_path.is_file():
        for line in tag_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            second = int(float(row["common_time_s"]))
            tag_values.setdefault(second, []).append(len(row.get("ids", [])))
    gripper_values: dict[int, list[int]] = {}
    gripper_path = directory / "gripper_stats.csv"
    if gripper_path.is_file():
        with gripper_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                second = int(float(row["common_time_s"]))
                gripper_values.setdefault(second, []).append(int(row["candidate_count"]))
    return {
        "tags": [[second, sum(values) / len(values)] for second, values in sorted(tag_values.items())],
        "gripper": [[second, sum(values) / len(values)] for second, values in sorted(gripper_values.items())],
    }


def suggest_segments(directory: Path) -> list[dict[str, float | str]]:
    path = directory / "review_bundle" / "timeline.json"
    if not path.is_file():
        return []
    timeline = json.loads(path.read_text(encoding="utf-8"))
    frames = timeline.get("frames", [])
    if len(frames) < 3:
        return []
    times = np.asarray([float(frame["t"]) for frame in frames])
    left = np.asarray([frame["left"]["p"] for frame in frames], dtype=float)
    right = np.asarray([frame["right"]["p"] for frame in frames], dtype=float)
    dt = np.maximum(np.diff(times), 1e-6)
    speed = np.maximum(
        np.linalg.norm(np.diff(left, axis=0), axis=1) / dt,
        np.linalg.norm(np.diff(right, axis=0), axis=1) / dt,
    )
    window = max(1, round(float(timeline.get("fps", 20)) * 0.35))
    smooth = np.convolve(speed, np.ones(window) / window, mode="same")
    median = float(np.median(smooth))
    mad = float(np.median(np.abs(smooth - median)))
    active = smooth > max(0.025, median + 3.0 * 1.4826 * mad)
    padded = np.pad(active.astype(np.int8), (1, 1))
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1)
    proposals = []
    for start, end in zip(starts, ends):
        start_s = max(0.0, float(times[start]) - 0.5)
        end_s = min(float(times[-1]), float(times[min(end, len(times) - 1)]) + 0.5)
        if end_s - start_s >= 1.0:
            proposals.append({"start_s": start_s, "end_s": end_s, "label": "move"})
    return proposals


def serve_review_ui(
    dataset_root: Path, *, host: str = "127.0.0.1", port: int = 7869,
    open_browser: bool = True, state_root: Path | None = None,
) -> None:
    store = ReviewStore(dataset_root, state_root=state_root)
    store.scan()

    class Handler(BaseHTTPRequestHandler):
        def json_response(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def file_response(self, path: Path, content_type: str) -> None:
            size = path.stat().st_size
            start, end = 0, size - 1
            range_header = self.headers.get("Range", "")
            status = HTTPStatus.OK
            if range_header.startswith("bytes="):
                bounds = range_header[6:].split("-", 1)
                start = int(bounds[0] or 0)
                end = min(int(bounds[1]) if bounds[1] else size - 1, size - 1)
                if start < 0 or start > end:
                    raise ValueError("invalid byte range")
                status = HTTPStatus.PARTIAL_CONTENT
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = end - start + 1
                while remaining:
                    chunk = handle.read(min(1024 * 1024, remaining))
                    if not chunk: break
                    self.wfile.write(chunk); remaining -= len(chunk)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            try:
                if path == "/":
                    body = PAGE.encode()
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if path == "/api/config":
                    return self.json_response({
                        "decisions": sorted(DECISIONS),
                        "reasons": REASONS,
                        "segment_labels": SEGMENT_LABELS,
                    })
                if path == "/api/items":
                    return self.json_response(store.scan())
                if path == "/api/reprocess-queue":
                    value = json.loads(store.queue_path.read_text()) if store.queue_path.is_file() else {"items": []}
                    return self.json_response(value)
                if path == "/api/keyframes-export":
                    if not store.keyframes_path.is_file():
                        store.keyframes_path.write_text("{}\n", encoding="utf-8")
                    return self.file_response(store.keyframes_path, "application/json; charset=utf-8")
                parts = path.strip("/").split("/")
                if len(parts) >= 3 and parts[:2] == ["api", "items"]:
                    pair_id = parts[2]
                    item = store.get_item(pair_id)
                    if len(parts) == 3:
                        return self.json_response(item)
                    if parts[3] == "history":
                        return self.json_response(store.history(pair_id))
                    if parts[3] == "alignment":
                        return self.json_response(store.get_alignment(pair_id))
                    if parts[3] == "timeline":
                        return self.json_response(pair_timeline(Path(item["source_dir"])))
                    if parts[3] == "segments":
                        return self.json_response(store.list_segments(pair_id))
                    if parts[3] == "keyframes":
                        return self.json_response(store.list_keyframes(pair_id))
                    if parts[3] == "suggestions":
                        return self.json_response(
                            suggest_segments(Path(item["source_dir"]))
                        )
                    if parts[3] == "video":
                        query = parse_qs(parsed.query)
                        role = query.get("role", [""])[0]
                        if role not in {"left", "right"}:
                            raise ValueError("invalid role")
                        video = Path(item["source_dir"]) / "video" / f"{role.title()}.mp4"
                        return self.file_response(video, "video/mp4")
                    if parts[3] == "frame":
                        query = parse_qs(parsed.query)
                        role = query.get("role", [""])[0]
                        index = int(query.get("index", ["0"])[0])
                        if role not in {"left", "right"}:
                            raise ValueError("invalid role")
                        npz = Path(item["source_dir"]) / "rgb_samples.npz"
                        arrays = load_rgb_samples(str(npz), npz.stat().st_mtime_ns)
                        if not 0 <= index < len(arrays[role]):
                            raise ValueError("invalid frame index")
                        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(arrays[role][index], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
                        if not ok:
                            raise RuntimeError("JPEG encoding failed")
                        body = encoded.tobytes()
                        self.send_response(HTTPStatus.OK)
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Cache-Control", "public, max-age=3600")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                        return
                self.json_response({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except (KeyError, ValueError, FileNotFoundError, RuntimeError) as error:
                self.json_response({"error": str(error)}, HTTPStatus.BAD_REQUEST)

        def do_POST(self) -> None:
            path = unquote(urlparse(self.path).path)
            parts = path.strip("/").split("/")
            item_action = (
                len(parts) == 4
                and parts[:2] == ["api", "items"]
                and parts[3] in {"review", "segments", "alignment", "keyframes"}
            )
            segment_action = (
                len(parts) == 4
                and parts[:2] == ["api", "segments"]
                and parts[3] == "review"
            )
            if not item_action and not segment_action:
                return self.json_response({"error": "not found"}, HTTPStatus.NOT_FOUND)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 65536:
                    raise ValueError("request is too large")
                payload = json.loads(self.rfile.read(length))
                if segment_action:
                    result = store.review_segment(
                        int(parts[2]), decision=str(payload.get("decision", "")),
                        reasons=list(payload.get("reasons", [])),
                        notes=str(payload.get("notes", "")),
                        reviewer=str(payload.get("reviewer", "")),
                    )
                elif parts[3] == "alignment":
                    result = store.save_alignment(
                        parts[2],
                        right_time_offset_s=float(payload.get("right_time_offset_s", 0)),
                        reviewer=str(payload.get("reviewer", "")),
                        notes=str(payload.get("notes", "")),
                    )
                elif parts[3] == "review":
                    result = store.add_review(
                        parts[2], decision=str(payload.get("decision", "")),
                        reasons=list(payload.get("reasons", [])),
                        notes=str(payload.get("notes", "")),
                        reviewer=str(payload.get("reviewer", "")),
                    )
                elif parts[3] == "keyframes":
                    result = store.add_keyframe(
                        parts[2], time_sec=float(payload.get("time_sec", -1)),
                        label=str(payload.get("label", "")),
                        reviewer=str(payload.get("reviewer", "")),
                    )
                else:
                    result = store.add_segment(
                        parts[2], start_s=float(payload.get("start_s", -1)),
                        end_s=float(payload.get("end_s", -1)),
                        label=str(payload.get("label", "")),
                        success=bool(payload.get("success", False)),
                        notes=str(payload.get("notes", "")),
                        reviewer=str(payload.get("reviewer", "")),
                    )
                self.json_response(result, HTTPStatus.CREATED)
            except (KeyError, ValueError, json.JSONDecodeError) as error:
                self.json_response({"error": str(error)}, HTTPStatus.BAD_REQUEST)

        def do_DELETE(self) -> None:
            parts = unquote(urlparse(self.path).path).strip("/").split("/")
            if len(parts) != 3 or parts[:2] != ["api", "keyframes"]:
                return self.json_response({"error": "not found"}, HTTPStatus.NOT_FOUND)
            try:
                store.delete_keyframe(parts[2])
                self.json_response({"deleted": True})
            except KeyError as error:
                self.json_response({"error": str(error)}, HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    print(f"UMI_REVIEW_READY {url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
