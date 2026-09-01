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
[hidden]{display:none!important}
.threed{margin-top:8px;padding:10px 12px;border-radius:10px;background:#fff0c9;color:#6d4300;font-weight:700}.threed.ready{background:#dff5e4;color:#17662b}.threed a{color:inherit}.segment{background:white;border-radius:16px;padding:18px;margin-top:14px}.segment-actions{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}.segment-actions button,.segment-actions select{padding:10px 12px;border:1px solid #cbd7dd;border-radius:9px;background:white}.segment-row{padding:8px;border-top:1px solid #e1e7ea}.actions button:disabled{opacity:.4;cursor:not-allowed}
:root{font-family:Inter,"Noto Sans SC",system-ui,sans-serif;color:#17202a;background:#f3f6f8}*{box-sizing:border-box}body{margin:0}button,input,textarea{font:inherit}.top{height:72px;background:#122431;color:white;display:flex;align-items:center;justify-content:space-between;padding:0 24px}.top h1{font-size:22px;margin:0}.top .hint{color:#b8c7d0}.layout{display:grid;grid-template-columns:340px 1fr;min-height:calc(100vh - 72px)}aside{background:white;border-right:1px solid #dce4e8;padding:16px;overflow:auto}.filters{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px}.filters button{border:1px solid #ccd7dd;background:white;border-radius:20px;padding:7px 11px;cursor:pointer}.filters button.active{background:#173d52;color:white}.item{border:2px solid #e2e9ed;border-radius:14px;padding:13px;margin:8px 0;cursor:pointer}.item.active{border-color:#168bd2;background:#eef8fe}.item h3{margin:0 0 7px;font-size:16px}.row{display:flex;align-items:center;justify-content:space-between;gap:10px}.badge{font-size:12px;padding:4px 9px;border-radius:20px;background:#e8edf0}.approved{background:#dff5e4;color:#17662b}.reprocess{background:#fff0c9;color:#7c4c00}.rejected{background:#fee1e1;color:#8d2020}.pending{background:#e9eef1;color:#53636d}main{padding:20px;overflow:auto}.empty{display:grid;place-items:center;height:60vh;color:#6c7d87}.step{font-size:18px;font-weight:700}.viewer{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:14px 0}.view{background:#0b1115;border-radius:16px;overflow:hidden;min-height:260px;position:relative}.view label{position:absolute;z-index:1;left:12px;top:10px;background:rgba(0,0,0,.65);color:white;padding:5px 10px;border-radius:12px}.view img{width:100%;height:100%;object-fit:contain;display:block}.timebox{background:white;border-radius:14px;padding:14px}.timebox input{width:100%}.chart{width:100%;height:110px;background:#f7fafb;border-radius:9px}.simple{background:white;border-radius:16px;padding:18px;margin-top:14px}.simple h2{margin:0 0 8px;font-size:20px}.reasons{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}.reason{border:2px solid #d8e1e6;background:white;border-radius:10px;padding:10px 13px;cursor:pointer}.reason.selected{border-color:#e18b00;background:#fff5dc}.name{display:grid;grid-template-columns:220px 1fr;gap:10px}.name input,.name textarea{width:100%;border:1px solid #cbd7dd;border-radius:9px;padding:10px}.actions{display:grid;grid-template-columns:1.4fr 1fr 1fr;gap:10px;margin-top:14px}.actions button{border:0;border-radius:13px;padding:17px 12px;font-weight:800;font-size:17px;cursor:pointer}.pass{background:#24a148;color:white}.redo{background:#f2a900;color:#2d2100}.reject{background:#da3b3b;color:white}.message{min-height:24px;margin-top:10px;color:#a33220;font-weight:600}details{margin-top:14px;background:white;border-radius:12px;padding:12px}pre{white-space:pre-wrap;font-size:12px}.history{font-size:13px;border-top:1px solid #e1e7ea;padding:7px 0}@media(max-width:900px){.layout{grid-template-columns:1fr}aside{max-height:300px;border-right:0}.viewer{grid-template-columns:1fr}.name,.actions{grid-template-columns:1fr}}
</style></head><body><header class="top"><div><h1>UMI 数据人工审核</h1><div class="hint">看画面，选结果，自动保存</div></div><div id="summary"></div></header><div class="layout"><aside><div class="filters" id="filters"></div><div id="list"></div></aside><main><div id="empty" class="empty">请从左边选择一条数据</div><div id="detail" hidden><div class="row"><div><div class="step" id="step"></div><div id="auto"></div><div id="threeD" class="threed"></div></div><button onclick="nextItem()">跳过，看下一条</button></div><div class="viewer"><div class="view"><label>左手画面</label><img id="left"></div><div class="view"><label>右手画面</label><img id="right"></div></div><div class="timebox"><div class="row"><b id="time">第 1 秒</b><span>拖动查看整段数据</span></div><input id="slider" type="range" min="0" max="0" value="0"><canvas id="chart" class="chart" width="900" height="110"></canvas></div><section class="segment"><h2>把长视频分成动作步骤</h2><div>拖到动作开始和结束的位置，各点一次。然后选择这一步做了什么。</div><div class="segment-actions"><button onclick="loadSuggestions()">自动找动作</button><button onclick="markStart()">从这里开始</button><button onclick="markEnd()">到这里结束</button><b id="segmentRange">还没有选择时间</b><select id="segmentLabel"></select><label><input id="segmentSuccess" type="checkbox" checked> 这一步成功了</label><button onclick="saveSegment()">保存这一步</button></div><div class="message" id="segmentMessage"></div><div id="segments"></div></section><section class="simple"><h2>这条数据能用吗？</h2><div>如果有问题，先选择原因。可以多选。</div><div class="reasons" id="reasons"></div><div class="name"><input id="reviewer" placeholder="审核人姓名（只填一次）"><textarea id="notes" rows="2" placeholder="补充说明（可不填）"></textarea></div><div class="actions"><button class="pass" id="passButton" onclick="save('approved')">画面正常，通过并看下一条</button><button class="redo" onclick="save('reprocess')">有问题，重新处理</button><button class="reject" onclick="save('rejected')">数据不能使用</button></div><div class="message" id="message"></div></section><details><summary>高级信息（普通审核员不用看）</summary><pre id="advanced"></pre><div id="history"></div></details></div></main></div><script>
let items=[],current=null,filter='all',selectedReasons=new Set(),timeline=null,segStart=null,segEnd=null;const labels={approved:'已通过',reprocess:'需重处理',rejected:'不能使用',reprocessed:'已重处理'};const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
async function api(url,opt){const r=await fetch(url,opt);const j=await r.json();if(!r.ok)throw Error(j.error||'操作失败');return j}function status(i){return i.decision&&!i.stale_review?i.decision:'pending'}
function renderFilters(){const f=[['all','全部'],['pending','待审核'],['approved','已通过'],['reprocess','需重处理'],['rejected','不能使用']];filters.innerHTML=f.map(x=>`<button class="${filter===x[0]?'active':''}" onclick="filter='${x[0]}';renderList()">${x[1]}</button>`).join('')}
function renderList(){renderFilters();const shown=items.filter(i=>filter==='all'||status(i)===filter);list.innerHTML=shown.map(i=>`<div class="item ${current===i.pair_id?'active':''}" onclick="openItem('${i.pair_id}')"><div class="row"><h3>${esc(i.pair_id)}</h3><span class="badge ${status(i)}">${labels[status(i)]||'待审核'}</span></div><div>画面 ${i.metrics.rgb_samples} 秒 · Tag可用 ${(i.metrics.tag_usable_ratio*100).toFixed(0)}%</div></div>`).join('');const done=items.filter(i=>status(i)!=='pending').length;summary.textContent=`已审核 ${done} / ${items.length}`}
async function load(){items=await api('/api/items');renderList();if(items.length)openItem(items.find(i=>status(i)==='pending')?.pair_id||items[0].pair_id)}
async function openItem(id){current=id;selectedReasons.clear();const data=await api(`/api/items/${id}`);const index=items.findIndex(i=>i.pair_id===id);empty.hidden=true;detail.hidden=false;step.textContent=`第 ${index+1} 条，共 ${items.length} 条：${id}`;auto.textContent=data.metrics.auto_status==='pass_candidate'?'系统初检：没有发现明显问题':'系统初检：建议认真检查';threeD.className='threed '+(data.metrics.three_d_ready?'ready':'');threeD.innerHTML=data.metrics.three_d_ready?`3D与视频同步审核已就绪：<a href="${esc(data.metrics.three_d_url)}" target="_blank">打开3D审核画面</a>`:'尚未生成3D与视频同步审核画面，当前数据不能点“通过”';passButton.disabled=!data.metrics.three_d_ready;slider.max=Math.max(0,data.metrics.rgb_samples-1);slider.value=0;reviewer.value=localStorage.reviewer||data.reviewer||'';notes.value=data.notes||'';renderReasons();advanced.textContent=JSON.stringify(data.metrics,null,2);timeline=await api(`/api/items/${id}/timeline`);drawChart();updateFrame();await loadSegments();const h=await api(`/api/items/${id}/history`);history.innerHTML='<h3>审核历史</h3>'+h.map(x=>`<div class="history">${esc(x.created_at)} · ${esc(x.reviewer)} · ${labels[x.decision]||x.decision}<br>${esc(x.notes)}</div>`).join('');renderList()}
function renderReasons(){reasons.innerHTML=Object.entries(window.REASONS||{}).map(([k,v])=>`<button class="reason ${selectedReasons.has(k)?'selected':''}" onclick="toggleReason('${k}')">${v}</button>`).join('')}function toggleReason(k){selectedReasons.has(k)?selectedReasons.delete(k):selectedReasons.add(k);renderReasons()}
function updateFrame(){if(!current)return;const i=slider.value;time.textContent=`第 ${Number(i)+1} 秒`;left.src=`/api/items/${current}/frame?role=left&index=${i}`;right.src=`/api/items/${current}/frame?role=right&index=${i}`}slider.oninput=updateFrame;
async function loadSuggestions(){const rows=await api(`/api/items/${current}/suggestions`);segments.innerHTML=rows.length?'<h3>系统建议的动作片段</h3>'+rows.map((x,i)=>`<button class="reason" onclick="useSuggestion(${x.start_s},${x.end_s})">建议 ${i+1}：${x.start_s.toFixed(1)}–${x.end_s.toFixed(1)}秒</button>`).join(''):'3D轨迹未就绪，暂时不能自动找动作'}function useSuggestion(a,b){segStart=a;segEnd=b;slider.value=Math.floor(a);updateFrame();renderSegmentRange()}function markStart(){segStart=Number(slider.value);renderSegmentRange()}function markEnd(){segEnd=Number(slider.value)+1;renderSegmentRange()}function renderSegmentRange(){segmentRange.textContent=segStart===null||segEnd===null?'还没有选择时间':`${segStart}秒 到 ${segEnd}秒`}async function loadSegments(){const rows=await api(`/api/items/${current}/segments`);segments.innerHTML=rows.length?rows.map(x=>`<div class="segment-row"><b>${x.start_s.toFixed(1)}–${x.end_s.toFixed(1)}秒 · ${esc((window.SEGMENT_LABELS||{})[x.label]||x.label)}</b> · ${labels[x.decision]||'待审核'}<br>${esc(x.notes||'')}<div class="segment-actions"><button onclick="reviewSegment(${x.id},'approved')">这一步通过</button><button onclick="reviewSegment(${x.id},'reprocess')">这一步重处理</button><button onclick="reviewSegment(${x.id},'rejected')">这一步不用</button></div></div>`).join(''):'还没有保存动作步骤'}async function saveSegment(){segmentMessage.textContent='';const name=reviewer.value.trim();if(!name){segmentMessage.textContent='请先填写审核人姓名';return}if(segStart===null||segEnd===null||segEnd<=segStart){segmentMessage.textContent='请先选择正确的开始和结束时间';return}try{await api(`/api/items/${current}/segments`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({start_s:segStart,end_s:segEnd,label:segmentLabel.value,success:segmentSuccess.checked,notes:notes.value,reviewer:name})});segStart=segEnd=null;renderSegmentRange();await loadSegments()}catch(e){segmentMessage.textContent=e.message}}
async function reviewSegment(id,decision){segmentMessage.textContent='';const name=reviewer.value.trim();if(!name){segmentMessage.textContent='请先填写审核人姓名';return}if(decision!=='approved'&&!selectedReasons.size){segmentMessage.textContent='请先选择问题原因';return}try{await api(`/api/segments/${id}/review`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({decision,reasons:[...selectedReasons],notes:notes.value,reviewer:name})});await loadSegments()}catch(e){segmentMessage.textContent=e.message}}function drawChart(){const c=chart,ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);if(!timeline)return;const series=[timeline.tags,timeline.gripper];const colors=['#168bd2','#e18b00'];series.forEach((s,n)=>{if(!s.length)return;const max=Math.max(...s.map(x=>x[1]),1);ctx.strokeStyle=colors[n];ctx.lineWidth=2;ctx.beginPath();s.forEach((p,i)=>{const x=i/(s.length-1||1)*c.width,y=c.height-8-p[1]/max*(c.height-16);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke()})}
async function save(decision){message.textContent='';const name=reviewer.value.trim();if(!name){message.textContent='请先填写审核人姓名';return}if(decision!=='approved'&&!selectedReasons.size){message.textContent='请先选择一个问题原因';return}localStorage.reviewer=name;try{await api(`/api/items/${current}/review`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({decision,reasons:[...selectedReasons],notes:notes.value,reviewer:name})});items=await api('/api/items');renderList();nextItem()}catch(e){message.textContent=e.message}}
function nextItem(){const i=items.findIndex(x=>x.pair_id===current);const pending=items.slice(i+1).find(x=>status(x)==='pending')||items.find(x=>status(x)==='pending');if(pending)openItem(pending.pair_id);else if(items.length)openItem(items[(i+1)%items.length].pair_id)}
(async()=>{const c=await api('/api/config');window.REASONS=c.reasons;window.SEGMENT_LABELS=c.segment_labels;segmentLabel.innerHTML=Object.entries(c.segment_labels).map(([k,v])=>`<option value="${k}">${v}</option>`).join('');await load()})().catch(e=>{empty.textContent=e.message});
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
    with (directory / "gripper_stats.csv").open(newline="", encoding="utf-8") as handle:
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
    open_browser: bool = True,
) -> None:
    store = ReviewStore(dataset_root)
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
                    return self.json_response(store.list_items())
                if path == "/api/reprocess-queue":
                    value = json.loads(store.queue_path.read_text()) if store.queue_path.is_file() else {"items": []}
                    return self.json_response(value)
                parts = path.strip("/").split("/")
                if len(parts) >= 3 and parts[:2] == ["api", "items"]:
                    pair_id = parts[2]
                    item = store.get_item(pair_id)
                    if len(parts) == 3:
                        return self.json_response(item)
                    if parts[3] == "history":
                        return self.json_response(store.history(pair_id))
                    if parts[3] == "timeline":
                        return self.json_response(pair_timeline(Path(item["source_dir"])))
                    if parts[3] == "segments":
                        return self.json_response(store.list_segments(pair_id))
                    if parts[3] == "suggestions":
                        return self.json_response(
                            suggest_segments(Path(item["source_dir"]))
                        )
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
                and parts[3] in {"review", "segments"}
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
                elif parts[3] == "review":
                    result = store.add_review(
                        parts[2], decision=str(payload.get("decision", "")),
                        reasons=list(payload.get("reasons", [])),
                        notes=str(payload.get("notes", "")),
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
