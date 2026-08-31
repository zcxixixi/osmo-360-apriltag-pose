from __future__ import annotations

import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


PAGE = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>UMI Pipeline Progress</title><style>
:root{color-scheme:dark;font-family:Inter,system-ui,sans-serif;background:#071018;color:#eaf2f7}*{box-sizing:border-box}body{margin:0}main{max-width:1180px;margin:auto;padding:32px}.eyebrow{color:#63d6ff;font-size:12px;letter-spacing:.18em}.head{display:flex;justify-content:space-between;gap:24px;align-items:end}.head h1{margin:8px 0 3px;font-size:32px}.muted{color:#91a4b5}.overall{min-width:280px}.bar{height:12px;background:#172630;border-radius:20px;overflow:hidden}.fill{height:100%;background:linear-gradient(90deg,#36c9ff,#75e46f);transition:width .4s}.summary{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}.metric,.card{background:#0d1a23;border:1px solid #21323e;border-radius:14px}.metric{padding:16px}.metric b{display:block;font-size:24px;margin-top:4px}.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}.card{padding:18px}.row{display:flex;justify-content:space-between;gap:12px}.name{font-weight:750}.badge{font-size:11px;padding:4px 8px;border-radius:20px;background:#23333f}.running{color:#ffd36a}.completed{color:#70e58b}.failed{color:#ff8181}.pending{color:#91a4b5}.card .bar{margin:14px 0 10px;height:7px}.message{font-size:13px;color:#b5c5d0;min-height:18px}.host{font-size:12px;color:#708695;margin-top:8px}@media(max-width:760px){.grid,.summary{grid-template-columns:1fr}.head{display:block}.overall{margin-top:20px}}
</style></head><body><main><div class="head"><div><div class="eyebrow">DUAL X5 · UMI DATA PIPELINE</div><h1 id="title">Loading…</h1><div class="muted" id="updated"></div></div><div class="overall"><div class="row"><span>Overall</span><b id="percent">0%</b></div><div class="bar"><div class="fill" id="overall" style="width:0"></div></div></div></div><div class="summary"><div class="metric"><span class="muted">Completed</span><b id="completed">0</b></div><div class="metric"><span class="muted">Running</span><b id="running">0</b></div><div class="metric"><span class="muted">Failed</span><b id="failed">0</b></div><div class="metric"><span class="muted">Elapsed</span><b id="elapsed">—</b></div></div><div class="grid" id="stages"></div></main><script>
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function render(s){document.getElementById('title').textContent=s.run_name||'UMI Pipeline';document.getElementById('updated').textContent=`Updated ${s.updated_at||'—'}`;const stages=s.stages||[];const counts=k=>stages.filter(x=>x.state===k).length;const done=counts('completed'),run=counts('running'),fail=counts('failed');const total=stages.length||1;const percent=Math.round(stages.reduce((a,x)=>a+Number(x.progress||0),0)/total);document.getElementById('percent').textContent=`${percent}%`;document.getElementById('overall').style.width=`${percent}%`;document.getElementById('completed').textContent=done;document.getElementById('running').textContent=run;document.getElementById('failed').textContent=fail;document.getElementById('elapsed').textContent=s.elapsed||'—';document.getElementById('stages').innerHTML=stages.map(x=>`<article class="card"><div class="row"><span class="name">${esc(x.name)}</span><span class="badge ${esc(x.state)}">${esc(x.state)}</span></div><div class="bar"><div class="fill" style="width:${Number(x.progress||0)}%"></div></div><div class="message">${esc(x.message||'')}</div><div class="host">${esc(x.host||'local')} · ${Number(x.progress||0)}%</div></article>`).join('')}
async function update(){try{const r=await fetch('/api/status',{cache:'no-store'});render(await r.json())}catch(e){document.getElementById('updated').textContent=e.message}}update();setInterval(update,2000);
</script></body></html>'''


def load_status(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"run_name": path.parent.name, "updated_at": None, "stages": []}
    return json.loads(path.read_text(encoding="utf-8"))


def serve_progress(
    status_path: Path, *, host: str = "127.0.0.1", port: int = 7868,
    open_browser: bool = True,
) -> None:
    status_path = status_path.resolve()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                body = PAGE.encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
            elif path == "/api/status":
                body = json.dumps(load_status(status_path), ensure_ascii=False).encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
            else:
                body = b'{"error":"not found"}'
                self.send_response(HTTPStatus.NOT_FOUND)
                self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    print(f"UMI_PROGRESS_READY {url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
