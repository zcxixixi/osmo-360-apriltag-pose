from __future__ import annotations

import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from .devices import DEFAULT_INVENTORY, assign_device, load_inventory, register_devices, scan_devices, sync_inventory
from .manifest import ManifestError


PAGE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>X5 设备管理</title><style>
*{box-sizing:border-box}body{margin:0;background:#071018;color:#edf5fb;font-family:Inter,"Noto Sans SC",system-ui,sans-serif}main{width:min(1100px,calc(100% - 32px));margin:0 auto;padding:34px 0 70px}.eyebrow{font-size:12px;font-weight:800;letter-spacing:.2em;color:#54d7f3}h1{margin:8px 0 6px;font-size:42px}.sub{color:#8fa4b5;line-height:1.65}.toolbar{display:flex;gap:12px;align-items:center;margin:25px 0 20px}.button{height:44px;padding:0 18px;border:0;border-radius:10px;background:#50d3f2;color:#041017;font-weight:900;cursor:pointer}.button.secondary{background:#17394d;color:#8de8fb}.button.save{height:36px;background:#78dc98}.button:disabled{opacity:.45;cursor:wait}.status{flex:1;color:#8fa4b5}.error{display:none;margin-bottom:14px;padding:12px;border:1px solid #8d4550;border-radius:10px;background:#29151a;color:#ff9eaa}.summary{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:16px}.metric,.card{background:#0c1923;border:1px solid #263d4e;border-radius:15px}.metric{padding:16px}.metric span{display:block;color:#7f93a4;font-size:12px}.metric b{display:block;margin-top:7px;font-size:24px}.devices{display:grid;gap:12px}.card{padding:19px}.head{display:flex;justify-content:space-between;gap:16px}.serial{font:800 19px ui-monospace,monospace}.badge{padding:5px 9px;border-radius:999px;background:#173126;color:#7cdd9b;font-size:11px;font-weight:900}.badge.new{background:#3a3016;color:#ffd86f}.meta{margin-top:7px;color:#8fa4b5;font-size:13px}.form{display:grid;grid-template-columns:1fr 150px 1fr auto;gap:10px;margin-top:16px}.form select,.form input{height:36px;border:1px solid #31495b;border-radius:8px;background:#091823;color:#eef6fb;padding:0 10px}.empty{padding:35px;text-align:center;color:#718697;border:1px dashed #31495b;border-radius:14px}@media(max-width:760px){h1{font-size:32px}.summary{grid-template-columns:1fr}.toolbar{align-items:stretch;flex-wrap:wrap}.status{width:100%;flex-basis:100%}.form{grid-template-columns:1fr}.button.save{width:100%}}
</style></head><body><main>
<div class="eyebrow">INSTA360 X5 · FLEET MANAGER</div><h1>X5 设备管理</h1><div class="sub">插入一台或一批 X5 后，点击扫描。登记只保存 serial、型号和固件，不运行视频处理。</div>
<div class="toolbar"><button class="button secondary" id="scan">扫描已连接 X5</button><button class="button" id="register">登记全部</button><button class="button secondary" id="sync">同步到服务器</button><div class="status" id="status">准备就绪</div></div><div class="error" id="error"></div>
<div class="summary"><div class="metric"><span>已登记</span><b id="registered">0</b></div><div class="metric"><span>本次扫描</span><b id="scanned">0</b></div><div class="metric"><span>已分配夹爪</span><b id="assigned">0</b></div></div>
<div class="devices" id="devices"><div class="empty">正在读取设备库存…</div></div>
</main><script>
let inventory={devices:{}},lastScan=[];const root=document.getElementById('devices'),status=document.getElementById('status'),errorBox=document.getElementById('error');
const esc=value=>String(value??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
async function api(path,method='GET',body){const response=await fetch(path,{method,headers:body?{'Content-Type':'application/json'}:{},body:body?JSON.stringify(body):undefined});const data=await response.json();if(!response.ok)throw new Error(data.error||`HTTP ${response.status}`);return data}
function setBusy(value,text){document.querySelectorAll('button').forEach(button=>button.disabled=value);status.textContent=text||'准备就绪'}
function showError(error){errorBox.textContent=error.message;errorBox.style.display='block';status.textContent='操作失败'}
function render(){const combined={...inventory.devices};for(const item of lastScan)if(!combined[item.serial])combined[item.serial]={...item,identity_status:'DISCOVERED_NOT_REGISTERED',assignment:null};const devices=Object.values(combined).sort((a,b)=>a.serial.localeCompare(b.serial));document.getElementById('registered').textContent=Object.keys(inventory.devices).length;document.getElementById('scanned').textContent=lastScan.length;document.getElementById('assigned').textContent=Object.values(inventory.devices).filter(item=>item.assignment).length;if(!devices.length){root.innerHTML='<div class="empty">未发现设备。确认相机为 Android/SDK USB 模式后点击扫描。</div>';return}root.innerHTML=devices.map(item=>{const a=item.assignment||{},registered=!!inventory.devices[item.serial];return `<article class="card"><div class="head"><div><div class="serial">${esc(item.serial)}</div><div class="meta">${esc(item.model)} · 固件 ${esc(item.firmware)}</div></div><span class="badge ${registered?'':'new'}">${registered?'SDK 已登记':'待登记'}</span></div><div class="form"><select data-field="role" data-serial="${esc(item.serial)}"><option value="">选择物理角色</option><option value="physical_left" ${a.role==='physical_left'?'selected':''}>物理左夹爪</option><option value="physical_right" ${a.role==='physical_right'?'selected':''}>物理右夹爪</option></select><select data-field="tag" data-serial="${esc(item.serial)}"><option value="">BaseTag</option><option value="2" ${a.base_tag_id===2?'selected':''}>BaseTag2</option><option value="3" ${a.base_tag_id===3?'selected':''}>BaseTag3</option></select><input data-field="label" data-serial="${esc(item.serial)}" value="${esc(a.label||'')}" placeholder="设备标签"><button class="button save" data-save="${esc(item.serial)}" ${registered?'':'disabled'}>保存分配</button></div></article>`}).join('');document.querySelectorAll('[data-save]').forEach(button=>button.onclick=()=>save(button.dataset.save))}
async function load(){inventory=await api('/api/inventory');render()}
async function scan(){errorBox.style.display='none';setBusy(true,'CameraSDK 正在扫描…');try{const result=await api('/api/scan','POST');lastScan=result.devices;status.textContent=`发现 ${lastScan.length} 台 X5`;render()}catch(error){showError(error)}finally{setBusy(false,status.textContent)}}
async function register(){errorBox.style.display='none';setBusy(true,'正在扫描并登记…');try{inventory=await api('/api/register','POST');lastScan=[];status.textContent='登记完成';render()}catch(error){showError(error)}finally{setBusy(false,status.textContent)}}
async function sync(){errorBox.style.display='none';setBusy(true,'正在上传设备信息…');try{const result=await api('/api/sync','POST');status.textContent=`已同步 ${result.count} 台到服务器`}catch(error){showError(error)}finally{setBusy(false,status.textContent)}}
async function save(serial){const get=field=>document.querySelector(`[data-field="${field}"][data-serial="${serial}"]`).value;const role=get('role'),baseTag=Number(get('tag')),label=get('label');if(!role||![2,3].includes(baseTag)){showError(new Error('请选择物理角色和 BaseTag'));return}setBusy(true,'保存分配…');try{inventory=await api('/api/assign','POST',{serial,role,base_tag_id:baseTag,label});status.textContent='分配已保存';render()}catch(error){showError(error)}finally{setBusy(false,status.textContent)}}
document.getElementById('scan').onclick=scan;document.getElementById('register').onclick=register;document.getElementById('sync').onclick=sync;load().catch(showError);
</script></body></html>'''


class DeviceUIHandler(BaseHTTPRequestHandler):
    server_version = "X5DeviceManager/1.0"

    def _json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 64 * 1024:
            raise ManifestError("request body too large")
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            body = PAGE.encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/inventory":
            self._json(HTTPStatus.OK, load_inventory())
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/api/scan":
                self._json(HTTPStatus.OK, {"devices": scan_devices()})
                return
            if path == "/api/register":
                self._json(HTTPStatus.OK, register_devices(scan_devices()))
                return
            if path == "/api/sync":
                self._json(HTTPStatus.OK, sync_inventory())
                return
            if path == "/api/assign":
                body = self._body()
                self._json(
                    HTTPStatus.OK,
                    assign_device(
                        str(body.get("serial", "")),
                        role=str(body.get("role", "")),
                        base_tag_id=int(body.get("base_tag_id", 0)),
                        label=str(body.get("label", "")).strip() or None,
                    ),
                )
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except (ManifestError, OSError, ValueError, json.JSONDecodeError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

    def log_message(self, format: str, *args: Any) -> None:
        return


def serve_device_ui(
    *, host: str = "127.0.0.1", port: int = 7866, open_browser: bool = True
) -> None:
    server = ThreadingHTTPServer((host, port), DeviceUIHandler)
    url = f"http://{host}:{port}/"
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    print(f"X5_DEVICE_MANAGER_READY {url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
