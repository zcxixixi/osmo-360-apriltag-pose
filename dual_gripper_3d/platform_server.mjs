#!/usr/bin/env node
import fs from 'node:fs';
import {promises as fsp} from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import {randomBytes,timingSafeEqual} from 'node:crypto';
import {fileURLToPath} from 'node:url';

const argv={};
for(let i=2;i<process.argv.length;i+=2)argv[process.argv[i].replace(/^--/,'')]=process.argv[i+1];
for(const key of ['data-dir','mesh-dir'])if(!argv[key])throw new Error(`missing --${key}`);

const root=path.dirname(fileURLToPath(import.meta.url));
const dataDir=path.resolve(argv['data-dir']);
const meshDir=path.resolve(argv['mesh-dir']);
const scene=path.resolve(argv.scene||path.join(root,'single_gripper_scene.html'));
const platform=path.resolve(argv.platform||path.join(root,'platform.html'));
const host=argv.host||'127.0.0.1';
const port=Number(argv.port||7865);
const configuredPublicBaseUrl=argv['public-base-url']?.replace(/\/+$/,'')||null;
let publicBaseUrl=null;
if(configuredPublicBaseUrl){
  const parsed=new URL(configuredPublicBaseUrl);
  if(!['http:','https:'].includes(parsed.protocol)||parsed.username||parsed.password||parsed.pathname!=='/'||parsed.search||parsed.hash)throw new Error('--public-base-url must be an HTTP(S) origin without credentials, path, query, or fragment');
  publicBaseUrl=parsed.origin
}
const MAX_JSON_BYTES=64*1024*1024;
const MAX_VIDEO_BYTES=8*1024*1024*1024;
const MAX_SCENE_BYTES=2*1024*1024;
const MAX_DEVICE_INVENTORY_BYTES=4*1024*1024;
const deviceInventoryPath=path.join(dataDir,'x5_device_inventory.json');
const PROJECT_ID=/^[a-z0-9-]{1,64}$/;
const WRITE_TOKEN_PATTERN=/^[A-Za-z0-9._~-]{43,256}$/;

function loadWriteToken(){
  const inline=String(process.env.OSMO_PLATFORM_WRITE_TOKEN||'').trim();
  const tokenFile=String(process.env.OSMO_PLATFORM_WRITE_TOKEN_FILE||'').trim();
  if(inline&&tokenFile)throw new Error('configure only one of OSMO_PLATFORM_WRITE_TOKEN or OSMO_PLATFORM_WRITE_TOKEN_FILE');
  let token=inline;
  if(tokenFile){
    const stat=fs.statSync(tokenFile);
    if((stat.mode&0o077)!==0)throw new Error('platform write-token file must not be accessible by group or others');
    token=fs.readFileSync(tokenFile,'utf8').trim();
  }
  if(!WRITE_TOKEN_PATTERN.test(token))throw new Error('platform write token is required and must contain at least 256 bits of random URL-safe text');
  return token
}
const writeToken=loadWriteToken();

await fsp.mkdir(dataDir,{recursive:true,mode:0o700});
const dataDirectoryStat=await fsp.lstat(dataDir);
if(!dataDirectoryStat.isDirectory()||dataDirectoryStat.isSymbolicLink()||(process.getuid&&dataDirectoryStat.uid!==process.getuid()))throw new Error('data directory must be a real directory owned by the service user');
await fsp.chmod(dataDir,0o700);
for(const entry of await fsp.readdir(dataDir,{withFileTypes:true})){
  const entryPath=path.join(dataDir,entry.name);
  if(entry.name==='x5_device_inventory.json'){
    const metadata=await fsp.lstat(entryPath);
    if(!metadata.isFile()||metadata.isSymbolicLink()||(process.getuid&&metadata.uid!==process.getuid()))throw new Error('device inventory must be a regular file owned by the service user');
    await fsp.chmod(entryPath,0o600);continue
  }
  if(!PROJECT_ID.test(entry.name))continue;
  const directoryMetadata=await fsp.lstat(entryPath);
  if(!directoryMetadata.isDirectory()||directoryMetadata.isSymbolicLink()||(process.getuid&&directoryMetadata.uid!==process.getuid()))throw new Error(`invalid project directory: ${entry.name}`);
  await fsp.chmod(entryPath,0o700);
  for(const child of await fsp.readdir(entryPath,{withFileTypes:true})){
    const childPath=path.join(entryPath,child.name),childMetadata=await fsp.lstat(childPath);
    if(childMetadata.isSymbolicLink()||(process.getuid&&childMetadata.uid!==process.getuid()))throw new Error(`invalid project asset: ${entry.name}/${child.name}`);
    if(childMetadata.isFile())await fsp.chmod(childPath,0o600)
  }
}

const mime=file=>file.endsWith('.html')?'text/html; charset=utf-8':file.endsWith('.json')?'application/json; charset=utf-8':file.endsWith('.js')?'text/javascript; charset=utf-8':file.endsWith('.stl')?'model/stl':file.endsWith('.mp4')?'video/mp4':'application/octet-stream';
const BASE_SECURITY_HEADERS={'X-Content-Type-Options':'nosniff','Referrer-Policy':'no-referrer','X-Frame-Options':'DENY','Cross-Origin-Resource-Policy':'same-origin','Permissions-Policy':'camera=(), microphone=(), geolocation=()'};
const HTML_SECURITY_HEADERS={...BASE_SECURITY_HEADERS,'Content-Security-Policy':"default-src 'self'; base-uri 'none'; object-src 'none'; frame-ancestors 'none'; form-action 'none'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self' data:; media-src 'self' blob:; connect-src 'self'; worker-src 'self' blob:"};
const sendJson=(response,status,payload,headers={})=>{const body=Buffer.from(JSON.stringify(payload));response.writeHead(status,{...BASE_SECURITY_HEADERS,'Content-Type':'application/json; charset=utf-8','Content-Length':body.length,'Cache-Control':'no-store',...headers});response.end(body)};
function safeAsset(base,relative){
  const resolved=path.resolve(base,relative),prefix=path.resolve(base)+path.sep;
  if(!resolved.startsWith(prefix))throw Object.assign(new Error('invalid asset path'),{status:400});
  return resolved
}
const sendError=(response,status,message)=>sendJson(response,status,{error:message});
function authorizedForWrite(request){
  const header=request.headers.authorization;
  if(typeof header!=='string'||!header.startsWith('Bearer '))return false;
  const presented=Buffer.from(header.slice(7),'utf8'),expected=Buffer.from(writeToken,'utf8');
  return presented.length===expected.length&&timingSafeEqual(presented,expected)
}
function requireWriteAuthorization(request,response){
  if(authorizedForWrite(request))return true;
  sendJson(response,401,{error:'write authorization required'},{'WWW-Authenticate':'Bearer realm="osmo-motion-studio"'});
  return false
}
const requestOrigin=request=>{
  if(publicBaseUrl)return publicBaseUrl;
  const address=String(request.socket.localAddress),hostLiteral=address.includes(':')?`[${address}]`:address;
  return `http://${hostLiteral}:${request.socket.localPort}`
};
const projectResponse=(request,metadata)=>{
  const origin=requestOrigin(request),base=`${origin}/api/projects/${metadata.id}`;
  return {...metadata,view_url:metadata.status==='ready'?`${origin}/view/${metadata.id}/?interactive=1`:null,links:{self:base,timeline_upload:`${base}/timeline`,video_upload:`${base}/video`,scene_upload:`${base}/scene`,publish:`${base}/publish`}}
};

async function readJsonBody(request,maxBytes=65536){
  const chunks=[];let bytes=0;
  for await(const chunk of request){bytes+=chunk.length;if(bytes>maxBytes)throw Object.assign(new Error('request body too large'),{status:413});chunks.push(chunk)}
  try{return JSON.parse(Buffer.concat(chunks).toString('utf8'))}catch{throw Object.assign(new Error('invalid JSON body'),{status:400})}
}

const projectDir=id=>path.join(dataDir,id);
const metadataPath=id=>path.join(projectDir(id),'project.json');
async function readMetadata(id){
  if(!PROJECT_ID.test(id))throw Object.assign(new Error('invalid project id'),{status:400});
  try{
    const directoryStat=await fsp.lstat(projectDir(id));
    if(!directoryStat.isDirectory()||directoryStat.isSymbolicLink()||(process.getuid&&directoryStat.uid!==process.getuid()))throw Object.assign(new Error('invalid project directory'),{status:400});
    const metadataStat=await fsp.lstat(metadataPath(id));
    if(!metadataStat.isFile()||metadataStat.isSymbolicLink()||(process.getuid&&metadataStat.uid!==process.getuid()))throw Object.assign(new Error('invalid project metadata file'),{status:400});
    const metadata=JSON.parse(await fsp.readFile(metadataPath(id),'utf8'));
    if(!metadata||metadata.id!==id||!PROJECT_ID.test(metadata.id)||typeof metadata.name!=='string')throw Object.assign(new Error('invalid project metadata'),{status:400});
    return metadata
  }catch(error){if(error.code==='ENOENT')throw Object.assign(new Error('project not found'),{status:404});throw error}
}
const temporaryPath=destination=>`${destination}.part-${process.pid}-${randomBytes(6).toString('hex')}`;
async function writeFileAtomic(destination,contents){
  const temporary=temporaryPath(destination);
  try{await fsp.writeFile(temporary,contents,{flag:'wx',mode:0o600});await fsp.rename(temporary,destination)}
  catch(error){await fsp.rm(temporary,{force:true});throw error}
}
async function writeMetadata(metadata){
  metadata.updated_at=new Date().toISOString();
  await writeFileAtomic(metadataPath(metadata.id),JSON.stringify(metadata,null,2)+'\n');
}

async function listProjects(){
  const entries=await fsp.readdir(dataDir,{withFileTypes:true});
  const projects=[];
  for(const entry of entries){
    if(!entry.isDirectory()||!PROJECT_ID.test(entry.name))continue;
    try{projects.push(await readMetadata(entry.name))}catch{}
  }
  return projects.sort((a,b)=>String(b.created_at).localeCompare(String(a.created_at)));
}

async function receiveFile(request,destination,maxBytes,validateTemporary=null){
  const declared=Number(request.headers['content-length']);
  if(Number.isFinite(declared)&&declared>maxBytes)throw Object.assign(new Error('file too large'),{status:413});
  const temporary=temporaryPath(destination);
  const output=fs.createWriteStream(temporary,{flags:'wx',mode:0o600});
  let bytes=0;
  try{
    for await(const chunk of request){
      bytes+=chunk.length;
      if(bytes>maxBytes)throw Object.assign(new Error('file too large'),{status:413});
      if(!output.write(chunk))await new Promise(resolve=>output.once('drain',resolve));
    }
    await new Promise((resolve,reject)=>{output.once('error',reject);output.end(resolve)});
    if(bytes===0)throw Object.assign(new Error('empty file'),{status:400});
    if(validateTemporary)await validateTemporary(temporary);
    await fsp.rename(temporary,destination);
    return bytes;
  }catch(error){output.destroy();await fsp.rm(temporary,{force:true});throw error}
}

async function validateMp4File(file){
  const handle=await fsp.open(file,'r');
  try{
    const header=Buffer.alloc(12),{bytesRead}=await handle.read(header,0,header.length,0);
    if(bytesRead<header.length||header.toString('ascii',4,8)!=='ftyp')throw Object.assign(new Error('front-video.mp4 is not an ISO BMFF/MP4 file'),{status:400})
  }finally{await handle.close()}
}

async function validateSceneFile(file){
  const source=await fsp.readFile(file,'utf8');
  if(!source.includes("fetch('timeline.json')")||!source.includes('front-video.mp4'))throw Object.assign(new Error('scene.html is not a processed-bundle renderer'),{status:400})
}

const finiteVector=(value,length)=>Array.isArray(value)&&value.length===length&&value.every(Number.isFinite);
function validateTimeline(timeline){
  if(!timeline||typeof timeline!=='object')throw Object.assign(new Error('timeline must be a JSON object'),{status:400});
  if(typeof timeline.schema_version!=='string')throw Object.assign(new Error('timeline.schema_version is required'),{status:400});
  if(!Number.isFinite(timeline.fps)||timeline.fps<=0||timeline.fps>1000)throw Object.assign(new Error('timeline.fps must be in (0, 1000]'),{status:400});
  if(!Array.isArray(timeline.frames)||timeline.frames.length===0)throw Object.assign(new Error('timeline.frames must be non-empty'),{status:400});
  let previousTime=-Infinity;
  for(let index=0;index<timeline.frames.length;index++){
    const frame=timeline.frames[index];
    if(!frame||!Number.isFinite(frame.t)||frame.t<previousTime)throw Object.assign(new Error(`frame ${index} has invalid/non-monotonic time`),{status:400});
    previousTime=frame.t;
    for(const side of ['left','right']){
      const pose=frame[side];
      if(!pose||!finiteVector(pose.p,3)||!finiteVector(pose.q,4)||!Array.isArray(pose.joints))throw Object.assign(new Error(`frame ${index}.${side} is incomplete`),{status:400});
    }
  }
  return {schema_version:timeline.schema_version,render_mode:timeline.render_mode||'unknown',fps:timeline.fps,frames:timeline.frames.length,duration_s:Number(timeline.duration_s??timeline.frames.at(-1).t)};
}

function validateDeviceInventory(inventory){
  if(!inventory||inventory.schema_version!=='x5-device-inventory/1.0'||typeof inventory.devices!=='object'||Array.isArray(inventory.devices))throw Object.assign(new Error('invalid X5 device inventory'),{status:400});
  for(const [serial,device] of Object.entries(inventory.devices)){
    if(!/^[A-Z0-9]{10,20}$/.test(serial)||!device||device.serial!==serial||typeof device.model!=='string'||typeof device.firmware!=='string')throw Object.assign(new Error(`invalid X5 device entry: ${serial}`),{status:400});
    if(device.assignment!==null&&device.assignment!==undefined){
      if(!['physical_left','physical_right'].includes(device.assignment.role)||![2,3].includes(device.assignment.base_tag_id))throw Object.assign(new Error(`invalid X5 assignment: ${serial}`),{status:400})
    }
  }
  return Object.keys(inventory.devices).length
}

async function readDeviceInventory(){
  try{
    const metadata=await fsp.lstat(deviceInventoryPath);
    if(!metadata.isFile()||metadata.isSymbolicLink()||(process.getuid&&metadata.uid!==process.getuid()))throw new Error('device inventory must be a regular file owned by the service user');
    return JSON.parse(await fsp.readFile(deviceInventoryPath,'utf8'))
  }
  catch(error){if(error.code==='ENOENT')return {schema_version:'x5-device-inventory/1.0',sdk_revision_id:null,devices:{}};throw error}
}

function serveFile(request,response,file,cache='no-store'){
  if(!fs.existsSync(file)){sendError(response,404,'not found');return}
  const stat=fs.lstatSync(file),range=request.headers.range;
  if(!stat.isFile()||stat.isSymbolicLink()){sendError(response,404,'not found');return}
  const security=file.endsWith('.html')?HTML_SECURITY_HEADERS:BASE_SECURITY_HEADERS;
  if(range&&file.endsWith('.mp4')){
    const match=/^bytes=(\d+)-(\d*)$/.exec(range);
    if(!match){response.writeHead(416,{'Content-Range':`bytes */${stat.size}`});response.end();return}
    const start=Number(match[1]),end=match[2]?Number(match[2]):stat.size-1;
    if(start<0||end>=stat.size||start>end){response.writeHead(416,{'Content-Range':`bytes */${stat.size}`});response.end();return}
    response.writeHead(206,{...security,'Content-Type':mime(file),'Content-Length':end-start+1,'Content-Range':`bytes ${start}-${end}/${stat.size}`,'Accept-Ranges':'bytes','Cache-Control':cache});
    fs.createReadStream(file,{start,end}).pipe(response);return
  }
  response.writeHead(200,{...security,'Content-Type':mime(file),'Content-Length':stat.size,'Accept-Ranges':'bytes','Cache-Control':cache});
  fs.createReadStream(file).pipe(response)
}

async function handle(request,response){
  let pathname;
  try{pathname=decodeURIComponent(new URL(request.url,'http://localhost').pathname)}
  catch{throw Object.assign(new Error('invalid request URL'),{status:400})}

  if(['POST','PUT','PATCH','DELETE'].includes(request.method)&&!requireWriteAuthorization(request,response))return;

  if(request.method==='GET'&&pathname==='/'){serveFile(request,response,platform);return}
  if(request.method==='GET'&&pathname==='/healthz'){sendJson(response,200,{status:'ok',service:'osmo-motion-studio',api_version:'v1',write_authentication:'bearer'});return}
  if(request.method==='GET'&&pathname==='/api/capabilities'){sendJson(response,200,{api_version:'v1',input_mode:'processed_bundle',write_authentication:{type:'bearer',required:true},required_files:{timeline:{name:'timeline.json',content_type:'application/json',max_bytes:MAX_JSON_BYTES},video:{name:'front-video.mp4',content_type:'video/mp4',max_bytes:MAX_VIDEO_BYTES},scene:{name:'scene.html',content_type:'text/html',max_bytes:MAX_SCENE_BYTES}},upload_sequence:['POST /api/projects','PUT {links.timeline_upload}','PUT {links.video_upload}','PUT {links.scene_upload}','POST {links.publish}'],renderer:{scene:'project-versioned',legacy_fallback:'single_gripper_scene',fixed_mesh_revision:'gripper_v52_new_r1'}});return}
  if(pathname==='/api/devices'&&request.method==='GET'){
    if(!requireWriteAuthorization(request,response))return;
    const inventory=await readDeviceInventory();validateDeviceInventory(inventory);sendJson(response,200,inventory);return
  }
  if(pathname==='/api/devices'&&request.method==='PUT'){
    const inventory=await readJsonBody(request,MAX_DEVICE_INVENTORY_BYTES),count=validateDeviceInventory(inventory);
    await writeFileAtomic(deviceInventoryPath,JSON.stringify(inventory,null,2)+'\n');sendJson(response,200,{status:'saved',count,inventory});return
  }
  if(request.method==='GET'&&pathname==='/api/projects'){sendJson(response,200,{projects:(await listProjects()).map(project=>projectResponse(request,project))});return}
  if(request.method==='POST'&&pathname==='/api/projects'){
    const input=await readJsonBody(request);
    const name=String(input.name||'未命名动画').trim().slice(0,80)||'未命名动画';
    const id=`${new Date().toISOString().replace(/\D/g,'').slice(0,14)}-${randomBytes(3).toString('hex')}`;
    await fsp.mkdir(projectDir(id),{mode:0o700});
    const metadata={id,name,status:'uploading',created_at:new Date().toISOString(),updated_at:new Date().toISOString(),error:null};
    await writeMetadata(metadata);sendJson(response,201,{project:projectResponse(request,metadata)});return
  }

  const projectMatch=/^\/api\/projects\/([^/]+)$/.exec(pathname);
  if(request.method==='GET'&&projectMatch){sendJson(response,200,{project:projectResponse(request,await readMetadata(projectMatch[1]))});return}

  const apiMatch=/^\/api\/projects\/([^/]+)\/(timeline|video|scene|publish)$/.exec(pathname);
  if(apiMatch){
    const [,id,action]=apiMatch,metadata=await readMetadata(id);
    if(action==='timeline'&&request.method==='PUT'){
      const bytes=await receiveFile(request,path.join(projectDir(id),'timeline.json'),MAX_JSON_BYTES);
      metadata.timeline_bytes=bytes;metadata.status='uploading';metadata.error=null;await writeMetadata(metadata);sendJson(response,200,{project:projectResponse(request,metadata),bytes});return
    }
    if(action==='video'&&request.method==='PUT'){
      const bytes=await receiveFile(request,path.join(projectDir(id),'front-video.mp4'),MAX_VIDEO_BYTES,validateMp4File);
      metadata.video_bytes=bytes;metadata.status='uploading';metadata.error=null;await writeMetadata(metadata);sendJson(response,200,{project:projectResponse(request,metadata),bytes});return
    }
    if(action==='scene'&&request.method==='PUT'){
      const destination=path.join(projectDir(id),'scene.html');
      const bytes=await receiveFile(request,destination,MAX_SCENE_BYTES,validateSceneFile);
      metadata.scene_bytes=bytes;metadata.status='uploading';metadata.error=null;await writeMetadata(metadata);sendJson(response,200,{project:projectResponse(request,metadata),bytes});return
    }
    if(action==='publish'&&request.method==='POST'){
      try{
        const timeline=JSON.parse(await fsp.readFile(path.join(projectDir(id),'timeline.json'),'utf8'));
        const summary=validateTimeline(timeline);
        const video=await fsp.stat(path.join(projectDir(id),'front-video.mp4'));
        const projectScene=await fsp.stat(path.join(projectDir(id),'scene.html'));
        if(video.size===0)throw Object.assign(new Error('front-video.mp4 is empty'),{status:400});
        if(projectScene.size===0)throw Object.assign(new Error('scene.html is empty'),{status:400});
        metadata.status='ready';metadata.error=null;metadata.summary=summary;metadata.video_bytes=video.size;metadata.scene_bytes=projectScene.size;await writeMetadata(metadata);
        sendJson(response,200,{project:projectResponse(request,metadata)});return
      }catch(error){
        metadata.status='failed';metadata.error=error.code==='ENOENT'?'timeline.json, front-video.mp4, and scene.html are required':error.message;await writeMetadata(metadata);
        throw Object.assign(new Error(metadata.error),{status:error.status||400})
      }
    }
    sendError(response,405,'method not allowed');return
  }

  const viewMatch=/^\/view\/([^/]+)\/(timeline\.json|front-video\.mp4)?$/.exec(pathname);
  if(request.method==='GET'&&viewMatch){
    const [,id,asset]=viewMatch,metadata=await readMetadata(id);
    if(metadata.status!=='ready'){sendError(response,409,'project is not ready');return}
    const projectScene=path.join(projectDir(id),'scene.html');
    const file=!asset?(fs.existsSync(projectScene)?projectScene:scene):path.join(projectDir(id),asset);
    serveFile(request,response,file);return
  }
  if(request.method==='GET'&&pathname.startsWith('/mesh/')){serveFile(request,response,path.join(meshDir,path.basename(pathname)),'public, max-age=86400');return}
  if(request.method==='GET'&&pathname.startsWith('/three/')){serveFile(request,response,safeAsset(path.join(root,'node_modules/three'),pathname.slice('/three/'.length)),'public, max-age=86400');return}
  sendError(response,404,'not found')
}

const dispatch=(request,response)=>handle(request,response).catch(error=>{if(!error.status)console.error(error);if(!response.headersSent)sendError(response,error.status||500,error.status?error.message:'internal server error');else response.destroy()});
const server=http.createServer(dispatch);
server.on('checkContinue',(request,response)=>{
  if(['POST','PUT','PATCH','DELETE'].includes(request.method)&&!authorizedForWrite(request)){requireWriteAuthorization(request,response);return}
  response.writeContinue();dispatch(request,response)
});
server.on('checkExpectation',(_request,response)=>sendError(response,417,'unsupported expectation'));
server.listen(port,host,()=>console.log(`READY http://${host}:${port}/`));
