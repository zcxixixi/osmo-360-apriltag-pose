#!/usr/bin/env node
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const argv={};for(let i=2;i<process.argv.length;i+=2)argv[process.argv[i].replace(/^--/,'')]=process.argv[i+1];
for(const key of ['timeline','mesh-dir'])if(!argv[key])throw new Error(`missing --${key}`);
const root=path.dirname(fileURLToPath(import.meta.url));
const scene=argv.scene?path.resolve(argv.scene):path.join(root,'scene.html');
const timeline=path.resolve(argv.timeline),meshDir=path.resolve(argv['mesh-dir']),frontVideo=argv['front-video']?path.resolve(argv['front-video']):null;
const port=Number(argv.port||7862),host=argv.host||'127.0.0.1';
const mime=file=>file.endsWith('.html')?'text/html; charset=utf-8':file.endsWith('.json')?'application/json':file.endsWith('.js')?'text/javascript':file.endsWith('.stl')?'model/stl':file.endsWith('.mp4')?'video/mp4':'application/octet-stream';
const server=http.createServer((request,response)=>{const urlPath=new URL(request.url,'http://127.0.0.1').pathname;let file;if(urlPath==='/')file=scene;else if(urlPath==='/timeline.json')file=timeline;else if(urlPath==='/front-video.mp4'&&frontVideo)file=frontVideo;else if(urlPath.startsWith('/mesh/'))file=path.join(meshDir,path.basename(urlPath));else if(urlPath.startsWith('/three/'))file=path.join(root,'node_modules/three',urlPath.slice('/three/'.length));else{response.writeHead(404);return response.end('not found')}if(!fs.existsSync(file)){response.writeHead(404);return response.end('missing')}const stat=fs.statSync(file),range=request.headers.range;if(range&&file.endsWith('.mp4')){const [startText,endText]=range.replace(/bytes=/,'').split('-'),start=Number(startText),end=endText?Number(endText):stat.size-1;if(!Number.isFinite(start)||start<0||end>=stat.size||start>end){response.writeHead(416,{'Content-Range':`bytes */${stat.size}`});return response.end()}response.writeHead(206,{'Content-Type':mime(file),'Content-Length':end-start+1,'Content-Range':`bytes ${start}-${end}/${stat.size}`,'Accept-Ranges':'bytes','Cache-Control':'no-store'});return fs.createReadStream(file,{start,end}).pipe(response)}response.writeHead(200,{'Content-Type':mime(file),'Content-Length':stat.size,'Accept-Ranges':'bytes','Cache-Control':'no-store'});fs.createReadStream(file).pipe(response)});
server.listen(port,host,()=>console.log(`READY http://${host}:${port}/?interactive=1`));
