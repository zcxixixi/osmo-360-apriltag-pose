#!/usr/bin/env node
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const argv={};for(let i=2;i<process.argv.length;i+=2)argv[process.argv[i].replace(/^--/,'')]=process.argv[i+1];
for(const key of ['timeline','mesh-dir'])if(!argv[key])throw new Error(`missing --${key}`);
const root=path.dirname(fileURLToPath(import.meta.url));
const scene=path.join(root,'scene.html');
const timeline=path.resolve(argv.timeline),meshDir=path.resolve(argv['mesh-dir']);
const port=Number(argv.port||7862);
const mime=file=>file.endsWith('.html')?'text/html; charset=utf-8':file.endsWith('.json')?'application/json':file.endsWith('.js')?'text/javascript':file.endsWith('.stl')?'model/stl':'application/octet-stream';
const server=http.createServer((request,response)=>{const urlPath=new URL(request.url,'http://127.0.0.1').pathname;let file;if(urlPath==='/')file=scene;else if(urlPath==='/timeline.json')file=timeline;else if(urlPath.startsWith('/mesh/'))file=path.join(meshDir,path.basename(urlPath));else if(urlPath.startsWith('/three/'))file=path.join(root,'node_modules/three',urlPath.slice('/three/'.length));else{response.writeHead(404);return response.end('not found')}if(!fs.existsSync(file)){response.writeHead(404);return response.end('missing')}response.writeHead(200,{'Content-Type':mime(file),'Cache-Control':'no-store'});fs.createReadStream(file).pipe(response)});
server.listen(port,'127.0.0.1',()=>console.log(`READY http://127.0.0.1:${port}/?interactive=1`));
