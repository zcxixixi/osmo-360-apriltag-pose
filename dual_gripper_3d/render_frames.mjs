import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {spawn} from 'node:child_process';
import puppeteer from 'puppeteer-core';

const argv=Object.fromEntries(process.argv.slice(2).map((value,index,array)=>value.startsWith('--')?[value.slice(2),array[index+1]]:null).filter(Boolean));
for(const key of ['timeline','mesh-dir','output','ffmpeg'])if(!argv[key])throw new Error(`missing --${key}`);
const root=path.dirname(fileURLToPath(import.meta.url)),timeline=JSON.parse(fs.readFileSync(argv.timeline,'utf8')),fps=Number(argv.fps||timeline.fps),startFrame=Math.max(0,Math.round(Number(argv['start-frame']||0))),availableFrames=Math.max(0,timeline.frames.length-startFrame),duration=Math.min(Number(argv.duration||timeline.duration_s),availableFrames/fps),frames=Math.min(availableFrames,Math.round(duration*fps));
const mime=file=>file.endsWith('.js')?'text/javascript':file.endsWith('.json')?'application/json':file.endsWith('.STL')?'application/octet-stream':'text/html';
const sceneFile=argv.scene?path.resolve(argv.scene):path.join(root,'scene.html');
const server=http.createServer((request,response)=>{const urlPath=new URL(request.url,'http://127.0.0.1').pathname;let file;if(urlPath==='/')file=sceneFile;else if(urlPath==='/timeline.json')file=argv.timeline;else if(urlPath.startsWith('/mesh/'))file=path.join(argv['mesh-dir'],path.basename(urlPath));else if(urlPath.startsWith('/three/'))file=path.join(root,'node_modules/three',urlPath.slice('/three/'.length));else{response.writeHead(404);return response.end()}try{response.writeHead(200,{'Content-Type':mime(file),'Cache-Control':'no-store'});fs.createReadStream(file).pipe(response)}catch(error){response.writeHead(500);response.end(String(error))}});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));const port=server.address().port;
const browser=await puppeteer.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--no-sandbox','--disable-dev-shm-usage','--use-gl=angle','--use-angle=swiftshader','--enable-webgl']});const page=await browser.newPage();await page.setViewport({width:1920,height:1080,deviceScaleFactor:1});const viewPreset=encodeURIComponent(argv['view-preset']||'operator');await page.goto(`http://127.0.0.1:${port}/?frames=${frames}&view-preset=${viewPreset}`,{waitUntil:'networkidle0'});await page.waitForFunction('window.rendererReady===true');
const ffmpeg=spawn(argv.ffmpeg,['-y','-hide_banner','-loglevel','error','-f','image2pipe','-framerate',String(fps),'-vcodec','png','-i','-','-an','-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',argv.output],{stdio:['pipe','inherit','inherit']});
for(let index=0;index<frames;index++){await page.evaluate(i=>window.renderFrame(i),startFrame+index);const png=await page.screenshot({type:'png'});if(!ffmpeg.stdin.write(png))await new Promise(resolve=>ffmpeg.stdin.once('drain',resolve));if(index%60===0)process.stderr.write(`render ${index}/${frames}\n`)}ffmpeg.stdin.end();const code=await new Promise(resolve=>ffmpeg.on('close',resolve));await browser.close();server.close();if(code!==0)throw new Error(`ffmpeg exited ${code}`);console.log(JSON.stringify({output:path.resolve(argv.output),start_frame:startFrame,frames,fps,duration_s:duration},null,2));
