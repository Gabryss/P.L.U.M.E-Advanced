/* PLUME network inspector. Native WebGL; no CDN, telemetry or runtime downloads. */
(() => {
  'use strict';
  const add=(a,b)=>a.map((v,i)=>v+b[i]), sub=(a,b)=>a.map((v,i)=>v-b[i]);
  const mul=(a,s)=>a.map(v=>v*s), dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
  const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
  const norm=a=>mul(a,1/(Math.hypot(...a)||1));
  function bounds(points) {
    const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
    for(const p of points) for(let i=0;i<3;i++){lo[i]=Math.min(lo[i],p[i]);hi[i]=Math.max(hi[i],p[i]);}
    return {lo,hi,center:mul(add(lo,hi),.5),span:sub(hi,lo)};
  }
  function prepare(network, sharedOrigin=null) {
    const all=network.segments.flatMap(s=>s.xyz), box=bounds(all), origin=sharedOrigin||box.center, a=network.flowAngle*Math.PI/180;
    const transform=p=>{const [x,y,z]=sub(p,origin);return [x*Math.cos(a)+y*Math.sin(a),z,x*Math.sin(a)-y*Math.cos(a)];};
    const segments=network.segments.map((s,i)=>({...s,pick:i+1,points:s.xyz.map(transform),
      length3d:s.xyz.slice(1).reduce((sum,p,j)=>sum+Math.hypot(...sub(p,s.xyz[j])),0)}));
    const positions=new Map();
    for(const s of segments) {positions.set(s.source,s.points[0]);positions.set(s.target,s.points.at(-1));}
    return {...network,segments,origin,nodes:network.nodes.filter(n=>positions.has(n.id)).map(n=>({...n,point:positions.get(n.id)})),
      box:bounds(segments.flatMap(s=>s.points))};
  }
  function lookAt(eye,target) {
    const z=norm(sub(eye,target)),x=norm(cross([0,1,0],z)),y=cross(z,x);
    return [x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-dot(x,eye),-dot(y,eye),-dot(z,eye),1];
  }
  function mm(a,b) {
    const out=Array(16).fill(0);
    for(let c=0;c<4;c++)for(let r=0;r<4;r++)for(let k=0;k<4;k++)out[c*4+r]+=a[k*4+r]*b[c*4+k];
    return out;
  }
  function mv(m,p) {return [0,1,2,3].map(r=>m[r]*p[0]+m[4+r]*p[1]+m[8+r]*p[2]+m[12+r]);}
  const rgb=hex=>[1,3,5].map(i=>parseInt(hex.slice(i,i+2),16)/255);
  const palette=['#4dc7bd','#f5a04e','#a294ed','#e39cc6'];
  function tube(points,radii,color,id,sides=8) {
    const rings=[], normals=[], out=[], pick=[(id&255)/255,((id>>8)&255)/255,((id>>16)&255)/255];
    for(let i=0;i<points.length;i++) {
      const tangent=norm(sub(points[Math.min(i+1,points.length-1)],points[Math.max(0,i-1)]));
      const n=norm(cross(tangent,Math.abs(tangent[1])<.9?[0,1,0]:[1,0,0])), b=cross(tangent,n);
      rings[i]=[];normals[i]=[];
      for(let j=0;j<sides;j++) {
        const angle=j*2*Math.PI/sides,v=add(mul(n,Math.cos(angle)),mul(b,Math.sin(angle)));
        rings[i].push(add(points[i],mul(v,radii[i])));normals[i].push(v);
      }
    }
    const vertex=(p,n)=>out.push(...p,...n,...color,...pick);
    for(let i=0;i<points.length-1;i++)for(let j=0;j<sides;j++) {
      const k=(j+1)%sides;
      for(const [row,col] of [[i,j],[i+1,j],[i+1,k],[i,j],[i+1,k],[i,k]])vertex(rings[row][col],normals[row][col]);
    }
    for(const i of [0,points.length-1]) {
      const n=norm(sub(points[i],points[i===0?1:i-1]));
      for(let j=0;j<sides;j++){vertex(points[i],n);vertex(rings[i][j],n);vertex(rings[i][(j+1)%sides],n);}
    }
    return out;
  }
  function visibleSegment(s,layers,ramps) {
    return layers.has(s.layer)&&layers.has(s.endLayer)&&(s.layer===s.endLayer||ramps);
  }
  if(typeof module!=='undefined'&&module.exports) {module.exports={prepare,bounds,lookAt,mm,mv,tube,visibleSegment};return;}
  const $=id=>document.getElementById(id), data=JSON.parse($('network-data').textContent);
  const canvas=$('scene'), overlay=$('labels'), ctx=overlay.getContext('2d');
  const gl=canvas.getContext('webgl',{antialias:true,alpha:false,preserveDrawingBuffer:true});
  function failure(message){$('error').hidden=false;$('error').textContent=message;}
  if(!gl){failure('3D graphics are unavailable in this browser. Enable hardware acceleration or open this page in another WebGL-capable browser.');return;}
  const vs=`attribute vec3 aPosition,aNormal,aColor,aPick;uniform mat4 uMatrix;uniform float uHeight;varying vec3 vNormal,vColor,vPick;void main(){gl_Position=uMatrix*vec4(aPosition*vec3(1.,uHeight,1.),1.);vNormal=aNormal/vec3(1.,uHeight,1.);vColor=aColor;vPick=aPick;}`;
  const fs=`precision highp float;varying vec3 vNormal,vColor,vPick;uniform float uPicking,uUnlit;uniform vec3 uSelected;void main(){if(uPicking>.5){gl_FragColor=vec4(vPick,1.);return;}float light=.48+.42*abs(dot(normalize(vNormal),normalize(vec3(.4,.8,.6))));if(uUnlit>.5)light=1.;vec3 color=vColor*light;if(length(vPick-uSelected)<.001&&length(uSelected)>.001)color=mix(color,vec3(1.,.85,.42),.8);gl_FragColor=vec4(color,1.);}`;
  function shader(type,source){const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;}
  let program;
  try{program=gl.createProgram();gl.attachShader(program,shader(gl.VERTEX_SHADER,vs));gl.attachShader(program,shader(gl.FRAGMENT_SHADER,fs));gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));}
  catch(error){failure(`The 3D renderer could not start: ${error.message}`);return;}
  gl.useProgram(program);gl.enable(gl.DEPTH_TEST);gl.disable(gl.CULL_FACE);gl.disable(gl.DITHER);
  const uniforms={};for(const key of ['Matrix','Height','Picking','Unlit','Selected'])uniforms[key]=gl.getUniformLocation(program,'u'+key);
  const attributes=['Position','Normal','Color','Pick'].map((n,i)=>({location:gl.getAttribLocation(program,'a'+n),offset:i*12}));
  const frame=gl.createFramebuffer(),pickTexture=gl.createTexture(),depth=gl.createRenderbuffer();
  let network,buffer,gridBuffer,vertexCount=0,gridCount=0,gridStep=100,visible=[],layers=new Set(),selected=0;
  let yaw=.30,elevation=.66,target=[0,0,0],scale=400,height=1,matrix,queued=false;
  const displayPoint=p=>[p[0],p[1]*height,p[2]];
  function view(){const dir=[Math.sin(yaw)*Math.cos(elevation),Math.sin(elevation),Math.cos(yaw)*Math.cos(elevation)];return lookAt(add(target,mul(dir,10000)),target);}
  function projection(){const a=canvas.width/canvas.height,s=scale;return [1/(s*a),0,0,0,0,1/s,0,0,0,0,-1/50000,0,0,0,0,1];}
  function fit(points){
    const displayed=points.map(displayPoint), box=bounds(displayed);target=box.center;
    const projected=displayed.map(p=>mv(view(),p)), b=bounds(projected), aspect=canvas.clientWidth/canvas.clientHeight;
    scale=Math.max(b.span[0]/Math.max(aspect,.1),b.span[1],20)*.62;request();
  }
  function bindBuffer(b){gl.bindBuffer(gl.ARRAY_BUFFER,b);for(const a of attributes){gl.enableVertexAttribArray(a.location);gl.vertexAttribPointer(a.location,3,gl.FLOAT,false,48,a.offset);}}
  function upload(old,array){if(old)gl.deleteBuffer(old);const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(array),gl.STATIC_DRAW);return b;}
  function rebuild(){
    visible=network.segments.filter(s=>visibleSegment(s,layers,$('ramps').checked));
    const vertices=[];const radius=Math.max(...network.box.span)*.001;
    for(const s of visible){
      const color=rgb(s.layer!==s.endLayer?'#c7d4de':palette[s.layer%palette.length]);
      const values=tube(s.points,s.widths.map(w=>$('mode').value==='width'?w/2:radius),color,s.pick);
      for(const value of values)vertices.push(value);
    }
    if($('nodes').checked){
      const connected=new Set(visible.flatMap(s=>[s.source,s.target]));
      const directions=[[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]];
      for(const n of network.nodes.filter(n=>connected.has(n.id))) {
        const r=radius*(n.kind==='junction'?2.3:3.2),c=rgb(n.kind==='terminal'?'#f48fa0':n.kind==='entry'?'#b4f1c0':'#e7eef6');
        for(const face of [[0,2,4],[4,2,1],[1,2,5],[5,2,0],[0,4,3],[4,1,3],[1,5,3],[5,0,3]])
          for(const i of face)vertices.push(...add(n.point,mul(directions[i],r)),...directions[i],...c,0,0,0);
      }
    }
    buffer=upload(buffer,vertices);vertexCount=vertices.length/12;
    $('visible').textContent=`${visible.length} / ${network.segments.length} passages visible · ${network.segments.reduce((n,s)=>n+s.points.length,0).toLocaleString()} original samples`;
    for(const option of $('segment').options)option.disabled=Boolean(option.value)&&!visible.some(s=>s.pick===Number(option.value));
    if(selected&&!visible.some(s=>s.pick===selected))select(0);
    request();
  }
  function makeGrid(){
    const b=network.box,extent=Math.max(b.span[0],b.span[2]),power=10**Math.floor(Math.log10(Math.max(extent/8,1)));
    gridStep=[1,2,5,10].map(n=>n*power).find(n=>n>=extent/8)||10*power;
    const x0=Math.floor(b.lo[0]/gridStep)*gridStep,x1=Math.ceil(b.hi[0]/gridStep)*gridStep;
    const z0=Math.floor(b.lo[2]/gridStep)*gridStep,z1=Math.ceil(b.hi[2]/gridStep)*gridStep,y=b.lo[1]-10;
    const vertices=[],c=rgb('#293c4b');
    const line=(a,b)=>{for(const p of [a,b])vertices.push(...p,0,1,0,...c,0,0,0);};
    for(let x=x0;x<=x1+1e-9;x+=gridStep)line([x,y,z0],[x,y,z1]);
    for(let z=z0;z<=z1+1e-9;z+=gridStep)line([x0,y,z],[x1,y,z]);
    gridBuffer=upload(gridBuffer,vertices);gridCount=vertices.length/12;
  }
  function render(picking=false){
    matrix=mm(projection(),view());gl.useProgram(program);gl.uniformMatrix4fv(uniforms.Matrix,false,new Float32Array(matrix));gl.uniform1f(uniforms.Height,height);
    gl.uniform1f(uniforms.Picking,picking?1:0);gl.uniform3fv(uniforms.Selected,[(selected&255)/255,((selected>>8)&255)/255,((selected>>16)&255)/255]);
    gl.clearColor(...(picking?[0,0,0]:[.048,.074,.10]),1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
    if($('grid').checked&&!picking){gl.uniform1f(uniforms.Unlit,1);bindBuffer(gridBuffer);gl.drawArrays(gl.LINES,0,gridCount);}
    gl.uniform1f(uniforms.Unlit,0);bindBuffer(buffer);gl.drawArrays(gl.TRIANGLES,0,vertexCount);
    if(!picking)labels();
  }
  function labels(){
    const w=overlay.width,h=overlay.height,dpr=w/overlay.clientWidth;ctx.clearRect(0,0,w,h);ctx.save();ctx.scale(dpr,dpr);
    const viewMatrix=view(),x=overlay.clientWidth-80,y=overlay.clientHeight-94;
    const axes=[[[1,0,0],'Along','#f1ad79'],[[0,1,0],'Up','#9aceba'],[[0,0,-1],'Lateral','#a89bef']];
    for(const [v,label,c] of axes){const dx=(viewMatrix[0]*v[0]+viewMatrix[4]*v[1]+viewMatrix[8]*v[2])*30,dy=-(viewMatrix[1]*v[0]+viewMatrix[5]*v[1]+viewMatrix[9]*v[2])*30;
      ctx.strokeStyle=c;ctx.fillStyle=c;ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x+dx,y+dy);ctx.stroke();ctx.font='10px system-ui';ctx.fillText(label,x+dx-8,y+dy+(dy>0?15:-7));}
    if(selected){const s=network.segments.find(s=>s.pick===selected);if(s){const point=mv(matrix,displayPoint(s.points[Math.floor(s.points.length/2)])),sx=(point[0]+1)*overlay.clientWidth/2,sy=(1-point[1])*overlay.clientHeight/2;
      ctx.font='12px system-ui';ctx.fillStyle='#ffd899';ctx.fillText(`Passage ${s.id}`,Math.max(8,Math.min(sx+12,overlay.clientWidth-100)),Math.max(25,Math.min(sy-12,overlay.clientHeight-40)));}}
    ctx.restore();
  }
  function request(){if(!queued){queued=true;requestAnimationFrame(()=>{queued=false;if(network)render();});}}
  function resize(){
    const dpr=Math.min(window.devicePixelRatio||1,2),w=Math.max(1,Math.round(canvas.clientWidth*dpr)),h=Math.max(1,Math.round(canvas.clientHeight*dpr));
    canvas.width=overlay.width=w;canvas.height=overlay.height=h;gl.viewport(0,0,w,h);
    gl.bindTexture(gl.TEXTURE_2D,pickTexture);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,w,h,0,gl.RGBA,gl.UNSIGNED_BYTE,null);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
    gl.bindRenderbuffer(gl.RENDERBUFFER,depth);gl.renderbufferStorage(gl.RENDERBUFFER,gl.DEPTH_COMPONENT16,w,h);
    gl.bindFramebuffer(gl.FRAMEBUFFER,frame);gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,pickTexture,0);gl.framebufferRenderbuffer(gl.FRAMEBUFFER,gl.DEPTH_ATTACHMENT,gl.RENDERBUFFER,depth);
    if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE)failure('Your graphics device cannot allocate the inspection buffer. Try a smaller browser window.');
    gl.bindFramebuffer(gl.FRAMEBUFFER,null);request();
  }
  function select(id){
    selected=id;const s=network.segments.find(s=>s.pick===id);$('segment').value=id?String(id):'';$('focus').disabled=!s;
    $('detail').replaceChildren();
    if(s){const dl=document.createElement('dl');
      const rows=[['3D length',`${s.length3d.toFixed(1)} m`],['Estimated width',`${Math.min(...s.widths).toFixed(1)}–${Math.max(...s.widths).toFixed(1)} m`],['Elevation',`${s.xyz[0][2].toFixed(1)} → ${s.xyz.at(-1)[2].toFixed(1)} m`],['Layer',s.layer===s.endLayer?`${s.layer+1}`:`${s.layer+1} → ${s.endLayer+1}`],['Connections',`${s.source} → ${s.target}`],['Passage type',s.role.replaceAll('_',' ')]];
      for(const [a,b] of rows){const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=a;dd.textContent=b;dl.append(dt,dd);}$('detail').append(dl);
    }else{const p=document.createElement('p');p.className='muted';p.textContent='Select a passage to see its length, width and connections.';$('detail').append(p);}request();
  }
  function pick(x,y){
    const rect=canvas.getBoundingClientRect(),px=Math.floor((x-rect.left)*canvas.width/rect.width),py=canvas.height-1-Math.floor((y-rect.top)*canvas.height/rect.height);
    gl.bindFramebuffer(gl.FRAMEBUFFER,frame);render(true);const pixel=new Uint8Array(4);gl.readPixels(px,py,1,1,gl.RGBA,gl.UNSIGNED_BYTE,pixel);gl.bindFramebuffer(gl.FRAMEBUFFER,null);
    select(pixel[0]+256*pixel[1]+65536*pixel[2]);render();
  }
  function load(index){
    const keepView=!!network&&$('keep-view').checked;
    network=prepare(data.networks[index],keepView?network.origin:null);if(!keepView){height=1;$('height').value='1';}heightLabel();selected=0;layers=new Set(network.layers);
    $('view-name').textContent=network.label;$('length').textContent=(network.segments.reduce((n,s)=>n+s.length3d,0)/1000).toFixed(2)+' km';
    $('connections').textContent=network.nodes.filter(n=>n.kind==='junction').length;$('sources').textContent=network.nodes.filter(n=>n.kind==='entry').length;
    $('status').textContent=network.quality===true?'● Network checks passed':network.quality===false?'Network checks did not pass':'Network qualification not recorded';
    $('status').style.color=network.quality===true?'#86cdb1':'#ffb780';
    $('identity').textContent=network.fingerprint?'Fingerprint '+network.fingerprint.slice(0,16):'';
    $('layers').replaceChildren();
    for(const id of network.layers){const label=document.createElement('label');label.className='check';const input=document.createElement('input');input.type='checkbox';input.checked=true;input.setAttribute('aria-label',`Layer ${id+1}`);
      const swatch=document.createElement('span');swatch.className='swatch';swatch.style.background=palette[id%palette.length];label.append(input,swatch,document.createTextNode(`Layer ${id+1}`));$('layers').append(label);
      input.addEventListener('change',()=>{input.checked?layers.add(id):layers.delete(id);rebuild();});}
    $('segment').replaceChildren(new Option('Click a passage in the view',''));
    for(const s of network.segments)$('segment').append(new Option(`Passage ${s.id} · ${s.role.replaceAll('_',' ')}`,s.pick));
    $('ramps').disabled=network.layers.length===1;
    select(0);makeGrid();rebuild();if(!keepView)preset('oblique');location.hash=`network=${index}`;
    $('scale-note').textContent=`Grid spacing: ${gridStep} m. All axes use metres.`;
  }
  function preset(kind){
    if(kind==='oblique'){yaw=.3;elevation=.66;}if(kind==='top'){yaw=0;elevation=Math.PI/2-1e-5;}if(kind==='side'){yaw=0;elevation=0;}
    for(const id of ['oblique','top','side'])$(id).classList.toggle('active',id===kind);
    fit(network.segments.flatMap(s=>s.points));
  }
  function heightLabel(){$('height-label').textContent=height===1?'1× · true scale':`${height}× · exaggerated`;$('height-label').style.color=height===1?'#9aceba':'#ffb780';}
  let drag=null;const pointers=new Map();let pinch=null;
  canvas.addEventListener('pointerdown',e=>{canvas.setPointerCapture(e.pointerId);pointers.set(e.pointerId,[e.clientX,e.clientY]);drag={x:e.clientX,y:e.clientY,startX:e.clientX,startY:e.clientY,button:e.button,moved:false};if(pointers.size===2){const p=[...pointers.values()];pinch=Math.hypot(...sub(p[0],p[1]));}});
  canvas.addEventListener('pointermove',e=>{
    if(!pointers.has(e.pointerId))return;pointers.set(e.pointerId,[e.clientX,e.clientY]);
    if(pointers.size===2){const p=[...pointers.values()],distance=Math.hypot(...sub(p[0],p[1]));if(pinch)scale=Math.max(.5,Math.min(100000,scale*pinch/Math.max(distance,1)));pinch=distance;if(drag)drag.moved=true;request();return;}
    if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.moved||=Math.hypot(e.clientX-drag.startX,e.clientY-drag.startY)>4;
    if(drag.button===2||e.shiftKey){const v=view(),factor=2*scale/canvas.clientHeight;target=add(target,[0,1,2].map(i=>-dx*factor*v[i*4]+dy*factor*v[i*4+1]));}
    else{yaw-=dx*.006;elevation=Math.max(-Math.PI/2+.01,Math.min(Math.PI/2-.01,elevation+dy*.006));}
    drag.x=e.clientX;drag.y=e.clientY;for(const id of ['oblique','top','side'])$(id).classList.remove('active');request();
  });
  canvas.addEventListener('pointerup',e=>{pointers.delete(e.pointerId);if(drag&&!drag.moved&&drag.button===0&&pointers.size===0)pick(e.clientX,e.clientY);drag=null;pinch=null;});
  canvas.addEventListener('pointercancel',e=>{pointers.delete(e.pointerId);drag=null;pinch=null;});
  canvas.addEventListener('contextmenu',e=>e.preventDefault());
  canvas.addEventListener('wheel',e=>{e.preventDefault();scale=Math.max(.5,Math.min(100000,scale*Math.exp(Math.max(-1,Math.min(1,e.deltaY*.001)))));request();},{passive:false});
  canvas.addEventListener('keydown',e=>{if(e.key==='f'||e.key==='Home'){e.preventDefault();fit(network.segments.flatMap(s=>s.points));}if(e.key==='Escape')select(0);if(e.key==='+'||e.key==='-'){scale*=e.key==='+'?.85:1.15;request();}});
  for(const id of ['oblique','top','side'])$(id).onclick=()=>preset(id);
  $('fit').onclick=()=>fit(network.segments.flatMap(s=>s.points));$('focus').onclick=()=>{const s=network.segments.find(s=>s.pick===selected);if(s)fit(s.points);};
  for(const id of ['ramps','nodes','mode'])$(id).onchange=rebuild;
  $('grid').onchange=request;$('height').oninput=()=>{height=Number($('height').value);heightLabel();fit(network.segments.flatMap(s=>s.points));};
  $('segment').onchange=()=>select(Number($('segment').value));
  function download(blob,name){const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
  $('download').onclick=()=>download(new Blob(['<!doctype html>\n'+document.documentElement.outerHTML],{type:'text/html'}),'plume-network-viewer.html');
  $('snapshot').onclick=()=>{render();const c=document.createElement('canvas');c.width=canvas.width;c.height=canvas.height;const dc=c.getContext('2d');dc.drawImage(canvas,0,0);dc.drawImage(overlay,0,0);dc.fillStyle='#edf0f3';dc.font=`${Math.max(14,c.width/75)}px system-ui`;dc.fillText(`PLUME · ${network.label} · vertical scale ${height}×`,24,35);c.toBlob(b=>{if(b)download(b,'plume-network.png');});};
  $('network').replaceChildren(...data.networks.map((n,i)=>new Option(n.label,i)));$('network').onchange=()=>load(Number($('network').value));
  canvas.addEventListener('webglcontextlost',e=>{e.preventDefault();failure('The graphics context was lost. Reload this page to restore the viewer.');});
  new ResizeObserver(resize).observe(canvas);resize();
  const requested=Number(new URLSearchParams(location.hash.slice(1)).get('network'));
  const initial=Number.isInteger(requested)&&requested>=0&&requested<data.networks.length?requested:0;
  $('network').value=String(initial);load(initial);
})();
