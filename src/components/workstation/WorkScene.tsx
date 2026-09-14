import {useEffect,useRef,useState} from 'react'
import * as T from 'three'
import {OrbitControls} from 'three/addons/controls/OrbitControls.js'
import {createCrystalTensor,createArrowStream,type Position3} from '../mmhvae/crystalPrimitives'
import {disposeGroup} from '../mmhvae/glyphs'
import type {Run,Tensor} from './engine'
import {coords} from './engine'
interface Props {run:Run;scope:string;index:number;progress:number;playing:boolean;reset:number;onScope:(s:string)=>void;onIndex:(i:number)=>void;onProgress:(p:number)=>void}
type Field={tensor:Tensor;crystals:ReturnType<typeof createCrystalTensor>;positions:Position3[];group:T.Group}
/** Last two axes occupy horizontal planes. All preceding coordinates address the slice stack. */
export function tensorPositions(t:Tensor):Position3[]{const w=t.shape.at(-1)!,h=t.shape.length>1?t.shape.at(-2)!:1;return t.values.map((_,i)=>[(i%w-(w-1)/2)*.57,Math.floor(i/(w*h))*.68,(Math.floor(i/w)%h-(h-1)/2)*.57])}
export default function WorkScene(props:Props){const host=useRef<HTMLDivElement>(null),latest=useRef(props);latest.current=props;const [error,setError]=useState(''),[hover,setHover]=useState('')
 useEffect(()=>{
  const el=host.current!;let renderer:T.WebGLRenderer
  try{renderer=new T.WebGLRenderer({antialias:true,powerPreference:'high-performance'})}catch{setError('无法启动 WebGL。仍可使用下方的逐项演算与完整张量表。');return}
  renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.setClearColor('#050c12');el.appendChild(renderer.domElement);renderer.domElement.tabIndex=0;renderer.domElement.setAttribute('aria-label','三维计算空间：点击模块进入，点击输出水晶选择数值，拖动旋转，滚轮缩放')
  const scene=new T.Scene(),camera=new T.PerspectiveCamera(40,1,.05,8000),controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.minDistance=2;controls.maxDistance=4000
  scene.add(new T.AmbientLight('#c6e7f0',1.7));const light=new T.DirectionalLight('#e7f8ff',3.1);light.position.set(20,35,25);scene.add(light);const rim=new T.DirectionalLight('#86bce6',2);rim.position.set(-20,8,-25);scene.add(rim)
  const labels=document.createElement('div');labels.className='ws-labels';el.appendChild(labels)
  let body=new T.Group(),fields:Field[]=[],clicks:T.Object3D[]=[],tags:{el:HTMLButtonElement;point:T.Vector3}[]=[],raf=0,run:Run|undefined,scope='',oldReset=-1,oldIndex=-1,oldProgress=-1,phase=0,last=0,dirty=true,visible=true,center=new T.Vector3(),extent=new T.Vector3(20,20,20)
  let connections:ReturnType<typeof createArrowStream>[]=[],lastNotify=0,products:Field|undefined
  let flow:ReturnType<typeof createArrowStream>|undefined,factorFlow:ReturnType<typeof createArrowStream>|undefined,traceLines:T.LineSegments|undefined,traceKey='',tween:{from:T.Vector3;to:T.Vector3;target:T.Vector3;start:number}|undefined
  scene.add(body)
  const reduced=matchMedia('(prefers-reduced-motion: reduce)')
  const fit=()=>{const direction=new T.Vector3(.45,.55,1).normalize(),distance=Math.max(extent.y,extent.x/camera.aspect,extent.z)*1.65+4,to=center.clone().addScaledVector(direction,distance);if(reduced.matches){camera.position.copy(to);controls.target.copy(center);controls.update()}else tween={from:camera.position.clone(),to,target:controls.target.clone(),start:performance.now()};dirty=true}
  const label=(text:string,point:T.Vector3,select:string)=>{const button=document.createElement('button');button.className='ws-space-label';button.textContent=text;button.onclick=()=>latest.current.onScope(select);labels.appendChild(button);tags.push({el:button,point})}
  const field=(t:Tensor,origin:T.Vector3,select?:string)=>{
   const group=new T.Group();group.position.copy(origin);body.add(group);const positions=tensorPositions(t),crystals=createCrystalTensor(group,t.values.length,.36);crystals.update(t.values,positions,{focus:-1});const f={tensor:t,group,positions,crystals};fields.push(f)
   for(const mesh of [crystals.body]){mesh.userData.tensor=t.id;mesh.userData.scope=select;clicks.push(mesh)}
   const b=new T.Box3().setFromObject(group);label(`${t.name}  [${t.shape.join(' × ')}]`,new T.Vector3(origin.x,b.max.y+1,origin.z),select??latest.current.scope);return b
  }
  const rebuild=()=>{
   const p=latest.current;scene.remove(body);disposeGroup(body);body=new T.Group();scene.add(body);labels.replaceChildren();fields=[];clicks=[];tags=[];flow=undefined;factorFlow=undefined;traceLines=undefined;traceKey='';oldIndex=-1;connections=[];products=undefined
   const groups=[...new Set(p.run.steps.map(s=>s.group))],box=new T.Box3();let y=0
   if(p.scope==='root'){
    for(const [i,name] of groups.entries()){
     const steps=p.run.steps.filter(s=>s.group===name),width=Math.max(4,Math.min(12,steps.length*.6)),height=2.2
     const shell=new T.Mesh(new T.BoxGeometry(width,height,4),new T.MeshPhysicalMaterial({color:'#92cfe5',transparent:true,opacity:.06,roughness:.15,metalness:.2,depthWrite:false}));shell.position.set(0,-i*8,0);shell.userData.scope=`group:${name}`;clicks.push(shell);body.add(shell)
     const edge=new T.LineSegments(new T.EdgesGeometry(shell.geometry),new T.LineBasicMaterial({color:'#80bace',transparent:true,opacity:.6}));edge.position.copy(shell.position);body.add(edge)
     // One address marker per actual operator; the enclosing volume is a collapsible group, not a tensor proxy.
     const markers=createCrystalTensor(body,steps.length,.36),positions:Position3[]=steps.map((_,j)=>[(j-(steps.length-1)/2)*.48,-i*8,0]);markers.update(steps.map(()=>.5),positions,{active:steps.map((_,j)=>j)});for(const mesh of [markers.body]){mesh.userData.scopes=steps.map(s=>s.id);clicks.push(mesh)}
     label(`${name} · ${steps.length} 个真实算子`,new T.Vector3(0,-i*8+2.4,0),`group:${name}`);box.expandByObject(shell)

    }
    const producer=new Map(p.run.steps.map(s=>[s.output.id,s.group])),seen=new Set<string>()
    for(const step of p.run.steps)for(const input of step.inputs){const from=producer.get(input.id),a=groups.indexOf(from??''),b=groups.indexOf(step.group),key=`${a}:${b}`;if(a<0||a===b||seen.has(key))continue;seen.add(key);const arrow=createArrowStream(body,'#a2d4e4',.15,3),side=b-a>1?6:0,points:Position3[]=[[0,-a*8-1.3,0],[side,-(a+b)*4,side?3:0],[0,-b*8+1.3,0]];arrow.update(points,.4);arrow.group.userData.path=points;connections.push(arrow)}
   }else if(p.scope.startsWith('group:')){
    const name=p.scope.slice(6),steps=p.run.steps.filter(s=>s.group===name)
    for(const step of steps){const positions=tensorPositions(step.output),height=Math.max(...positions.map(v=>v[1]))+5,origin=new T.Vector3(0,y,0);const b=field(step.output,origin,step.id);box.union(b);y-=height}
    for(const step of steps)for(const input of step.inputs){const from=fields.find(f=>f.tensor.id===input.id),to=fields.find(f=>f.tensor.id===step.output.id);if(!from||!to)continue;const arrow=createArrowStream(body,'#a9d8e8',.14,2),points:Position3[]=[from.group.position.toArray() as Position3,[3,(from.group.position.y+to.group.position.y)/2,2],to.group.position.toArray() as Position3];arrow.update(points,.4);arrow.group.userData.path=points;connections.push(arrow)}
   }else{
    for(const stream of connections)stream.update(stream.group.userData.path,phase)
   const step=p.run.steps.find(s=>s.id===p.scope)??p.run.steps[0],inputs=[...new Map(step.inputs.map(t=>[t.id,t])).values()];let left=-6
    for(const t of inputs){const pos=tensorPositions(t),width=Math.max(...pos.map(v=>v[0]))-Math.min(...pos.map(v=>v[0]))+3,origin=t.parameter?new T.Vector3(left-width/2,0,0):new T.Vector3(inputs.filter(v=>!v.parameter).indexOf(t)*Math.max(8,width),5,0);const b=field(t,origin);box.union(b);if(t.parameter)left-=width}
    const terms=step.trace(Math.min(p.index,step.output.values.length-1))
    if(terms.length&&terms.every(t=>t.factor!==undefined)){const tensor:Tensor={id:'products',name:'逐项乘积与常量贡献',shape:[1,terms.length],values:terms.map(t=>t.value*t.factor!)};box.union(field(tensor,new T.Vector3(0,-.5,0)));products=fields.at(-1)}
    const outputBounds=field(step.output,new T.Vector3(0,-6,0));box.union(outputBounds)
    const hull=new T.Box3().setFromObject(body),dimensions=hull.getSize(new T.Vector3()).addScalar(3),shell=new T.LineSegments(new T.EdgesGeometry(new T.BoxGeometry(dimensions.x,dimensions.y,dimensions.z)),new T.LineBasicMaterial({color:'#779dac',transparent:true,opacity:.16}));shell.position.copy(hull.getCenter(new T.Vector3()));body.add(shell)
    flow=createArrowStream(body,'#d4f3fc',.14,2);factorFlow=createArrowStream(body,'#f1ca9e',.12,2)
   }
   box.getCenter(center);box.getSize(extent);extent.addScalar(5);body.updateMatrixWorld(true);el.dataset.scope=p.scope;el.dataset.cells=String(fields.reduce((n,f)=>n+f.tensor.values.length,0));fit();dirty=true
  }
  const resize=()=>{const w=el.clientWidth,h=el.clientHeight;if(!w||!h)return;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();fit()};const ro=new ResizeObserver(resize);ro.observe(el);const io=new IntersectionObserver(([e])=>visible=e.isIntersecting);io.observe(el)
  const ray=new T.Raycaster(),pointer=new T.Vector2();let down=[0,0]
  const pick=(event:PointerEvent)=>{const r=renderer.domElement.getBoundingClientRect();pointer.set((event.clientX-r.left)/r.width*2-1,-(event.clientY-r.top)/r.height*2+1);ray.setFromCamera(pointer,camera);return ray.intersectObjects(clicks,false)[0]}
  const pointerDown=(e:PointerEvent)=>{down=[e.clientX,e.clientY];tween=undefined}
  const pointerUp=(e:PointerEvent)=>{if(Math.hypot(e.clientX-down[0],e.clientY-down[1])>6)return;const hit=pick(e);if(!hit)return;const data=hit.object.userData;if(data.scope)latest.current.onScope(data.scope);else if(data.scopes)latest.current.onScope(data.scopes[hit.instanceId??0]);else if(data.tensor){const t=latest.current.run.tensors.find(t=>t.id===data.tensor)??fields.find(f=>f.tensor.id===data.tensor)?.tensor;if(!t)return;const index=hit.instanceId??0;const step=latest.current.run.steps.find(s=>s.id===latest.current.scope);if(step?.output.id===t.id)latest.current.onIndex(index);setHover(`${t.name}[${coords(index,t.shape).join(', ')}] = ${t.values[index]?.toPrecision(7)}`)}}
  renderer.domElement.addEventListener('keydown',event=>{if(event.key==='+'||event.key==='='){camera.position.lerp(controls.target,.15);dirty=true;event.preventDefault()}else if(event.key==='-'){camera.position.sub(controls.target).multiplyScalar(1.18).add(controls.target);dirty=true;event.preventDefault()}else if(event.key==='Home'){fit();event.preventDefault()}})
  renderer.domElement.addEventListener('pointerdown',pointerDown);renderer.domElement.addEventListener('pointerup',pointerUp)
  controls.addEventListener('change',()=>dirty=true)
  const tick=(now:number)=>{raf=requestAnimationFrame(tick);const p=latest.current;if(run!==p.run||scope!==p.scope){run=p.run;scope=p.scope;rebuild()}
   if(p.reset!==oldReset){oldReset=p.reset;fit()}if(p.progress!==oldProgress){oldProgress=p.progress;phase=p.progress;dirty=true}
   const dt=Math.min(.05,(now-last)/1000);last=now;if(!visible||document.hidden)return
   if(p.playing&&!reduced.matches){phase=(phase+dt*.12)%1;dirty=true;if(now-lastNotify>120){lastNotify=now;p.onProgress(phase)}}
   if(tween){const f=Math.min(1,(now-tween.start)/650),k=1-(1-f)**4;camera.position.lerpVectors(tween.from,tween.to,k);controls.target.lerpVectors(tween.target,center,k);if(f===1)tween=undefined;dirty=true}controls.update()
   if(p.index!==oldIndex){oldIndex=p.index;dirty=true}
   if(!dirty)return
   for(const stream of connections)stream.update(stream.group.userData.path,phase)
   const step=p.run.steps.find(s=>s.id===p.scope)
   if(step){const index=Math.min(p.index,step.output.values.length-1),terms=step.trace(index),current=Math.min(terms.length-1,Math.floor(phase*terms.length)),term=terms[current],out=fields.find(f=>f.tensor.id===step.output.id)!,target=out.group.localToWorld(new T.Vector3(...out.positions[index]))
    if(products){products.tensor.values=terms.map(t=>t.value*(t.factor??1));products.crystals.update(products.tensor.values,products.positions,{focus:current,active:Array.from({length:current+1},(_,i)=>i)})}
    for(const f of fields){if(f===products)continue;const active=terms.filter(t=>t.tensor===f.tensor.id&&t.index>=0).map(t=>t.index);active.push(...terms.filter(t=>t.factorTensor===f.tensor.id).map(t=>t.factorIndex!));f.crystals.update(f.tensor.values,f.positions,{active,focus:f===out?index:term?.tensor===f.tensor.id?term.index:term?.factorTensor===f.tensor.id?term.factorIndex:-1})}
    const point=(tensor:string,i:number)=>{const f=fields.find(v=>v.tensor.id===tensor);return f&&i>=0?f.group.localToWorld(new T.Vector3(...f.positions[i])):null}
    const start=term?point(term.tensor,term.index):null;flow!.group.visible=!!start;if(start)flow!.update([start.toArray() as Position3,...products?[products.group.localToWorld(new T.Vector3(...products.positions[current])).toArray() as Position3]:[],target.toArray() as Position3],(phase*Math.max(1,terms.length))%1)
    const factor=term?.factorTensor?point(term.factorTensor,term.factorIndex!):null;factorFlow!.group.visible=!!factor;if(factor)factorFlow!.update([factor.toArray() as Position3,target.toArray() as Position3],(phase*Math.max(1,terms.length))%1)
    const key=`${p.scope}:${index}`;if(traceKey!==key){traceKey=key;if(traceLines){body.remove(traceLines);traceLines.geometry.dispose();(traceLines.material as T.Material).dispose()}const points:T.Vector3[]=[];for(const t of terms){const source=point(t.tensor,t.index);if(source)points.push(source,target.clone())}traceLines=new T.LineSegments(new T.BufferGeometry().setFromPoints(points),new T.LineBasicMaterial({color:'#80b7c9',transparent:true,opacity:.12}));body.add(traceLines)}
    el.dataset.term=String(current);el.dataset.output=String(step.output.values[index])
   }
   renderer.render(scene,camera);const used:{x:number;y:number;w:number;h:number}[]=[];for(const tag of tags){const q=tag.point.clone().project(camera),x=(q.x+1)*el.clientWidth/2,y=(1-q.y)*el.clientHeight/2,w=tag.el.offsetWidth||180,h=tag.el.offsetHeight||26;let show=q.z<1&&q.z>0&&x>w/2&&x<el.clientWidth-w/2&&y>24&&y<el.clientHeight-24;if(used.some(r=>Math.abs(x-r.x)<(w+r.w)/2+8&&Math.abs(y-r.y)<(h+r.h)/2+6))show=false;tag.el.style.visibility=show?'visible':'hidden';tag.el.tabIndex=show?0:-1;tag.el.style.left=`${x}px`;tag.el.style.top=`${y}px`;if(show)used.push({x,y,w,h})}dirty=false
  };resize();raf=requestAnimationFrame(tick)
  return()=>{cancelAnimationFrame(raf);ro.disconnect();io.disconnect();controls.dispose();disposeGroup(scene);renderer.dispose();renderer.domElement.remove();labels.remove()}
 },[])
 return <div className="ws-scene" ref={host}>{error&&<p className="ws-scene-error" role="status">{error}</p>}{hover&&<output className="ws-cell-readout">{hover}</output>}</div>
}
