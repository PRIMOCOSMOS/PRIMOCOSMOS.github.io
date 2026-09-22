import {forwardSceneWheel} from './sceneWheel'
import { useEffect, useRef, useState } from 'react'
import * as T from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { LEVELS, MODALITIES, NODE_MAP, type ModelNode } from './model'
import { glyphFor, type AnatomyGraph, type AnatomyPart } from './anatomy'
import { labelHTML, MathLabel } from './MathLabel'
import { detailLayout } from './detailLayout'
import { atomId, moduleName, partName } from './navigation'
import { disposeGroup } from './glyphs'
import { modelTopology, levelY } from './topology'
import { makeMathVisual, type MathVisual } from './mathVisuals'
import { createArrowStream } from './crystalPrimitives'
import {holographicLabel,sizeHolographicLabel} from './holographicLabel'
import {sourceTensor,sourceWindow} from './sourceTensor'
import {moduleAssembly} from '../researchLab/ModuleAssembly'
import {mmhvaeModule} from '../researchLab/sourceArchitecture'

export interface SceneProps {
  windowPage:number;
  labelMode:'hover'|'all'|'none';selected:string; observed:string[]; target:string; level:number; isolate:boolean; spread:number
  playing:boolean; stage:number; temperature:number; view:'orbit'|'front'|'top'; reset:number; zoom:number; focus:number
  reducedMotion:boolean; onNavigate:(id:string)=>void; onSelect:(id:string)=>void
  detailId?:string; detailGraph:AnatomyGraph|null; activePart:string|null; detailMotion:boolean; detailZoom:number; onPart:(id:string)=>void
  mathProbe:number; mathProgress:number;mathFocus:boolean
}
type Visual={assembly:ReturnType<typeof moduleAssembly>;node:ModelNode;group:T.Group;label?:T.Sprite;materials:Map<T.Material,number>;hit:T.Mesh;origin:T.Vector3}
type Flow={from:string;to:string;curve:T.CatmullRomCurve3;line:T.Line;arrow:T.Mesh;stream:ReturnType<typeof createArrowStream>;label?:T.Sprite;l?:number;mod?:string}
type PartVisual={part:AnatomyPart;group:T.Group;dest:T.Vector3;label:T.Sprite;dim:T.Sprite;hit:T.Mesh;mathematics?:MathVisual;readout?:T.Sprite;axes?:T.Sprite[];assembly?:ReturnType<typeof moduleAssembly>;holo:T.Group}
const vector=(p:[number,number,number])=>new T.Vector3(...p)

export default function OrbitScene(props:SceneProps){
 const host=useRef<HTMLDivElement>(null),latest=useRef(props);latest.current=props
 const [error,setError]=useState(false),[hover,setHover]=useState('')
 useEffect(()=>{
  const container=host.current!;let renderer:T.WebGLRenderer
  try{renderer=new T.WebGLRenderer({antialias:true,alpha:true,powerPreference:'low-power'})}catch{setError(true);return}
  renderer.setPixelRatio(Math.min(devicePixelRatio,1.65));renderer.setClearColor('#060e15',1);renderer.outputColorSpace=T.SRGBColorSpace;container.appendChild(renderer.domElement)
  renderer.domElement.tabIndex=0;renderer.domElement.setAttribute('role','img');renderer.domElement.setAttribute('aria-label','MMHVAE 同一三维场景：拖动旋转，点击模块原位展开，方向键旋转，加减号缩放。')
  const scene=new T.Scene(),overview=new T.Group();scene.add(overview)
  scene.add(new T.AmbientLight('#c5e0ec',1.6));const sun=new T.DirectionalLight('#e4f6ff',2.4);sun.position.set(-20,70,50);scene.add(sun)
  const rim=new T.DirectionalLight('#75accc',1.3);rim.position.set(20,20,-40);scene.add(rim)
  const camera=new T.PerspectiveCamera(42,1,.01,5000),controls=new OrbitControls(camera,renderer.domElement)
  controls.enableDamping=true;controls.enableZoom=true;controls.zoomToCursor=true;controls.zoomSpeed=.85;controls.minDistance=.25;controls.maxDistance=1800;controls.screenSpacePanning=true;controls.dampingFactor=.09;controls.minPolarAngle=.18;controls.maxPolarAngle=1.5
  const topology=modelTopology(),visuals=new Map<string,Visual>(),overviewFlows:Flow[]=[],decorations:T.Object3D[]=[]
  const labelLayer=document.createElement('div');labelLayer.className='mm-scene-labels';container.appendChild(labelLayer);forwardSceneWheel(labelLayer,renderer.domElement)
  const labels=new Map<T.Sprite,HTMLSpanElement>()
  const label=(value:string,color:string,_height=.7,_width=6,math=false)=>{
   const element=document.createElement('span');element.className='mm-scene-label';element.style.color=color;element.innerHTML=labelHTML(value,math);element.style.visibility='hidden';labelLayer.appendChild(element)
   const sprite=new T.Sprite(new T.SpriteMaterial({transparent:true,opacity:0,depthWrite:false}));sprite.scale.set(1,.5,1);sprite.userData.caption=value;labels.set(sprite,element);const activate=()=>{if(sprite.userData.part)latest.current.onPart(sprite.userData.part);else if(sprite.userData.route)latest.current.onNavigate(sprite.userData.route);else if(sprite.userData.id)latest.current.onSelect(sprite.userData.id)};element.onclick=activate;element.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();activate()}};return sprite
  }
  const colorFor=(node:ModelNode)=>MODALITIES.find(m=>m.id===node.mod)?.color??(node.kind==='poe'?'#ebdfb9':node.kind==='sample'?'#d8ff45':'#8bc6ec')
  const addFlow=(parent:T.Group,from:string,to:string,a:T.Vector3,b:T.Vector3,color:string,caption?:string,residual=false):Flow=>{
   const dir=b.clone().sub(a).normalize(),start=a.clone().addScaledVector(dir,.64),end=b.clone().addScaledVector(dir,-.65),mid=start.clone().lerp(end,.5)
   if(residual){mid.x+=3.4;mid.z+=2.5}else if(Math.abs(a.x-b.x)>2){mid.z+=.85}
   const curve=new T.CatmullRomCurve3([start,mid,end]),line=new T.Line(new T.BufferGeometry().setFromPoints(curve.getPoints(32)),new T.LineBasicMaterial({color,transparent:true,opacity:.45}))
   const arrow=new T.Mesh(new T.ConeGeometry(.12,.35,8),new T.MeshBasicMaterial({color,transparent:true}));arrow.position.copy(end);arrow.quaternion.setFromUnitVectors(new T.Vector3(0,1,0),curve.getTangent(1).normalize())
   const stream=createArrowStream(parent,'#c6edf4',.12,2);stream.update(curve,.2);parent.add(line,arrow)
   let tag:T.Sprite|undefined;if(caption){tag=label(caption,color,.45,5);tag.position.copy(mid).add(new T.Vector3(.7,.25,0));parent.add(tag)}
   return {from,to,curve,line,arrow,stream,label:tag}
  }
  for(const [id,point] of topology.positions){
   const node=NODE_MAP.get(id)!,color=colorFor(node),group=new T.Group(),assembly=moduleAssembly(mmhvaeModule({id,child:id,title:node.title,glyph:glyphFor(node.kind),shape:node.shape,detail:node.description,sourceName:id},id,MODALITIES.map(m=>m.id),1),color,true,latest.current.windowPage),glyph=assembly.group,large=['encoder','output','input','image'].includes(node.kind)
   glyph.scale.setScalar((large?3.4:2)/assembly.height);glyph.position.y=(assembly.height/2-5)*glyph.scale.y;group.position.copy(vector(point));group.add(glyph)
   const tag=label(['input','image'].includes(node.kind)?`${node.kind==='input'?'INPUT':'OUTPUT'} / ${MODALITIES.find(m=>m.id===node.mod)!.label}`:moduleName(node),color,large?.9:.68,large?6:4.5);tag.position.set(large?2.6:1.65,.25,.4);tag.userData.id=id;group.add(tag)
   const materials=new Map<T.Material,number>();group.traverse(o=>{const m=o as T.Mesh;if(m.material)(Array.isArray(m.material)?m.material:[m.material]).forEach(mat=>materials.set(mat,mat.opacity))})
   const hit=new T.Mesh(new T.BoxGeometry(large?3.4:1.6,large?1.5:1.3,large?3.4:1.6),new T.MeshBasicMaterial({transparent:true,opacity:0,depthWrite:false}));hit.userData.id=id;group.add(hit)
   overview.add(group);visuals.set(id,{assembly,node,group,label:tag,materials,hit,origin:group.position.clone()})
  }
  const overviewLanes=new Map<string,Visual[]>();for(const v of visuals.values()){const key=`${v.origin.x.toFixed(1)}:${v.origin.z.toFixed(1)}`;overviewLanes.set(key,[...overviewLanes.get(key)??[],v])}for(const lane of overviewLanes.values()){lane.sort((a,b)=>b.origin.y-a.origin.y);lane.forEach((v,i)=>{const y=v.origin.y,top=i?(lane[i-1].origin.y+y)/2:y+2,bottom=i<lane.length-1?(y+lane[i+1].origin.y)/2:y-2;v.hit.geometry.dispose();v.hit.geometry=new T.BoxGeometry(4,Math.max(1,top-bottom),4);v.hit.position.y=(top+bottom)/2-y})}
  for(const e of topology.links){const n=NODE_MAP.get(e.from)!;const f=addFlow(overview,e.from,e.to,vector(topology.positions.get(e.from)!),vector(topology.positions.get(e.to)!),e.mod?colorFor(n):'#81b4ca');f.l=e.l;f.mod=e.mod;overviewFlows.push(f)}
  for(const m of MODALITIES){const point=topology.positions.get(`${m.id}-encoder-7`)!,tag=label(`${m.label} 编码塔`,m.color);tag.position.set(point[0],point[1]+5,point[2]);tag.userData.route=`encoder:${m.id}`;tag.userData.priority=4;overview.add(tag);decorations.push(tag)}
  for(const v of LEVELS){const tag=label(`z${v.l} · ${v.l===7?'256-vector':`${v.channels} × ${v.size}²`}`,'#e4ead6');tag.position.set(7,levelY(v.l)-2.2,2);tag.userData.route=`layer-${v.l}`;tag.userData.priority=2;overview.add(tag);decorations.push(tag)}
  for(const [name,route,point] of [['层次潜变量 · 先验 / 融合 / 采样','core',[0,123,0]],['观测输入 · 二维切片','encoders',[-26,3,14]],['四模态独立影像生成器','outputs',[12,-9,14]]] as const){const tag=label(name,'#b9d7e3');tag.position.set(point[0],point[1],point[2]);tag.userData.route=route;tag.userData.priority=5;overview.add(tag);decorations.push(tag)}
  for(const m of MODALITIES){const point=topology.positions.get(`${m.id}-encoder-7`)!,holo=holographicLabel(`${m.label} 编码塔`,'二维切片 · 七级特征',10);holo.position.set(point[0],point[1]+8,point[2]);overview.add(holo);decorations.push(holo)}
  const floor=new T.GridHelper(62,16,'#3c5c6a','#18313e');floor.position.y=-6;overview.add(floor);decorations.push(floor)
  let detail:T.Group|null=null,cage:T.LineSegments|null=null,parts:PartVisual[]=[],detailFlows:Flow[]=[],connectors:Flow[]=[]
  let detailKey:AnatomyGraph|null=null,detailStart=0,anchor=new T.Vector3(),detailExtent=new T.Vector3(),cageExtent=new T.Vector3(),detailCenter=new T.Vector3(),hoverPart='',hoverNode='',hoverModule='',hoverClock=0
  const anchors=new Map<string,T.Vector3>()
  type CameraPose={position:T.Vector3;target:T.Vector3;zoom:number}
  let saved:CameraPose|null=null,tween:{from:CameraPose;to:CameraPose;start:number}|null=null,lastView='',lastReset=-1,lastZoom=1,lastDetailZoom=1
  const pose=():CameraPose=>({position:camera.position.clone(),target:controls.target.clone(),zoom:camera.zoom})
  const transition=(to:CameraPose)=>{tween={from:pose(),to,start:performance.now()};controls.enabled=false}
  const fit=(center:T.Vector3,size:T.Vector3,zoom=1,overviewView=false)=>{
   const direction=new T.Vector3(overviewView?.25:.3,overviewView?1.05:.48,1).normalize(),right=new T.Vector3(0,1,0).cross(direction).normalize(),up=direction.clone().cross(right).normalize(),tan=Math.tan(T.MathUtils.degToRad(camera.fov/2))
   let distance=0
   for(const x of [-1,1])for(const y of [-1,1])for(const z of [-1,1]){const p=new T.Vector3(x*size.x/2,y*size.y/2,z*size.z/2);distance=Math.max(distance,Math.abs(p.dot(up))/(tan*Math.max(.62,(container.clientHeight-150)/container.clientHeight))+p.dot(direction),Math.abs(p.dot(right))/(tan*camera.aspect)+p.dot(direction))}
   const target=center.clone().addScaledVector(up,distance*tan*.09);return {position:target.clone().addScaledVector(direction,distance*1.1),target,zoom}
  }
  const overviewPose=()=>fit(new T.Vector3(0,52,0),new T.Vector3(78,146,40),1,true)
  const initial=overviewPose();camera.position.copy(initial.position);controls.target.copy(initial.target);controls.update()
  const clearDetail=()=>{if(detail){detail.traverse(o=>{const e=labels.get(o as T.Sprite);if(e){e.remove();labels.delete(o as T.Sprite)}});scene.remove(detail);disposeGroup(detail)}detail=null;parts=[];detailFlows=[];connectors=[];cage=null}
  const buildDetail=(graph:AnatomyGraph,id:string)=>{
   let nextAnchor=graph.origin?vector(graph.origin):anchors.get(id)?.clone()
   if(!nextAnchor){const nested=parts.find(p=>p.part.child===id);nextAnchor=nested&&detail?nested.group.position.clone().add(detail.position):id.startsWith('layer-')?new T.Vector3(0,levelY(Number(id.slice(6))),0):visuals.get(id.split('/')[0])?.origin.clone()??visuals.get(`${NODE_MAP.get(id.split('/')[0])?.mod}-output`)?.origin.clone()??new T.Vector3(0,levelY(latest.current.level),0);anchors.set(id,nextAnchor.clone())}
   anchor.copy(nextAnchor);clearDetail();for(const v of visuals.values())v.assembly.clear();hoverPart='';hoverNode='';hoverModule='';detail=new T.Group();detail.position.copy(anchor);scene.add(detail);detailStart=performance.now()
   const layout=detailLayout(graph,container.clientWidth<600)
   for(const p of graph.parts){
    const dest=vector(layout.positions.get(p.id)!)
    const color=MODALITIES.find(m=>m.id===p.mod)?.color??(p.role==='input'?'#91ddf5':p.role==='output'?'#dcff91':p.glyph==='poe'?'#eaddb5':'#9dc9dd')
    const group=new T.Group(),mathematics=p.math?makeMathVisual(p.math):undefined,assembly=!p.math&&!p.role?moduleAssembly(mmhvaeModule(p,id,latest.current.observed,1),color,true,latest.current.windowPage):undefined,glyph=mathematics?.group??assembly?.group??sourceTensor(p.glyph,p.shape,color,latest.current.windowPage);glyph.scale.setScalar(mathematics?1:assembly?(graph.layout==='overview'?3.4:5)/assembly.height:1.7);if(assembly)glyph.position.y=(assembly.height/2-5)*glyph.scale.y;group.add(glyph)
    mathematics?.update({phase:latest.current.mathProgress,probe:latest.current.mathProbe,temperature:latest.current.temperature})
    const holo=holographicLabel(partName(p),p.math?'可核算数值示例':p.shape,p.math?6:4.5);holo.position.set(p.math?4.3:2.5,.9,0);group.add(holo)
    const caption=p.role?`${p.role==='input'?'来自':'流向'} · ${partName(p)}`:partName(p)
    const tag=label(caption,color,.68,p.role?7:5.5);tag.position.set(0,p.math?4.3:1.65,0);tag.userData.baseScale=tag.scale.clone();tag.userData.part=p.id;tag.userData.priority=p.math?5:0;group.add(tag)
    const dim=label(p.math?'':p.shape,'#b3cbd4',.48,5.4,true);dim.position.set(0,.8,0);dim.userData.baseScale=dim.scale.clone();dim.userData.part=p.id;dim.userData.dimension=true;group.add(dim)
    let readout:T.Sprite|undefined,axes:T.Sprite[]|undefined
    if(p.math){readout=label('','#e7d5b0',.5,7,true);readout.position.set(0,-3.7,1);readout.userData.priority=6;group.add(readout);const kind=p.math.kind,names=kind==='activation'?['x','f(x)','C']:kind==='distribution'?['z₁','8p(z₁,z₂)','z₂']:kind==='sampling'?['ε₁','ε₂','ε₃']:kind==='linear'?['输入 n','输出 m','权重 W']:kind==='fusion'?[]:['W','C','H'];axes=names.map((name,i)=>{const a=label(name,'#a2bcc6');a.position.copy(vector([[3.5,-.6,0],[-3.5,2,0],[0,-.6,3.6]][i] as [number,number,number]));group.add(a);return a});for(const item of mathematics?.group.userData.annotations??[]){const a=label('\\displaystyle '+item.text,'#c6e6f0',.5,5,true);a.position.copy(vector(item.position));a.userData.priority=4;group.add(a);axes.push(a)}}
    const hit=new T.Mesh(new T.BoxGeometry(p.math?12:2.6,p.math?5:1.5,p.math?8:2.6),new T.MeshBasicMaterial({transparent:true,opacity:0,depthWrite:false}));hit.userData.part=p.id;group.add(hit);detail.add(group);parts.push({part:p,group,dest,label:tag,dim,hit,mathematics,readout,axes,holo,assembly})
   }
   const lanes=new Map<number,PartVisual[]>();for(const v of parts){const key=Math.round(v.dest.x/6);lanes.set(key,[...lanes.get(key)??[],v])}for(const lane of lanes.values()){lane.sort((a,b)=>b.dest.y-a.dest.y);lane.forEach((v,i)=>{const y=v.dest.y,top=i?(lane[i-1].dest.y+y)/2:y+3,bottom=i<lane.length-1?(y+lane[i+1].dest.y)/2:y-3;v.hit.geometry.dispose();v.hit.geometry=new T.BoxGeometry(v.part.math?12:8,Math.max(1,top-bottom),v.part.math?9:7);v.hit.position.y=(top+bottom)/2-y})}
   const coords=new Map(parts.map(p=>[p.part.id,p.dest]))
   for(const e of graph.edges)detailFlows.push(addFlow(detail,e.from,e.to,coords.get(e.from)!,coords.get(e.to)!,e.residual?'#e8ad7c':graph.parts.find(p=>p.id===e.from)?.role==='input'?'#87cbe6':'#84b0ba',e.label,e.residual))
   const bounds=new T.Box3(vector(layout.min),vector(layout.max));bounds.getSize(cageExtent)
   cage=new T.LineSegments(new T.EdgesGeometry(new T.BoxGeometry(1,1,1)),new T.LineBasicMaterial({color:'#c4dae1',transparent:true,opacity:.2}));cage.position.copy(bounds.getCenter(new T.Vector3()));detail.add(cage)
   // The expanded boundary is the selected module itself. Context stays outside it.
   const frameBounds=bounds.clone()
   for(const v of parts){frameBounds.expandByPoint(v.dest.clone().add(new T.Vector3(9,5,5)));frameBounds.expandByPoint(v.dest.clone().sub(new T.Vector3(9,5,5)));anchors.set(atomId(id,v.part.id),v.dest.clone().add(anchor))}
   frameBounds.getSize(detailExtent);detailCenter.copy(frameBounds.getCenter(new T.Vector3()).add(anchor));transition(fit(detailCenter,detailExtent,latest.current.detailZoom,graph.layout==='overview'))
   renderer.domElement.dataset.layout=graph.layout??'vertical';renderer.domElement.dataset.spine=JSON.stringify(layout.spine.map(key=>({id:key,position:layout.positions.get(key)})));renderer.domElement.dataset.envelope=JSON.stringify({min:layout.min,max:layout.max,ports:parts.filter(v=>v.part.role).map(v=>({id:v.part.id,position:v.dest.toArray()}))})
   renderer.domElement.dataset.anatomy='in-place-3d';renderer.domElement.dataset.depthRange=String(bounds.max.z-bounds.min.z);renderer.domElement.dataset.mathKind=graph.mathematics?.kind??''
  }
  let dirty=true,visible=true,frame=0,lastTime=0,elapsed=0,signature='',mathClock=0,lastProgress=-1,lastMath='',lastMathFocus=''
  const mark=()=>{dirty=true};controls.addEventListener('change',mark)
  let firstSize=true
  const resize=()=>{
   const w=container.clientWidth,h=container.clientHeight;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix()
   if(detail){if(detailKey&&latest.current.detailId){
    buildDetail(detailKey,latest.current.detailId)
    // A responsive rebuild changes the camera fit even if the selected step did not change.
    lastMathFocus='';detailStart=performance.now()-900
   }}else if(firstSize){const next=overviewPose();camera.position.copy(next.position);controls.target.copy(next.target)}
   firstSize=false;dirty=true
  }
  const ro=new ResizeObserver(resize);ro.observe(container);resize()
  const io=new IntersectionObserver(([e])=>{visible=e.isIntersecting;dirty=true},{rootMargin:'80px'});io.observe(container)
  const ray=new T.Raycaster(),pointer=new T.Vector2();let down=[0,0]
  const pick=(e:PointerEvent)=>{const r=renderer.domElement.getBoundingClientRect();pointer.set((e.clientX-r.left)/r.width*2-1,1-(e.clientY-r.top)/r.height*2);ray.setFromCamera(pointer,camera);
   const child=ray.intersectObjects(detail?parts.flatMap(p=>p.assembly?.cells.map(c=>c.hit)??[]):[...visuals.values()].filter(v=>v.group.visible).flatMap(v=>v.assembly.cells.map(c=>c.hit)))[0];
   if(child){const module=child.object.userData.moduleId;if(detail)return {part:parts.find(p=>p.assembly?.cells.some(c=>c.hit===child.object))!.part.id,module};return {id:[...visuals.values()].find(v=>v.assembly.cells.some(c=>c.hit===child.object))!.node.id,module}}
   return ray.intersectObjects(detail?parts.flatMap(p=>[p.hit,...p.label.visible&&p.label.userData.labelShown?[p.label]:[]]):[...visuals.values()].filter(v=>v.group.visible).map(v=>v.hit).concat(decorations.filter(d=>d.visible&&d.userData.route&&d.userData.labelShown) as T.Mesh[]))[0]?.object.userData}
  const pointerdown=(e:PointerEvent)=>{down=[e.clientX,e.clientY]}
  const pointerup=(e:PointerEvent)=>{if(Math.hypot(e.clientX-down[0],e.clientY-down[1])<5){const hit=pick(e);if(hit?.module)latest.current.onNavigate(hit.module);else if(hit?.part)latest.current.onPart(hit.part);else if(hit?.route)latest.current.onNavigate(hit.route);else if(hit?.id)latest.current.onSelect(hit.id)}}
  const move=(e:PointerEvent)=>{if(e.buttons)return;const hit=pick(e);if(hoverPart!==(hit?.part??'')||hoverNode!==(hit?.id??'')||hoverModule!==(hit?.module??'')){hoverClock=0}hoverPart=hit?.part??'';hoverNode=hit?.id??'';hoverModule=hit?.module??'';setHover(hit?.part?partName(parts.find(p=>p.part.id===hit.part)!.part):hit?.id?`${moduleName(NODE_MAP.get(hit.id)!)} · ${sourceWindow(NODE_MAP.get(hit.id)!.shape,glyphFor(NODE_MAP.get(hit.id)!.kind),latest.current.windowPage).label}`:hit?.route?[...labels.keys()].find(v=>v.userData.route===hit.route)?.userData.caption??'':'');renderer.domElement.style.cursor=hit?'pointer':'grab';dirty=true}
  const leave=()=>{hoverPart='';hoverNode='';hoverModule='';setHover('');dirty=true}
  const key=(e:KeyboardEvent)=>{const offset=camera.position.clone().sub(controls.target),s=new T.Spherical().setFromVector3(offset);if(e.key==='ArrowLeft')s.theta-=.12;else if(e.key==='ArrowRight')s.theta+=.12;else if(e.key==='ArrowUp')s.phi=Math.max(.2,s.phi-.1);else if(e.key==='ArrowDown')s.phi=Math.min(1.5,s.phi+.1);else if(['+','='].includes(e.key))camera.zoom=Math.min(100,camera.zoom*1.15);else if(e.key==='-')camera.zoom=Math.max(.05,camera.zoom/1.15);else return;e.preventDefault();camera.position.copy(controls.target).add(new T.Vector3().setFromSpherical(s));camera.updateProjectionMatrix();dirty=true}
  for(const [name,fn] of [['pointerdown',pointerdown],['pointerup',pointerup],['pointermove',move],['pointerleave',leave],['keydown',key]] as const)renderer.domElement.addEventListener(name,fn as EventListener)
  const lost=(e:Event)=>{e.preventDefault();setError(true)};renderer.domElement.addEventListener('webglcontextlost',lost)
  const animate=(now:number)=>{
   frame=requestAnimationFrame(animate);const dt=Math.min((now-lastTime)/1000,.05);lastTime=now;if(!visible||document.hidden)return
   const p=latest.current
   if(p.detailGraph!==detailKey){
    const previous=detailKey;detailKey=p.detailGraph
    if(p.detailGraph&&p.detailId){if(!previous)saved=pose();buildDetail(p.detailGraph,p.detailId);lastDetailZoom=p.detailZoom}
    else{clearDetail();anchors.clear();if(saved)transition(saved);saved=null;delete renderer.domElement.dataset.anatomy;delete renderer.domElement.dataset.depthRange}
    signature='';dirty=true
   }
   if(lastView!==p.view||lastReset!==p.reset){lastView=p.view;lastReset=p.reset;if(!detail){const next=overviewPose();if(p.view==='front')next.position.copy(next.target).add(new T.Vector3(0,0,150));if(p.view==='top')next.position.copy(next.target).add(new T.Vector3(0,150,10));transition(next)}dirty=true}
   if(!detail&&lastZoom!==p.zoom){lastZoom=p.zoom;camera.zoom=p.zoom;camera.updateProjectionMatrix();dirty=true}
   if(detail&&lastDetailZoom!==p.detailZoom){lastDetailZoom=p.detailZoom;camera.zoom=p.detailZoom*(p.mathFocus?1.4:1);camera.updateProjectionMatrix();if(tween)tween.to.zoom=camera.zoom;dirty=true}
   const focusKey=`${p.detailId}/${p.activePart}/${p.mathFocus}`
   if(detailKey?.mathematics&&focusKey!==lastMathFocus){lastMathFocus=focusKey;const stage=parts.find(v=>v.part.id===p.activePart);if(p.mathFocus&&stage)transition(fit(stage.dest.clone().add(anchor),new T.Vector3(13,9,11),p.detailZoom*1.4));else transition(fit(detailCenter,detailExtent,p.detailZoom));dirty=true}
   if(tween){const t=p.reducedMotion?1:Math.min((now-tween.start)/850,1),k=t*t*(3-2*t);camera.position.lerpVectors(tween.from.position,tween.to.position,k);controls.target.lerpVectors(tween.from.target,tween.to.target,k);camera.zoom=T.MathUtils.lerp(tween.from.zoom,tween.to.zoom,k);camera.updateProjectionMatrix();dirty=true;if(t===1){tween=null;controls.enabled=true}}
   const sig=[p.selected,p.observed.join(),p.level,p.isolate,p.spread,p.playing,p.stage,p.activePart,p.detailMotion,p.detailId,p.mathProbe,p.mathProgress,p.temperature,p.labelMode,hoverPart,hoverNode,hoverModule].join('|')
   if(sig!==signature||dirty){signature=sig;dirty=true
    for(const v of visuals.values()){
     const observed=!v.node.mod||p.observed.includes(v.node.mod)||['output','image'].includes(v.node.kind),local=v.node.l===p.level
     v.group.visible=detail?v.node.id===p.detailId?.split('/')[0]:!p.isolate||local||['input','image','output'].includes(v.node.kind)
     for(const [mat,base] of v.materials)mat.opacity=base*(detail?.025:observed?1:.19)
     if(v.label)v.label.visible=p.labelMode!=='none'&&!detail&&(p.labelMode==='all'||v.node.id===hoverNode||['input','image'].includes(v.node.kind))
     if(v.node.mod)v.group.position.x=v.origin.x*(1+p.spread*.12)
    }
    decorations.forEach(d=>d.visible=!detail&&!p.isolate&&(d instanceof T.GridHelper||p.labelMode==='all'||p.labelMode==='hover'&&!!d.userData.route&&!d.userData.holographicLabel))
    for(const f of overviewFlows){const active=(!f.mod||p.observed.includes(f.mod)||f.to.includes('output')||f.to.includes('image'))&&(!p.isolate||f.l===p.level||f.l===0),focused=f.l===p.level
     f.line.visible=!detail&&active&&p.spread===0;(f.line.material as T.LineBasicMaterial).opacity=focused?.65:.12;f.arrow.visible=f.line.visible&&(focused||!f.mod);f.stream.group.visible=f.line.visible&&(focused||p.stage===0&&f.mod!==undefined)
    }
    for(const v of parts){const active=v.part.id===(hoverPart||p.activePart);v.holo.visible=p.labelMode==='hover'&&active;v.label.visible=p.labelMode==='all';v.dim.visible=v.label.visible&&!v.part.math;v.group.scale.setScalar(1);if(v.readout)v.readout.visible=p.labelMode!=='none'&&active;v.axes?.forEach(a=>a.visible=p.labelMode!=='none'&&active&&p.mathFocus)}
    for(const f of [...detailFlows,...connectors]){const active=!p.activePart||f.from===p.activePart||f.to===p.activePart;(f.line.material as T.LineBasicMaterial).opacity=active?.67:.16;f.arrow.visible=true;f.stream.group.visible=active;if(f.label)f.label.visible=p.labelMode==='all'||p.labelMode==='hover'&&!!hoverPart&&[f.from,f.to].includes(hoverPart)}
   }
   if(detail){const t=p.reducedMotion?1:Math.min((now-detailStart)/900,1),k=1-(1-t)**3
    if(t<1||parts.some(v=>!v.group.userData.arrived)){for(const v of parts){v.group.position.copy(v.dest).multiplyScalar(v.part.position?1:k);v.group.userData.arrived=t===1}if(cage)cage.scale.copy(cageExtent).multiplyScalar(.08+.92*k);dirty=true}
   }
   if((detail?p.detailMotion:p.playing)&&!p.reducedMotion){if(!hoverPart&&!hoverNode)elapsed+=dt;dirty=true}
   if((detail?p.detailMotion:p.playing)&&!p.reducedMotion&&(hoverPart||hoverNode))hoverClock+=dt/3.6;
   const assemblyOwners=detail?parts.filter(v=>v.assembly).map(v=>({id:v.part.id,assembly:v.assembly!})): [...visuals.values()].filter(v=>v.group.visible&&(!v.node.mod||p.observed.includes(v.node.mod)||['output','image'].includes(v.node.kind))).map(v=>({id:v.node.id,assembly:v.assembly}));
   const functions=assemblyOwners.flatMap(v=>v.assembly.cells.map(c=>({owner:v.id,cell:c.module.id}))),cursor=elapsed/3.6,automatic=functions[Math.floor(cursor)%Math.max(1,functions.length)],owner=detail?(hoverPart||p.activePart||automatic?.owner):(hoverNode||(!p.playing?p.selected:automatic?.owner));
   if(dirty&&p.labelMode==='hover'){for(const v of parts)v.holo.visible=v.part.id===(hoverPart||p.activePart||owner);if(!detail)for(const v of visuals.values())if(v.label&&v.node.id===owner)v.label.visible=true}
   if(dirty)for(const v of assemblyOwners){if(v.id===owner){const selected=hoverModule||((hoverPart||hoverNode)?v.assembly.cells[Math.floor(hoverClock)%v.assembly.cells.length]?.module.id:v.assembly.cells.find(c=>c.module.id===automatic?.cell)?.module.id)||v.assembly.cells[0]?.module.id;v.assembly.setActive(selected,{phase:hoverPart||hoverNode?hoverClock%1:cursor%1,probe:p.mathProbe,temperature:p.temperature});renderer.domElement.dataset.activeFunction=selected;renderer.domElement.dataset.localPhase=(hoverPart||hoverNode?hoverClock%1:cursor%1).toFixed(4);renderer.domElement.dataset.activeRegion=owner??''}else v.assembly.clear()}
   if(dirty)for(const [i,f] of (detail?[...detailFlows,...connectors]:overviewFlows).entries()){const local=f.from===owner||f.to===owner;f.stream.group.visible=f.line.visible&&local;f.arrow.visible=f.line.visible&&local;if(local&&f.line.visible)f.stream.update(f.curve,((hoverPart||hoverNode?hoverClock:elapsed*.36)+i*.17)%1)}
   if(detailKey?.mathematics){
    if(lastProgress!==p.mathProgress||lastMath!==p.detailId){mathClock=p.mathProgress;lastProgress=p.mathProgress;lastMath=p.detailId??'';dirty=true}
    if(p.detailMotion&&!p.reducedMotion){mathClock=(mathClock+dt*.15)%1;dirty=true}
    if(dirty){parts.forEach(v=>{const active=v.part.id===(hoverPart||p.activePart||parts[0]?.part.id);if(active)v.mathematics?.update({phase:mathClock,probe:p.mathProbe,temperature:p.temperature});v.mathematics?.group.traverse(o=>{if(o.userData.directionalFlow&&!active)o.visible=false});if(v.readout){const text=v.mathematics?.group.userData.readout??'';const element=labels.get(v.readout)!;if(element.dataset.value!==text){element.dataset.value=text;element.innerHTML=labelHTML('\\displaystyle '+text,true)}}});renderer.domElement.dataset.mathPhase=mathClock.toFixed(4);renderer.domElement.dataset.mathProbe=String(p.mathProbe)}
   }
   controls.update();if(dirty){
    // DOM labels retain KaTeX/MathML and fixed text size at every camera distance.
    scene.updateMatrixWorld(true);camera.updateMatrixWorld(true);for(const v of parts)if(v.holo.visible)sizeHolographicLabel(v.holo,camera,container.clientHeight,container.clientWidth<600?145:220)
    const w=container.clientWidth,h=container.clientHeight,occupied:{x:number;y:number;w:number;h:number}[]=[]
    const visibleInScene=(o:T.Object3D)=>{let cursor:T.Object3D|null=o;while(cursor){if(!cursor.visible)return false;if(cursor===scene)return true;cursor=cursor.parent}return false}
    const candidates=[...labels].filter(([tag,element])=>{element.style.visibility='hidden';element.tabIndex=-1;return p.labelMode!=='none'&&visibleInScene(tag)}).sort(([a],[b])=>Number(b.userData.part===(hoverPart||p.activePart))*10+(b.userData.priority??0)-Number(a.userData.part===(hoverPart||p.activePart))*10-(a.userData.priority??0))
    for(const [tag,element] of candidates){
     const world=tag.getWorldPosition(new T.Vector3()),point=world.clone().project(camera);if(point.z<=-1||point.z>=1)continue
     const width=element.offsetWidth,height=element.offsetHeight,x=(point.x+1)*w/2,y=(1-point.y)*h/2
     const shifts=tag.userData.dimension?[0,24,-24,48]:tag.userData.priority?[0,-28,28,-56,56,-84,84]:[0,-24,24]
     let placed=false
     for(const shift of shifts){const rect={x:Math.max(width/2+6,Math.min(w-width/2-6,x)),y:y+shift,w:width+8,h:height+6}
      if(rect.y<90||rect.y>h-36||occupied.some(r=>Math.abs(r.x-rect.x)<(r.w+rect.w)/2&&Math.abs(r.y-rect.y)<(r.h+rect.h)/2))continue
      element.style.transform=`translate(${rect.x}px,${rect.y}px) translate(-50%,-50%)`;element.style.visibility='visible';element.dataset.route=tag.userData.route??'';element.dataset.part=tag.userData.part??'';if(tag.userData.part||tag.userData.route||tag.userData.id){element.setAttribute('role','button');element.tabIndex=0;element.style.pointerEvents='auto'};occupied.push(rect);placed=true;break
     }
     // Match transparent hit geometry to the rendered annotation size.
     const distance=camera.position.distanceTo(world),unit=h/(2*Math.tan(T.MathUtils.degToRad(camera.fov/2))*distance)*camera.zoom
     tag.scale.set(width/unit,height/unit,1);tag.userData.labelShown=placed
    }
    renderer.domElement.dataset.parts=JSON.stringify(parts.map(v=>{const q=v.group.getWorldPosition(new T.Vector3()).project(camera);return {id:v.part.id,child:v.part.child,role:v.part.role,x:(q.x+1)*w/2,y:(1-q.y)*h/2,label:v.label.visible}}))
renderer.domElement.dataset.detailId=p.detailId??'';renderer.domElement.dataset.holograms=String(parts.filter(v=>v.holo.visible).length);renderer.domElement.dataset.nodes=JSON.stringify([...visuals.values()].map(v=>{const q=v.group.getWorldPosition(new T.Vector3()).project(camera);return {id:v.node.id,x:(q.x+1)*w/2,y:(1-q.y)*h/2}}));renderer.domElement.dataset.windowPage=String(p.windowPage);renderer.domElement.dataset.camera=JSON.stringify([...camera.position.toArray(),...controls.target.toArray(),camera.zoom]);renderer.render(scene,camera);dirty=false}
  };frame=requestAnimationFrame(animate)
  return()=>{cancelAnimationFrame(frame);ro.disconnect();io.disconnect();controls.dispose();disposeGroup(scene);renderer.dispose();renderer.domElement.remove();labelLayer.remove();labels.clear()}
 },[props.windowPage])
 return <div className="mm-orbit-host" ref={host}>{error&&<div className="mm-webgl-fallback"><h4>当前浏览器无法显示三维场景</h4><p>下方模块目录与内部算子序列仍可查看完整结构、来源、去向和公式。</p><a href="#mm-module-directory">打开模块目录</a></div>}{hover&&!error&&<div className="mm-hover-label"><MathLabel value={hover}/><span>{props.detailGraph?.mathematics?'数学步骤 · 已选中后可调参数或恢复完整演算':props.detailGraph?'点击进入组件 · 边界外节点跳转到来源 / 去向':'点击原位拆解 · 镜头将移入模块'}</span></div>}</div>
}
