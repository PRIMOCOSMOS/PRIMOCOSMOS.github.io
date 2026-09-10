import {useEffect,useRef,useState} from 'react'
import * as T from 'three'
import {OrbitControls} from 'three/addons/controls/OrbitControls.js'
import katex from 'katex'
import {detailLayout} from '../mmhvae/detailLayout'
import {makeGlyph,disposeGroup} from '../mmhvae/glyphs'
import {createArrowStream} from '../mmhvae/crystalPrimitives'
import {makeScientificVisual} from './scientificVisuals'
import type {MathVisual} from '../mmhvae/mathVisuals'
import type {AnatomyGraph,AnatomyPart} from '../mmhvae/anatomy'

interface Props {graph:AnatomyGraph;selected:string;playing:boolean;reduced:boolean;probe:number;progress:number;step:number;focus:boolean;view:'orbit'|'front'|'top';reset:number;zoom:number;onSelect:(id:string)=>void;onStep:(n:number)=>void}
type Label={el:HTMLButtonElement;point:T.Object3D;priority:number;id:string;always?:boolean}
type Node={part:AnatomyPart;group:T.Group;hit:T.Mesh;math?:MathVisual}
const vec=(p:number[])=>new T.Vector3(p[0],p[1],p[2])
const tex=(s:string)=>katex.renderToString(s,{throwOnError:false,trust:false,output:'htmlAndMathml'})

export default function ResearchScene(props:Props){
 const host=useRef<HTMLDivElement>(null),latest=useRef(props);latest.current=props
 const [error,setError]=useState(false),[hover,setHover]=useState('')
 useEffect(()=>{
  const container=host.current!;let renderer:T.WebGLRenderer
  try{renderer=new T.WebGLRenderer({alpha:false,antialias:true,powerPreference:'low-power'})}catch{setError(true);return}
  renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.setClearColor('#060e15');renderer.outputColorSpace=T.SRGBColorSpace
  container.appendChild(renderer.domElement);renderer.domElement.tabIndex=0;renderer.domElement.setAttribute('aria-label','三维模型：拖动旋转、滚轮缩放；点击结构逐层进入。下方目录提供等效键盘操作。')
  const scene=new T.Scene(),camera=new T.PerspectiveCamera(42,1,.1,5000),controls=new OrbitControls(camera,renderer.domElement)
  controls.enableDamping=true;controls.dampingFactor=.12;controls.minDistance=5;controls.maxDistance=1500
  scene.add(new T.AmbientLight('#c8e4f2',1.7));const key=new T.DirectionalLight('#edfbff',2.6);key.position.set(-30,70,50);scene.add(key)
  const rim=new T.DirectionalLight('#84bfdf',1.8);rim.position.set(40,10,-35);scene.add(rim)
  const labelLayer=document.createElement('div');labelLayer.className='rl-label-layer';container.appendChild(labelLayer)
  let group=new T.Group();scene.add(group)
  let labels:Label[]=[],nodes:Node[]=[],flows:{stream:ReturnType<typeof createArrowStream>;curve:T.CatmullRomCurve3}[]=[],ghost:T.Group|undefined
  let graph:AnatomyGraph|null=null,sceneKey='',focusKey='',viewKey='',zoom=1,reset=-1,visible=true,hoverId='',raf=0,dirty=true,phase=0,lastTime=0,lastProgress=-1
  let center=new T.Vector3(),extent=new T.Vector3(30,50,20),anchor=new T.Vector3(),tween:{position:T.Vector3;target:T.Vector3;from:T.Vector3;fromTarget:T.Vector3;start:number}|null=null
  const addLabel=(parent:T.Object3D,title:string,pos:number[],id:string,priority=1,math=false,always=false)=>{
   const point=new T.Object3D();point.position.copy(vec(pos));parent.add(point)
   const el=document.createElement('button');el.type='button';el.className=`rl-scene-label${math?' rl-formula':''}`
   if(math)el.innerHTML=tex(title);else el.textContent=title
   el.onclick=()=>activate(id);el.onpointerenter=()=>{hoverId=id;dirty=true};el.onpointerleave=()=>{hoverId='';dirty=true};el.style.visibility='hidden';labelLayer.appendChild(el);labels.push({el,point,priority,id,always});return el
  }
  const activate=(id:string)=>{const node=nodes.find(n=>n.part.id===id);if(!node)return;if(node.part.math)latest.current.onStep(node.part.math.stage);else if(node.part.child)latest.current.onSelect(node.part.child)}
  const fit=()=>{
   const p=latest.current;let c=center.clone(),size=extent.clone()
   const active=nodes.find(n=>n.part.math?.stage===p.step)
   if(p.focus&&active){c=active.group.getWorldPosition(new T.Vector3());size.set(18,14,14)}
   const dir=p.view==='front'?new T.Vector3(0,.06,1):p.view==='top'?new T.Vector3(.03,1,.06):new T.Vector3(.3,.32,1).normalize()
   const right=new T.Vector3(0,1,0).cross(dir).normalize(),up=dir.clone().cross(right).normalize(),tan=Math.tan(T.MathUtils.degToRad(camera.fov/2));let dist=10
   for(const x of [-1,1])for(const y of [-1,1])for(const z of [-1,1]){const q=new T.Vector3(x*size.x/2,y*size.y/2,z*size.z/2);dist=Math.max(dist,Math.abs(q.dot(up))/(tan*.76)+q.dot(dir),Math.abs(q.dot(right))/(tan*camera.aspect*.83)+q.dot(dir))}
   const position=c.clone().addScaledVector(dir,dist/p.zoom)
   if(p.reduced){camera.position.copy(position);controls.target.copy(c);controls.update();tween=null}else tween={position,target:c,from:camera.position.clone(),fromTarget:controls.target.clone(),start:performance.now()}
   dirty=true
  }
  const build=(g:AnatomyGraph)=>{
   const previous=nodes.find(n=>n.part.child===latest.current.selected),nextAnchor=previous?.group.getWorldPosition(new T.Vector3())??new T.Vector3()
   if(ghost){scene.remove(ghost);disposeGroup(ghost);ghost=undefined}
   if(latest.current.selected!=='root'&&previous){ghost=makeGlyph(previous.part.glyph,'#7795a5');ghost.position.copy(nextAnchor);ghost.traverse(o=>{const m=o as T.Mesh;if(m.material)(Array.isArray(m.material)?m.material:[m.material]).forEach(mat=>{mat.transparent=true;mat.opacity=.055;mat.depthWrite=false})});scene.add(ghost)}
   scene.remove(group);disposeGroup(group);labels.forEach(l=>l.el.remove());labels=[];nodes=[];flows=[];group=new T.Group();scene.add(group)
   anchor=latest.current.selected==='root'||g.layout==='overview'?new T.Vector3():nextAnchor;group.position.copy(anchor)
   const layout=detailLayout(g,container.clientWidth<600),bounds=new T.Box3()
   for(const part of g.parts){
    const position=layout.positions.get(part.id);if(!position)continue
    const item=new T.Group();item.position.copy(vec(position));group.add(item)
    const color=part.role==='input'?'#8cd4ed':part.role==='output'?'#d6e2b1':part.color??'#aacfdc',math=part.math?makeScientificVisual(part.math):undefined
    const glyph=math?.group??makeGlyph(part.glyph,color);glyph.scale.setScalar(math?1:g.layout==='overview'?1.5:1.8);item.add(glyph)
    const hit=new T.Mesh(new T.BoxGeometry(math?12:3,math?7:2.6,math?8:3),new T.MeshBasicMaterial({transparent:true,opacity:0,depthWrite:false}));item.add(hit);hit.userData.id=part.id
    nodes.push({part,group:item,hit,math});bounds.expandByPoint(vec(position).add(new T.Vector3(math?7:3,math?5:2.5,math?5:3)));bounds.expandByPoint(vec(position).sub(new T.Vector3(math?7:3,math?5:2.5,math?5:3)))
    const title=part.role?`${part.role==='input'?'来自':'流向'} · ${part.title}`:part.title
    addLabel(item,title,[0,math?4.7:1.7,0],part.id,part.role?4:math?5:1)
    if(g.layout!=='overview'&&!math)addLabel(item,part.shape,[0,-1.6,0],part.id,0)
    if(math){
     addLabel(item,g.mathematics!.steps[part.math!.stage].formula,[0,-4.3,0],part.id,6,true)
     for(const annotation of math.group.userData.annotations??[])addLabel(item,'\\displaystyle '+annotation.text,annotation.position,part.id,2,true)
    }
   }
   for(const edge of g.edges){
    const a=layout.positions.get(edge.from),b=layout.positions.get(edge.to);if(!a||!b)continue
    const start=vec(a),end=vec(b),direction=end.clone().sub(start).normalize();start.addScaledVector(direction,1.1);end.addScaledVector(direction,-1.1)
    const mid=start.clone().lerp(end,.5);if(edge.residual){mid.x+=5;mid.z+=4}else if(Math.abs(a[0]-b[0])>2)mid.z+=2
    const curve=new T.CatmullRomCurve3([start,mid,end]),stream=createArrowStream(group,edge.residual?'#e4b896':'#93c5d7',g.mathematics?.16:.2,3);stream.update(curve,.3);flows.push({stream,curve})
    if(edge.label&&g.layout!=='overview')addLabel(group,edge.label,[mid.x+1,mid.y,mid.z],edge.to,0)
   }
   if(latest.current.selected!=='root'){
    const cageSize=vec(layout.max).sub(vec(layout.min)),cageCenter=vec(layout.min).add(vec(layout.max)).multiplyScalar(.5)
    const box=new T.BoxGeometry(cageSize.x,cageSize.y,cageSize.z),cage=new T.LineSegments(new T.EdgesGeometry(box),new T.LineBasicMaterial({color:'#75929e',transparent:true,opacity:.2}));box.dispose();cage.position.copy(cageCenter);group.add(cage)
    if(ghost){ghost.position.copy(cageCenter).add(anchor);ghost.scale.set(cageSize.x*.7,cageSize.z*.6,cageSize.y*.5)}
   }
   bounds.getCenter(center).add(anchor);bounds.getSize(extent);extent.addScalar(3);group.updateMatrixWorld(true)
   container.dataset.nodeCount=String(nodes.length);container.dataset.selected=latest.current.selected;container.dataset.math=g.mathematics?.operation??''
   fit()
  }
  const resize=()=>{const w=container.clientWidth,h=container.clientHeight;if(!w||!h)return;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();fit()}
  const observer=new ResizeObserver(resize);observer.observe(container)
  const intersection=new IntersectionObserver(([e])=>{visible=e.isIntersecting;dirty=true});intersection.observe(container)
  const ray=new T.Raycaster(),pointer=new T.Vector2();let down:[number,number]=[0,0]
  const pick=(e:PointerEvent)=>{const rect=renderer.domElement.getBoundingClientRect();pointer.set((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/rect.height*2+1);ray.setFromCamera(pointer,camera);return ray.intersectObjects(nodes.map(n=>n.hit))[0]?.object.userData.id as string|undefined}
  const pointerDown=(e:PointerEvent)=>{down=[e.clientX,e.clientY];tween=null;dirty=true}
  const pointerUp=(e:PointerEvent)=>{if(Math.hypot(e.clientX-down[0],e.clientY-down[1])<7){const id=pick(e);if(id)activate(id)}}
  const pointerMove=(e:PointerEvent)=>{const id=pick(e)??'';if(id!==hoverId){hoverId=id;setHover(nodes.find(n=>n.part.id===id)?.part.title??'');dirty=true}}
  const keydown=(e:KeyboardEvent)=>{if(e.key==='+'||e.key==='='){camera.position.lerp(controls.target,.12);dirty=true;e.preventDefault()}else if(e.key==='-'){camera.position.sub(controls.target).multiplyScalar(1.12).add(controls.target);dirty=true;e.preventDefault()}}
  renderer.domElement.addEventListener('pointerdown',pointerDown);renderer.domElement.addEventListener('pointerup',pointerUp);renderer.domElement.addEventListener('pointermove',pointerMove);renderer.domElement.addEventListener('keydown',keydown)
  controls.addEventListener('change',()=>{dirty=true})
  const projectLabels=()=>{
   const rects:{x:number;y:number;w:number;h:number}[]=[],w=container.clientWidth,h=container.clientHeight
   for(const l of [...labels].sort((a,b)=>(b.id===hoverId?100:b.priority)-(a.id===hoverId?100:a.priority))){
    const p=l.point.getWorldPosition(new T.Vector3()).project(camera),x=(p.x+1)*w/2,y=(1-p.y)*h/2,focusNode=latest.current.focus?nodes.find(n=>n.part.id===l.id):null
    let show=p.z<1&&p.z>0&&x>8&&x<w-8&&y>36&&y<h-24&&(!latest.current.focus||focusNode?.part.math?.stage===latest.current.step)
    const lw=Math.min(l.el.offsetWidth||150,w-20),lh=l.el.offsetHeight||24,r={x:x-lw/2,y:y-lh/2,w:lw,h:lh}
    if(r.x<4||r.x+r.w>w-4)show=false
    if(show&&rects.some(a=>r.x<a.x+a.w+8&&r.x+r.w+8>a.x&&r.y<a.y+a.h+6&&r.y+r.h+6>a.y))show=false
    l.el.style.visibility=show?'visible':'hidden';l.el.tabIndex=show?0:-1;l.el.style.left=`${x}px`;l.el.style.top=`${y}px`;if(show)rects.push(r)
   }
  }
  const tick=(now:number)=>{
   raf=requestAnimationFrame(tick);const p=latest.current
   if(graph!==p.graph||sceneKey!==p.selected){graph=p.graph;sceneKey=p.selected;build(graph);focusKey=''}
   if(focusKey!==`${p.focus}:${p.step}`||viewKey!==p.view||zoom!==p.zoom||reset!==p.reset){focusKey=`${p.focus}:${p.step}`;viewKey=p.view;zoom=p.zoom;reset=p.reset;fit()}
   if(p.progress!==lastProgress){phase=p.progress;lastProgress=p.progress;dirty=true}
   if(!visible||document.hidden){lastTime=now;return}
   const dt=Math.min((now-lastTime)/1000,.06);lastTime=now
   if(p.playing&&!p.reduced){phase=(phase+dt*.16)%1;dirty=true}
   if(tween){const t=Math.min(1,(now-tween.start)/650),k=1-(1-t)**4;camera.position.lerpVectors(tween.from,tween.position,k);controls.target.lerpVectors(tween.fromTarget,tween.target,k);if(t===1)tween=null;dirty=true}
   controls.update()
   if(dirty||p.probe!==Number(container.dataset.probe)){
    for(const n of nodes)n.math?.update({phase,probe:p.probe,temperature:1})
    for(const f of flows)f.stream.update(f.curve,phase)
    container.dataset.probe=String(p.probe);container.dataset.focus=p.focus?String(p.step):'all';container.dataset.phase=phase.toFixed(3)
    renderer.render(scene,camera);projectLabels();dirty=false
   }
  }
  resize();raf=requestAnimationFrame(tick)
  return ()=>{cancelAnimationFrame(raf);observer.disconnect();intersection.disconnect();controls.dispose();disposeGroup(scene);renderer.dispose();renderer.domElement.remove();labelLayer.remove()}
 },[])
 return <div ref={host} className="mm-orbit-host rl-scene" data-testid="research-scene">{error&&<div className="mm-webgl-fallback"><h4>当前设备未能启动三维显示</h4><p>仍可通过下方层级目录阅读完整模型、数学原理与源码；启用浏览器硬件加速后可重新载入。</p></div>}{hover&&<div className="mm-hover-label">{hover}<span>点击进入 · 拖动旋转</span></div>}</div>
}
