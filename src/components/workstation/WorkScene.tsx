import {useEffect,useRef,useState} from 'react'
import * as T from 'three'
import {OrbitControls} from 'three/addons/controls/OrbitControls.js'
import {createCrystalTensor,createArrowStream,type Position3} from '../mmhvae/crystalPrimitives'
import {disposeGroup} from '../mmhvae/glyphs'
import type {Run,Tensor} from './engine'
import {coords} from './engine'
import {buildScaffold,childSteps,principalSteps,stepLabel,tensorLayout} from './scaffold'
interface Props {run:Run;scope:string;index:number;progress:number;playing:boolean;reset:number;focus:string;highlight:string;onScope:(s:string)=>void;onIndex:(i:number)=>void;onProgress:(p:number)=>void}
type Field={tensor:Tensor;crystals:ReturnType<typeof createCrystalTensor>;positions:Position3[];group:T.Group}
/** Every coordinate remains a crystal; leading coordinates identify separate planes. */
export const tensorPositions=(t:Tensor):Position3[]=>tensorLayout(t).positions
export default function WorkScene(props:Props){const host=useRef<HTMLDivElement>(null),latest=useRef(props);latest.current=props;const [error,setError]=useState(''),[hover,setHover]=useState('')
 useEffect(()=>{
  const el=host.current!;let renderer:T.WebGLRenderer
  try{renderer=new T.WebGLRenderer({antialias:true,powerPreference:'high-performance'})}catch{setError('无法启动 WebGL。仍可使用下方的逐项演算与完整张量表。');return}
  renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.setClearColor('#050c12');el.appendChild(renderer.domElement);renderer.domElement.tabIndex=0;renderer.domElement.setAttribute('aria-label','三维计算空间：点击模块进入，点击输出水晶选择数值，拖动旋转，滚轮缩放')
  const scene=new T.Scene(),camera=new T.PerspectiveCamera(40,1,.05,8000),controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.minDistance=2;controls.maxDistance=4000
  scene.add(new T.AmbientLight('#c6e7f0',1.7));const light=new T.DirectionalLight('#e7f8ff',3.1);light.position.set(20,35,25);scene.add(light);const rim=new T.DirectionalLight('#86bce6',2);rim.position.set(-20,8,-25);scene.add(rim)
  const labels=document.createElement('div');labels.className='ws-labels';el.appendChild(labels);const leaders=document.createElementNS('http://www.w3.org/2000/svg','svg');leaders.classList.add('ws-label-leaders');labels.appendChild(leaders)
  let body=new T.Group(),fields:Field[]=[],clicks:T.Object3D[]=[],tags:{el:HTMLButtonElement;point:T.Vector3;line:SVGLineElement;priority:number;scope:string}[]=[],raf=0,run:Run|undefined,scope='',oldReset=-1,oldFocus='',oldHighlight='',oldIndex=-1,oldProgress=-1,phase=0,last=0,dirty=true,visible=true,center=new T.Vector3(),extent=new T.Vector3(20,20,20)
  let connections:ReturnType<typeof createArrowStream>[]=[],lastNotify=0,products:Field|undefined
  const fieldBounds=new Map<string,T.Box3>();
  let flow:ReturnType<typeof createArrowStream>|undefined,factorFlow:ReturnType<typeof createArrowStream>|undefined,traceLines:T.LineSegments|undefined,traceKey='',tween:{from:T.Vector3;to:T.Vector3;target:T.Vector3;start:number}|undefined
  scene.add(body)
  const reduced=matchMedia('(prefers-reduced-motion: reduce)')
  const fit=()=>{const overview=latest.current.scope==='root'||latest.current.scope.startsWith('group:')||latest.current.scope.startsWith('layer:');camera.up.set(overview&&camera.aspect>1?1.15:0,1,0).normalize();
   const direction=new T.Vector3(1,1.05,1.55).normalize(),right=new T.Vector3().crossVectors(camera.up,direction).normalize(),up=new T.Vector3().crossVectors(direction,right).normalize();
   const focused=latest.current.focus?new Set(latest.current.run.steps.filter(s=>s.group===latest.current.focus).map(s=>s.output.id)):undefined;
   const boxes=[...fieldBounds].filter(([id])=>!focused||focused.has(id)).map(([,b])=>b),corners:T.Vector3[]=[];
   for(const b of boxes)for(const x of [b.min.x,b.max.x])for(const y of [b.min.y,b.max.y])for(const z of [b.min.z,b.max.z])corners.push(new T.Vector3(x,y,z).sub(center));
   const tan=Math.tan(T.MathUtils.degToRad(camera.fov/2)),usableX=Math.max(.52,1-140/el.clientWidth),usableY=Math.max(.65,1-85/el.clientHeight);
   let distance=3;for(const q of corners){const near=q.dot(direction);distance=Math.max(distance,near+Math.abs(q.dot(right))/(tan*camera.aspect*usableX),near+Math.abs(q.dot(up))/(tan*usableY))}
   const to=center.clone().addScaledVector(direction,distance*1.05+1);

   if(reduced.matches){camera.position.copy(to);controls.target.copy(center);controls.update()}else tween={from:camera.position.clone(),to,target:controls.target.clone(),start:performance.now()};dirty=true}

  const label=(text:string,point:T.Vector3,select:string,priority=0)=>{const button=document.createElement('button');button.className='ws-space-label';button.textContent=text;button.title=text;button.dataset.scope=select;button.onclick=()=>latest.current.onScope(select);labels.appendChild(button);const line=document.createElementNS('http://www.w3.org/2000/svg','line');leaders.appendChild(line);tags.push({el:button,point,line,priority,scope:select})}

  const field=(t:Tensor,origin:T.Vector3,select?:string,title?:string)=>{
   const group=new T.Group();group.position.copy(origin);body.add(group);const positions=tensorPositions(t),crystals=createCrystalTensor(group,t.values.length,.36);crystals.update(t.values,positions,{focus:-1});const f={tensor:t,group,positions,crystals};fields.push(f)
   for(const mesh of [crystals.body]){mesh.userData.tensor=t.id;mesh.userData.scope=select;clicks.push(mesh)}
   const layout=tensorLayout(t),segments:T.Vector3[]=[];
   for(const c of layout.centers){const w=layout.planeWidth/2,d=layout.planeDepth/2;const corners=[[-w,-.24,-d],[w,-.24,-d],[w,-.24,d],[-w,-.24,d]];for(let i=0;i<4;i++){const a=corners[i],b=corners[(i+1)%4];segments.push(new T.Vector3(c[0]+a[0],a[1],c[2]+a[2]),new T.Vector3(c[0]+b[0],b[1],c[2]+b[2]))}}
   group.add(new T.LineSegments(new T.BufferGeometry().setFromPoints(segments),new T.LineBasicMaterial({color:t.parameter?'#ddb889':'#76bdcf',transparent:true,opacity:.48})));
   const b=new T.Box3().setFromObject(group);fieldBounds.set(t.id,b.clone());
   label(`${title??t.name}\n[${t.shape.join(' × ')}]`,new T.Vector3(b.max.x+.35,origin.y+.4,origin.z),select??latest.current.scope,t.id===latest.current.run.input.id||t.id===latest.current.run.output.id?100:0);return b
  }
  const rebuild=()=>{
   const p=latest.current;setHover('');fieldBounds.clear();scene.remove(body);disposeGroup(body);body=new T.Group();scene.add(body);labels.replaceChildren(leaders);leaders.replaceChildren();fields=[];clicks=[];tags=[];flow=undefined;factorFlow=undefined;traceLines=undefined;traceKey='';oldIndex=-1;connections=[];products=undefined
   const box=new T.Box3();
   if(p.scope==='root'||p.scope.startsWith('group:')||p.scope.startsWith('layer:')){
    const principal=principalSteps(p.run),parent=p.run.steps.find(s=>s.id===p.scope.slice(6));
    const steps=p.scope==='root'?principal:p.scope.startsWith('layer:')&&parent?childSteps(p.run,parent):p.run.steps.filter(s=>s.group===p.scope.slice(6));
    const graph=buildScaffold(p.run,steps);
    for(const n of graph.nodes){
     const hasChildren=n.step&&childSteps(p.run,n.step).length>1;
     const destination=n.step?(hasChildren&&!p.scope.startsWith('layer:')?`layer:${n.step.id}`:n.step.id):p.run.steps.find(s=>s.output.id===n.tensor.id)?.id??p.scope;
     const name=n.step?stepLabel(n.step):n.tensor.id===p.run.input.id?'输入 X':`外部输入 · ${n.tensor.name}`;
     box.union(field(n.tensor,new T.Vector3(...n.position),destination,name+(n.tensor.id===p.run.output.id?' · 输出':'')));
    }
    const map=new Map(graph.nodes.map(n=>[n.tensor.id,n]));
    for(const e of graph.edges){const a=map.get(e.from)!,b=map.get(e.to)!;
     const side=Math.max(a.width,b.width)/2+2,from:Position3=[a.position[0],a.position[1]-.4,a.position[2]],to:Position3=[b.position[0],b.position[1]+.4,b.position[2]];
     const points:Position3[]=e.skip?[from,[side,from[1]-1,1],[side,to[1]+1,1],to]:[from,to];
     const arrow=createArrowStream(body,e.skip?'#e3c392':'#a2d4e4',.15,3);arrow.update(points,.4);arrow.group.userData.path=points;arrow.group.userData.to=e.to;connections.push(arrow);
    }
    el.dataset.layers=String(graph.nodes.length);el.dataset.levels=String(graph.levels);
   }else{
    for(const stream of connections)stream.update(stream.group.userData.path,phase)
   if(p.scope==='root'||p.scope.startsWith('group:')||p.scope.startsWith('layer:')){
    const visibleSteps=p.run.steps.filter(s=>fields.some(f=>f.tensor.id===s.output.id)),cursor=phase*visibleSteps.length,active=visibleSteps[Math.min(visibleSteps.length-1,Math.floor(cursor))];
    if(active){const index=Math.min(active.output.values.length-1,Math.floor((cursor%1)*active.output.values.length)),terms=active.trace(index);for(const f of fields){const indices=terms.filter(t=>t.tensor===f.tensor.id&&t.index>=0).map(t=>t.index);f.crystals.update(f.tensor.values,f.positions,{active:indices,focus:f.tensor.id===active.output.id?index:-1})}el.dataset.activeLayer=active.id;}
   }
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
  const focusStage=()=>{const p=latest.current;let bounds=new T.Box3();const targets=p.focus?p.run.steps.filter(s=>s.group===p.focus):[];for(const s of targets){const b=fieldBounds.get(s.output.id);if(b)bounds.union(b)}if(bounds.isEmpty())for(const b of fieldBounds.values())bounds.union(b);if(!bounds.isEmpty()){bounds.getCenter(center);bounds.getSize(extent);extent.addScalar(5);fit()}};
  const resize=()=>{const w=el.clientWidth,h=el.clientHeight;if(!w||!h)return;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();fit()};const ro=new ResizeObserver(resize);ro.observe(el);const io=new IntersectionObserver(([e])=>visible=e.isIntersecting);io.observe(el)
  const ray=new T.Raycaster(),pointer=new T.Vector2();let down=[0,0]
  const pick=(event:PointerEvent)=>{const r=renderer.domElement.getBoundingClientRect();pointer.set((event.clientX-r.left)/r.width*2-1,-(event.clientY-r.top)/r.height*2+1);ray.setFromCamera(pointer,camera);return ray.intersectObjects(clicks,false)[0]}
  const pointerDown=(e:PointerEvent)=>{down=[e.clientX,e.clientY];tween=undefined}
  const pointerUp=(e:PointerEvent)=>{if(Math.hypot(e.clientX-down[0],e.clientY-down[1])>6)return;const hit=pick(e);if(!hit)return;const data=hit.object.userData;if(data.scope&&data.scope!==latest.current.scope)latest.current.onScope(data.scope);else if(data.scopes)latest.current.onScope(data.scopes[hit.instanceId??0]);else if(data.tensor){const t=latest.current.run.tensors.find(t=>t.id===data.tensor)??fields.find(f=>f.tensor.id===data.tensor)?.tensor;if(!t)return;const index=hit.instanceId??0;const step=latest.current.run.steps.find(s=>s.id===latest.current.scope);if(step?.output.id===t.id)latest.current.onIndex(index);setHover(`${t.name}[${coords(index,t.shape).join(', ')}] = ${t.values[index]?.toPrecision(7)}`)}}
  renderer.domElement.addEventListener('keydown',event=>{if(event.key==='+'||event.key==='='){camera.position.lerp(controls.target,.15);dirty=true;event.preventDefault()}else if(event.key==='-'){camera.position.sub(controls.target).multiplyScalar(1.18).add(controls.target);dirty=true;event.preventDefault()}else if(event.key==='Home'){fit();event.preventDefault()}})
  renderer.domElement.addEventListener('pointerdown',pointerDown);renderer.domElement.addEventListener('pointerup',pointerUp)
  controls.addEventListener('change',()=>dirty=true)
  const tick=(now:number)=>{raf=requestAnimationFrame(tick);const p=latest.current;if(run!==p.run||scope!==p.scope){run=p.run;scope=p.scope;rebuild()}
   if(p.focus!==oldFocus){oldFocus=p.focus;focusStage()}if(p.reset!==oldReset){oldReset=p.reset;focusStage()}if(p.progress!==oldProgress){oldProgress=p.progress;phase=p.progress;dirty=true}
   const dt=Math.min(.05,(now-last)/1000);last=now;if(!visible||document.hidden)return
   if(p.playing&&!reduced.matches){phase=(phase+dt*.12)%1;dirty=true;if(now-lastNotify>120){lastNotify=now;p.onProgress(phase)}}
   if(tween){const f=Math.min(1,(now-tween.start)/650),k=1-(1-f)**4;camera.position.lerpVectors(tween.from,tween.to,k);controls.target.lerpVectors(tween.target,center,k);if(f===1)tween=undefined;dirty=true}controls.update()
   if(p.highlight!==oldHighlight){oldHighlight=p.highlight;dirty=true}
   if(p.index!==oldIndex){oldIndex=p.index;dirty=true}
   if(!dirty)return
   for(const stream of connections)stream.update(stream.group.userData.path,phase)
   if(p.scope==='root'||p.scope.startsWith('group:')||p.scope.startsWith('layer:')){
    const visibleSteps=p.run.steps.filter(s=>fields.some(f=>f.tensor.id===s.output.id)),cursor=phase*visibleSteps.length,active=visibleSteps[Math.min(visibleSteps.length-1,Math.floor(cursor))];
    if(active){const index=Math.min(active.output.values.length-1,Math.floor((cursor%1)*active.output.values.length)),terms=active.trace(index);for(const f of fields){const indices=terms.filter(t=>t.tensor===f.tensor.id&&t.index>=0).map(t=>t.index);f.crystals.update(f.tensor.values,f.positions,{active:indices,focus:f.tensor.id===active.output.id?index:-1})}el.dataset.activeLayer=active.id;}
   }
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
   const highlighted=p.run.steps.find(s=>s.id===p.highlight)?.output.id;for(const f of fields){for(const child of f.group.children)if(child instanceof T.LineSegments&&(child.material instanceof T.LineBasicMaterial)){child.material.opacity=f.tensor.id===highlighted?1:.48;child.material.color.set(f.tensor.id===highlighted?'#effaff':f.tensor.parameter?'#ddb889':'#76bdcf')}}
   renderer.render(scene,camera);const used:{x:number;y:number;w:number;h:number}[]=[];
   const ordered=[...tags].sort((a,b)=>(b.priority+(b.scope.replace('layer:','')===p.highlight?200:0))-(a.priority+(a.scope.replace('layer:','')===p.highlight?200:0)));
   for(const tag of ordered){const q=tag.point.clone().project(camera),anchorX=(q.x+1)*el.clientWidth/2,anchorY=(1-q.y)*el.clientHeight/2,w=tag.el.offsetWidth||150,h=tag.el.offsetHeight||42;let x=anchorX,y=anchorY,show=false;
    if(q.z<1&&q.z>0&&((!p.focus&&tag.priority>0)||tag.scope.replace('layer:','')===p.highlight||(anchorX>-10&&anchorX<el.clientWidth+10&&anchorY>0&&anchorY<el.clientHeight))){
     for(const dy of [0,-45,45,-90,90,-135,135]){for(const dx of [w/2+10,-w/2-10,0,w+20,-w-20]){const cx=Math.max(w/2+8,Math.min(el.clientWidth-w/2-8,anchorX+dx)),cy=Math.max(h/2+8,Math.min(el.clientHeight-h/2-8,anchorY+dy));if(!used.some(r=>Math.abs(cx-r.x)<(w+r.w)/2+7&&Math.abs(cy-r.y)<(h+r.h)/2+5)){x=cx;y=cy;show=true;break}}if(show)break}
    }
    tag.el.style.visibility=show?'visible':'hidden';tag.el.tabIndex=show?0:-1;tag.el.style.left=`${x}px`;tag.el.style.top=`${y}px`;tag.el.classList.toggle('is-linked',tag.scope.replace('layer:','')===p.highlight);tag.line.style.visibility=show?'visible':'hidden';if(show){used.push({x,y,w,h});tag.line.setAttribute('x1',String(anchorX));tag.line.setAttribute('y1',String(anchorY));tag.line.setAttribute('x2',String(x));tag.line.setAttribute('y2',String(y));}
   }
   dirty=false
  };resize();raf=requestAnimationFrame(tick)
  return()=>{cancelAnimationFrame(raf);ro.disconnect();io.disconnect();controls.dispose();disposeGroup(scene);renderer.dispose();renderer.domElement.remove();labels.remove()}
 },[])
 return <div className="ws-scene" ref={host}>{error&&<p className="ws-scene-error" role="status">{error}</p>}{hover&&<output className="ws-cell-readout">{hover}</output>}</div>
}
