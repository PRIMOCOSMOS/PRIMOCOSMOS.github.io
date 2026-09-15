import {useEffect,useRef,useState} from 'react'
import * as T from 'three'
import katex from 'katex'
import {OrbitControls} from 'three/addons/controls/OrbitControls.js'
import {createCrystalTensor,createArrowStream,type Position3} from '../mmhvae/crystalPrimitives'
import {disposeGroup} from '../mmhvae/glyphs'
import type {Run,Tensor,Step} from './engine'
import {coords} from './engine'
import {computation,scalarEquation} from './computation'
import {connectionFabric,operatorBridge,operationFamily,motionNames,type Connection} from './operationMotion'
import {buildScaffold,childSteps,principalSteps,stepLabel,tensorLayout} from './scaffold'
interface Props {run:Run;scope:string;index:number;progress:number;playing:boolean;reset:number;focus:string;highlight:string;zoom:number;pan:boolean;showLabels:boolean;onScope:(s:string)=>void;onIndex:(i:number)=>void;onProgress:(p:number)=>void}
type Field={tensor:Tensor;crystals:ReturnType<typeof createCrystalTensor>;positions:Position3[];group:T.Group;selectionKey?:string}
/** Every coordinate remains a crystal; leading coordinates identify separate planes. */
export const tensorPositions=(t:Tensor):Position3[]=>tensorLayout(t).positions
export default function WorkScene(props:Props){const host=useRef<HTMLDivElement>(null),latest=useRef(props);latest.current=props;const [error,setError]=useState(''),[hover,setHover]=useState(''),[live,setLive]=useState({title:'',equation:'',step:''})
 useEffect(()=>{
  const el=host.current!;let renderer:T.WebGLRenderer
  try{renderer=new T.WebGLRenderer({antialias:true,powerPreference:'high-performance'})}catch{setError('无法启动 WebGL。仍可使用下方的逐项演算与完整张量表。');return}
  renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.setClearColor('#050c12');el.appendChild(renderer.domElement);renderer.domElement.tabIndex=0;renderer.domElement.setAttribute('aria-label','三维计算空间：点击模块进入，点击输出水晶选择数值，拖动旋转，滚轮缩放')
  const scene=new T.Scene(),camera=new T.PerspectiveCamera(40,1,.05,8000),controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.minDistance=2;controls.maxDistance=4000;controls.zoomToCursor=true;controls.zoomSpeed=.7;controls.screenSpacePanning=true;controls.touches.TWO=T.TOUCH.DOLLY_PAN
  scene.add(new T.AmbientLight('#c6e7f0',1.7));const light=new T.DirectionalLight('#e7f8ff',3.1);light.position.set(20,35,25);scene.add(light);const rim=new T.DirectionalLight('#86bce6',2);rim.position.set(-20,8,-25);scene.add(rim)
  const labels=document.createElement('div');labels.className='ws-labels';el.appendChild(labels);const leaders=document.createElementNS('http://www.w3.org/2000/svg','svg');leaders.classList.add('ws-label-leaders');labels.appendChild(leaders)
  let body=new T.Group(),fields:Field[]=[],clicks:T.Object3D[]=[],tags:{el:HTMLButtonElement;point:T.Vector3;line:SVGLineElement;priority:number;scope:string}[]=[],raf=0,run:Run|undefined,scope='',oldReset=-1,oldFocus='',layoutFocus='',liveStep='',oldHighlight='',hoveredScope='',oldZoom=0,oldPan=false,oldLabels=false,oldIndex=-1,oldProgress=-1,phase=0,last=0,dirty=true,visible=true,center=new T.Vector3(),extent=new T.Vector3(20,20,20)
  let lastLive=0;
  let connections:ReturnType<typeof createArrowStream>[]=[],lastNotify=0,products:Field|undefined,prefix:Field|undefined,sums:Field|undefined
  const fieldBounds=new Map<string,T.Box3>();let manualFrame:T.Box3|undefined;
  let fabrics:{step:Step;view:ReturnType<typeof connectionFabric>}[]=[],bridges:{step:Step;view:ReturnType<typeof operatorBridge>}[]=[],linearFabric:ReturnType<typeof connectionFabric>|undefined,activeFabric:ReturnType<typeof connectionFabric>|undefined,activeKey='';
  let flow:ReturnType<typeof createArrowStream>|undefined,factorFlow:ReturnType<typeof createArrowStream>|undefined,traceLines:T.LineSegments|undefined,traceKey='',tween:{from:T.Vector3;to:T.Vector3;target:T.Vector3;start:number}|undefined
  scene.add(body)
  const reduced=matchMedia('(prefers-reduced-motion: reduce)')
  const fit=()=>{const overview=latest.current.scope==='root'&&!latest.current.focus;camera.up.set(overview&&camera.aspect>1?1.15:0,1,0).normalize();
   const direction=new T.Vector3(1,1.05,1.55).normalize(),right=new T.Vector3().crossVectors(camera.up,direction).normalize(),up=new T.Vector3().crossVectors(direction,right).normalize();
   // Focus uses a local stage layout with boundary tensor ports, so framing also includes its inputs.

   const boxes=manualFrame?[manualFrame]:[...fieldBounds].map(([,b])=>b),corners:T.Vector3[]=[];
   for(const b of boxes)for(const x of [b.min.x,b.max.x])for(const y of [b.min.y,b.max.y])for(const z of [b.min.z,b.max.z])corners.push(new T.Vector3(x,y,z).sub(center));
   const tan=Math.tan(T.MathUtils.degToRad(camera.fov/2)),usableX=Math.max(.52,1-140/el.clientWidth),usableY=Math.max(.55,1-((latest.current.scope==='root'||latest.current.scope.startsWith('group:')||latest.current.scope.startsWith('layer:'))?160:85)/el.clientHeight);
   let distance=3;for(const q of corners){const near=q.dot(direction);distance=Math.max(distance,near+Math.abs(q.dot(right))/(tan*camera.aspect*usableX),near+Math.abs(q.dot(up))/(tan*usableY))}
   const to=center.clone().addScaledVector(direction,distance*1.05+1);

   if(reduced.matches){camera.position.copy(to);controls.target.copy(center);controls.update()}else tween={from:camera.position.clone(),to,target:controls.target.clone(),start:performance.now()};dirty=true}

  const label=(text:string,point:T.Vector3,select:string,priority=0,formula?:string)=>{const button=document.createElement('button');button.className='ws-space-label';button.textContent=text;if(formula){const math=document.createElement('span');math.className='ws-label-formula';math.innerHTML=katex.renderToString(formula,{throwOnError:false,trust:false});button.appendChild(math)}button.title=text;button.dataset.scope=select;button.onclick=()=>latest.current.onScope(select);labels.appendChild(button);const line=document.createElementNS('http://www.w3.org/2000/svg','line');leaders.appendChild(line);tags.push({el:button,point,line,priority,scope:select})}

  const field=(t:Tensor,origin:T.Vector3,select?:string,title?:string,formula?:string)=>{
   const group=new T.Group();group.position.copy(origin);body.add(group);const positions=tensorPositions(t),crystals=createCrystalTensor(group,t.values.length,.36,{valueEdges:true,bodyOpacity:.23,edgeOpacity:.5});crystals.update(t.values,positions,{focus:-1});const f={tensor:t,group,positions,crystals};fields.push(f)
   for(const mesh of [crystals.body]){mesh.userData.tensor=t.id;mesh.userData.scope=select;clicks.push(mesh)}
   const layout=tensorLayout(t),segments:T.Vector3[]=[];
   for(const c of layout.centers){const w=layout.planeWidth/2,d=layout.planeDepth/2;const corners=[[-w,-.24,-d],[w,-.24,-d],[w,-.24,d],[-w,-.24,d]];for(let i=0;i<4;i++){const a=corners[i],b=corners[(i+1)%4];segments.push(new T.Vector3(c[0]+a[0],a[1],c[2]+a[2]),new T.Vector3(c[0]+b[0],b[1],c[2]+b[2]))}}
   group.add(new T.LineSegments(new T.BufferGeometry().setFromPoints(segments),new T.LineBasicMaterial({color:t.parameter?'#ddb889':'#76bdcf',transparent:true,opacity:.48})));
   const b=new T.Box3().setFromObject(group);fieldBounds.set(t.id,b.clone());
   label(`${title??t.name}${layout.wrapped?' · 连续索引折行':''}\n[${t.shape.join(' × ')}]`,new T.Vector3(b.max.x+.35,origin.y+.4,origin.z),select??latest.current.scope,t.id===latest.current.run.input.id||t.id===(latest.current.run.steps.find(s=>s.id===latest.current.scope)?.output.id??latest.current.run.output.id)?100:0,formula);return b
  }
  const paint=(f:Field,selection:{focus?:number;active?:number[]})=>{const key=String(selection.focus??-1)+':'+(selection.active??[]).join(',');if(f.selectionKey===key)return;f.selectionKey=key;f.crystals.update(f.tensor.values,f.positions,selection)};
  const rebuild=()=>{
   const p=latest.current;setHover('');setLive({title:'',equation:'',step:''});lastLive=-Infinity;manualFrame=undefined;fabrics=[];bridges=[];linearFabric=undefined;activeFabric=undefined;activeKey='';hoveredScope='';fieldBounds.clear();scene.remove(body);disposeGroup(body);body=new T.Group();scene.add(body);labels.replaceChildren(leaders);leaders.replaceChildren();fields=[];clicks=[];tags=[];flow=undefined;factorFlow=undefined;traceLines=undefined;traceKey='';oldIndex=-1;connections=[];products=undefined;prefix=undefined;sums=undefined
   const box=new T.Box3();
   if(p.scope==='root'||p.scope.startsWith('group:')||p.scope.startsWith('layer:')){
    const principal=principalSteps(p.run),parent=p.run.steps.find(s=>s.id===p.scope.slice(6));
    const steps=p.scope==='root'?(p.focus?principal.filter(s=>s.group===p.focus):principal):p.scope.startsWith('layer:')&&parent?childSteps(p.run,parent):p.run.steps.filter(s=>s.group===p.scope.slice(6));
    const graph=buildScaffold(p.run,steps),stageView=p.scope!=='root'||!!p.focus;
    const outputs=new Set(graph.nodes.filter(n=>n.step&&!graph.edges.some(e=>e.from===n.tensor.id)).map(n=>n.tensor.id));
    for(const n of graph.nodes){
     const hasChildren=n.step&&childSteps(p.run,n.step).length>1;
     const destination=n.step?(hasChildren&&!p.scope.startsWith('layer:')?`layer:${n.step.id}`:n.step.id):p.run.steps.find(s=>s.output.id===n.tensor.id)?.id??p.scope;
     const boundary=stageView&&(n.external||outputs.has(n.tensor.id));const name=stageView&&n.external?`阶段输入 · ${n.tensor.name}`:stageView&&outputs.has(n.tensor.id)?`阶段输出 · ${stepLabel(n.step!)}`:n.step?stepLabel(n.step):n.tensor.id===p.run.input.id?'输入 X':`外部输入 · ${n.tensor.name}`;
     box.union(field(n.tensor,new T.Vector3(...n.position),destination,name+(n.tensor.id===p.run.output.id&&!stageView?' · 输出':'')));if(boundary)tags.at(-1)!.priority=100;
    }
    const map=new Map(graph.nodes.map(n=>[n.tensor.id,n]));
    for(const e of graph.edges){const a=map.get(e.from)!,b=map.get(e.to)!;
     const side=Math.max(...graph.nodes.map(n=>n.position[0]+n.width/2))+2,from:Position3=[a.position[0],a.position[1]-.4,a.position[2]],to:Position3=[b.position[0],b.position[1]+.4,b.position[2]];
     const points:Position3[]=e.skip?[from,[side,from[1]-1,1],[side,to[1]+1,1],to]:[from,to];
     if(e.skip){const arrow=createArrowStream(body,'#e3c392',.15,3);arrow.update(points,.4);arrow.group.userData.path=points;arrow.group.userData.to=e.to;arrow.group.userData.skip=true;connections.push(arrow)}
     else if(b.step){const sf=fields.find(f=>f.tensor.id===a.tensor.id)!,tf=fields.find(f=>f.tensor.id===b.tensor.id)!;if(b.step.kind==='Linear'&&a.tensor.id===b.step.inputs[0].id){const step=b.step,n=step.inputs[0].shape.at(-1)!,out=step.output.shape.at(-1)!,links:Connection[]=[];for(let o=0;o<step.output.values.length;o++)for(let i=0;i<n;i++)links.push({from:sf.group.localToWorld(new T.Vector3(...sf.positions[Math.floor(o/out)*n+i])).toArray() as Position3,to:tf.group.localToWorld(new T.Vector3(...tf.positions[o])).toArray() as Position3,weight:step.inputs[1].values[o%out*n+i],output:o,term:i});fabrics.push({step,view:connectionFabric(body,links)})}else bridges.push({step:b.step,view:operatorBridge(body,b.step,from,to,Math.min(a.width,b.width))})}
    }
    el.dataset.layers=String(graph.nodes.length);el.dataset.levels=String(graph.levels);

   }else{
    const step=p.run.steps.find(s=>s.id===p.scope)??p.run.steps[0],inputs=[...new Map(step.inputs.map(t=>[t.id,t])).values()],calc=computation(step,Math.min(p.index,step.output.values.length-1));
    if(step.kind==='Linear'){
     const [x,w,bias]=step.inputs,gap=Math.max(4,tensorLayout(calc.products!).depth*.45+1.5),position=(f:Field,i:number)=>f.group.localToWorld(new T.Vector3(...f.positions[i])).toArray() as Position3;
     box.union(field(x,new T.Vector3(0,gap*2,0)));const input=fields.at(-1)!;
     box.union(field(calc.products!,new T.Vector3(0,gap,0),undefined,'逐项乘积',String.raw`x_{r,i}W_{o,i}`));products=fields.at(-1)!;
     box.union(field(calc.sums!,new T.Vector3(0,0,0),undefined,'点积 · 尚未加偏置',String.raw`\sum_i x_{r,i}W_{o,i}`));sums=fields.at(-1)!;
     let biasField:Field|undefined;if(bias){box.union(field(bias,new T.Vector3(0,-gap*.55,0),undefined,'偏置 b · 按 batch 广播'));biasField=fields.at(-1)!}
     box.union(field(step.output,new T.Vector3(0,-gap*1.35,0),undefined,'全连接输出',String.raw`y_{r,o}=s_{r,o}+b_o`));const output=fields.at(-1)!,n=x.shape.at(-1)!,out=w.shape[0],links:Connection[]=[];
     for(let o=0;o<step.output.values.length;o++){for(let i=0;i<n;i++){const pi=o*n+i;links.push({from:position(input,Math.floor(o/out)*n+i),to:position(products,pi),weight:w.values[o%out*n+i],output:o,term:i},{from:position(products,pi),to:position(sums,o),weight:1,output:o,term:i})}links.push({from:position(sums,o),to:position(output,o),weight:1,output:o,term:n});if(biasField)links.push({from:position(biasField,o%out),to:position(output,o),weight:bias!.values[o%out],output:o,term:n})}
     linearFabric=connectionFabric(body,links);label(`加权连接 W [${w.shape.join(' × ')}]\n颜色表示权重正负`,new T.Vector3(tensorLayout(calc.products!).width/2+.3,gap*1.5,0),p.scope);
    }else{
    const data=inputs.filter(t=>!t.parameter&&!t.constant),parameters=inputs.filter(t=>t.parameter||t.constant);
    const depth=Math.max(2,...inputs.map(t=>tensorLayout(t).depth),tensorLayout(step.output).depth),gap=Math.max(5,depth*.65+2);
    let x=-(data.reduce((n,t)=>n+tensorLayout(t).width+3,0)-3)/2;
    for(const t of data){const width=tensorLayout(t).width;box.union(field(t,new T.Vector3(x+width/2,gap*2,0)));x+=width+3}
    let py=gap*3;
    for(const t of parameters){const l=tensorLayout(t),isBias=step.kind==='Linear'&&step.inputs[2]?.id===t.id;box.union(field(t,new T.Vector3(0,isBias?-gap:py,0)));if(!isBias)py+=Math.max(4,l.depth*.65+2)}
    let outputY=0;
    if(calc.products){box.union(field(calc.products,new T.Vector3(0,gap,0)));products=fields.at(-1);box.union(field(calc.prefix!,new T.Vector3(0,0,0)));prefix=fields.at(-1);outputY=-gap;if(calc.sums){box.union(field(calc.sums,new T.Vector3(0,-gap,0)));sums=fields.at(-1);outputY=-gap*2}}
    box.union(field(step.output,new T.Vector3(0,outputY,0),undefined,stepLabel(step)+' · 输出'));
    }
    if(step.kind!=='Linear'){const src=fields.find(f=>f.tensor.id===step.inputs[0]?.id),dst=fields.find(f=>f.tensor.id===step.output.id);if(src&&dst){const to=(products?.group.position??dst.group.position).toArray() as Position3;bridges.push({step,view:operatorBridge(body,step,src.group.position.toArray() as Position3,to,Math.min(tensorLayout(src.tensor).width,tensorLayout(dst.tensor).width))})}}
    flow=createArrowStream(body,'#d4f3fc',.14,2);factorFlow=createArrowStream(body,'#f1ca9e',.12,2);
   }

   box.getCenter(center);box.getSize(extent);extent.addScalar(5);body.updateMatrixWorld(true);el.dataset.scope=p.scope;el.dataset.cells=String(fields.reduce((n,f)=>n+f.tensor.values.length,0));fit();dirty=true
  }
  const focusStage=()=>{manualFrame=undefined;let bounds=new T.Box3();for(const b of fieldBounds.values())bounds.union(b);if(!bounds.isEmpty()){bounds.getCenter(center);bounds.getSize(extent);extent.addScalar(5);fit()}};
  const resize=()=>{const w=el.clientWidth,h=el.clientHeight;if(!w||!h)return;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();fit()};const ro=new ResizeObserver(resize);ro.observe(el);const io=new IntersectionObserver(([e])=>visible=e.isIntersecting);io.observe(el)
  const ray=new T.Raycaster(),pointer=new T.Vector2();let down=[0,0]
  const pick=(event:PointerEvent)=>{const r=renderer.domElement.getBoundingClientRect();pointer.set((event.clientX-r.left)/r.width*2-1,-(event.clientY-r.top)/r.height*2+1);ray.setFromCamera(pointer,camera);return ray.intersectObjects(clicks.filter(o=>{for(let n:T.Object3D|null=o;n;n=n.parent)if(!n.visible)return false;return true}),false)[0]}
  const pointerDown=(e:PointerEvent)=>{down=[e.clientX,e.clientY];tween=undefined}
  const selectHit=(e:PointerEvent)=>{if(Math.hypot(e.clientX-down[0],e.clientY-down[1])>6)return;const hit=pick(e);if(!hit)return;const data=hit.object.userData;if(data.scope&&data.scope!==latest.current.scope)latest.current.onScope(data.scope);else if(data.scopes)latest.current.onScope(data.scopes[hit.instanceId??0]);else if(data.tensor){const t=latest.current.run.tensors.find(t=>t.id===data.tensor)??fields.find(f=>f.tensor.id===data.tensor)?.tensor;if(!t)return;const index=hit.instanceId??0;const step=latest.current.run.steps.find(s=>s.id===latest.current.scope);if(step?.output.id===t.id)latest.current.onIndex(index);setHover(`${t.name}[${coords(index,t.shape).join(', ')}] = ${t.values[index]?.toPrecision(7)}`)}}
  let clickTimer:ReturnType<typeof setTimeout>|undefined;const pointerUp=(e:PointerEvent)=>{if(Math.hypot(e.clientX-down[0],e.clientY-down[1])>6)return;if(clickTimer)clearTimeout(clickTimer);clickTimer=setTimeout(()=>selectHit(e),230)};
  renderer.domElement.addEventListener('keydown',event=>{if(event.key==='+'||event.key==='='){camera.position.lerp(controls.target,.15);dirty=true;event.preventDefault()}else if(event.key==='-'){camera.position.sub(controls.target).multiplyScalar(1.18).add(controls.target);dirty=true;event.preventDefault()}else if(event.key==='Home'){fit();event.preventDefault()}})
  let lastHover=0;
  renderer.domElement.addEventListener('pointermove',event=>{if(event.buttons||performance.now()-lastHover<70)return;lastHover=performance.now();const hit=pick(event),next=hit?.object.userData.scope??'';if(next!==hoveredScope){hoveredScope=next;dirty=true}renderer.domElement.style.cursor=hit?'pointer':latest.current.pan?'grab':'default'});
  renderer.domElement.addEventListener('pointerleave',()=>{hoveredScope='';dirty=true});
  renderer.domElement.addEventListener('dblclick',event=>{if(clickTimer)clearTimeout(clickTimer);tween=undefined;const hit=pick(event as unknown as PointerEvent),id=hit?.object.userData.tensor,bounds=id?fieldBounds.get(id):undefined;if(bounds){manualFrame=bounds.clone();bounds.getCenter(center);fit()}});
  renderer.domElement.addEventListener('pointerdown',pointerDown);renderer.domElement.addEventListener('pointerup',pointerUp)
  controls.addEventListener('change',()=>dirty=true)
  const tick=(now:number)=>{raf=requestAnimationFrame(tick);const p=latest.current;if(run!==p.run||scope!==p.scope||layoutFocus!==p.focus){run=p.run;scope=p.scope;layoutFocus=p.focus;liveStep='';rebuild()}
   if(p.focus!==oldFocus){oldFocus=p.focus;focusStage()}if(p.reset!==oldReset){oldReset=p.reset;focusStage()}if(p.progress!==oldProgress){oldProgress=p.progress;phase=p.progress;dirty=true}
   const dt=Math.min(.05,(now-last)/1000);last=now;if(!visible||document.hidden)return
   if(p.playing&&!reduced.matches){phase=(phase+dt*.4/((p.scope==='root'||p.scope.startsWith('group:')||p.scope.startsWith('layer:'))?Math.max(1,p.run.steps.filter(s=>(!p.focus||s.group===p.focus)&&fields.some(f=>f.tensor.id===s.output.id)).length):2))%1;dirty=true;if(now-lastNotify>120){lastNotify=now;p.onProgress(phase)}}
   if(tween){const f=Math.min(1,(now-tween.start)/650),k=1-(1-f)**4;camera.position.lerpVectors(tween.from,tween.to,k);controls.target.lerpVectors(tween.target,center,k);if(f===1)tween=undefined;dirty=true}controls.update()
   if(p.zoom!==oldZoom){tween=undefined;const delta=p.zoom-oldZoom;oldZoom=p.zoom;camera.position.sub(controls.target).multiplyScalar(Math.exp(-delta*.23)).clampLength(controls.minDistance,controls.maxDistance).add(controls.target);dirty=true}
   if(p.pan!==oldPan){oldPan=p.pan;controls.mouseButtons.LEFT=p.pan?T.MOUSE.PAN:T.MOUSE.ROTATE}
   if(p.showLabels!==oldLabels){oldLabels=p.showLabels;dirty=true}
   if(p.highlight!==oldHighlight){oldHighlight=p.highlight;dirty=true}
   if(p.index!==oldIndex){oldIndex=p.index;for(const f of fields)f.selectionKey=undefined;dirty=true}
   if(!dirty)return
   const activeHighlight=p.highlight||hoveredScope.replace('layer:','');
   for(const stream of connections)stream.update(stream.group.userData.path,phase)
   if(p.scope==='root'||p.scope.startsWith('group:')||p.scope.startsWith('layer:')){
    const visibleSteps=p.run.steps.filter(s=>(!p.focus||s.group===p.focus)&&fields.some(f=>f.tensor.id===s.output.id)),cursor=phase*visibleSteps.length,active=visibleSteps.find(s=>s.id===activeHighlight)??visibleSteps[Math.min(visibleSteps.length-1,Math.floor(cursor))];
    if(active){liveStep=active.id;for(const stream of connections)stream.group.visible=!stream.group.userData.skip||stream.group.userData.to===active.output.id;const local=activeHighlight?phase:cursor%1,index=Math.min(p.index,active.output.values.length-1),terms=active.trace(index),j=Math.min(terms.length-1,Math.floor(local*terms.length)),term=terms[j];for(const f of fields){if(!f.group.visible)continue;const indices=terms.filter(t=>t.tensor===f.tensor.id&&t.index>=0).map(t=>t.index);paint(f,{active:indices,focus:f.tensor.id===active.output.id?index:term?.tensor===f.tensor.id?term.index:-1})}el.dataset.activeLayer=active.id;
     const nextKey=active.id+':'+index;if(nextKey!==activeKey){activeKey=nextKey;activeFabric?.dispose();const target=fields.find(f=>f.tensor.id===active.output.id)!;const links:Connection[]=terms.flatMap((t,i)=>{const source=fields.find(f=>f.tensor.id===t.tensor);return source&&t.index>=0?[{from:source.group.localToWorld(new T.Vector3(...source.positions[t.index])).toArray() as Position3,to:target.group.localToWorld(new T.Vector3(...target.positions[index])).toArray() as Position3,weight:t.factor??1,output:index,term:i}]:[]});activeFabric=connectionFabric(body,links)}activeFabric?.update(local,index,j);
     for(const item of fabrics)item.view.update(local,item.step.id===active.id?index:-2,j,.9);for(const item of bridges)item.view.update(local,item.step.id===active.id);
     if(now-lastLive>120){lastLive=now;setLive({title:active.title+' · '+motionNames[operationFamily(active)],equation:scalarEquation(active,index,local),step:active.id})}
    }

   }
   const step=p.run.steps.find(s=>s.id===p.scope)
   if(step){const index=Math.min(p.index,step.output.values.length-1),terms=step.trace(index),current=Math.min(terms.length-1,Math.floor(phase*terms.length)),term=terms[current],out=fields.find(f=>f.tensor.id===step.output.id)!,target=out.group.localToWorld(new T.Vector3(...out.positions[index]))
    const calc=computation(step,index),biasTerm=step.kind==='Linear'&&term?.tensor===step.inputs[2]?.id,productIndex=calc.elementwise?index:(calc.productOffset??0)+Math.min(current,(calc.width??terms.length)-1);
    if(products&&calc.products){products.tensor.values=calc.products.values;paint(products,{focus:productIndex,active:Array.from({length:Math.min(current+1,calc.width!)},(_,i)=>(calc.productOffset??0)+i)})}
    if(prefix&&calc.prefix){prefix.tensor.values=calc.prefix.values;paint(prefix,{focus:calc.elementwise?index:Math.min(current,calc.width!-1),active:Array.from({length:Math.min(current+1,calc.width!)},(_,i)=>i)})}
    linearFabric?.update(phase,index,current);for(const item of bridges)item.view.update(phase,true);
    if(sums)paint(sums,{focus:index});
    for(const f of fields){if(f===products||f===prefix||f===sums)continue;const active=terms.filter(t=>t.tensor===f.tensor.id&&t.index>=0).map(t=>t.index);active.push(...terms.filter(t=>t.factorTensor===f.tensor.id).map(t=>t.factorIndex!));paint(f,{active,focus:f===out?index:term?.tensor===f.tensor.id?term.index:term?.factorTensor===f.tensor.id?term.factorIndex:-1})}
    const point=(tensor:string,i:number)=>{const f=fields.find(v=>v.tensor.id===tensor);return f&&i>=0?f.group.localToWorld(new T.Vector3(...f.positions[i])):null}
    if(step.kind!=='Linear'&&products&&!calc.elementwise){const nextKey=step.id+':'+index;if(activeKey!==nextKey){activeKey=nextKey;activeFabric?.dispose();const links:Connection[]=[];for(let i=0;i<terms.length;i++){const t=terms[i],src=point(t.tensor,t.index),pr=products.group.localToWorld(new T.Vector3(...products.positions[i])),to=prefix?prefix.group.localToWorld(new T.Vector3(...prefix.positions[i])):target;if(src)links.push({from:src.toArray() as Position3,to:pr.toArray() as Position3,weight:t.factor??1,output:index,term:i});const w=t.factorTensor?point(t.factorTensor,t.factorIndex!):null;if(w)links.push({from:w.toArray() as Position3,to:pr.toArray() as Position3,weight:t.factor??1,output:index,term:i});links.push({from:pr.toArray() as Position3,to:to.toArray() as Position3,output:index,term:i});if(prefix&&i>0)links.push({from:prefix.group.localToWorld(new T.Vector3(...prefix.positions[i-1])).toArray() as Position3,to:to.toArray() as Position3,output:index,term:i});if(i===terms.length-1)links.push({from:to.toArray() as Position3,to:target.toArray() as Position3,output:index,term:i})}activeFabric=connectionFabric(body,links)}activeFabric?.update(phase,index,current)}
    const start=term?point(term.tensor,term.index):null;flow!.group.visible=!!start;if(start)flow!.update([start.toArray() as Position3,...products&&!biasTerm?[products.group.localToWorld(new T.Vector3(...products.positions[productIndex])).toArray() as Position3]:[],...prefix&&!biasTerm?[prefix.group.localToWorld(new T.Vector3(...prefix.positions[calc.elementwise?index:Math.min(current,prefix.positions.length-1)])).toArray() as Position3]:[],...sums&&!biasTerm?[sums.group.localToWorld(new T.Vector3(...sums.positions[index])).toArray() as Position3]:[],target.toArray() as Position3],(phase*Math.max(1,terms.length))%1)
    const factor=term?.factorTensor?point(term.factorTensor,term.factorIndex!):null;factorFlow!.group.visible=!!factor;if(factor)factorFlow!.update([factor.toArray() as Position3,...products&&!biasTerm?[products.group.localToWorld(new T.Vector3(...products.positions[productIndex])).toArray() as Position3]:[],target.toArray() as Position3],(phase*Math.max(1,terms.length))%1)
    const key=`${p.scope}:${index}`;if(traceKey!==key&&!products){traceKey=key;if(traceLines){body.remove(traceLines);traceLines.geometry.dispose();(traceLines.material as T.Material).dispose()}const points:T.Vector3[]=[];for(const t of terms){const source=point(t.tensor,t.index);if(source)points.push(source,target.clone())}traceLines=new T.LineSegments(new T.BufferGeometry().setFromPoints(points),new T.LineBasicMaterial({color:'#80b7c9',transparent:true,opacity:.12}));body.add(traceLines)}
    if(now-lastLive>120){lastLive=now;setLive({title:step.title+' · '+motionNames[operationFamily(step)],equation:scalarEquation(step,index,phase),step:step.id})}
    el.dataset.term=String(current);el.dataset.output=String(step.output.values[index])
   }
   const sceneHighlight=activeHighlight||liveStep;const highlighted=p.run.steps.find(s=>s.id===sceneHighlight)?.output.id;for(const f of fields){for(const child of f.group.children)if(child instanceof T.LineSegments&&(child.material instanceof T.LineBasicMaterial)){child.material.opacity=f.tensor.id===highlighted?1:.48;child.material.color.set(f.tensor.id===highlighted?'#effaff':f.tensor.parameter?'#ddb889':'#76bdcf')}}
   for(const f of fields){const own=p.run.steps.find(s=>s.output.id===f.tensor.id);const relevant=!p.focus||own?.group===p.focus||p.run.steps.some(s=>s.group===p.focus&&s.inputs.some(t=>t.id===f.tensor.id));f.group.visible=relevant;}for(const item of bridges)item.view.group.visible=!p.focus||item.step.group===p.focus;for(const item of fabrics)item.view.mesh.visible=!p.focus||item.step.group===p.focus;
   el.dataset.distance=camera.position.distanceTo(controls.target).toFixed(4);renderer.render(scene,camera);const used:{x:number;y:number;w:number;h:number}[]=[];const hud=el.querySelector('.ws-live-math')?.getBoundingClientRect(),hostRect=el.getBoundingClientRect();if(hud)used.push({x:hud.x-hostRect.x,y:hud.y-hostRect.y,w:hud.width,h:hud.height});
   const ordered=[...tags].sort((a,b)=>(b.priority+(b.scope.replace('layer:','')===sceneHighlight?200:0))-(a.priority+(a.scope.replace('layer:','')===sceneHighlight?200:0)));
   for(const tag of ordered){const q=tag.point.clone().project(camera),anchorX=(q.x+1)*el.clientWidth/2,anchorY=(1-q.y)*el.clientHeight/2,w=tag.el.offsetWidth||150,h=tag.el.offsetHeight||42;let x=anchorX,y=anchorY,show=false;
    if((!(p.scope==='root'||p.scope.startsWith('group:')||p.scope.startsWith('layer:'))||p.showLabels||fields.length<=6||tag.priority>0||tag.scope.replace('layer:','')===sceneHighlight)&&q.z<1&&q.z>0&&((!p.focus&&tag.priority>0)||tag.scope.replace('layer:','')===sceneHighlight||(anchorX>-10&&anchorX<el.clientWidth+10&&anchorY>0&&anchorY<el.clientHeight))){
     for(const dy of [0,-45,45,-90,90,-135,135]){for(const dx of [w/2+10,-w/2-10,0,w+20,-w-20]){const cx=Math.max(w/2+8,Math.min(el.clientWidth-w/2-8,anchorX+dx)),cy=Math.max(h/2+8,Math.min(el.clientHeight-h/2-8,anchorY+dy));if(!used.some(r=>Math.abs(cx-r.x)<(w+r.w)/2+7&&Math.abs(cy-r.y)<(h+r.h)/2+5)){x=cx;y=cy;show=true;break}}if(show)break}
    }
    tag.el.style.visibility=show?'visible':'hidden';tag.el.tabIndex=show?0:-1;tag.el.style.left=`${x}px`;tag.el.style.top=`${y}px`;tag.el.classList.toggle('is-linked',tag.scope.replace('layer:','')===sceneHighlight);tag.line.style.visibility=show?'visible':'hidden';if(show){used.push({x,y,w,h});tag.line.setAttribute('x1',String(anchorX));tag.line.setAttribute('y1',String(anchorY));tag.line.setAttribute('x2',String(x));tag.line.setAttribute('y2',String(y));}
   }
   dirty=false
  };resize();raf=requestAnimationFrame(tick)
  return()=>{if(clickTimer)clearTimeout(clickTimer);cancelAnimationFrame(raf);ro.disconnect();io.disconnect();controls.dispose();disposeGroup(scene);renderer.dispose();renderer.domElement.remove();labels.remove()}
 },[])
 return <div className="ws-scene" ref={host}>{error&&<p className="ws-scene-error" role="status">{error}</p>}{live.title&&<div className="ws-live-math"><strong>{live.title}</strong><output>{live.equation}</output>{props.scope!==live.step&&<button onClick={()=>props.onScope(live.step)}>展开本次演算</button>}</div>}{hover&&<output className="ws-cell-readout">{hover}</output>}</div>
}
