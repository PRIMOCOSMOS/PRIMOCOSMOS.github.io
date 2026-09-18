import {useEffect,useRef,useState} from 'react'
import * as T from 'three'
import katex from 'katex'
import {ChevronLeft,ChevronRight} from 'lucide-react'
import {OrbitControls} from 'three/addons/controls/OrbitControls.js'
import {createCrystalTensor,createArrowStream,type Position3} from '../mmhvae/crystalPrimitives'
import {disposeGroup} from '../mmhvae/glyphs'
import type {Run,Tensor,Step} from './engine'
import {coords} from './engine'
import {scalarEquation} from './computation'
import {executionGraph,streamedIndex,executionCoordinates,activationKinds} from './execution'
import {activationTransfer} from './activationTransfer'
import {connectionFabric,type Connection} from './operationMotion'
import {valueExtent,numericPalette} from '../mmhvae/numericPalette'
import {buildScaffold,childSteps,principalSteps,stepLabel,tensorLayout} from './scaffold'
interface Props {run:Run;scope:string;index:number;progress:number;playing:boolean;reset:number;focus:string;highlight:string;zoom:number;pan:boolean;showLabels:boolean;onScope:(s:string)=>void;onIndex:(i:number)=>void;onProgress:(p:number)=>void;onPlaying:(p:boolean)=>void}
type Field={tensor:Tensor;crystals:ReturnType<typeof createCrystalTensor>;positions:Position3[];group:T.Group;selectionKey?:string}
/** Every coordinate remains a crystal; leading coordinates identify separate planes. */
export const tensorPositions=(t:Tensor):Position3[]=>tensorLayout(t).positions
export default function WorkScene(props:Props){const viewAction=useRef<(action:string,value?:number)=>void>(()=>{}),[following,setFollowing]=useState(false),[stageCount,setStageCount]=useState(1),[stageCursor,setStageCursor]=useState(0);const host=useRef<HTMLDivElement>(null),latest=useRef(props);latest.current=props;const [error,setError]=useState(''),[hover,setHover]=useState(''),[live,setLive]=useState({title:'',equation:'',step:''})
 useEffect(()=>{
  const el=host.current!;let renderer:T.WebGLRenderer
  try{renderer=new T.WebGLRenderer({antialias:true,powerPreference:'high-performance'})}catch{setError('无法启动 WebGL。仍可使用下方的逐项演算与完整张量表。');return}
  renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.setClearColor('#050c12');el.appendChild(renderer.domElement);renderer.domElement.tabIndex=0;renderer.domElement.setAttribute('aria-label','三维计算空间：点击模块进入，点击输出水晶选择数值，拖动旋转，滚轮缩放')
  const scene=new T.Scene(),camera=new T.PerspectiveCamera(40,1,.05,8000),controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.minDistance=2;controls.maxDistance=4000;controls.zoomToCursor=true;controls.zoomSpeed=.7;controls.screenSpacePanning=true;controls.touches.TWO=T.TOUCH.DOLLY_PAN
  scene.add(new T.AmbientLight('#c6e7f0',1.7));const light=new T.DirectionalLight('#e7f8ff',3.1);light.position.set(20,35,25);scene.add(light);const rim=new T.DirectionalLight('#86bce6',2);rim.position.set(-20,8,-25);scene.add(rim)
  const labels=document.createElement('div');labels.className='ws-labels';el.appendChild(labels);const leaders=document.createElementNS('http://www.w3.org/2000/svg','svg');leaders.classList.add('ws-label-leaders');labels.appendChild(leaders)
  let body=new T.Group(),fields:Field[]=[],clicks:T.Object3D[]=[],tags:{el:HTMLButtonElement;point:T.Vector3;line:SVGLineElement;priority:number;scope:string}[]=[],raf=0,run:Run|undefined,scope='',oldReset=-1,oldFocus='',layoutFocus='',liveStep='',oldHighlight='',hoveredScope='',oldZoom=0,oldPan=false,oldLabels=false,oldIndex=-1,oldProgress=-1,phase=0,last=0,dirty=true,visible=true,center=new T.Vector3(),extent=new T.Vector3(20,20,20)
  let lastLive=0,lastEmitted=-1;
  let coordinateKey='',coordinateMap=new Map<string,Set<number>>();let neuronTensors=new Set<string>();let streamedKeys=new Map<string,number>();let stageOrder=new Map<string,number>();let transfers=new Map<string,ReturnType<typeof activationTransfer>>();let lastNotify=0,pass=0,execution:ReturnType<typeof executionGraph>,visibleSteps:Step[]=[]
  const fieldMap=new Map<string,Field>();let follow=false,framedOwner='';const fieldBounds=new Map<string,T.Box3>();let manualFrame:T.Box3|undefined;
  let fabrics:{step:Step;view:ReturnType<typeof connectionFabric>}[]=[],activeFabric:ReturnType<typeof connectionFabric>|undefined,activeKey='';
  let flow:ReturnType<typeof createArrowStream>|undefined,tween:{from:T.Vector3;to:T.Vector3;target:T.Vector3;start:number}|undefined
  scene.add(body)
  const reduced=matchMedia('(prefers-reduced-motion: reduce)')
  const fit=()=>{camera.up.set(0,1,0);
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
   const group=new T.Group();group.position.copy(origin);body.add(group);const stretch=neuronTensors.has(t.id)?3.6:1,positions=tensorPositions(t).map(p=>[p[0]*stretch,p[1],p[2]*(stretch>1?1.7:1)] as Position3),crystals=createCrystalTensor(group,t.values.length,stretch>1?.58:.36,{valueEdges:true,bodyOpacity:.3,edgeOpacity:.72,valueScale:()=>valueExtent(t.values)});crystals.update(t.values,positions,{focus:-1});const f={tensor:t,group,positions,crystals};fields.push(f);fieldMap.set(t.id,f)
   for(const mesh of [crystals.body]){mesh.userData.tensor=t.id;mesh.userData.scope=select;clicks.push(mesh)}
   const layout=tensorLayout(t),segments:T.Vector3[]=[];
   for(const c of layout.centers){const w=layout.planeWidth/2,d=layout.planeDepth/2;const corners=[[-w,-.24,-d],[w,-.24,-d],[w,-.24,d],[-w,-.24,d]];for(let i=0;i<4;i++){const a=corners[i],b=corners[(i+1)%4];segments.push(new T.Vector3((c[0]+a[0])*stretch,a[1],(c[2]+a[2])*(stretch>1?1.7:1)),new T.Vector3((c[0]+b[0])*stretch,b[1],(c[2]+b[2])*(stretch>1?1.7:1)))}}
   group.add(new T.LineSegments(new T.BufferGeometry().setFromPoints(segments),new T.LineBasicMaterial({color:t.parameter?numericPalette.relation:'#76bdcf',transparent:true,opacity:.48})));
   const b=new T.Box3().setFromObject(group);fieldBounds.set(t.id,b.clone());
   label(`${title??t.name}${layout.wrapped?' · 连续索引折行':''}\n[${t.shape.join(' × ')}]`,new T.Vector3(b.max.x+.35,origin.y+.4,origin.z),select??latest.current.scope,t.id===latest.current.run.input.id||t.id===(latest.current.run.steps.find(s=>s.id===latest.current.scope)?.output.id??latest.current.run.output.id)?100:0,formula);tags.at(-1)!.el.dataset.tensor=t.id;return b
  }
  const paint=(f:Field,selection:{focus?:number;active?:number[]})=>{const key=String(selection.focus??-1)+':'+(selection.active??[]).join(',');if(f.selectionKey===key)return;f.selectionKey=key;f.crystals.update(f.tensor.values,f.positions,selection)};
  const point=(id:string,i:number)=>{const f=fieldMap.get(id),p=f?.positions[i];return f&&p?[p[0]+f.group.position.x,p[1]+f.group.position.y,p[2]+f.group.position.z] as Position3:undefined};
  const scaleCache=new WeakMap<Tensor,number>();
  const coefficientScale=(step:Step,t:ReturnType<Step['trace']>[number])=>{
   const factor=step.inputs.find(input=>input.id===t.factorTensor)??(t.factorTensor?latest.current.run.tensors.find(input=>input.id===t.factorTensor):undefined);
   if(!factor)return 1;
   let scale=scaleCache.get(factor);if(scale===undefined){scale=valueExtent(factor.values);scaleCache.set(factor,scale)}
   const value=factor.values[t.factorIndex??0];return scale*(value?Math.abs((t.factor??value)/value):Math.abs(Number(step.settings.scale??1)));
  };
  const linksFor=(step:Step,index:number,terms=step.trace(index)):Connection[]=>{const to=point(step.output.id,index),via=transfers.get(step.id)?.positions[index];if(!to)return [];return terms.flatMap((t,j)=>{const from=point(t.tensor,t.index),weight=t.factorTensor?point(t.factorTensor,t.factorIndex!):undefined,numeric=t.factorTensor!==undefined||(t.factor!==undefined&&t.factor!==1),style={weight:numeric?t.factor:undefined,weightScale:coefficientScale(step,t),output:index,term:j,...via?{via:[via]}:{}};return [...from?[{from,to,...style}]:[],...weight?[{from:weight,to,...style}]:[]]})};
  const rebuild=()=>{
   const p=latest.current;setHover('');setLive({title:'',equation:'',step:''});lastLive=-Infinity;manualFrame=undefined;fabrics=[];activeFabric=undefined;activeKey='';hoveredScope='';fieldBounds.clear();fieldMap.clear();transfers=new Map();streamedKeys=new Map();scene.remove(body);disposeGroup(body);body=new T.Group();scene.add(body);labels.replaceChildren(leaders);leaders.replaceChildren();fields=[];clicks=[];tags=[];oldIndex=-1;pass=0;coordinateKey=''
   const box=new T.Box3(),parent=p.run.steps.find(s=>s.id===p.scope.slice(6)),atom=p.run.steps.find(s=>s.id===p.scope);
   const selected=atom?[atom]:p.scope.startsWith('layer:')&&parent?childSteps(p.run,parent):p.scope.startsWith('group:')?p.run.steps.filter(s=>s.group===p.scope.slice(6)):p.focus?p.run.steps.filter(s=>s.group===p.focus):p.run.steps;
   neuronTensors=new Set(selected.flatMap(s=>(s.kind==='Linear'||activationKinds.has(s.kind))?[s.inputs[0],s.output,...s.kind==='Linear'&&s.inputs[2]?[s.inputs[2]]:[]].filter(t=>t.shape.length<=3&&t.values.length<=256).map(t=>t.id):[]));
   execution=executionGraph(p.run,selected,true);visibleSteps=execution.run.steps;stageOrder=new Map(visibleSteps.map((s,i)=>[s.id,i]));
   const graph=buildScaffold(execution.run,visibleSteps,true),stageView=p.scope!=='root'||!!p.focus;
   // Dense weights are the complete connection fabric itself. Bias stays
   // docked to its output layer rather than becoming an unrelated tower.
   const weights=new Set(selected.filter(s=>s.kind==='Linear').map(s=>s.inputs[1].id));graph.nodes=graph.nodes.filter(n=>!weights.has(n.tensor.id));graph.edges=graph.edges.filter(e=>!weights.has(e.from));
   for(const n of graph.nodes)if(neuronTensors.has(n.tensor.id)){n.width*=3.6;n.depth*=1.7}
   for(let level=0;level<graph.levels;level++){const row=graph.nodes.filter(n=>n.level===level),main=row.find(n=>n.backbone)??row[0];if(!main)continue;main.position[0]=0;let left=-main.width/2-2,right=main.width/2+2;row.filter(n=>n!==main).forEach((n,i)=>{if(i%2){n.position[0]=left-n.width/2;left-=n.width+2}else{n.position[0]=right+n.width/2;right+=n.width+2}})}

   const heights=[0];for(let level=1;level<graph.levels;level++){const row=graph.nodes.filter(n=>n.level===level),dense=row.some(n=>n.step?.kind==='Linear'),activation=row.some(n=>activationKinds.has(n.step?.kind??''));heights[level]=heights[level-1]+Math.max(dense?5:activation?4:2.5,...row.map(n=>n.depth*.4+1.5))}for(const n of graph.nodes)n.position[1]=-heights[n.level];
   for(const source of selected){if(source.kind==='Linear'&&source.inputs[2]){const bias=graph.nodes.find(n=>n.tensor.id===source.inputs[2].id),out=graph.nodes.find(n=>n.tensor.id===source.output.id);if(bias&&out)bias.position=[out.position[0],out.position[1]+.75,out.position[2]-.7]}}
   for(const block of execution.blocks){if(!block.streamed)continue;const first=graph.nodes.find(n=>n.tensor.id===block.steps.find(s=>s.title.includes('当前'))?.output.id);if(!first)continue;const input=graph.nodes.find(n=>n.tensor.id===block.source.inputs[0]?.id),aux=graph.nodes.filter(n=>n.external&&(n.tensor.parameter||n.tensor.constant)&&block.steps.some(s=>s.inputs.some(t=>t.id===n.tensor.id)));for(const [i,n] of aux.entries()){if(n.tensor.id.endsWith(':padding')&&input)n.position=[input.position[0]+input.width/2+.65,input.position[1]-.5,input.position[2]];else n.position=[first.position[0]+(i%2?-1:1)*((first.width+n.width)/2+1.1),first.position[1]+1.2,first.position[2]]}}
   if(el.clientWidth<600&&graph.nodes.length<=24){const core=graph.nodes.filter(n=>!n.tensor.parameter&&!n.tensor.constant),params=graph.nodes.filter(n=>n.tensor.parameter||n.tensor.constant);for(const n of params){const near=core.reduce((a,b)=>Math.abs(b.position[1]-n.position[1])<Math.abs(a.position[1]-n.position[1])?b:a,core[0]);n.position=[near.position[0],near.position[1]+1.2,near.position[2]-Math.max(2,n.depth/2+near.depth/2+.5)]}}
   const outputs=new Set(graph.nodes.filter(n=>n.step&&!graph.edges.some(e=>e.from===n.tensor.id)).map(n=>n.tensor.id));
   for(const n of graph.nodes){
    const owner=String(n.step?.settings.owner??''),destination=owner||p.run.steps.find(s=>s.output.id===n.tensor.id)?.id||p.scope;
    const boundary=n.tensor.id===p.run.input.id||n.external&&p.run.steps.some(s=>s.output.id===n.tensor.id)||outputs.has(n.tensor.id);
    const name=n.step?(n.step.kind==='Linear'?`${n.step.group} · 全连接`:n.step.title):n.tensor.parameter?`参数 · ${n.tensor.name}`:n.tensor.constant?`常量 · ${n.tensor.name}`:stageView?`阶段输入 · ${n.tensor.name}`:n.tensor.id===p.run.input.id?'输入 X':n.tensor.name;
    box.union(field(n.tensor,new T.Vector3(...n.position),destination,name,n.step?.formula));if(boundary)tags.at(-1)!.priority=100;else if(n.step?.kind==='Linear'&&selected.length<=12)tags.at(-1)!.priority=30;
   }
   body.updateMatrixWorld(true);
   for(const step of visibleSteps)if(activationKinds.has(step.kind)){const input=fieldMap.get(step.inputs[0].id),output=fieldMap.get(step.output.id);if(input&&output){const transfer=activationTransfer(body,step,input.positions.map((_,i)=>point(input.tensor.id,i)!),output.positions.map((_,i)=>point(output.tensor.id,i)!));transfers.set(step.id,transfer);clicks.push(transfer.body)}}
   // Small and dense layers retain every coordinate edge. Large contractions
   // use an exact, output-by-output stream, with the entire tensor still present.
   for(const block of execution.blocks){if(block.streamed)streamedKeys.set(block.source.id,0);for(const step of block.steps){let count=0;const links:Connection[]=[];for(let i=0;i<(block.streamed&&step.output.id===block.source.output.id?1:step.output.values.length);i++){const terms=step.trace(i);count+=terms.length;if(count>20000){links.length=0;break}links.push(...linksFor(step,i,terms))}if(links.length)fabrics.push({step,view:connectionFabric(body,links)})}}
   flow=createArrowStream(body,'#e8f8ff',.12,2);flow.group.visible=false;
   follow=graph.levels>24;framedOwner='';setFollowing(follow);setStageCount(visibleSteps.length);el.dataset.layers=String(graph.nodes.length);el.dataset.levels=String(graph.levels);el.dataset.mathStages=String(visibleSteps.length);el.dataset.weightEdges=String(selected.filter(s=>s.kind==='Linear').reduce((n,s)=>n+s.output.values.length*s.inputs[0].shape.at(-1)!,0));

   box.getCenter(center);box.getSize(extent);extent.addScalar(5);body.updateMatrixWorld(true);el.dataset.scope=p.scope;el.dataset.cells=String(fields.reduce((n,f)=>n+f.tensor.values.length,0));fit();dirty=true
  }
  const frameOwner=(owner:string)=>{const block=execution.blocks.find(b=>b.source.id===owner);if(!block)return;const bounds=new T.Box3();for(const s of block.steps){const b=fieldBounds.get(s.output.id);if(b)bounds.union(b)}const top=bounds.max.y;for(const t of block.source.inputs){const b=fieldBounds.get(t.id);if(b&&b.max.y<top+18)bounds.union(b)}if(!bounds.isEmpty()){manualFrame=bounds;bounds.getCenter(center);fit()}};
  viewAction.current=(action,value)=>{if(action==='all'){follow=false;setFollowing(false);focusStage()}else if(action==='follow'){follow=!follow;setFollowing(follow);if(follow)frameOwner(liveStep)}else if(action==='stage'&&value!==undefined){follow=true;setFollowing(true);framedOwner='';latest.current.onPlaying(false);latest.current.onProgress((value+.35)/visibleSteps.length)}};
  const focusStage=()=>{manualFrame=undefined;let bounds=new T.Box3();for(const b of fieldBounds.values())bounds.union(b);if(!bounds.isEmpty()){bounds.getCenter(center);bounds.getSize(extent);extent.addScalar(5);fit()}};
  const resize=()=>{const w=el.clientWidth,h=el.clientHeight;if(!w||!h)return;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();fit()};const ro=new ResizeObserver(resize);ro.observe(el);const io=new IntersectionObserver(([e])=>visible=e.isIntersecting);io.observe(el)
  const ray=new T.Raycaster(),pointer=new T.Vector2();let down=[0,0]
  const pick=(event:PointerEvent)=>{const r=renderer.domElement.getBoundingClientRect();pointer.set((event.clientX-r.left)/r.width*2-1,-(event.clientY-r.top)/r.height*2+1);ray.setFromCamera(pointer,camera);return ray.intersectObjects(clicks.filter(o=>{for(let n:T.Object3D|null=o;n;n=n.parent)if(!n.visible)return false;return true}),false)[0]}
  const pointerDown=(e:PointerEvent)=>{down=[e.clientX,e.clientY];tween=undefined;follow=false;setFollowing(false)}
  const selectHit=(e:PointerEvent)=>{if(Math.hypot(e.clientX-down[0],e.clientY-down[1])>6)return;const hit=pick(e);if(!hit)return;const data=hit.object.userData;if(data.scope&&data.scope!==latest.current.scope)latest.current.onScope(data.scope);else if(data.scopes)latest.current.onScope(data.scopes[hit.instanceId??0]);else if(data.tensor){const t=latest.current.run.tensors.find(t=>t.id===data.tensor)??fields.find(f=>f.tensor.id===data.tensor)?.tensor;if(!t)return;const index=hit.instanceId??0;const step=latest.current.run.steps.find(s=>s.id===latest.current.scope);if(step?.output.id===t.id)latest.current.onIndex(index);setHover(`${t.name}[${coords(index,t.shape).join(', ')}] = ${t.values[index]?.toPrecision(7)}`)}}
  let clickTimer:ReturnType<typeof setTimeout>|undefined;const pointerUp=(e:PointerEvent)=>{if(Math.hypot(e.clientX-down[0],e.clientY-down[1])>6)return;if(clickTimer)clearTimeout(clickTimer);clickTimer=setTimeout(()=>selectHit(e),230)};
  renderer.domElement.addEventListener('keydown',event=>{if(event.key==='+'||event.key==='='){camera.position.lerp(controls.target,.15);dirty=true;event.preventDefault()}else if(event.key==='-'){camera.position.sub(controls.target).multiplyScalar(1.18).add(controls.target);dirty=true;event.preventDefault()}else if(event.key==='Home'){fit();event.preventDefault()}})
  let lastHover=0;
  renderer.domElement.addEventListener('pointermove',event=>{if(event.buttons||performance.now()-lastHover<70)return;lastHover=performance.now();const hit=pick(event),next=hit?.object.userData.scope??'';if(next!==hoveredScope){hoveredScope=next;dirty=true}renderer.domElement.style.cursor=hit?'pointer':latest.current.pan?'grab':'default'});
  renderer.domElement.addEventListener('pointerleave',()=>{hoveredScope='';dirty=true});
  renderer.domElement.addEventListener('dblclick',event=>{if(clickTimer)clearTimeout(clickTimer);tween=undefined;const hit=pick(event as unknown as PointerEvent),id=hit?.object.userData.tensor,bounds=id?fieldBounds.get(id):undefined;if(bounds){manualFrame=bounds.clone();bounds.getCenter(center);fit()}});
  renderer.domElement.addEventListener('wheel',()=>{follow=false;setFollowing(false)},{passive:true});
  renderer.domElement.addEventListener('pointerdown',pointerDown);renderer.domElement.addEventListener('pointerup',pointerUp)
  controls.addEventListener('change',()=>dirty=true)
  const tick=(now:number)=>{raf=requestAnimationFrame(tick);const p=latest.current;if(run!==p.run||scope!==p.scope||layoutFocus!==p.focus){run=p.run;scope=p.scope;layoutFocus=p.focus;liveStep='';rebuild()}
   if(p.focus!==oldFocus){oldFocus=p.focus;focusStage()}if(p.reset!==oldReset){oldReset=p.reset;focusStage()}if(p.progress!==oldProgress){oldProgress=p.progress;if(p.progress!==lastEmitted)phase=p.progress;dirty=true}
   const dt=Math.min(.05,(now-last)/1000);last=now;if(!visible||document.hidden)return
   if(p.playing&&!reduced.matches){const next=phase+dt/Math.max(4,visibleSteps.length*1.1);if(next>=1){pass++;const atom=p.run.steps.find(s=>s.id===p.scope);if(atom)p.onIndex((p.index+1)%atom.output.values.length)}phase=next%1;dirty=true;if(now-lastNotify>120){lastNotify=now;lastEmitted=phase;p.onProgress(phase)}}
   if(tween){const f=Math.min(1,(now-tween.start)/650),k=1-(1-f)**4;camera.position.lerpVectors(tween.from,tween.to,k);controls.target.lerpVectors(tween.target,center,k);if(f===1)tween=undefined;dirty=true}controls.update()
   if(p.zoom!==oldZoom){tween=undefined;const delta=p.zoom-oldZoom;oldZoom=p.zoom;camera.position.sub(controls.target).multiplyScalar(Math.exp(-delta*.23)).clampLength(controls.minDistance,controls.maxDistance).add(controls.target);dirty=true}
   if(p.pan!==oldPan){oldPan=p.pan;controls.mouseButtons.LEFT=p.pan?T.MOUSE.PAN:T.MOUSE.ROTATE}
   if(p.showLabels!==oldLabels){oldLabels=p.showLabels;dirty=true}
   if(p.highlight!==oldHighlight){oldHighlight=p.highlight;dirty=true}
   if(p.index!==oldIndex){oldIndex=p.index;for(const f of fields)f.selectionKey=undefined;dirty=true}
   if(!dirty)return
   const activeHighlight=p.highlight||hoveredScope.replace('layer:','');
   const timeline=activeHighlight?visibleSteps.filter(s=>s.settings.owner===activeHighlight):visibleSteps,cursor=phase*timeline.length,active=timeline[Math.min(timeline.length-1,Math.floor(cursor))];
   if(active){
    const owner=String(active.settings.owner),block=execution.blocks.find(b=>b.source.id===owner)!,atom=p.run.steps.some(s=>s.id===p.scope),outputIndex=streamedIndex(block.source.output.values.length,atom?0:pass,p.index,p.playing),local=cursor%1;
    block.select(outputIndex);if(block.streamed&&streamedKeys.get(owner)!==outputIndex){streamedKeys.set(owner,outputIndex);for(const item of fabrics.filter(f=>f.step.settings.owner===owner))item.view.dispose();fabrics=fabrics.filter(f=>f.step.settings.owner!==owner);for(const step of block.steps){const indices=step.output.id===block.source.output.id?[outputIndex]:step.output.values.map((_,i)=>i),links=indices.flatMap(i=>linksFor(step,i));fabrics.push({step,view:connectionFabric(body,links)})}}liveStep=owner;if(follow&&!activeHighlight&&framedOwner!==owner){framedOwner=owner;frameOwner(owner)}
    const ck=owner+':'+outputIndex;if(ck!==coordinateKey){coordinateKey=ck;coordinateMap=executionCoordinates(block,outputIndex)}
    const coordinates=[...(coordinateMap.get(active.output.id)??new Set([Math.min(outputIndex,active.output.values.length-1)]))].sort((a,b)=>a-b);
    const index=coordinates[Math.min(coordinates.length-1,Math.floor(local*coordinates.length))];
    const terms=active.trace(index),j=Math.min(terms.length-1,Math.floor(local*terms.length)),current=terms[j];
    for(const f of fields){const indices=terms.filter(t=>t.tensor===f.tensor.id&&t.index>=0).map(t=>t.index);indices.push(...terms.filter(t=>t.factorTensor===f.tensor.id).map(t=>t.factorIndex!));if(block.streamed&&block.steps.some(s=>s.output.id===f.tensor.id))f.selectionKey=undefined;paint(f,{active:indices,focus:f.tensor.id===active.output.id?index:current?.tensor===f.tensor.id?current.index:current?.factorTensor===f.tensor.id?current.factorIndex:-1})}
    const nextKey=active.id+':'+index+':'+outputIndex;
    if(nextKey!==activeKey){activeKey=nextKey;activeFabric?.dispose();activeFabric=connectionFabric(body,linksFor(active,index,terms))}
    activeFabric?.update(local,index,j,.45);for(const item of fabrics){const ordinal=stageOrder.get(item.step.id)!,distance=Math.abs(ordinal-cursor),strength=.2+.8*Math.exp(-distance*distance/3),sweep=p.playing?((phase*2+ordinal*.075)%1)*item.step.output.values.length:index;item.view.update(phase*3+ordinal*.1,sweep,p.playing?local*Math.max(1,terms.length):j,strength,visibleSteps.length<30?1:Math.max(.12,1/(1+distance*.35)))}for(const [id,transfer] of transfers)transfer.update((phase*2+(stageOrder.get(id)??0)*.07)%1);
    const from=current?point(current.tensor,current.index):undefined,to=point(active.output.id,index);flow!.group.visible=!!from&&!!to;if(from&&to)flow!.update([from,to],(local*Math.max(1,terms.length))%1);
    el.dataset.activeLayer=owner;el.dataset.activeMath=active.title;el.dataset.term=String(j);el.dataset.outputIndex=String(outputIndex);el.dataset.output=String(block.source.output.values[outputIndex]);
    if(now-lastLive>100){lastLive=now;setStageCursor(visibleSteps.indexOf(active));const format=(v:number)=>Number.isFinite(v)?v.toPrecision(4):String(v);setLive({title:active.title,equation:active.kind==='Linear'?scalarEquation(active,index,local):`输出 [${coords(outputIndex,block.source.output.shape).join(', ')}] · ${current?`读取 ${format(current.value)}${current.factor!==undefined?' × '+format(current.factor):''} → `:''}${format(active.output.values[index])}${block.streamed?' · 逐输出流式演算':''}`,step:owner})}
   }
   const sceneHighlight=activeHighlight||liveStep,highlighted=active?.output.id;for(const f of fields){for(const child of f.group.children)if(child instanceof T.LineSegments&&child.material instanceof T.LineBasicMaterial){child.material.opacity=f.tensor.id===highlighted?1:.34;child.material.color.set(f.tensor.id===highlighted?'#effaff':f.tensor.parameter?'#ddb889':'#76bdcf')}}
   el.dataset.distance=camera.position.distanceTo(controls.target).toFixed(4);renderer.render(scene,camera);const used:{x:number;y:number;w:number;h:number}[]=[],mobile=el.clientWidth<600;
   if(mobile&&highlighted){const bounds=fieldBounds.get(highlighted);if(bounds){const corners:T.Vector3[]=[];for(const x of [bounds.min.x,bounds.max.x])for(const y of [bounds.min.y,bounds.max.y])for(const z of [bounds.min.z,bounds.max.z])corners.push(new T.Vector3(x,y,z).project(camera));const xs=corners.map(q=>(q.x+1)*el.clientWidth/2),ys=corners.map(q=>(1-q.y)*el.clientHeight/2),left=Math.max(0,Math.min(...xs)),right=Math.min(el.clientWidth,Math.max(...xs)),top=Math.max(0,Math.min(...ys)),bottom=Math.min(el.clientHeight,Math.max(...ys));if(right>left&&bottom>top)used.push({x:(left+right)/2,y:(top+bottom)/2,w:right-left+10,h:bottom-top+10})}}
   const secondary=active?.inputs.find(t=>!t.parameter&&!t.constant)?.id;
   const rank=(tag:typeof tags[number])=>tag.priority+(tag.el.dataset.tensor===highlighted?400:tag.el.dataset.tensor===secondary?200:0);
   const ordered=[...tags].sort((a,b)=>rank(b)-rank(a));
   for(const tag of ordered){const q=tag.point.clone().project(camera),anchorX=(q.x+1)*el.clientWidth/2,anchorY=(1-q.y)*el.clientHeight/2,w=tag.el.offsetWidth||150,h=tag.el.offsetHeight||42;let x=anchorX,y=anchorY,show=false;
    if((!mobile||p.showLabels||tag.el.dataset.tensor===highlighted||tag.el.dataset.tensor===secondary||!!hoveredScope&&tag.scope===hoveredScope)&&(!(p.scope==='root'||p.scope.startsWith('group:')||p.scope.startsWith('layer:'))||p.showLabels||fields.length<=6||tag.priority>0||tag.scope.replace('layer:','')===sceneHighlight)&&q.z<1&&q.z>0&&((!p.focus&&tag.priority>0)||tag.scope.replace('layer:','')===sceneHighlight||(anchorX>-10&&anchorX<el.clientWidth+10&&anchorY>0&&anchorY<el.clientHeight))){
     for(const dy of [0,-45,45,-90,90,-135,135,-180,180,-225,225]){for(const dx of [w/2+10,-w/2-10,0,w+20,-w-20]){const cx=Math.max(w/2+8,Math.min(el.clientWidth-w/2-8,anchorX+dx)),cy=Math.max(h/2+8,Math.min(el.clientHeight-h/2-8,anchorY+dy));if(!used.some(r=>Math.abs(cx-r.x)<(w+r.w)/2+7&&Math.abs(cy-r.y)<(h+r.h)/2+5)){x=cx;y=cy;show=true;break}}if(show)break}
    }
    tag.el.style.visibility=show?'visible':'hidden';tag.el.tabIndex=show?0:-1;tag.el.style.left=`${x}px`;tag.el.style.top=`${y}px`;tag.el.classList.toggle('is-linked',tag.el.dataset.tensor===highlighted);tag.line.style.visibility=show?'visible':'hidden';if(show){used.push({x,y,w,h});tag.line.setAttribute('x1',String(anchorX));tag.line.setAttribute('y1',String(anchorY));tag.line.setAttribute('x2',String(x));tag.line.setAttribute('y2',String(y));}
   }
   dirty=false
  };resize();raf=requestAnimationFrame(tick)
  return()=>{if(clickTimer)clearTimeout(clickTimer);cancelAnimationFrame(raf);ro.disconnect();io.disconnect();controls.dispose();disposeGroup(scene);renderer.dispose();renderer.forceContextLoss();renderer.domElement.remove();labels.remove()}
 },[])
 return <div className="ws-scene-container"><div className="ws-scene" ref={host}>{error&&<p className="ws-scene-error" role="status">{error}</p>}{hover&&<output className="ws-cell-readout">{hover}</output>}</div>{live.title&&<div className="ws-live-math"><div className="ws-execution-nav"><button onClick={()=>viewAction.current('stage',Math.max(0,stageCursor-1))} disabled={stageCursor===0} aria-label="上一个数学步骤"><ChevronLeft size={14}/></button><label>计算步骤 {stageCursor+1} / {stageCount}<input aria-label="浏览全部计算步骤" type="range" min={0} max={stageCount-1} value={stageCursor} onChange={e=>viewAction.current('stage',Number(e.target.value))}/></label><button onClick={()=>viewAction.current('stage',Math.min(stageCount-1,stageCursor+1))} disabled={stageCursor===stageCount-1} aria-label="下一个数学步骤"><ChevronRight size={14}/></button><button aria-pressed={following} onClick={()=>viewAction.current('follow')}>{following?'停止跟随':'跟随计算'}</button><button onClick={()=>viewAction.current('all')}>适配全图</button></div><strong>{live.title}</strong><output>{live.equation}</output>{props.scope!==live.step&&<button onClick={()=>props.onScope(live.step)}>展开本次演算</button>}</div>}</div>
}
