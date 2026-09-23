import {useEffect,useRef,useState} from 'react'
import * as T from 'three'
import katex from 'katex'
import {ChevronLeft,ChevronRight} from 'lucide-react'
import {overviewGlyphBatch,type OverviewItem} from '../researchLab/overviewGlyphBatch'
import type {OverviewAppearance} from '../researchLab/overviewAppearance'
import {layoutSourceScaffold} from './sourceScaffold'
import {forwardSceneWheel} from '../mmhvae/sceneWheel'
import {OrbitControls} from 'three/addons/controls/OrbitControls.js'
import {createCrystalTensor,createArrowStream,type Position3} from '../mmhvae/crystalPrimitives'
import {disposeGroup} from '../mmhvae/glyphs'
import type {Run,Tensor,Step} from './engine'
import {coords} from './engine'
import {scalarEquation} from './computation'
import {executionGraph,streamedIndex,executionCoordinates} from './execution'
import {activationTransfer,hasTransferPlot} from './activationTransfer'
import {functionalRegions,regionSteps,contiguousRegions,type FunctionalRegion} from './functionalRegions'
import {connectionFabric,type Connection} from './operationMotion'
import {holographicLabel,sizeHolographicLabel} from '../mmhvae/holographicLabel'
import {receptiveField,receptiveFieldOverlay} from './receptiveField'
import {valueExtent,numericPalette} from '../mmhvae/numericPalette'
import {buildScaffold,childSteps,principalSteps,stepLabel,tensorLayout,stratifyScaffold} from './scaffold'
interface Props {focusOnly?:boolean;view?:'orbit'|'front'|'top';run:Run;scope:string;index:number;progress:number;playing:boolean;reset:number;focus:string;highlight:string;zoom:number;pan:boolean;labelMode:'hover'|'all'|'none';onScope:(s:string)=>void;onIndex:(i:number)=>void;onProgress:(p:number)=>void;onPlaying:(p:boolean)=>void}
type Field={tensor:Tensor;crystals?:ReturnType<typeof createCrystalTensor>;positions:Position3[];group:T.Group;selectionKey?:string}
/** Every coordinate remains a crystal; leading coordinates identify separate planes. */
export const tensorPositions=(t:Tensor):Position3[]=>tensorLayout(t).positions
export default function WorkScene(props:Props){const viewAction=useRef<(action:string,value?:number)=>void>(()=>{}),[following,setFollowing]=useState(false),[stageCount,setStageCount]=useState(1),[stageCursor,setStageCursor]=useState(0);const host=useRef<HTMLDivElement>(null),latest=useRef(props);latest.current=props;const [error,setError]=useState(''),[hover,setHover]=useState(''),[live,setLive]=useState({title:'',equation:'',step:''})
 useEffect(()=>{
  const el=host.current!;let renderer:T.WebGLRenderer
  try{renderer=new T.WebGLRenderer({antialias:true,powerPreference:'high-performance'})}catch{setError('无法启动 WebGL。仍可使用下方的逐项演算与完整张量表。');return}
  renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.setClearColor('#050c12');el.appendChild(renderer.domElement);renderer.domElement.tabIndex=0;renderer.domElement.setAttribute('aria-label','三维计算空间：点击模块进入，点击输出水晶选择数值，拖动旋转，滚轮缩放')
  const scene=new T.Scene(),camera=new T.PerspectiveCamera(40,1,.05,8000),controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.minDistance=2;controls.maxDistance=4000;controls.zoomToCursor=true;controls.zoomSpeed=.7;controls.screenSpacePanning=true;controls.touches.TWO=T.TOUCH.DOLLY_PAN
  scene.add(new T.AmbientLight('#c6e7f0',1.7));const light=new T.DirectionalLight('#e7f8ff',3.1);light.position.set(20,35,25);scene.add(light);const rim=new T.DirectionalLight('#86bce6',2);rim.position.set(-20,8,-25);scene.add(rim)
  const labels=document.createElement('div');labels.className='ws-labels';el.appendChild(labels);const leaders=document.createElementNS('http://www.w3.org/2000/svg','svg');leaders.classList.add('ws-label-leaders');labels.appendChild(leaders);forwardSceneWheel(labels,renderer.domElement)
  let overviewVisual:ReturnType<typeof overviewGlyphBatch>|undefined;
  let sourcePool:ReturnType<typeof createCrystalTensor>|undefined,sourceHighlights:ReturnType<typeof createCrystalTensor>|undefined,sourceCells:{field:Field;index:number}[]=[],oldView='';
  let body=new T.Group(),fields:Field[]=[],clicks:T.Object3D[]=[],tags:{el:HTMLButtonElement;point:T.Vector3;line:SVGLineElement;priority:number;scope:string}[]=[],raf=0,run:Run|undefined,scope='',oldReset=-1,oldFocus='',layoutFocus='',liveStep='',oldHighlight='',hoveredScope='',hoveredTensor='',oldZoom=0,oldPan=false,oldLabels='hover',oldIndex=-1,oldProgress=-1,phase=0,last=0,dirty=true,visible=true,center=new T.Vector3(),extent=new T.Vector3(20,20,20)
  let regions:FunctionalRegion[]=[],regionHits:T.Mesh[]=[],hoveredRegion='',hoverClock=0,hoverOwner='';let ownerRegion=new Map<string,string>();
  let paintedFields=new Set<string>();let lastLive=0,lastEmitted=-1,selective=false,fabricOwner='';let receptive:ReturnType<typeof receptiveFieldOverlay>|undefined;let holograms:{group:T.Group;owner:string}[]=[];
  let sourceTransferOwner='';
  let coordinateKey='',coordinateMap=new Map<string,Set<number>>();let neuronTensors=new Set<string>();let streamedKeys=new Map<string,number>();let stageOrder=new Map<string,number>();let transfers=new Map<string,ReturnType<typeof activationTransfer>>();let lastNotify=0,pass=0,execution:ReturnType<typeof executionGraph>,visibleSteps:Step[]=[]
  const fieldMap=new Map<string,Field>();let follow=false,framedOwner='',autoViewportFit=true;const fieldBounds=new Map<string,T.Box3>();let manualFrame:T.Box3|undefined;
  let fabrics:{step:Step;view:ReturnType<typeof connectionFabric>}[]=[],activeFabric:ReturnType<typeof connectionFabric>|undefined,activeKey='';
  let flow:ReturnType<typeof createArrowStream>|undefined,tween:{from:T.Vector3;to:T.Vector3;target:T.Vector3;start:number}|undefined
  scene.add(body)
  const reduced=matchMedia('(prefers-reduced-motion: reduce)')
  const fit=()=>{camera.up.set(0,1,0);
   const direction=(latest.current.view==='front'?new T.Vector3(0,.12,2):latest.current.view==='top'?new T.Vector3(.05,2,.1):new T.Vector3(.5,.55,2)).normalize(),right=new T.Vector3().crossVectors(camera.up,direction).normalize(),up=new T.Vector3().crossVectors(direction,right).normalize();
   // Focus uses a local stage layout with boundary tensor ports, so framing also includes its inputs.

   const boxes=manualFrame?[manualFrame]:[...fieldBounds].map(([,b])=>b),corners:T.Vector3[]=[];
   for(const b of boxes)for(const x of [b.min.x,b.max.x])for(const y of [b.min.y,b.max.y])for(const z of [b.min.z,b.max.z])corners.push(new T.Vector3(x,y,z).sub(center));
   const tan=Math.tan(T.MathUtils.degToRad(camera.fov/2)),usableX=Math.max(.52,1-140/el.clientWidth),usableY=Math.max(.55,1-((latest.current.scope==='root'||latest.current.scope.startsWith('group:')||latest.current.scope.startsWith('layer:'))?160:85)/el.clientHeight);
   let distance=3;for(const q of corners){const near=q.dot(direction);distance=Math.max(distance,near+Math.abs(q.dot(right))/(tan*camera.aspect*usableX),near+Math.abs(q.dot(up))/(tan*usableY))}
   const to=center.clone().addScaledVector(direction,distance*1.05+1);

   if(reduced.matches){camera.position.copy(to);controls.target.copy(center);controls.update()}else tween={from:camera.position.clone(),to,target:controls.target.clone(),start:performance.now()};dirty=true}

  const activateHover=(owner:string,region:string)=>{
   const changed=region!==hoveredRegion||owner&&owner!==hoverOwner;
   hoveredRegion=region;hoverOwner=owner;if(region)tween=undefined;
   if(changed&&region){const local=visibleSteps.filter(s=>ownerRegion.get(String(s.settings.owner))===region);const at=local.findIndex(s=>s.settings.owner===owner);hoverClock=(Math.max(0,at)+.18)/Math.max(1,local.length);dirty=true}
   if(!region)dirty=true;
  };
  const label=(text:string,point:T.Vector3,select:string,priority=0,formula?:string)=>{const button=document.createElement('button');button.className='ws-space-label';button.textContent=text;if(formula&&!latest.current.run.sourceGraph){const math=document.createElement('span');math.className='ws-label-formula';math.innerHTML=katex.renderToString(formula,{throwOnError:false,trust:false});button.appendChild(math)}if(formula)button.dataset.formula=formula;button.title=text;button.dataset.scope=select;button.onpointerenter=()=>{if(selective)activateHover(select,ownerRegion.get(select)??'');hoveredScope=select;hoveredTensor=button.dataset.tensor??'';dirty=true};button.onpointerleave=()=>{hoveredScope='';hoveredTensor='';activateHover('','');dirty=true};button.onclick=()=>chooseScope(select);labels.appendChild(button);const line=document.createElementNS('http://www.w3.org/2000/svg','line');leaders.appendChild(line);tags.push({el:button,point,line,priority,scope:select})}

  const field=(t:Tensor,origin:T.Vector3,select?:string,title?:string,formula?:string)=>{
   const group=new T.Group();group.position.copy(origin);body.add(group);const stretch=neuronTensors.has(t.id)?3.6:1,positions=tensorPositions(t).map(p=>[p[0]*stretch,p[1],p[2]*(stretch>1?1.7:1)] as Position3),crystals=t.window?undefined:createCrystalTensor(group,t.values.length,stretch>1?.58:.36,{valueEdges:true,bodyOpacity:t.shape.length===5?.075:.3,edgeOpacity:t.shape.length===5?.23:.72,...t.window?{}:{valueScale:()=>valueExtent(t.values)}});crystals?.update(t.values,positions,{focus:-1});const f={tensor:t,group,positions,crystals};fields.push(f);fieldMap.set(t.id,f)
   group.userData.tensor=t.id;group.userData.scope=select;for(const mesh of crystals?[crystals.body]:[]){mesh.userData.tensor=t.id;mesh.userData.scope=select;clicks.push(mesh)}
   const layout=tensorLayout(t),segments:T.Vector3[]=[];
   for(const c of layout.centers){const w=layout.planeWidth/2,d=layout.planeDepth/2;const y=layout.volume?-layout.height!/2:-.24,corners=[[-w,y,-d],[w,y,-d],[w,y,d],[-w,y,d]];for(let i=0;i<4;i++){const a=corners[i],b=corners[(i+1)%4];segments.push(new T.Vector3((c[0]+a[0])*stretch,a[1],(c[2]+a[2])*(stretch>1?1.7:1)),new T.Vector3((c[0]+b[0])*stretch,b[1],(c[2]+b[2])*(stretch>1?1.7:1)))}}
   if(layout.volume&&!t.window){const volumeEdges=new T.EdgesGeometry(new T.BoxGeometry(layout.planeWidth,layout.height!,layout.planeDepth));for(const c of layout.centers){const frame=new T.LineSegments(volumeEdges,new T.LineBasicMaterial({color:'#6facc3',transparent:true,opacity:.24}));frame.position.set(...c);group.add(frame)}}
   if(!t.window)group.add(new T.LineSegments(new T.BufferGeometry().setFromPoints(segments),new T.LineBasicMaterial({color:t.parameter?numericPalette.relation:'#76bdcf',transparent:true,opacity:.48})));
   const b=t.window?new T.Box3().setFromPoints(positions.map(p=>new T.Vector3(...p).add(origin))).expandByScalar(latest.current.run.sourceGraph?.overview?3:.22):new T.Box3().setFromObject(group);fieldBounds.set(t.id,b.clone());
   label(`${title??t.name}${layout.wrapped?' · 连续索引折行':''}\n[${t.window?.shapeLabel??t.shape.join(' × ')}]${t.window&&!latest.current.run.sourceGraph?.overview?' · '+t.window.label:''}`,new T.Vector3(b.max.x+.35,origin.y+.4,origin.z),select??latest.current.scope,t.id===latest.current.run.input.id||t.id===(latest.current.run.steps.find(s=>s.id===latest.current.scope)?.output.id??latest.current.run.output.id)?100:0,formula);tags.at(-1)!.el.dataset.tensor=t.id;if(latest.current.run.sourceGraph?.overview){const appearance=latest.current.run.steps.find(s=>s.output.id===t.id)?.settings.appearance as OverviewAppearance|undefined;if(appearance){tags.at(-1)!.el.style.borderColor=appearance.color;tags.at(-1)!.el.dataset.glyph=appearance.glyph}}return b
  }
  const paint=(f:Field,selection:{focus?:number;active?:number[]})=>{const key=String(selection.focus??-1)+':'+(selection.active??[]).join(',');if(f.selectionKey===key)return;f.selectionKey=key;f.crystals?.update(f.tensor.values,f.positions,selection)};
  const point=(id:string,i:number)=>{const f=fieldMap.get(id),p=f?.positions[i];return f&&p?[p[0]+f.group.position.x,p[1]+f.group.position.y,p[2]+f.group.position.z] as Position3:undefined};
  const scaleCache=new WeakMap<Tensor,number>();
  const tensorScale=(t:Tensor)=>{let scale=scaleCache.get(t);if(scale===undefined){scale=valueExtent(t.values);scaleCache.set(t,scale)}return scale};
  const coefficientScale=(step:Step,t:ReturnType<Step['trace']>[number])=>{
   const factor=step.inputs.find(input=>input.id===t.factorTensor)??(t.factorTensor?latest.current.run.tensors.find(input=>input.id===t.factorTensor):undefined);
   if(!factor)return 1;
   let scale=scaleCache.get(factor);if(scale===undefined){scale=valueExtent(factor.values);scaleCache.set(factor,scale)}
   const value=factor.values[t.factorIndex??0];return scale*(value?Math.abs((t.factor??value)/value):Math.abs(Number(step.settings.scale??1)));
  };
  const linksFor=(step:Step,index:number,terms=step.trace(index)):Connection[]=>{const to=point(step.output.id,index),via=transfers.get(step.id)?.positions[index];if(!to)return [];return terms.flatMap((t,j)=>{const from=point(t.tensor,t.index),weight=t.factorTensor?point(t.factorTensor,t.factorIndex!):undefined,numeric=t.factorTensor!==undefined||(t.factor!==undefined&&t.factor!==1),style={weight:numeric?(Number.isFinite(t.factor)?t.factor:Number.isFinite(step.inputs.find(input=>input.id===t.factorTensor)?.values[t.factorIndex??-1])?step.inputs.find(input=>input.id===t.factorTensor)!.values[t.factorIndex??-1]:undefined):undefined,weightScale:coefficientScale(step,t),output:index,term:j,...via?{via:[via]}:{}};return [...from?[{from,to,...style}]:[],...weight?[{from:weight,to,...style}]:[]]})};
  const rebuild=()=>{
   const p=latest.current;autoViewportFit=true;overviewVisual=undefined;sourcePool=undefined;sourceHighlights=undefined;coordinateKey='';coordinateMap.clear();sourceTransferOwner='';selective=!!p.run.sourceGraph||p.run.steps.some(s=>s.settings.transposeB!==undefined);fabricOwner='';paintedFields.clear();regions=[];regionHits=[];ownerRegion.clear();hoveredRegion='';hoverClock=0;holograms=[];setHover('');setLive({title:'',equation:'',step:''});lastLive=-Infinity;manualFrame=undefined;fabrics=[];activeFabric=undefined;activeKey='';hoveredScope='';hoveredTensor='';fieldBounds.clear();fieldMap.clear();transfers=new Map();streamedKeys=new Map();scene.remove(body);disposeGroup(body);body=new T.Group();scene.add(body);labels.replaceChildren(leaders);leaders.replaceChildren();fields=[];clicks=[];tags=[];oldIndex=-1;pass=0;coordinateKey=''
   const box=new T.Box3(),parent=p.run.steps.find(s=>s.id===p.scope.slice(6)),atom=p.run.steps.find(s=>s.id===p.scope);
   const regional=regionSteps(p.run,p.scope);const selected=p.focusOnly?p.run.steps:regional?p.run.steps.filter(s=>regional.steps.includes(s.id)):atom?[atom]:p.scope.startsWith('layer:')&&parent?childSteps(p.run,parent):p.scope.startsWith('group:')?p.run.steps.filter(s=>s.group===p.scope.slice(6)):p.focus?p.run.steps.filter(s=>s.group===p.focus):p.run.steps;
   regions=functionalRegions(p.run,selected);if(regional&&regions.length===1)regions[0]={...regions[0],id:p.scope,title:regional.title};ownerRegion=new Map(regions.flatMap(r=>r.steps.map(id=>[id,r.id])));
   neuronTensors=new Set((p.run.sourceGraph?[]:selected).flatMap(s=>(s.kind==='Linear'||hasTransferPlot(s))?[s.inputs[0],s.output,...s.kind==='Linear'&&s.inputs[2]?[s.inputs[2]]:[]].filter(t=>t.shape.length<=3&&t.values.length<=256).map(t=>t.id):[]));
   execution=executionGraph(p.run,selected,true);visibleSteps=execution.run.steps;stageOrder=new Map(visibleSteps.map((s,i)=>[s.id,i]));
   const graph=buildScaffold(execution.run,visibleSteps,true),stageView=!p.focusOnly&&(p.scope!=='root'||!!p.focus);
   // Dense weights are the complete connection fabric itself. Bias stays
   // docked to its output layer rather than becoming an unrelated tower.
   const weights=new Set((p.run.sourceGraph?[]:selected).filter(s=>s.kind==='Linear').map(s=>s.inputs[1].id));graph.nodes=graph.nodes.filter(n=>!weights.has(n.tensor.id));graph.edges=graph.edges.filter(e=>!weights.has(e.from));
   for(const n of graph.nodes)if(neuronTensors.has(n.tensor.id)){n.width*=3.6;n.depth*=1.7}
   if(p.run.sourceGraph?.overview){for(const [i,n] of graph.nodes.entries()){const position=n.step?.settings.overviewPosition as Position3|undefined;n.position=position?[position[0]*2,position[1]*2,position[2]*2]:[(i%4-1.5)*14,-Math.floor(i/4)*14,0];n.width=2;n.depth=2}graph.levels=12}else if(p.run.sourceGraph)layoutSourceScaffold(graph,p.run);else stratifyScaffold(graph);
   el.style.setProperty('--ws-scene-height',`${Math.max(580,Math.min(1100,graph.levels*36))}px`);
   const outputs=new Set(graph.nodes.filter(n=>n.step&&!graph.edges.some(e=>e.from===n.tensor.id)).map(n=>n.tensor.id));
   for(const n of graph.nodes){
    const owner=String(n.step?.settings.owner??''),destination=owner||p.run.steps.find(s=>s.output.id===n.tensor.id)?.id||selected.find(s=>s.inputs.some(t=>t.id===n.tensor.id))?.id||p.scope;
    const boundary=n.tensor.id===p.run.input.id||n.external&&p.run.steps.some(s=>s.output.id===n.tensor.id)||outputs.has(n.tensor.id);
    const name=n.step?(n.step.kind==='Linear'?`${n.step.group} · 全连接`:n.step.title):n.tensor.parameter?`参数 · ${n.tensor.name}`:n.tensor.constant?`常量 · ${n.tensor.name}`:stageView?`阶段输入 · ${n.tensor.name}`:n.tensor.id===p.run.input.id?'输入 X':n.tensor.name;
    box.union(field(n.tensor,new T.Vector3(...n.position),destination,name,n.step?.formula));if(boundary)tags.at(-1)!.priority=100;else if(n.step?.kind==='Linear'&&selected.length<=12)tags.at(-1)!.priority=30;
   }
   if(selective){const boxes=regions.map(region=>{const b=new T.Box3();for(const n of graph.nodes){const owner=String(n.step?.settings.owner??n.step?.id??'');if(region.steps.includes(owner)||region.steps.some(id=>p.run.steps.find(s=>s.id===id)?.output.id===n.tensor.id)){const bounds=fieldBounds.get(n.tensor.id);if(bounds)b.union(bounds)}}return {id:region.id,min:b.min.toArray() as Position3,max:b.max.toArray() as Position3}}).filter(b=>b.min.every(Number.isFinite));
    const areas=contiguousRegions(boxes);if(areas.length){areas[0].max[1]=Math.max(areas[0].max[1],box.max.y+1.2);areas[areas.length-1].min[1]=Math.min(areas.at(-1)!.min[1],box.min.y-1.2)}for(const area of areas){const b=new T.Box3(new T.Vector3(...area.min),new T.Vector3(...area.max)),hit=new T.Mesh(new T.BoxGeometry(...b.getSize(new T.Vector3()).toArray()),new T.MeshBasicMaterial({transparent:true,opacity:0,depthWrite:false}));hit.position.copy(b.getCenter(new T.Vector3()));hit.userData.region=area.id;hit.visible=false;body.add(hit);regionHits.push(hit)}
   }
   receptive=receptiveFieldOverlay(body);
   const titled=new Set<string>();for(const n of graph.nodes){const source=selected.find(s=>s.output.id===n.tensor.id);if(!source||titled.has(source.group))continue;titled.add(source.group);const hologram=holographicLabel(String(source.settings.groupTitle??source.group),n.tensor.shape.length===5?'N × C × D × H × W · '+n.tensor.shape.join(' × '):n.tensor.shape.join(' × '),5);hologram.position.set(n.position[0]+n.width/2+1,n.position[1]+.6,n.position[2]);body.add(hologram);holograms.push({group:hologram,owner:source.id});}
   const inputNode=graph.nodes.find(n=>n.tensor.id===p.run.input.id);if(inputNode){const h=holographicLabel(inputNode.tensor.shape.length===5?'输入体 · D / H / W':inputNode.tensor.shape.length>=4?'输入特征图 · H / W':'输入张量',inputNode.tensor.shape.join(' × '),5);h.position.set(inputNode.position[0]+inputNode.width/2+1,inputNode.position[1]+.6,inputNode.position[2]);body.add(h);holograms.push({group:h,owner:''});}
   body.updateMatrixWorld(true);
   for(const step of visibleSteps)if(hasTransferPlot(step)&&!p.run.sourceGraph){const input=fieldMap.get(step.inputs[0].id),output=fieldMap.get(step.output.id);if(input&&output){const transfer=activationTransfer(body,step,input.positions.map((_,i)=>point(input.tensor.id,i)!),output.positions.map((_,i)=>point(output.tensor.id,i)!));transfers.set(step.id,transfer);clicks.push(transfer.body)}}
   // Small and dense layers retain every coordinate edge. Large contractions
   // use an exact, output-by-output stream, with the entire tensor still present.
   for(const block of selective?[]:execution.blocks){if(block.streamed)streamedKeys.set(block.source.id,0);for(const step of block.steps){let count=0;const links:Connection[]=[];for(let i=0;i<(block.streamed&&step.output.id===block.source.output.id?1:step.output.values.length);i++){const terms=step.trace(i);count+=terms.length;if(count>20000){links.length=0;break}links.push(...linksFor(step,i,terms))}if(links.length)fabrics.push({step,view:connectionFabric(body,links)})}}
   if(p.run.sourceGraph){
    sourceCells=[];const positions:Position3[]=[],values:number[]=[],framePoints:number[]=[];
    for(const f of fields){f.positions.forEach((v,index)=>{positions.push([v[0]+f.group.position.x,v[1]+f.group.position.y,v[2]+f.group.position.z]);values.push(f.tensor.values[index]/tensorScale(f.tensor));sourceCells.push({field:f,index})});const b=fieldBounds.get(f.tensor.id)!;const v=[[b.min.x,b.min.y,b.min.z],[b.max.x,b.min.y,b.min.z],[b.max.x,b.min.y,b.max.z],[b.min.x,b.min.y,b.max.z]];for(let j=0;j<4;j++)framePoints.push(...v[j],...v[(j+1)%4]);
     for(const child of f.group.children)if(child instanceof T.LineSegments){const a=child.geometry.getAttribute('position');for(let i=0;i<a.count;i++){const v=new T.Vector3(a.getX(i),a.getY(i),a.getZ(i));child.localToWorld(v);framePoints.push(v.x,v.y,v.z)}child.visible=false}
    }
    if(p.run.sourceGraph.overview){
     const items:OverviewItem[]=graph.nodes.filter(n=>n.step).map(n=>({id:n.step!.id,position:n.position,...n.step!.settings.appearance as OverviewAppearance}));overviewVisual=overviewGlyphBatch(body,items);el.dataset.overviewGlyphs=JSON.stringify(items.map(i=>({id:i.id,glyph:i.glyph,color:i.color})));clicks=[];
    }else{
     sourcePool=createCrystalTensor(body,positions.length,.36,{valueEdges:true,bodyOpacity:.18,edgeOpacity:.5,valueScale:1});sourcePool.update(values,positions,-1);sourceHighlights=createCrystalTensor(body,4096,.38,{valueEdges:true,bodyOpacity:.65,edgeOpacity:.95,valueScale:1});sourcePool.body.userData.sourcePool=true;clicks=[sourcePool.body];el.dataset.overviewGlyphs='[]';
    }
    el.dataset.numericCells=String(fields.reduce((count,f)=>count+f.tensor.values.filter(Number.isFinite).length,0));
    const geometry=new T.BufferGeometry();geometry.setAttribute('position',new T.Float32BufferAttribute(framePoints,3));if(!p.run.sourceGraph.overview)body.add(new T.LineSegments(geometry,new T.LineBasicMaterial({color:'#76bdcf',transparent:true,opacity:.3})));else geometry.dispose();
    // Sparse module-to-module backbone is always visible; arithmetic fabrics
    // are instantiated for the current complete function only.
    const path:T.Vector3[]=[],pathColors:number[]=[];for(const edge of graph.edges){const a=fieldMap.get(edge.from),b=fieldMap.get(edge.to);if(a&&b&&!a.tensor.parameter){path.push(a.group.position.clone(),b.group.position.clone());const color=new T.Color(p.run.sourceGraph.overview?(p.run.steps.find(s=>s.output.id===a.tensor.id)?.settings.appearance as OverviewAppearance)?.color??'#75a9ba':'#75a9ba');pathColors.push(...color.toArray(),...color.toArray())}}
    const pathGeometry=new T.BufferGeometry().setFromPoints(path);pathGeometry.setAttribute('color',new T.Float32BufferAttribute(pathColors,3));body.add(new T.LineSegments(pathGeometry,new T.LineBasicMaterial({vertexColors:true,transparent:true,opacity:p.run.sourceGraph.overview?.4:.14})));
   }
   flow=createArrowStream(body,'#e8f8ff',.12,2);flow.group.visible=false;
   follow=p.run.sourceGraph?!p.run.sourceGraph.overview:graph.levels>80;framedOwner='';setFollowing(follow);setStageCount(visibleSteps.length);el.dataset.layers=String(graph.nodes.length);el.dataset.levels=String(graph.levels);el.dataset.mathStages=String(visibleSteps.length);el.dataset.weightEdges=String(selected.filter(s=>s.kind==='Linear').reduce((n,s)=>n+s.output.values.length*s.inputs[0].shape.at(-1)!,0));

   box.getCenter(center);box.getSize(extent);extent.addScalar(5);camera.far=Math.max(8000,extent.length()*20);camera.updateProjectionMatrix();controls.maxDistance=Math.max(4000,extent.length()*8);controls.minDistance=p.run.sourceGraph?.2:2;body.updateMatrixWorld(true);el.dataset.scope=p.scope;el.dataset.cells=String(fields.reduce((n,f)=>n+f.tensor.values.length,0));fit();dirty=true
  }
  const frameOwner=(owner:string)=>{const block=execution.blocks.find(b=>b.source.id===owner);if(!block)return;const bounds=new T.Box3();for(const s of block.steps){const b=fieldBounds.get(s.output.id);if(b)bounds.union(b)}const top=bounds.max.y;for(const t of block.source.inputs){const b=fieldBounds.get(t.id);if(b&&b.max.y<top+18)bounds.union(b)}if(!bounds.isEmpty()){manualFrame=bounds;bounds.getCenter(center);fit()}};
  const sourceBounds=(id:string)=>{const region=regions.find(r=>r.id===id),ids=new Set(latest.current.run.sourceGraph?.paths[id]??region?.steps??[id]),bounds=new T.Box3();for(const step of latest.current.run.steps)if(ids.has(step.id)){const b=fieldBounds.get(step.output.id);if(b)bounds.union(b)}return bounds};
  const chooseScope=(id:string)=>{if(latest.current.focusOnly){follow=false;setFollowing(false);const bounds=sourceBounds(id);if(!bounds.isEmpty()){manualFrame=bounds;bounds.getCenter(center);fit()}const source=latest.current.run.steps.find(s=>s.id===id);latest.current.onScope(String(source?.settings.sourceId??id));}else latest.current.onScope(id)};
  viewAction.current=(action,value)=>{if(action==='all'){follow=false;setFollowing(false);focusStage()}else if(action==='follow'){follow=!follow;setFollowing(follow);if(follow)frameOwner(liveStep)}else if(action==='stage'&&value!==undefined){follow=true;setFollowing(true);framedOwner='';latest.current.onPlaying(false);latest.current.onProgress((value+.35)/visibleSteps.length)}};
  const focusStage=()=>{autoViewportFit=true;manualFrame=undefined;let bounds=new T.Box3();for(const b of fieldBounds.values())bounds.union(b);if(!bounds.isEmpty()){bounds.getCenter(center);bounds.getSize(extent);extent.addScalar(5);fit()}};
  let sized=false,lastWidth=0,lastFullscreen=document.fullscreenElement;const resize=()=>{const w=el.clientWidth,h=el.clientHeight;if(!w||!h)return;const viewportChanged=Math.abs(w-lastWidth)>2||lastFullscreen!==document.fullscreenElement;lastWidth=w;lastFullscreen=document.fullscreenElement;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();if(!sized||autoViewportFit&&viewportChanged){sized=true;fit()}else dirty=true};const ro=new ResizeObserver(resize);ro.observe(el);const io=new IntersectionObserver(([e])=>visible=e.isIntersecting);io.observe(el)
  const ray=new T.Raycaster(),pointer=new T.Vector2();let down=[0,0]
  const pick=(event:PointerEvent)=>{const r=renderer.domElement.getBoundingClientRect();pointer.set((event.clientX-r.left)/r.width*2-1,-(event.clientY-r.top)/r.height*2+1);ray.setFromCamera(pointer,camera);if(latest.current.run.sourceGraph){const owner=projectedOwner(event),candidates=fields.filter(f=>f.group.userData.scope===owner);let nearest=Infinity,result:{object:T.Object3D;instanceId:number}|undefined;for(const f of candidates)for(let i=0;i<f.positions.length;i++){const p=new T.Vector3(...f.positions[i]).add(f.group.position),d=ray.ray.distanceSqToPoint(p);if(d<nearest){nearest=d;result={object:f.group,instanceId:i}}}return result}const hit=ray.intersectObjects(clicks.filter(o=>{for(let n:T.Object3D|null=o;n;n=n.parent)if(!n.visible)return false;return true}),false)[0];if(hit?.object.userData.sourcePool){const cell=sourceCells[hit.instanceId??0];hit.object.userData.tensor=cell.field.tensor.id;hit.object.userData.scope=cell.field.group.userData.scope;hit.instanceId=cell.index}return hit}
  const regionAt=(event:PointerEvent)=>{pick(event);const direct=ray.intersectObjects(regionHits,false)[0];if(direct)return direct.object.userData.region as string;
   // At full-model zoom a long trunk can be only a few pixels wide. Preserve a
   // 28 px interaction corridor without stretching a function into another tier.
   let nearest=Infinity,region:string|undefined;for(const hit of regionHits){const bounds=new T.Box3().setFromObject(hit),center=bounds.getCenter(new T.Vector3()),radius=14*2*Math.tan(T.MathUtils.degToRad(camera.fov/2))*camera.position.distanceTo(center)/Math.max(1,el.clientHeight)/camera.zoom;bounds.min.x=Math.min(bounds.min.x,center.x-radius);bounds.max.x=Math.max(bounds.max.x,center.x+radius);bounds.min.z=Math.min(bounds.min.z,center.z-radius);bounds.max.z=Math.max(bounds.max.z,center.z+radius);const point=ray.ray.intersectBox(bounds,new T.Vector3());if(point){const distance=point.distanceTo(ray.ray.origin);if(distance<nearest){nearest=distance;region=hit.userData.region}}}return region};
  const pointerDown=(e:PointerEvent)=>{autoViewportFit=false;down=[e.clientX,e.clientY];tween=undefined;follow=false;setFollowing(false)}
  const selectHit=(e:PointerEvent)=>{if(Math.hypot(e.clientX-down[0],e.clientY-down[1])>6)return;const hit=pick(e);if(!hit){const id=regionAt(e);if(id){const r=regions.find(r=>r.id===id)!;chooseScope(r.id===latest.current.scope?r.steps[0]:r.id)}return}const data=hit.object.userData;if(data.scope&&data.scope!==latest.current.scope)chooseScope(data.scope);else if(data.scopes)chooseScope(data.scopes[hit.instanceId??0]);else if(data.tensor){const t=latest.current.run.tensors.find(t=>t.id===data.tensor)??fields.find(f=>f.tensor.id===data.tensor)?.tensor;if(!t)return;const index=hit.instanceId??0;const step=latest.current.run.steps.find(s=>s.id===latest.current.scope);if(step?.output.id===t.id)latest.current.onIndex(index);setHover(t.window?`${t.name}[${t.window.coordinates[index]?.join(', ')}]  · ${Number.isFinite(t.values[index])?'= '+t.values[index].toPrecision(6):'数值未加载'}`:`${t.name}[${coords(index,t.shape).join(', ')}] = ${t.values[index]?.toPrecision(7)}`)}}
  let clickTimer:ReturnType<typeof setTimeout>|undefined;const pointerUp=(e:PointerEvent)=>{if(Math.hypot(e.clientX-down[0],e.clientY-down[1])>6)return;if(clickTimer)clearTimeout(clickTimer);clickTimer=setTimeout(()=>selectHit(e),230)};
  renderer.domElement.addEventListener('keydown',event=>{if(['+','=','-'].includes(event.key)){autoViewportFit=false;tween=undefined;follow=false;setFollowing(false)}if(event.key==='+'||event.key==='='){camera.position.lerp(controls.target,.15);dirty=true;event.preventDefault()}else if(event.key==='-'){camera.position.sub(controls.target).multiplyScalar(1.18).add(controls.target);dirty=true;event.preventDefault()}else if(event.key==='Home'){fit();event.preventDefault()}})
  // Hit-test the projected tensor footprints first. A foreground 3D slab can
  // otherwise steal a ray from another tier under a perspective camera.
  const projectedOwner=(event:PointerEvent)=>{
   const rect=renderer.domElement.getBoundingClientRect(),x=event.clientX-rect.left,y=event.clientY-rect.top;
   let best='',distance=Infinity;
   for(const f of fields){const owner=String(f.group.userData.scope??'');if(!ownerRegion.has(owner))continue;
    const bounds=fieldBounds.get(f.tensor.id)!;const corners:T.Vector3[]=[];
    for(const a of [bounds.min.x,bounds.max.x])for(const b of [bounds.min.y,bounds.max.y])for(const c of [bounds.min.z,bounds.max.z])corners.push(new T.Vector3(a,b,c).project(camera));
    if(corners.every(q=>q.z>1||q.z<0))continue;
    const xs=corners.map(q=>(q.x+1)*rect.width/2),ys=corners.map(q=>(1-q.y)*rect.height/2),left=Math.min(...xs),right=Math.max(...xs),top=Math.min(...ys),bottom=Math.max(...ys);
    const dx=Math.max(left-x,0,x-right),dy=Math.max(top-y,0,y-bottom),center=f.group.position.clone().project(camera),d=Math.hypot(dx,dy)+(latest.current.run.sourceGraph?.overview?Math.hypot(x-(center.x+1)*rect.width/2,y-(1-center.y)*rect.height/2)*.5:Math.abs(y-(top+bottom)/2)*.08);
    if(dx<=22&&dy<=22&&d<distance){best=owner;distance=d}
   }return best;
  };
  const pointerMove=(event:PointerEvent)=>{if(event.buttons)return;
   const target=(event.target as HTMLElement).closest?.('.ws-space-label') as HTMLElement|null;
   if(target){const owner=target.dataset.scope??'';activateHover(owner,ownerRegion.get(owner)??'');hoveredScope=owner;hoveredTensor=target.dataset.tensor??'';dirty=true;return}
   if(event.target!==renderer.domElement)return;
   const hit=pick(event),next=String(hit?.object.userData.scope??projectedOwner(event)),tensor=String(hit?.object.userData.tensor??'');
   const region=selective?(ownerRegion.get(next)||regionAt(event)||''):'';
   activateHover(next,region);hoveredScope=next;hoveredTensor=tensor;dirty=true;
   renderer.domElement.style.cursor=hit||region?'pointer':latest.current.pan?'grab':'default';
  };
  el.addEventListener('pointermove',pointerMove,{capture:true});
  renderer.domElement.addEventListener('pointerleave',event=>{if((event.relatedTarget as HTMLElement)?.closest?.('.ws-space-label'))return;hoveredScope='';hoveredTensor='';activateHover('','');dirty=true});
  renderer.domElement.addEventListener('dblclick',event=>{if(clickTimer)clearTimeout(clickTimer);tween=undefined;const hit=pick(event as unknown as PointerEvent),id=hit?.object.userData.tensor,bounds=id?fieldBounds.get(id):undefined;if(bounds){manualFrame=bounds.clone();bounds.getCenter(center);fit()}});
  renderer.domElement.addEventListener('wheel',()=>{autoViewportFit=false;tween=undefined;follow=false;setFollowing(false)},{passive:true});
  renderer.domElement.addEventListener('pointerdown',pointerDown);renderer.domElement.addEventListener('pointerup',pointerUp)
  controls.addEventListener('change',()=>dirty=true)
  const tick=(now:number)=>{raf=requestAnimationFrame(tick);const p=latest.current;if(run!==p.run||!p.focusOnly&&(scope!==p.scope||layoutFocus!==p.focus)){run=p.run;scope=p.scope;layoutFocus=p.focus;liveStep='';rebuild()}
   if((p.view??'')!==oldView){oldView=p.view??'';fit()}if(p.focusOnly&&p.scope!==scope){scope=p.scope;if(p.scope==='root')focusStage();else{const bounds=sourceBounds(p.scope);if(!bounds.isEmpty()){follow=false;setFollowing(false);phase=0;manualFrame=bounds;bounds.getCenter(center);fit()}}}if(p.focus!==oldFocus){oldFocus=p.focus;if(!p.focusOnly)focusStage()}if(p.reset!==oldReset){oldReset=p.reset;focusStage()}if(p.progress!==oldProgress){oldProgress=p.progress;if(p.progress!==lastEmitted)phase=p.progress;dirty=true}
   const dt=Math.min(.05,(now-last)/1000);last=now;if(!visible||document.hidden)return
   if(p.playing&&!reduced.matches&&hoveredRegion){const count=visibleSteps.filter(s=>ownerRegion.get(String(s.settings.owner))===hoveredRegion).length;hoverClock=(hoverClock+dt/Math.max(4,count*1.1))%1;dirty=true}
   if(p.playing&&!reduced.matches&&!hoveredRegion){const count=p.focusOnly&&p.scope!=='root'?(p.run.sourceGraph?.paths[p.scope]?.length??1):visibleSteps.length;const next=phase+dt/Math.max(4,count*1.1);if(next>=1){pass++;const atom=p.run.steps.find(s=>s.id===p.scope);if(atom)p.onIndex((p.index+1)%atom.output.values.length)}phase=next%1;dirty=true;if(now-lastNotify>120){lastNotify=now;lastEmitted=phase;p.onProgress(phase)}}
   if(tween){const f=Math.min(1,(now-tween.start)/650),k=1-(1-f)**4;camera.position.lerpVectors(tween.from,tween.to,k);controls.target.lerpVectors(tween.target,center,k);if(f===1)tween=undefined;dirty=true}controls.update()
   if(p.zoom!==oldZoom){autoViewportFit=false;tween=undefined;follow=false;setFollowing(false);const delta=p.zoom-oldZoom;oldZoom=p.zoom;camera.position.sub(controls.target).multiplyScalar(Math.exp(-delta*.23)).clampLength(controls.minDistance,controls.maxDistance).add(controls.target);dirty=true}
   if(p.pan!==oldPan){oldPan=p.pan;controls.mouseButtons.LEFT=p.pan?T.MOUSE.PAN:T.MOUSE.ROTATE}
   if(p.labelMode!==oldLabels){oldLabels=p.labelMode;dirty=true}
   if(p.highlight!==oldHighlight){oldHighlight=p.highlight;dirty=true}
   if(p.index!==oldIndex){oldIndex=p.index;for(const f of fields)f.selectionKey=undefined;dirty=true}
   if(!dirty)return
   const activeHighlight=p.highlight||(!selective?hoveredScope.replace('layer:',''):'');
   const focusIds=p.focusOnly&&p.scope!=='root'?new Set(p.run.sourceGraph?.paths[p.scope]??[p.scope]):undefined;const focused=focusIds?visibleSteps.filter(s=>focusIds.has(s.id)):undefined;
   const motionPhase=hoveredRegion?hoverClock:phase;const timeline=hoveredRegion?visibleSteps.filter(s=>ownerRegion.get(String(s.settings.owner))===hoveredRegion):activeHighlight?visibleSteps.filter(s=>s.settings.owner===activeHighlight):focused?.length?focused:visibleSteps,cursor=motionPhase*timeline.length,active=timeline[Math.min(timeline.length-1,Math.floor(cursor))];
   if(active){
    const owner=String(active.settings.owner),block=execution.blocks.find(b=>b.source.id===owner)!,atom=p.run.steps.some(s=>s.id===p.scope),outputIndex=streamedIndex(block.source.output.values.length,atom?0:pass,p.run.sourceGraph?Math.floor(cursor%1*block.source.output.values.length):p.index,p.playing),local=cursor%1;
    if(p.run.sourceGraph){const select=block.source.settings.selectOutput as ((i:number)=>void)|undefined;select?.(Math.floor(local*fields.find(f=>f.tensor.id===block.source.settings.receptiveOutput)!.tensor.values.length));
     if(sourceTransferOwner!==owner){for(const transfer of transfers.values()){body.remove(transfer.group);disposeGroup(transfer.group)}transfers.clear();sourceTransferOwner=owner;if(hasTransferPlot(active)){const input=fieldMap.get(active.inputs[0]?.id),output=fieldMap.get(active.output.id);if(input&&output&&input.positions.length===output.positions.length){const t=activationTransfer(body,active,input.positions.map((_,i)=>point(input.tensor.id,i)!),output.positions.map((_,i)=>point(output.tensor.id,i)!));transfers.set(active.id,t)}}}
    }
    block.select(outputIndex);if(selective&&fabricOwner!==owner){for(const item of fabrics)item.view.dispose();fabrics=[];fabricOwner=owner;streamedKeys.delete(owner);if(!block.streamed)for(const step of block.steps){const links=step.output.values.flatMap((_,i)=>linksFor(step,i));if(links.length)fabrics.push({step,view:connectionFabric(body,links)})}}if(block.streamed&&streamedKeys.get(owner)!==outputIndex){streamedKeys.set(owner,outputIndex);for(const item of fabrics.filter(f=>f.step.settings.owner===owner))item.view.dispose();fabrics=fabrics.filter(f=>f.step.settings.owner!==owner);for(const step of block.steps){const indices=step.output.id===block.source.output.id?[outputIndex]:step.output.values.map((_,i)=>i),links=indices.flatMap(i=>linksFor(step,i));fabrics.push({step,view:connectionFabric(body,links)})}}liveStep=owner;overviewVisual?.update(owner,local);
    // A source operation is a coherent calculation (e.g. convolution's
    // products, reduction and writeback), not the entire encoder tower.
    const frameKey=p.run.sourceGraph?String(active.settings.sourceId):owner;
    if(follow&&!hoveredRegion&&!activeHighlight&&framedOwner!==frameKey){framedOwner=frameKey;if(p.run.sourceGraph){const steps=p.run.steps.filter(s=>s.settings.sourceId===active.settings.sourceId),b=new T.Box3();for(const s of steps){const box=fieldBounds.get(s.output.id);if(box)b.union(box)}const top=b.max.y,bottom=b.min.y;for(const s of steps)for(const t of s.inputs){const box=fieldBounds.get(t.id);if(box&&box.max.y<top+18&&box.min.y>bottom-18)b.union(box)}if(!b.isEmpty()){manualFrame=b;b.getCenter(center);fit()}}else frameOwner(owner)}
    const ck=owner+':'+outputIndex;if(ck!==coordinateKey){coordinateKey=ck;coordinateMap=executionCoordinates(block,outputIndex)}
    const coordinates=[...(coordinateMap.get(active.output.id)??new Set([Math.min(outputIndex,active.output.values.length-1)]))].sort((a,b)=>a-b);
    const index=coordinates[Math.min(coordinates.length-1,Math.floor(local*coordinates.length))];
    receptive!.group.visible=false;const rf=!p.run.sourceGraph&&block.source.kind.startsWith('Conv')?receptiveField(block.source,outputIndex):undefined;
    const terms=active.trace(index),j=Math.min(terms.length-1,Math.floor(local*terms.length)),current=terms[j];
    const selections=new Map<string,{active:number[];focus?:number}>();
    const select=(id:string,i:number)=>{if(i<0)return;const state=selections.get(id)??{active:[]};state.active.push(i);selections.set(id,state)};
    for(const term of terms){select(term.tensor,term.index);if(term.factorTensor)select(term.factorTensor,term.factorIndex!)}
    selections.set(active.output.id,{active:[],focus:index});
    if(current){const id=current.tensor===active.output.id?'':current.tensor,state=selections.get(id);if(state)state.focus=current.index;const factor=selections.get(current.factorTensor??'');if(factor)factor.focus=current.factorIndex}
    if(p.run.sourceGraph&&block.source.settings.sourceConvolution){const input=fieldMap.get(String(block.source.settings.receptiveInput)),destination=String(block.source.settings.receptiveOutput),targetIndex=(block.source.settings.receptiveTarget as ()=>number)(),target=point(destination,targetIndex),terms=block.source.output.values.flatMap((_,i)=>block.source.trace(i)),ids=[...new Set(terms.filter(t=>t.index>=0&&t.tensor===input?.tensor.id).map(t=>t.index))];if(input&&target&&ids.length){selections.set(input.tensor.id,{active:ids,focus:ids[Math.floor(local*ids.length)]});const patches=new Map<number,Position3[]>();for(const i of ids){const channel=input.tensor.window!.coordinates[i][0];patches.set(channel,[...patches.get(channel)??[],point(input.tensor.id,i)!])}receptive!.update(owner+':'+targetIndex,[...patches.values()],target,local);el.dataset.receptiveCount=String(ids.length)}}
    if(rf){const input=fieldMap.get(block.source.inputs[0].id),target=point(block.source.output.id,outputIndex);if(input&&target){selections.set(input.tensor.id,{active:rf.indices,focus:rf.indices[Math.floor(local*rf.indices.length)]});const patches=[...rf.channels.values()].map(ids=>ids.map(i=>point(input.tensor.id,i)!));receptive!.update(owner+':'+outputIndex,patches,target,local);el.dataset.receptiveCount=String(rf.indices.length);el.dataset.receptiveOutput=rf.output.join(',');el.dataset.receptivePadding=String(rf.padding)}}else if(!block.source.settings.sourceConvolution)el.dataset.receptiveCount='0';
    const changed=new Set([...paintedFields,...selections.keys()]);if(block.streamed)for(const s of block.steps){changed.add(s.output.id);const field=fieldMap.get(s.output.id);if(field)field.selectionKey=undefined}
    for(const id of changed){const field=fieldMap.get(id);if(field)paint(field,selections.get(id)??{focus:-1})}paintedFields=new Set(selections.keys());
    if(p.run.sourceGraph&&sourceHighlights){const positions:Position3[]=[],values:number[]=[];for(const [id,state] of selections){const f=fieldMap.get(id);if(!f)continue;for(const i of new Set([...state.active,...state.focus!==undefined?[state.focus]:[]])){const q=point(id,i);if(q){positions.push(q);values.push(f.tensor.values[i]/tensorScale(f.tensor))}}}sourceHighlights.update(values,positions,{active:positions.map((_,i)=>i)});}
    el.dataset.updatedFields=String(changed.size);el.dataset.totalFields=String(fields.length);
    const nextKey=active.id+':'+index+':'+outputIndex;
    if(nextKey!==activeKey){activeKey=nextKey;activeFabric?.dispose();activeFabric=connectionFabric(body,linksFor(active,index,terms))}
    activeFabric?.update(local,index,j,.45);for(const item of fabrics){const ordinal=hoveredRegion?timeline.findIndex(s=>s.id===item.step.id):stageOrder.get(item.step.id)!,distance=Math.abs(ordinal-cursor),strength=.2+.8*Math.exp(-distance*distance/3),sweep=p.playing?((motionPhase*2+ordinal*.075)%1)*item.step.output.values.length:index;item.view.update(motionPhase*3+ordinal*.1,sweep,p.playing?local*Math.max(1,terms.length):j,strength,visibleSteps.length<30?1:Math.max(.12,1/(1+distance*.35)))}for(const [id,transfer] of transfers){transfer.motion.visible=!selective||id===active.id;if(transfer.motion.visible)transfer.update((motionPhase*2+(stageOrder.get(id)??0)*.07)%1);}
    const from=current?point(current.tensor,current.index):undefined,to=point(active.output.id,index);flow!.group.visible=!!from&&!!to;if(from&&to)flow!.update([from,to],(local*Math.max(1,terms.length))%1);
    el.dataset.activeRegion=ownerRegion.get(owner)??'';el.dataset.hoveredRegion=hoveredRegion;el.dataset.activeLayer=owner;el.dataset.activeMath=active.title;el.dataset.localPhase=String(local);el.dataset.sourceId=String(active.settings.sourceId??'');el.dataset.term=String(j);el.dataset.outputIndex=String(outputIndex);el.dataset.output=String(block.source.output.values[outputIndex]);
    if(now-lastLive>100){lastLive=now;setStageCursor(visibleSteps.indexOf(active));const format=(v:number)=>Number.isFinite(v)?v.toPrecision(4):String(v);setLive({title:active.title,equation:active.output.window?p.run.sourceGraph?.overview?`${active.settings.sourceSteps} 个数学步骤 · 点击进入计算结构`:`${active.output.name}[${active.output.window.coordinates[index]?.join(', ')}] ← ${terms.filter(t=>t.index>=0).length} 条窗口内依赖 · 完整形状 ${active.output.window.shapeLabel}`:active.kind==='Linear'?scalarEquation(active,index,local):`输出 [${coords(outputIndex,block.source.output.shape).join(', ')}] · ${current?`读取 ${format(current.value)}${current.factor!==undefined?' × '+format(current.factor):''} → `:''}${format(active.output.values[index])}${block.streamed?' · 逐输出流式演算':''}`,step:owner})}
   }
   const highlighted=active?.output.id;for(const f of fields){for(const child of f.group.children)if(child instanceof T.LineSegments&&child.material instanceof T.LineBasicMaterial){if(p.run.sourceGraph)continue;child.material.opacity=f.tensor.id===highlighted?1:.34;child.material.color.set(f.tensor.id===highlighted?'#effaff':f.tensor.parameter?'#ddb889':'#76bdcf')}}
   for(const h of holograms){h.group.visible=!p.run.sourceGraph?.overview&&p.labelMode==='hover'&&(p.run.sourceGraph?ownerRegion.get(h.owner)===ownerRegion.get(liveStep):(!selective||!h.owner||h.owner===liveStep||p.run.steps.find(s=>s.id===h.owner)?.group===p.run.steps.find(s=>s.id===liveStep)?.group));if(h.group.visible)sizeHolographicLabel(h.group,camera,el.clientHeight,el.clientWidth<600?145:220);}
   el.dataset.fields=JSON.stringify(fields.flatMap(f=>{const q=f.group.position.clone().project(camera),first=new T.Vector3(...f.positions[0]).add(f.group.position).project(camera);return q.z>0&&q.z<1?[{id:f.tensor.id,owner:f.group.userData.scope,pickX:(first.x+1)*el.clientWidth/2,pickY:(1-first.y)*el.clientHeight/2,x:(q.x+1)*el.clientWidth/2,y:(1-q.y)*el.clientHeight/2}]:[]}));
   el.dataset.regions=JSON.stringify(regionHits.map(h=>{const q=h.getWorldPosition(new T.Vector3()).project(camera);return {id:h.userData.region,x:(q.x+1)*el.clientWidth/2,y:(1-q.y)*el.clientHeight/2,size:(h.geometry as T.BoxGeometry).parameters}}));
   el.dataset.connectionMode=selective?'selected-layer':'complete';el.dataset.connectionLayers=String(new Set(fabrics.map(f=>f.step.settings.owner)).size);el.dataset.drawCalls=String(renderer.info.render.calls);
   el.dataset.distance=camera.position.distanceTo(controls.target).toFixed(4);renderer.render(scene,camera);const used:{x:number;y:number;w:number;h:number}[]=[],mobile=el.clientWidth<600;
   // Keep overview labels outside every module's hit footprint. A label
   // for a lower tier must never intercept a tap on the tower behind it.
   if(p.run.sourceGraph?.overview)for(const f of fields){const bounds=fieldBounds.get(f.tensor.id)!;const corners:T.Vector3[]=[];for(const x of [bounds.min.x,bounds.max.x])for(const y of [bounds.min.y,bounds.max.y])for(const z of [bounds.min.z,bounds.max.z])corners.push(new T.Vector3(x,y,z).project(camera));if(corners.every(q=>q.z<0||q.z>1))continue;const xs=corners.map(q=>(q.x+1)*el.clientWidth/2),ys=corners.map(q=>(1-q.y)*el.clientHeight/2),left=Math.min(...xs),right=Math.max(...xs),top=Math.min(...ys),bottom=Math.max(...ys);used.push({x:(left+right)/2,y:(top+bottom)/2,w:Math.max(14,right-left)+4,h:Math.max(14,bottom-top)+4})}
   if(mobile&&highlighted){const bounds=fieldBounds.get(highlighted);if(bounds){const corners:T.Vector3[]=[];for(const x of [bounds.min.x,bounds.max.x])for(const y of [bounds.min.y,bounds.max.y])for(const z of [bounds.min.z,bounds.max.z])corners.push(new T.Vector3(x,y,z).project(camera));const xs=corners.map(q=>(q.x+1)*el.clientWidth/2),ys=corners.map(q=>(1-q.y)*el.clientHeight/2),left=Math.max(0,Math.min(...xs)),right=Math.min(el.clientWidth,Math.max(...xs)),top=Math.max(0,Math.min(...ys)),bottom=Math.min(el.clientHeight,Math.max(...ys));if(right>left&&bottom>top)used.push({x:(left+right)/2,y:(top+bottom)/2,w:right-left+10,h:bottom-top+10})}}
   const secondary=active?.inputs.find(t=>!t.parameter&&!t.constant)?.id;
   const rank=(tag:typeof tags[number])=>tag.priority+(tag.el.dataset.tensor===highlighted?400:tag.el.dataset.tensor===secondary?200:0);
   const ordered=[...tags].sort((a,b)=>rank(b)-rank(a));
   for(const tag of ordered){const q=tag.point.clone().project(camera),anchorX=(q.x+1)*el.clientWidth/2,anchorY=(1-q.y)*el.clientHeight/2,w=tag.el.offsetWidth||150,h=tag.el.offsetHeight||42;let x=anchorX,y=anchorY,show=false;
    const requested=p.highlight?latest.current.run.steps.find(s=>s.id===p.highlight)?.output.id:undefined;
    const primaryOverview=p.run.sourceGraph?.overview&&!hoveredTensor&&(tag.priority>=100||tag.el.dataset.tensor===highlighted||tag.el.dataset.tensor===secondary||tags.indexOf(tag)%Math.max(1,Math.ceil(tags.length/8))===0);
    const showTag=p.labelMode==='all'||p.labelMode==='hover'&&(primaryOverview||tag.el.dataset.tensor===(hoveredTensor||requested));
    if(showTag&&q.z<1&&q.z>0&&anchorX>-10&&anchorX<el.clientWidth+10&&anchorY>0&&anchorY<el.clientHeight){
     for(const dy of [0,-45,45,-90,90,-135,135,-180,180,-225,225]){for(const dx of [w/2+10,-w/2-10,0,w+20,-w-20]){const cx=Math.max(w/2+8,Math.min(el.clientWidth-w/2-8,anchorX+dx)),cy=Math.max(h/2+8,Math.min(el.clientHeight-h/2-8,anchorY+dy));if(!used.some(r=>Math.abs(cx-r.x)<(w+r.w)/2+7&&Math.abs(cy-r.y)<(h+r.h)/2+5)){x=cx;y=cy;show=true;break}}if(show)break}
    }
    tag.el.style.visibility=show?'visible':'hidden';tag.el.tabIndex=show?0:-1;tag.el.style.left=`${x}px`;tag.el.style.top=`${y}px`;tag.el.classList.toggle('is-linked',tag.el.dataset.tensor===highlighted);tag.line.style.visibility=show?'visible':'hidden';if(show){if(p.run.sourceGraph&&tag.el.dataset.formula&&!tag.el.querySelector('.ws-label-formula')){const math=document.createElement('span');math.className='ws-label-formula';math.innerHTML=katex.renderToString(tag.el.dataset.formula,{throwOnError:false,trust:false});tag.el.appendChild(math)}used.push({x,y,w,h});tag.line.setAttribute('x1',String(anchorX));tag.line.setAttribute('y1',String(anchorY));tag.line.setAttribute('x2',String(x));tag.line.setAttribute('y2',String(y));}
   }
   dirty=false
  };resize();raf=requestAnimationFrame(tick)
  return()=>{if(clickTimer)clearTimeout(clickTimer);cancelAnimationFrame(raf);el.removeEventListener('pointermove',pointerMove,{capture:true});ro.disconnect();io.disconnect();controls.dispose();disposeGroup(scene);renderer.dispose();renderer.forceContextLoss();renderer.domElement.remove();labels.remove()}
 },[])
 return <div className="ws-scene-container"><div className="ws-scene" ref={host}>{error&&<p className="ws-scene-error" role="status">{error}</p>}{hover&&<output className="ws-cell-readout">{hover}</output>}</div>{live.title&&<div className="ws-live-math"><div className="ws-execution-nav"><button onClick={()=>viewAction.current('stage',Math.max(0,stageCursor-1))} disabled={stageCursor===0} aria-label="上一个数学步骤"><ChevronLeft size={14}/></button><label>计算步骤 {stageCursor+1} / {stageCount}<input aria-label="浏览全部计算步骤" type="range" min={0} max={stageCount-1} value={stageCursor} onChange={e=>viewAction.current('stage',Number(e.target.value))}/></label><button onClick={()=>viewAction.current('stage',Math.min(stageCount-1,stageCursor+1))} disabled={stageCursor===stageCount-1} aria-label="下一个数学步骤"><ChevronRight size={14}/></button><button aria-pressed={following} onClick={()=>viewAction.current('follow')}>{following?'停止跟随':'跟随计算'}</button><button onClick={()=>viewAction.current('all')}>适配全图</button></div><strong>{live.title}</strong><output>{live.equation}</output>{props.scope!==live.step&&<button onClick={()=>props.onScope(String(props.run.steps.find(s=>s.id===live.step)?.settings.sourceId??live.step))}>{props.focusOnly?'聚焦本次演算':'展开本次演算'}</button>}</div>}</div>
}
