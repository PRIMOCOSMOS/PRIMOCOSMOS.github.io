import type {Run,Step,Tensor} from './engine'
import type {Position3} from '../mmhvae/crystalPrimitives'

// Inspired by TensorSpace's LayerLocator + FeatureMap: tensor planes, then ordered
// layers. Unlike its uniform sequential spacing, this layout uses the real DAG.
// No numerical tensor is sampled, resized, or replaced by an operator marker.
export function tensorLayout(t:Tensor){
 const w=t.shape.at(-1)!, h=t.shape.length>1?t.shape.at(-2)!:1;
 const count=t.values.length/(w*h), columns=Math.ceil(Math.sqrt(count)),rows=Math.ceil(count/columns),pitch=.57,gap=1.05;
 const width=columns*(w*pitch+gap)-gap,depth=rows*(h*pitch+gap)-gap;
 const centers:Position3[]=Array.from({length:count},(_,p)=>[(p%columns)*(w*pitch+gap)-(width-w*pitch)/2,0,Math.floor(p/columns)*(h*pitch+gap)-(depth-h*pitch)/2]);
 const positions:Position3[]=t.values.map((_,i)=>{const p=Math.floor(i/(w*h)),c=centers[p];return [c[0]+(i%w-(w-1)/2)*pitch,0,c[2]+(Math.floor(i/w)%h-(h-1)/2)*pitch]});
 return {positions,centers,width,depth,planeWidth:w*pitch,planeDepth:h*pitch};
}
const statistics=new Set(['norm-mean','norm-var','row-max','row-exp','row-sum']);
export function principalSteps(run:Run){return run.steps.filter(s=>!statistics.has(String(s.settings.operation)))}
export function childSteps(run:Run,step:Step){
 const result:Step[]=[],seen=new Set<string>(),byOutput=new Map(run.steps.map(s=>[s.output.id,s]));
 const visit=(s:Step)=>{for(const t of s.inputs){const p=byOutput.get(t.id);if(p&&statistics.has(String(p.settings.operation))&&!seen.has(p.id)){seen.add(p.id);visit(p);result.push(p)}}};visit(step);return [...result,step];
}
export interface ScaffoldNode {tensor:Tensor;step?:Step;position:Position3;width:number;depth:number;level:number;external:boolean}
export interface ScaffoldEdge {from:string;to:string;skip:boolean}
export function buildScaffold(run:Run,selected:Step[]=principalSteps(run)){
 const byOutput=new Map(run.steps.map(s=>[s.output.id,s])),chosen=new Set(selected.map(s=>s.id));
 const resolve=(t:Tensor):Tensor[]=>{const p=byOutput.get(t.id);return p&&!chosen.has(p.id)&&statistics.has(String(p.settings.operation))?p.inputs.flatMap(resolve):[t]};
 const nodes:ScaffoldNode[]=[],edges:ScaffoldEdge[]=[],seen=new Set<string>();
 const add=(tensor:Tensor,step?:Step)=>{if(seen.has(tensor.id))return;seen.add(tensor.id);const l=tensorLayout(tensor);nodes.push({tensor,step,position:[0,0,0],width:l.width,depth:l.depth,level:0,external:!step})};
 for(const step of selected){
  const data=[...new Map(step.inputs.flatMap(resolve).filter(t=>!t.parameter).map(t=>[t.id,t])).values()];
  for(const t of data){add(t,selected.find(s=>s.output.id===t.id));edges.push({from:t.id,to:step.output.id,skip:false})}add(step.output,step);
 }
 // Topological depth keeps Q/K/V and shortcut branches adjacent, not serialized.
 const map=new Map(nodes.map(n=>[n.tensor.id,n]));
 for(let pass=0;pass<nodes.length;pass++){let changed=false;for(const e of edges){const a=map.get(e.from)!,b=map.get(e.to)!;if(b.level<=a.level){b.level=a.level+1;changed=true}}if(!changed)break}
 const levels=Math.max(0,...nodes.map(n=>n.level))+1;
 const maxDepth=Math.max(...nodes.map(n=>n.depth),3),spacing=Math.max(2.8,Math.min(6,maxDepth*.24+1.6));
 for(let level=0;level<levels;level++){
  const row=nodes.filter(n=>n.level===level),width=row.reduce((n,v)=>n+v.width+3,0)-3;let x=-width/2;
  for(const n of row){n.position=[x+n.width/2,-level*spacing,0];x+=n.width+3}
 }
 for(const e of edges)e.skip=map.get(e.to)!.level-map.get(e.from)!.level>1;
 return {nodes,edges,levels,spacing};
}
export function stepLabel(s:Step){
 if(['BatchNorm2d','InstanceNorm2d','LayerNorm','RMSNorm','GroupNorm'].includes(s.kind))return s.kind;
 if(s.kind==='Linear')return s.title==='全连接投影'?'Linear 全连接':s.title;
 if(s.kind.startsWith('Conv'))return `${s.title} · ${s.kind}`;
 return s.title;
}
