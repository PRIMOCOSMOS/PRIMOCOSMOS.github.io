import type {Run,Step,Tensor} from './engine'
import type {Position3} from '../mmhvae/crystalPrimitives'

// Inspired by TensorSpace's LayerLocator + FeatureMap: tensor planes, then ordered
// layers. Unlike its uniform sequential spacing, this layout uses the real DAG.
// No numerical tensor is sampled, resized, or replaced by an operator marker.
export function tensorLayout(t:Tensor){
 if(t.values.length>32&&(t.shape.length===1||t.shape.length===2&&t.shape[0]===1)){const cols=32,rows=Math.ceil(t.values.length/cols),width=cols*.57,depth=rows*.57;return {positions:t.values.map((_,i)=>[(i%cols-(cols-1)/2)*.57,0,(Math.floor(i/cols)-(rows-1)/2)*.57] as Position3),centers:[[0,0,0] as Position3],width,depth,planeWidth:width,planeDepth:depth,wrapped:true}}

 const w=t.shape.at(-1)!, h=t.shape.length>1?t.shape.at(-2)!:1;
 const count=t.values.length/(w*h), columns=Math.ceil(Math.sqrt(count)),rows=Math.ceil(count/columns),pitch=.57,gap=1.05;
 const width=columns*(w*pitch+gap)-gap,depth=rows*(h*pitch+gap)-gap;
 const centers:Position3[]=Array.from({length:count},(_,p)=>[(p%columns)*(w*pitch+gap)-(width-w*pitch)/2,0,Math.floor(p/columns)*(h*pitch+gap)-(depth-h*pitch)/2]);
 const positions:Position3[]=t.values.map((_,i)=>{const p=Math.floor(i/(w*h)),c=centers[p];return [c[0]+(i%w-(w-1)/2)*pitch,0,c[2]+(Math.floor(i/w)%h-(h-1)/2)*pitch]});
 return {positions,centers,width,depth,planeWidth:w*pitch,planeDepth:h*pitch,wrapped:false};
}
const statistics=new Set(['norm-mean','norm-var','row-max','row-exp','row-sum']);
export function principalSteps(run:Run){return run.steps.filter(s=>!statistics.has(String(s.settings.operation)))}
export function childSteps(run:Run,step:Step){
 const result:Step[]=[],seen=new Set<string>(),byOutput=new Map(run.steps.map(s=>[s.output.id,s]));
 const visit=(s:Step)=>{for(const t of s.inputs){const p=byOutput.get(t.id);if(p&&statistics.has(String(p.settings.operation))&&!seen.has(p.id)){seen.add(p.id);visit(p);result.push(p)}}};visit(step);return [...result,step];
}
export interface ScaffoldNode {tensor:Tensor;step?:Step;position:Position3;width:number;depth:number;level:number;external:boolean;backbone?:boolean}
export interface ScaffoldEdge {from:string;to:string;skip:boolean}
export function buildScaffold(run:Run,selected:Step[]=principalSteps(run)){
 const byOutput=new Map(run.steps.map(s=>[s.output.id,s])),chosen=new Set(selected.map(s=>s.id));
 const resolve=(t:Tensor):Tensor[]=>{const p=byOutput.get(t.id);return p&&!chosen.has(p.id)&&statistics.has(String(p.settings.operation))?p.inputs.flatMap(resolve):[t]};
 const nodes:ScaffoldNode[]=[],edges:ScaffoldEdge[]=[],seen=new Set<string>();
 const add=(tensor:Tensor,step?:Step)=>{if(seen.has(tensor.id))return;seen.add(tensor.id);const l=tensorLayout(tensor);nodes.push({tensor,step,position:[0,0,0],width:l.width,depth:l.depth,level:0,external:!step})};
 for(const step of selected){
  const data=[...new Map(step.inputs.flatMap(resolve).filter(t=>!t.parameter&&!t.constant).map(t=>[t.id,t])).values()];
  for(const t of data){add(t,selected.find(s=>s.output.id===t.id));edges.push({from:t.id,to:step.output.id,skip:false})}add(step.output,step);
 }
 // Topological depth keeps Q/K/V and shortcut branches adjacent, not serialized.
 const map=new Map(nodes.map(n=>[n.tensor.id,n]));
 for(let pass=0;pass<nodes.length;pass++){let changed=false;for(const e of edges){const a=map.get(e.from)!,b=map.get(e.to)!;if(b.level<=a.level){b.level=a.level+1;changed=true}}if(!changed)break}
 const levels=Math.max(0,...nodes.map(n=>n.level))+1;
 const spacing=3;
 const rowDepths=Array.from({length:levels},(_,i)=>Math.max(1,...nodes.filter(n=>n.level===i).map(n=>n.depth)));
 const heights=[0];for(let i=1;i<levels;i++)heights[i]=heights[i-1]+Math.max(2,(rowDepths[i-1]+rowDepths[i])*.32+1);
 // Anchor the longest input-to-output dependency path at x=0; auxiliary
 // branches occupy stable side lanes and never shift the central trunk.
 const backbone=new Set<string>();let cursor=selected.at(-1)?.output.id;
 while(cursor){backbone.add(cursor);const parents=edges.filter(e=>e.to===cursor).map(e=>map.get(e.from)!).sort((a,b)=>b.level-a.level||b.tensor.values.length-a.tensor.values.length);cursor=parents[0]?.tensor.id}
 for(let level=0;level<levels;level++){
  const row=nodes.filter(n=>n.level===level),main=row.find(n=>backbone.has(n.tensor.id))??row[0];if(!main)continue;
  main.position=[0,-heights[level],0];main.backbone=true;let left=-main.width/2-3,right=main.width/2+3;
  row.filter(n=>n!==main).forEach((n,i)=>{if(i%2===0){n.position=[right+n.width/2,-heights[level],0];right+=n.width+3}else{n.position=[left-n.width/2,-heights[level],0];left-=n.width+3}})
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

export function initialScope(run:Run){const steps=principalSteps(run);return steps.length===1?(childSteps(run,steps[0]).length>1?`layer:${steps[0].id}`:steps[0].id):'root'}
