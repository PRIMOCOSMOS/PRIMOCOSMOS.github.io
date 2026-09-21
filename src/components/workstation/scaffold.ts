import type {Run,Step,Tensor} from './engine'
import type {Position3} from '../mmhvae/crystalPrimitives'

// Inspired by TensorSpace's LayerLocator + FeatureMap: tensor planes, then ordered
// layers. Unlike its uniform sequential spacing, this layout uses the real DAG.
// No numerical tensor is sampled, resized, or replaced by an operator marker.
export function tensorLayout(t:Tensor){
 if(t.shape.length===5){
  const [outer,channels,d,h,w]=t.shape,pitch=.57,slicePitch=.82,count=outer*channels,[rows,columns]=exactRectangle(count),gap=1.6;
  const width=columns*(w*pitch+gap)-gap,depth=rows*(h*pitch+gap)-gap,height=(d-1)*slicePitch+pitch;
  const centers:Position3[]=Array.from({length:count},(_,i)=>[(i%columns)*(w*pitch+gap)-(width-w*pitch)/2,0,Math.floor(i/columns)*(h*pitch+gap)-(depth-h*pitch)/2]);
  const positions:Position3[]=t.values.map((_,i)=>{const c=centers[Math.floor(i/(d*h*w))];return [c[0]+(i%w-(w-1)/2)*pitch,(Math.floor(i/(h*w))%d-(d-1)/2)*slicePitch,c[2]+(Math.floor(i/w)%h-(h-1)/2)*pitch]});
  return {positions,centers,width,depth,height,planeWidth:w*pitch,planeDepth:h*pitch,wrapped:false,volume:true};
 }
 if(t.values.length>32&&(t.shape.length===1||t.shape.length===2&&t.shape[0]===1)){const [rows,cols]=exactRectangle(t.values.length),width=cols*.57,depth=rows*.57;return {positions:t.values.map((_,i)=>[(i%cols-(cols-1)/2)*.57,0,(Math.floor(i/cols)-(rows-1)/2)*.57] as Position3),centers:[[0,0,0] as Position3],width,depth,planeWidth:width,planeDepth:depth,wrapped:true}}

 const w=t.shape.at(-1)!, h=t.shape.length>1?t.shape.at(-2)!:1;
 const count=t.values.length/(w*h), [rows,columns]=exactRectangle(count),pitch=.57,gap=1.05;
 const width=columns*(w*pitch+gap)-gap,depth=rows*(h*pitch+gap)-gap;
 const centers:Position3[]=Array.from({length:count},(_,p)=>[(p%columns)*(w*pitch+gap)-(width-w*pitch)/2,0,Math.floor(p/columns)*(h*pitch+gap)-(depth-h*pitch)/2]);
 const positions:Position3[]=t.values.map((_,i)=>{const p=Math.floor(i/(w*h)),c=centers[p];return [c[0]+(i%w-(w-1)/2)*pitch,0,c[2]+(Math.floor(i/w)%h-(h-1)/2)*pitch]});
 return {positions,centers,width,depth,planeWidth:w*pitch,planeDepth:h*pitch,wrapped:false};
}
export function exactRectangle(count:number):[number,number]{let rows=Math.floor(Math.sqrt(count));while(count%rows)rows--;return [rows,count/rows]}
const statistics=new Set(['norm-mean','norm-var','row-max','row-exp','row-sum']);
export function principalSteps(run:Run){return run.steps.filter(s=>!statistics.has(String(s.settings.operation)))}
export function childSteps(run:Run,step:Step){
 const result:Step[]=[],seen=new Set<string>(),byOutput=new Map(run.steps.map(s=>[s.output.id,s]));
 const visit=(s:Step)=>{for(const t of s.inputs){const p=byOutput.get(t.id);if(p&&statistics.has(String(p.settings.operation))&&!seen.has(p.id)){seen.add(p.id);visit(p);result.push(p)}}};visit(step);return [...result,step];
}
export interface ScaffoldNode {tensor:Tensor;step?:Step;position:Position3;width:number;depth:number;height?:number;level:number;external:boolean;backbone?:boolean}
export interface ScaffoldEdge {from:string;to:string;skip:boolean}
/** A shared plane means a shared operation AND data role, not merely equal DAG depth.
 * Ordering independent lanes vertically is visual organisation, never a new edge. */
export function stratifyScaffold(graph:{nodes:ScaffoldNode[];edges:ScaffoldEdge[];levels:number}){
 const byId=new Map(graph.nodes.map(n=>[n.tensor.id,n]));
 // Keep the longest data path fixed. Delay independent branches (e.g. random
 // noise) towards their consumer instead of stretching them from the input roof.
 const trunk=new Set<string>();let cursor=[...graph.nodes].sort((a,b)=>b.level-a.level)[0];
 while(cursor){trunk.add(cursor.tensor.id);const parents=graph.edges.filter(e=>e.to===cursor.tensor.id).map(e=>byId.get(e.from)!).filter(n=>!n.tensor.parameter&&!n.tensor.constant).sort((a,b)=>b.level-a.level);cursor=parents[0]}
 for(const n of [...graph.nodes].sort((a,b)=>b.level-a.level)){n.backbone=trunk.has(n.tensor.id);if(n.backbone||n.step?.kind==='Linear')continue;const children=graph.edges.filter(e=>e.from===n.tensor.id).map(e=>byId.get(e.to)!.level);if(children.length)n.level=Math.max(n.level,Math.min(...children)-1)}
 const consumerRole=(n:ScaffoldNode)=>graph.edges.filter(e=>e.from===n.tensor.id).map(e=>{const consumer=byId.get(e.to)?.step;return `${consumer?.group??''}/${consumer?.title??''}`}).sort();
 const signature=(n:ScaffoldNode)=>JSON.stringify([
  n.tensor.parameter?'parameter':n.tensor.constant?'constant':n.step?'computed':'input',
  n.step?.kind??'',n.step?.settings.operation??'',n.step?.settings.layoutRole??n.step?.title??n.tensor.name,n.tensor.shape,
  n.tensor.parameter?consumerRole(n):null,
 ]);
 const original=[...new Set(graph.nodes.map(n=>n.level))].sort((a,b)=>a-b);
 const originalRows=new Map(original.map(depth=>[depth,graph.nodes.filter(n=>n.level===depth)]));
 let level=0,y=0,previousDepth=0,previousHeight=0;
 for(const depth of original){
  const groups=new Map<string,ScaffoldNode[]>();
  // External data precede constants/parameters, then computed outputs.
  const order=(n:ScaffoldNode)=>n.step?3:n.tensor.parameter?2:n.tensor.constant?1:0;
  for(const n of originalRows.get(depth)!.sort((a,b)=>order(a)-order(b))){const key=signature(n);groups.set(key,[...groups.get(key)??[],n])}
  for(const row of groups.values()){
   const rowDepth=Math.max(...row.map(n=>n.depth)),rowHeight=Math.max(...row.map(n=>n.height??.6)),kind=row[0].step?.kind;
   const bias=row.every(n=>n.tensor.parameter&&n.tensor.name==='b');
   if(level)y-=Math.max(bias?1.2:kind==='Linear'?4.5:kind&&['ReLU','ReLU6','LeakyReLU','Sigmoid','Tanh','SiLU','GELU','Hardswish','Hardsigmoid','Softplus'].includes(kind)?4:2.5,(previousDepth+rowDepth)*.42+(bias?.4:1.4),(previousHeight+rowHeight)/2+1.8);
   const total=row.reduce((n,item)=>n+item.width,0)+(row.length-1)*2;let x=-total/2;
   for(const n of row){n.level=level;n.position=[x+n.width/2,y,0];x+=n.width+2}
   previousDepth=rowDepth;previousHeight=rowHeight;level++;
  }
 }
 graph.levels=level;
 for(const edge of graph.edges)edge.skip=byId.get(edge.to)!.level-byId.get(edge.from)!.level>1;
 return graph;
}
export function buildScaffold(run:Run,selected:Step[]=principalSteps(run),includeParameters=false){
 const byOutput=new Map(run.steps.map(s=>[s.output.id,s])),chosen=new Set(selected.map(s=>s.id));
 const resolve=(t:Tensor):Tensor[]=>{const p=byOutput.get(t.id);return p&&!chosen.has(p.id)&&statistics.has(String(p.settings.operation))?p.inputs.flatMap(resolve):[t]};
 const nodes:ScaffoldNode[]=[],edges:ScaffoldEdge[]=[],seen=new Set<string>();
 const add=(tensor:Tensor,step?:Step)=>{if(seen.has(tensor.id))return;seen.add(tensor.id);const l=tensorLayout(tensor);nodes.push({tensor,step,position:[0,0,0],width:l.width,depth:l.depth,height:l.height??.6,level:0,external:!step})};
 for(const step of selected){
  const data=[...new Map(step.inputs.flatMap(resolve).filter(t=>includeParameters||!t.parameter&&!t.constant).map(t=>[t.id,t])).values()];
  for(const t of data){add(t,selected.find(s=>s.output.id===t.id));edges.push({from:t.id,to:step.output.id,skip:false})}add(step.output,step);
 }
 // Topological depth keeps Q/K/V and shortcut branches adjacent, not serialized.
 const map=new Map(nodes.map(n=>[n.tensor.id,n]));
 for(let pass=0;pass<nodes.length;pass++){let changed=false;for(const e of edges){const a=map.get(e.from)!,b=map.get(e.to)!;if(b.level<=a.level){b.level=a.level+1;changed=true}}if(!changed)break}
 // Parameters sit beside their first consumer, not in a detached top row.
 if(includeParameters)for(const n of nodes)if(n.external&&n.tensor.id!==run.input.id){const children=edges.filter(e=>e.from===n.tensor.id).map(e=>map.get(e.to)!.level);if(children.length)n.level=Math.max(0,Math.min(...children)-1)}
 const levels=Math.max(0,...nodes.map(n=>n.level))+1;
 const spacing=3;
 const rowDepths=Array.from({length:levels},(_,i)=>Math.max(1,...nodes.filter(n=>n.level===i).map(n=>n.depth)));
 const heights=[0];for(let i=1;i<levels;i++)heights[i]=heights[i-1]+Math.max(2,(rowDepths[i-1]+rowDepths[i])*.32+1);
 // Anchor the longest input-to-output dependency path at x=0; auxiliary
 // branches occupy stable side lanes and never shift the central trunk.
 const backbone=new Set<string>();let cursor=selected.at(-1)?.output.id;
 while(cursor){backbone.add(cursor);const parents=edges.filter(e=>e.to===cursor).map(e=>map.get(e.from)!).sort((a,b)=>Number(!!a.tensor.parameter)-Number(!!b.tensor.parameter)||b.level-a.level||b.tensor.values.length-a.tensor.values.length);cursor=parents[0]?.tensor.id}
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
