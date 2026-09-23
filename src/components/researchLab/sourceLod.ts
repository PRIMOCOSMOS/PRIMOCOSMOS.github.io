import type {Run,Step,Tensor} from '../workstation/engine'
import type {PaperModel} from './catalog'
import {modelTopology} from '../mmhvae/topology'
import {NODE_MAP} from '../mmhvae/model'
import {overviewAppearance} from './overviewAppearance'
import {moduleName} from '../mmhvae/navigation'
const unitOf=(s:Step,model?:PaperModel,selected='root')=>{
 const path=s.settings.ancestors as string[]|undefined,id=String(s.settings.sourceId);
 if(selected!=='root'&&path?.includes(selected)){const next=path[path.indexOf(selected)+1];if(next)return next===id||next.startsWith('atom:')?`direct:${selected}`:next}
 if(!model){const lane=String(s.settings.lane);return selected==='root'&&/^(us|t2|cet1|flair)-(resnet|output|image)/.test(lane)?lane.split('-')[0]+'-output':lane}
 if(model.id==='pnp'&&selected==='root'){
  if(path?.[1]==='networks')return path[3]??path[2]??id;
  if(path?.[1]==='reconstruction')return path[2]??id;
 }
 if(path?.[1]&&['encoder','decoder'].includes(path[1])&&path[2]?.includes('/block'))return path[2];
 return path?.[1]??id;
};
/** Performance LOD is an explicit interface-level graph. A module marker is not
 * advertised as a scalar. Full mathematical tensors live in the same renderer
 * at the second level, including the function's external input interfaces. */
export function overviewRun(full:Run,model?:PaperModel,selected='root'):Run{
 const units=new Map<string,Step[]>(),producer=new Map(full.steps.map(s=>[s.output.id,s]));
 for(const s of full.steps){const id=unitOf(s,model,selected);units.set(id,[...units.get(id)??[],s])}
 const tensors=new Map<string,Tensor>(),positions=modelTopology().positions;
 for(const [id,steps] of units){const output=steps.at(-1)!.output,node=NODE_MAP.get(id),title=model?.entries[id]?.title??(node?moduleName(node):String(steps[0].settings.groupTitle??id));tensors.set(id,{id:`module:${id}`,name:title,shape:[1],values:[0],window:{shapeLabel:output.window?.shapeLabel??output.shape.join(' × '),dimensions:[1],coordinates:[[0]],positions:[[0,0,0]],label:'功能模块 · 进入后显示逐元素数学结构',symbolic:true}})}
 const overviewSteps:Step[]=[];
 for(const [id,steps] of units){const inputs=new Map<string,Tensor>();for(const s of steps)for(const t of s.inputs){const prev=producer.get(t.id),owner=prev&&unitOf(prev,model,selected);if(owner&&owner!==id&&tensors.has(owner))inputs.set(owner,tensors.get(owner)!)}const output=tensors.get(id)!,entry=model?.entries[id],children=entry?.children.map(key=>model!.entries[key].position).filter((p):p is [number,number,number]=>!!p)??[],node=NODE_MAP.get(id);
  let pos=entry?.position??(children.length?[0,1,2].map(axis=>children.reduce((n,p)=>n+p[axis],0)/children.length) as [number,number,number]:undefined)??positions.get(id);
  if(model?.id==='pnp'&&id.startsWith('reconstruction/')){const order=['init','cc','dc','cr'].indexOf(id.split('/')[1]);if(order>=0)pos=[0,42-order*17,-10]}
  overviewSteps.push({id,group:id,title:output.name,kind:'Module',inputs:[...inputs.values()],output,formula:'',detail:`${steps.length} 个已展开数学步骤`,source:steps[0].source,trace:()=>[...inputs.values()].map(t=>({tensor:t.id,index:0,value:NaN})),settings:{symbolic:true,sourceId:id,groupTitle:output.name,lane:id,overviewPosition:pos,ancestors:['root',id],sourceSteps:steps.length,stage:node?.kind,appearance:overviewAppearance(id,steps,model)}});
 }
 return {tensors:[...tensors.values()],steps:overviewSteps,input:overviewSteps[0].output,output:overviewSteps.at(-1)!.output,parameters:full.parameters,scalars:full.scalars,sourceGraph:{paths:{root:overviewSteps.map(s=>s.id),...Object.fromEntries([...units.keys()].map(id=>[id,[id]]))},warnings:full.sourceGraph!.warnings,overview:true}};
}
export function sourceSlice(full:Run,id:string):Run{
 const ids=new Set(full.sourceGraph?.paths[id]??[]),steps=full.steps.filter(s=>{if(id.startsWith('direct:')){const parent=id.slice(7),path=s.settings.ancestors as string[];return path?.[path.indexOf(parent)+1]===s.settings.sourceId}if(/^(us|t2|cet1|flair)-output$/.test(id)){const prefix=id.split('-')[0];return ids.has(s.id)||new RegExp(`^${prefix}-(resnet|output|image)`).test(String(s.settings.lane))}return ids.has(s.id)}),tensors=[...new Map(steps.flatMap(s=>[...s.inputs,s.output]).map(t=>[t.id,t])).values()];
 if(!steps.length)return full;
 return {...full,steps,tensors,input:steps[0].inputs[0]??steps[0].output,output:steps.at(-1)!.output,sourceGraph:{paths:Object.fromEntries(Object.entries(full.sourceGraph!.paths).map(([key,ids])=>[key,ids.filter(id=>steps.some(s=>s.id===id))])),warnings:full.sourceGraph!.warnings}};
}
