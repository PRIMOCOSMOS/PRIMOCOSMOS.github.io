import type {AnatomyGraph,AnatomyEdge,Glyph,AnatomyPart} from '../mmhvae/anatomy'
import type {MathSpec} from '../mmhvae/mathematics'
import {mathematicalGraph} from '../mmhvae/mathematics'
export type Point=[number,number,number]
export interface Entry {
 id:string;parent?:string;title:string;glyph:Glyph;shape:string;purpose:string;implementation:string;formula:string;file:string;line:number;symbol:string
 children:string[];edges:AnatomyEdge[];position?:Point;color?:string;overview?:boolean;math?:MathSpec
}
export interface PaperModel {
 id:'pnp'|'mmvae'|'ssdiff'|'metsc'|'pigment';name:string;repo:string;commit:string;entries:Record<string,Entry>;overview:string[];overviewEdges:AnatomyEdge[];sourceMode?:'paper';paperUrl?:string;training?:{title:string;detail:string}[]
 intro:string;config:string;equations:{title:string;formula:string;explanation:string}[];audit:{title:string;detail:string;file:string;line:number}[]
}
export function sourceLink(model:PaperModel,entry:Pick<Entry,'file'|'line'>){return model.sourceMode==='paper'?`${model.paperUrl}#page=${entry.line}`:`${model.repo}/blob/${model.commit}/${entry.file}#L${entry.line}`}
export function pathFor(model:PaperModel,id:string){const path:string[]=[];let next:string|undefined=id;while(next){path.unshift(next);next=model.entries[next]?.parent}return path}
export function asPart(e:Entry,role?:'input'|'output'):AnatomyPart{return {id:e.id,title:e.title,glyph:e.glyph,shape:e.shape,detail:e.implementation,sourceName:e.symbol,child:e.id,color:e.color,position:e.position,role}}
export function graphForEntry(model:PaperModel,id:string):AnatomyGraph{
 const e=model.entries[id],parent=e.parent?model.entries[e.parent]:null
 const context:AnatomyPart[]=[],contextEdges:AnatomyEdge[]=[]
 const allEdges=Object.values(model.entries).flatMap(v=>v.edges)
 const descendants=new Set(Object.keys(model.entries).filter(key=>pathFor(model,key).includes(id)))
 const inside=(key:string)=>descendants.has(key)
 const addBoundary=(link:AnatomyEdge,role:'input'|'output',endpoint:string)=>{
  const other=model.entries[role==='input'?link.from:link.to];if(!other)return
  const portId=`port:${role}:${other.id}`
  if(!context.some(p=>p.id===portId)){const part=asPart(other,role);part.id=portId;part.position=undefined;context.push(part)}
  const edge={from:role==='input'?portId:endpoint,to:role==='input'?endpoint:portId,label:link.label??'张量传递',residual:link.residual}
  if(!contextEdges.some(v=>v.from===edge.from&&v.to===edge.to))contextEdges.push(edge)
 }
 if(parent){
  for(const link of allEdges){
   if(inside(link.to)&&!inside(link.from))addBoundary(link,'input',link.to)
   if(inside(link.from)&&!inside(link.to))addBoundary(link,'output',link.from)
  }
  for(const role of ['input','output'] as const){
   if(context.some(p=>p.role===role)||(role==='input'&&id.endsWith('/spectral')))continue
   let boundary=e
   while(boundary.parent){
    const owner=model.entries[boundary.parent],edges=allEdges.filter(l=>role==='input'?l.to===boundary.id:l.from===boundary.id)
    if(edges.length){for(const link of edges)addBoundary(link,role,id);break}
    boundary=owner
   }
  }
 }
 if(e.math){
  const graph=mathematicalGraph(asPart(e),context,contextEdges,e.title,e.formula)
  graph.mathematics=e.math
  graph.parts=graph.parts.filter(p=>p.role)
  e.math.steps.forEach((step,i)=>graph.parts.unshift({id:`math-${i}`,title:step.title,glyph:'tensor',shape:e.shape,detail:step.explanation,sourceName:e.symbol,math:{...e.math!,stage:i}}))
  graph.parts.sort((a,b)=>a.role?1:b.role?-1:(a.math!.stage-b.math!.stage))
  graph.edges=e.math.steps.slice(1).map((_,i)=>({from:`math-${i}`,to:`math-${i+1}`,label:'数学推导'}))
  for(const p of context)graph.edges.push(p.role==='input'?{from:p.id,to:'math-0',label:'输入数据'}:{from:`math-${e.math.steps.length-1}`,to:p.id,label:'输出结果'})
  return graph
 }
 const subset=e.overview?model.overview.filter(key=>pathFor(model,key).includes(id)):[]
 const keys=id==='root'?model.overview:subset.length?subset:e.children
 const edges=id==='root'||subset.length?model.overviewEdges:e.edges
 const parts=keys.map(key=>asPart(model.entries[key]));const internalIds=new Set(keys)
 // Overview subsets retain their exact world arrangement. Smaller operators get a fresh vertical layout.
 const exact=e.overview||id==='root'
 if(exact)parts.forEach(p=>{const owner=model.entries[model.entries[p.id].parent??''];if(owner?.overview&&owner.children[0]===p.id)p.title=`${owner.title} · ${p.title}`})
 // Explicit operator layouts (e.g. AdaIN side conditioning) remain in local coordinates.
 const links=edges.filter(link=>internalIds.has(link.from)&&internalIds.has(link.to))
 parts.push(...context)
 const visible=(key:string,input:boolean)=>keys.find(k=>pathFor(model,key).includes(k))??(input?(keys.find(k=>/\/(input|conv|normalize)$/.test(k))??keys[0]):keys.at(-1)!)
 for(const link of contextEdges)links.push({...link,from:link.from.startsWith('port:')?link.from:visible(link.from,false),to:link.to.startsWith('port:')?link.to:visible(link.to,true)})
 return {title:e.title,tensor:e.shape,note:e.implementation,parts,edges:links,layout:exact?'overview':undefined}
}
export class ModelBuilder {
 entries:Record<string,Entry>={}
 add(id:string,parent:string|undefined,title:string,glyph:Glyph,shape:string,purpose:string,implementation:string,formula:string,file:string,line:number,symbol=id,math?:MathSpec){
  const e:Entry={id,parent,title,glyph,shape,purpose,implementation,formula,file,line,symbol,children:[],edges:[],math};this.entries[id]=e
  if(parent)this.entries[parent].children.push(id)
  return e
 }
 link(parent:string,from:string,to:string,label?:string,residual=false){this.entries[parent].edges.push({from,to,label,residual})}
 chain(parent:string){const e=this.entries[parent];e.children.slice(1).forEach((id,i)=>this.link(parent,e.children[i],id))}
}
