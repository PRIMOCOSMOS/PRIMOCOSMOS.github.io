import type {Run,Step} from './engine'
import type {Position3} from '../mmhvae/crystalPrimitives'

export interface FunctionalRegion {id:string;title:string;group:string;steps:string[]}
/** Source-level functions own regions; arithmetic microsteps never create regions. */
export function functionalRegions(run:Run,selected:Step[]=run.steps):FunctionalRegion[]{
 const groups=[...new Set(selected.map(s=>s.group))];
 if(groups.length>1)return groups.map(group=>({id:`group:${group}`,title:String(selected.find(s=>s.group===group)?.settings.groupTitle??group),group,steps:selected.filter(s=>s.group===group).map(s=>s.id)}));
 const scores=selected.findIndex(s=>s.settings.transposeB===true),values=selected.findIndex((s,i)=>i>scores&&s.settings.transposeB===false);
 if(scores<0||values<0)return groups.map(group=>({id:`group:${group}`,title:String(selected.find(s=>s.group===group)?.settings.groupTitle??group),group,steps:selected.map(s=>s.id)}));
 const group=groups[0];
 return [[0,scores,'查询、键与值投影'],[scores,values,'相似度与注意力归一化'],[values,selected.length,'上下文汇聚与输出投影']].map(([start,end,title],i)=>({id:`region:${encodeURIComponent(group)}:${i}`,title:String(title),group,steps:selected.slice(Number(start),Number(end)).map(s=>s.id)}));
}
export function regionSteps(run:Run,scope:string){
 if(!scope.startsWith('region:'))return undefined;
 const [,group]=scope.split(':');return functionalRegions(run,run.steps.filter(s=>s.group===decodeURIComponent(group))).find(r=>r.id===scope);
}
export interface RegionExtent {id:string;min:Position3;max:Position3}
/** Continuous slabs cover the whole trunk, including the spaces between layers.
 * Geometry picking can override a slab at an interleaved branch; the fallback
 * never leaves an unassigned gap between functional regions. */
export function contiguousRegions(extents:RegionExtent[]):RegionExtent[]{
 const sorted=[...extents].sort((a,b)=>(b.min[1]+b.max[1])-(a.min[1]+a.max[1]));
 if(!sorted.length)return [];
 const minX=Math.min(...sorted.map(e=>e.min[0]))-1.2,maxX=Math.max(...sorted.map(e=>e.max[0]))+1.2,minZ=Math.min(...sorted.map(e=>e.min[2]))-1.2,maxZ=Math.max(...sorted.map(e=>e.max[2]))+1.2;
 const boundaries=sorted.slice(1).map((e,i)=>(sorted[i].min[1]+sorted[i].max[1]+e.min[1]+e.max[1])/4);
 return sorted.map((e,i)=>({id:e.id,min:[minX,i===sorted.length-1?Math.min(...sorted.map(v=>v.min[1]))-1.2:boundaries[i],minZ],max:[maxX,i===0?Math.max(...sorted.map(v=>v.max[1]))+1.2:boundaries[i-1],maxZ]}));
}
