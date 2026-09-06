import type { AnatomyGraph } from './anatomy'
import type { Point3 } from './topology'

/** Containment bounds exclude context references. Ports occupy separate outside rails. */
export function detailLayout(graph:AnatomyGraph,compact=false){
  const internal=graph.parts.filter(p=>!p.role),ids=new Set(internal.map(p=>p.id))
  const depth=new Map(internal.map(p=>[p.id,0])),pending=new Set(ids)
  for(let pass=0;pass<internal.length;pass++)for(const p of internal){
    if(!pending.has(p.id))continue
    const incoming=graph.edges.filter(e=>e.to===p.id&&!e.residual&&ids.has(e.from))
    if(incoming.every(e=>!pending.has(e.from))){depth.set(p.id,incoming.length?1+Math.max(...incoming.map(e=>depth.get(e.from)!)):0);pending.delete(p.id)}
  }
  const positions=new Map<string,Point3>()
  for(const p of internal){
    const d=depth.get(p.id)!,tier=Math.floor(d/4),column=d%4,siblings=internal.filter(v=>depth.get(v.id)===d),branch=siblings.indexOf(p)-(siblings.length-1)/2
    positions.set(p.id,p.position&&!compact?[p.position[0]*1.25,p.position[1]*2.1,p.position[2]*1.3]:[(tier%2?1.5-column:column-1.5)*6,-tier*7,Math.sin(column*Math.PI/2)*2.8+branch*7])
  }
  const values=[...positions.values()]
  const min=([0,1,2] as const).map(i=>Math.min(...values.map(v=>v[i]))-[3.6,2.6,3.6][i]) as Point3
  const max=([0,1,2] as const).map(i=>Math.max(...values.map(v=>v[i]))+[3.6,2.6,3.6][i]) as Point3
  for(const role of ['input','output'] as const){
    const ports=graph.parts.filter(p=>p.role===role)
    ports.forEach((p,i)=>positions.set(p.id,compact?[(i%2-.5)*10,role==='input'?max[1]+5+Math.floor(i/2)*5:min[1]-5-Math.floor(i/2)*5,max[2]+1]:[role==='input'?min[0]-7:max[0]+7,(min[1]+max[1])/2+((ports.length-1)/2-i)*5.5,max[2]+1]))
  }
  return {positions,min,max}
}
